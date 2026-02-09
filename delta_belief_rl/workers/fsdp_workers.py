# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import datetime

import logging
import os
import warnings
from typing import Union
from dataclasses import asdict
import re
import json
import numpy as np

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.utils.save_and_load import set_peft_model_state_dict

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.logger.aggregate_logger import log_with_rank
from delta_belief_rl.llm_agent.prompts import get_judge_prompt, get_judge_system_prompt
from delta_belief_rl.utils.lora_adapters import _use_adapter

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# Find all <answer>…</answer> spans, case‐insensitive, across lines
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# Remove any characters except letters, digits, dot, comma, and space
CLEAN_PATTERN = re.compile(r"[^a-zA-Z0-9\., ]+")


def create_device_mesh(world_size, fsdp_size):
    # Ensure CUDA is properly initialized before creating device mesh
    import torch

    if torch.cuda.is_available():
        # Force proper CUDA initialization
        torch.cuda.init()

    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(
            f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2"
        )
    return sharding_strategy


class JudgeRolloutWorker(Worker):
    """This worker is used as a standaline rolout rollout for PPO training."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        import torch.distributed

        # Ensure CUDA is available and initialized before proceeding
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This worker requires GPU.")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name
        if rollout_name == "vllm":
            from delta_belief_rl.workers.rollout.vllm_rollout import (
                vllm_mode,
                vLLMAsyncRollout,
                vLLMRollout,
            )
            from delta_belief_rl.workers.sharding_manager.fsdp_vllm import (
                VLLMShardingManager,
            )

            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            local_path = copy_to_local(self.config.model.path)

            if vllm_mode == "spmd":
                vllm_rollout_cls = (
                    vLLMRollout
                    if self.config.rollout.mode == "sync"
                    else vLLMAsyncRollout
                )
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.model_hf_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                )
            else:
                raise NotImplementedError(
                    "vllm_mode must be 'spmd' upgrade your vllm version"
                )
        else:
            raise NotImplementedError(f"Rollout {rollout_name} is not supported")

        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"

        rollout_sharding_manager = VLLMShardingManager(
            inference_engine=rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            seed=self.config.get("seed", 42),
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.model import get_generation_config

        trust_remote_code = self.config.model.get("trust_remote_code", False)
        local_path = copy_to_local(self.config.model.path)
        input_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.input_tokenizer = hf_tokenizer(
            input_tokenizer_local_path, trust_remote_code=trust_remote_code
        )

        self.model_hf_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        self.rollout, self.rollout_sharding_manager = self._build_rollout(
            trust_remote_code=trust_remote_code
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_rollout(self):
        self.rollout_sharding_manager.__enter__()  # enter the sharing manager context

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def exit_rollout(self):
        self.rollout_sharding_manager.__exit__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        prompts = self._switch_chat_template(prompts)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
            "validate": True,
        }

        prompts.meta_info.update(meta_info)

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

        del prompts
        output = output.to("cpu")

        output = self._decode_output(output)

        return output

    def _decode_output(self, output: DataProto) -> DataProto:
        output_str = self.tokenizer.batch_decode(
            output.batch["responses"], skip_special_tokens=True
        )

        # Log only the first few judge responses to avoid log spam
        for idx, output_s in enumerate(output_str[:3]):
            print(
                f"[DEBUG] Judge output cot + answer, idx {idx}:\n{output_s}\n{'=' * 60}"
            )

        processed = []
        for resp in output_str:
            if self.config.cot or self.config.thinking:
                # findall returns a list of all inner matches
                matches = ANSWER_PATTERN.findall(resp)
                if matches:
                    # join them in case there are multiple <answer> blocks
                    answer = " ".join(m.strip() for m in matches)
                else:
                    # if no <answer> tags found, use the whole response
                    answer = ""
            else:
                answer = resp

            # clean out unwanted chars
            answer = CLEAN_PATTERN.sub("", answer).strip()

            processed.append(answer)

        # store as object array of strings
        output.non_tensor_batch["answers"] = np.array(processed, dtype=object)
        output.non_tensor_batch["responses_str"] = np.array(output_str, dtype=object)
        return output

    def _switch_chat_template(self, data: DataProto) -> DataProto:
        max_length = self.config.rollout.get("prompt_length")
        gt_str = data.non_tensor_batch["golden_answers"]

        # support both keys; upstream _ask_question uses 'scenarios'
        scenarios = data.non_tensor_batch.get("scenarios", None)
        history = data.meta_info.get("history", None)

        prompt_input_ids = []
        prompt_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            response = self.input_tokenizer.decode(
                data.batch["input_ids"][i], skip_special_tokens=True
            )
            if history is not None:
                chat = [
                    {
                        "role": "system",
                        "content": get_judge_system_prompt(
                            repeated=True,
                            env=self.config.get("env", "twenty_questions"),
                        ),
                    },
                    {
                        "role": "user",
                        "content": get_judge_prompt(
                            gt_str[i],
                            response,
                            history[i],
                            thinking=self.config.thinking,
                            cot=self.config.cot,
                            env=self.config.get("env", "twenty_questions"),
                        ),
                    },
                ]
            else:
                chat = [
                    {
                        "role": "system",
                        "content": get_judge_system_prompt(
                            env=self.config.get("env", "twenty_questions")
                        ),
                    },
                    {
                        "role": "user",
                        "content": get_judge_prompt(
                            gt_str[i],
                            response,
                            thinking=self.config.thinking,
                            cot=self.config.cot,
                            env=self.config.get("env", "twenty_questions"),
                            scenario=scenarios[i] if scenarios is not None else None,
                        ),
                    },
                ]
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.config.thinking,
            )

            model_inputs = self.tokenizer(
                prompt_with_chat_template,
                return_tensors="pt",
                add_special_tokens=False,
                padding=False,
            )

            input_ids, attention_mask = (
                model_inputs.pop("input_ids"),
                model_inputs.pop("attention_mask"),
            )

            try:
                input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.config.get("truncation", "error"),
                )  # truncate from the right
            except NotImplementedError as e:
                prompt_str = self.tokenizer.decode(
                    input_ids[0], skip_special_tokens=True
                )
                logger.error(
                    f"Error in processing prompt {i} with length {input_ids.shape[1]}: {prompt_str}"
                )
                raise e

            prompt_input_ids.append(input_ids)
            prompt_attention_mask.append(attention_mask)

        input_ids = torch.cat(prompt_input_ids, dim=0)
        del prompt_input_ids
        attention_mask = torch.cat(prompt_attention_mask, dim=0)
        del prompt_attention_mask

        position_ids = compute_position_id_with_mask(attention_mask)

        prompts = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return DataProto.from_dict(tensors=prompts)


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        self.val_only = self.config.get("val_only", False)
        self.loading_chkpt = self.config.get("resume_mode", "disable") != "disable"
        import torch.distributed

        # Ensure CUDA is available and initialized before proceeding
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This worker requires GPU.")

        # Force CUDA initialization to ensure device is ready
        torch.cuda.current_device()

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(
                    seconds=self.config.get("nccl_timeout", 600)
                ),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor",
                dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )
        else:
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.rank, is_collect=True
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0
        self.active_adapter_name = None  # to store the active adapter name

        self.role = role
        logger.info(f"ActorRolloutRefWorker role: {self.role}")
        logger.info(f"lora state {self._is_lora} with rank {self._lora_rank}")
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.gen_n
            self.config.actor.ppo_mini_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = (
                    self.config.actor.ppo_micro_batch_size
                )

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert (
                    self.config.actor.ppo_mini_batch_size
                    % self.config.actor.ppo_micro_batch_size_per_gpu
                    == 0
                ), (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert (
                    self.config.actor.ppo_mini_batch_size
                    // self.config.actor.ppo_micro_batch_size_per_gpu
                    > 0
                ), (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if (
            self._is_rollout
            and self.config.rollout.log_prob_micro_batch_size is not None
        ):
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size
            )
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.ref.log_prob_micro_batch_size_per_gpu = (
                self.config.ref.log_prob_micro_batch_size
            )

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        use_fused_kernels=False,
        role="actor",
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import (
            get_generation_config,
            print_model_size,
            update_model_config,
        )
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = copy_to_local(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )

        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        if self._is_actor or self._is_ref:
            init_context = get_init_weight_context_manager(
                use_meta_tensor=not actor_model_config.tie_word_embeddings,
                mesh=self.device_mesh,
            )

            with init_context(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if (
                    type(actor_model_config)
                    in AutoModelForVision2Seq._model_mapping.keys()
                ):
                    actor_module_class = AutoModelForVision2Seq
                else:
                    actor_module_class = AutoModelForCausalLM

                actor_module = actor_module_class.from_pretrained(
                    pretrained_model_name_or_path=local_path,
                    torch_dtype=torch_dtype,
                    config=actor_model_config,
                    trust_remote_code=trust_remote_code,
                )

                # Apply Liger kernel to the model if use_liger is set to True
                if use_liger:
                    from liger_kernel.transformers.monkey_patch import (
                        _apply_liger_kernel_to_instance,
                    )

                    _apply_liger_kernel_to_instance(model=actor_module)

                fused_kernel_options = self.config.model.get(
                    "fused_kernel_options", None
                )
                fused_kernels_backend = (
                    fused_kernel_options.get("impl_backend", None)
                    if fused_kernel_options is not None
                    else None
                )

                apply_monkey_patch(
                    model=actor_module,
                    use_remove_padding=use_remove_padding,
                    ulysses_sp_size=self.ulysses_sequence_parallel_size,
                    use_fused_kernels=use_fused_kernels,
                    fused_kernels_backend=fused_kernels_backend,
                )

                # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
                actor_module.to(torch_dtype)

                if enable_gradient_checkpointing:
                    actor_module.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )

                if self._is_lora:
                    logger.info("Applying LoRA to actor module")
                    actor_module.enable_input_require_grads()

                    adapters = self.config.model.get("adapters", None)
                    logger.info(f"adapters: {adapters}")

                    if adapters:
                        first = adapters[0]
                        self.active_adapter_name = first["name"]
                        first_cfg = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            r=first["r"],
                            lora_alpha=first["alpha"],
                            target_modules=convert_to_regular_types(
                                first.get("target_modules")
                            ),
                            exclude_modules=convert_to_regular_types(
                                first.get("exclude_modules")
                            ),
                            bias="none",
                        )
                        actor_module = get_peft_model(
                            actor_module, first_cfg, adapter_name=first["name"]
                        )

                        for ad in adapters[1:]:
                            ad_cfg = LoraConfig(
                                task_type=TaskType.CAUSAL_LM,
                                r=ad["r"],
                                lora_alpha=ad["alpha"],
                                target_modules=convert_to_regular_types(
                                    ad.get("target_modules")
                                ),
                                exclude_modules=convert_to_regular_types(
                                    ad.get("exclude_modules")
                                ),
                                bias="none",
                            )
                            actor_module.add_adapter(ad["name"], ad_cfg)

                        actor_module.set_adapter(self.active_adapter_name)
                        logger.info(f"Active LoRA adapter: {self.active_adapter_name}")

                        all_names = [a["name"] for a in adapters]
                        inactive_names = [
                            n for n in all_names if n != self.active_adapter_name
                        ]

                        if inactive_names:
                            for module in actor_module.modules():
                                if isinstance(module, LoraLayer):
                                    # Each LoraLayer keeps dicts: lora_A[name], lora_B[name]
                                    srcA = module.lora_A[
                                        self.active_adapter_name
                                    ].weight.data
                                    srcB = module.lora_B[
                                        self.active_adapter_name
                                    ].weight.data

                                    for name in inactive_names:
                                        # Shapes are guaranteed equal by PEFT for a given layer
                                        module.lora_A[name].weight.data.copy_(srcA)
                                        module.lora_B[name].weight.data.copy_(srcB)
                                        # If the layer uses per-adapter scaling maps, keep them aligned too
                                        if hasattr(
                                            module, "lora_dropout"
                                        ) and isinstance(
                                            getattr(module, "lora_dropout"), dict
                                        ):
                                            # usually same dropout is shared; nothing to copy
                                            pass

                        for n, p in actor_module.named_parameters():
                            # PEFT names adapter params like "...lora_A.<adapter>..." / "...lora_B.<adapter>..."
                            if any(f".{iname}." in n for iname in inactive_names):
                                p.requires_grad_(False)
                            elif f".{self.active_adapter_name}." in n:
                                p.requires_grad_(True)
                            else:
                                # Base model params: keep frozen for LoRA fine-tuning
                                # (flip to True only if you want full fine-tuning)
                                p.requires_grad_(False)

                        try:
                            actor_module.print_trainable_parameters()
                        except Exception:
                            pass
                    else:
                        # Backward‑compatible single‑adapter path
                        lora_config = {
                            "task_type": TaskType.CAUSAL_LM,
                            "r": self.config.model.lora_rank,
                            "lora_alpha": self.config.model.lora_alpha,
                            "target_modules": convert_to_regular_types(
                                self.config.model.target_modules
                            ),
                            "exclude_modules": convert_to_regular_types(
                                self.config.model.exclude_modules
                            ),
                            "bias": "none",
                        }
                        actor_module = get_peft_model(
                            actor_module, LoraConfig(**lora_config)
                        )

            torch.distributed.barrier()
            if self.rank == 0:
                print_model_size(actor_module)

            log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

            # We wrap FSDP for rollout as well
            mixed_precision_config = fsdp_config.get("mixed_precision", None)
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(
                    mixed_precision_config.get("param_dtype", "bf16")
                )
                reduce_dtype = PrecisionType.to_dtype(
                    mixed_precision_config.get("reduce_dtype", "fp32")
                )
                buffer_dtype = PrecisionType.to_dtype(
                    mixed_precision_config.get("buffer_dtype", "fp32")
                )
            else:
                param_dtype = PrecisionType.to_dtype(
                    self.config.actor.get("dtype", "float16")
                )
                reduce_dtype = torch.float32
                buffer_dtype = torch.float32

            mixed_precision = MixedPrecision(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                buffer_dtype=buffer_dtype,
            )

            auto_wrap_policy = get_fsdp_wrap_policy(
                module=actor_module,
                config=fsdp_config.get("wrap_policy", None),
                is_lora=self.config.model.get("lora_rank", 0) > 0,
            )

            if self._is_rollout and self.config.rollout.name == "hf":
                auto_wrap_policy = None

            if self.rank == 0:
                print(f"wrap_policy: {auto_wrap_policy}")

            fsdp_mesh = self.device_mesh
            sharding_strategy = get_sharding_strategy(fsdp_mesh)

            cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=fsdp_config.get("use_orig_params", False),
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )

            log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)
        else:
            # inference only state
            actor_module_fsdp = None

        if self._is_actor and optim_config is not None:
            from verl.utils.torch_functional import (
                get_constant_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
            )

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            logger.info(
                f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}"
            )

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                )
            else:
                raise NotImplementedError(
                    f"Warmup style {warmup_style} is not supported"
                )

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(
                module=self.actor_module_fsdp, config=self.config.rollout
            )
            rollout_sharding_manager = BaseShardingManager()

        elif rollout_name == "vllm":
            from delta_belief_rl.workers.rollout.vllm_rollout import (
                vllm_mode,
                vLLMAsyncRollout,
                vLLMRollout,
            )
            from delta_belief_rl.workers.sharding_manager.fsdp_vllm import (
                FSDPVLLMShardingManager,
            )

            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            local_path = copy_to_local(self.config.model.path)

            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                )
            elif vllm_mode == "spmd":
                lora_kwargs = (
                    {
                        "lora_kwargs": {
                            "enable_lora": True,
                            "max_loras": 1,
                            "max_lora_rank": self.config.model.lora_rank,
                        }
                    }
                    if self.config.model.lora_rank > 0
                    else {}
                )
                logger.info(f"lora kwargs {lora_kwargs}")

                vllm_rollout_cls = (
                    vLLMRollout
                    if self.config.rollout.mode == "sync"
                    else vLLMAsyncRollout
                )
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(
                f"After building {rollout_name} rollout", logger=logger
            )

            if self._is_actor:
                full_params = torch.distributed.get_world_size() == 1
                rollout_sharding_manager = FSDPVLLMShardingManager(
                    module=self.actor_module_fsdp,
                    inference_engine=rollout.inference_engine,
                    model_config=self.actor_model_config,
                    rollout_config=self.config.rollout,
                    full_params=full_params,
                    device_mesh=rollout_device_mesh,
                    offload_param=self._is_offload_param,
                    load_format=self.config.rollout.load_format,
                    layered_summon=self.config.rollout.get("layered_summon", False),
                    adapter_name=self.active_adapter_name,
                )
            else:
                from delta_belief_rl.workers.sharding_manager.fsdp_vllm import (
                    VLLMShardingManager,
                )

                rollout_sharding_manager = VLLMShardingManager(
                    inference_engine=rollout.inference_engine,
                    device_mesh=rollout_device_mesh,
                )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            from verl.workers.sharding_manager.fsdp_sglang import (
                FSDPSGLangShardingManager,
            )

            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            local_path = copy_to_local(self.config.model.path)

            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
            )
            log_gpu_memory_usage(
                f"After building {rollout_name} rollout", logger=logger
            )

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )

            log_gpu_memory_usage("After building sharding manager", logger=logger)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from delta_belief_rl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )

            if self.actor_module_fsdp is not None:
                # get the original unwrapped module
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                    log_gpu_memory_usage(
                        "After offload actor model during init", logger=logger
                    )

                if self._is_offload_optimizer:
                    offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                    log_gpu_memory_usage(
                        "After offload actor optimizer during init", logger=logger
                    )
        # load from checkpoint
        if self._is_actor and (not self.val_only or self.loading_chkpt):
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )
            assert hasattr(self.ref_policy, "actor_module"), (
                f"Ref model must have actor_module as an attribute, current ref model: {self.ref_policy.actor_module}"
            )

        if self._is_actor and (not self.val_only or self.loading_chkpt):
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor
                if self.processor is not None
                else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
            )

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops
                * self.config.actor.ppo_epochs
                / promised_flops
                / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (
                1024**3
            )

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during update_actor", logger=logger
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor optimizer during update_actor", logger=logger
            )

        # clear kv cache
        torch.cuda.empty_cache()

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_ema(self, beta: float = 0.0):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        from delta_belief_rl.utils.lora_adapters import (
            _peft_under_fsdp,
            ema_update_adapter,
            _maybe_summon_full_params_if_fsdp,
            _adapter_signature,
            _adapter_distance,
        )

        # Perform EMA update once at the end of all epochs and mini-batches
        peft_model = _peft_under_fsdp(self.actor_module_fsdp)

        # Only perform EMA update if 'ema' adapter exists and if attribute peft_config exists
        if peft_model is not None and hasattr(peft_model, "peft_config"):
            if "ema" in peft_model.peft_config:
                with (
                    torch.no_grad(),
                    _maybe_summon_full_params_if_fsdp(self.actor_module_fsdp),
                ):
                    sig_before = _adapter_signature(
                        peft_model, adapter=self.active_adapter_name
                    )

                    ema_update_adapter(
                        peft_model=peft_model,
                        src_name=self.active_adapter_name,
                        dst_name="ema",
                        beta=beta,
                    )
                    sig_after = _adapter_signature(
                        peft_model, adapter=self.active_adapter_name
                    )
                    assert sig_before == sig_after, (
                        f"EMA mutated actor_lora! before={sig_before} after={sig_after}"
                    )
                    dist = _adapter_distance(
                        peft_model, src=self.active_adapter_name, dst="ema"
                    )
                    ema_metrics = {
                        "ema/beta": beta,
                        "ema/changed_layers_ratio": dist.get(
                            "changed_layers_ratio", 0.0
                        ),
                        "ema/A_rel_l2": dist.get("A/rel_l2", 0.0),
                        "ema/B_rel_l2": dist.get("B/rel_l2", 0.0),
                        "ema/A_cos": dist.get("A/cos", 0.0),
                        "ema/B_cos": dist.get("B/cos", 0.0),
                        "ema/A_linf": dist.get("A/linf", 0.0),
                        "ema/B_linf": dist.get("B/linf", 0.0),
                        "ema/EA_rel_l2": dist.get("EA/rel_l2", 0.0),
                        "ema/EB_rel_l2": dist.get("EB/rel_l2", 0.0),
                    }

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return ema_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_rollout(self):
        self.rollout_sharding_manager.__enter__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def exit_rollout(self):
        self.rollout_sharding_manager.__exit__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = (
            self.config.rollout.log_prob_micro_batch_size_per_gpu
        )
        data.meta_info["max_token_len"] = (
            self.config.rollout.log_prob_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz

        # for logprob_secret set to 1
        if "logprob_secret" in data.meta_info:
            data.meta_info["temperature"] = data.meta_info["logprob_secret"][
                "temperature"
            ]
            compute_entropy = data.meta_info["logprob_secret"]["calculate_entropy"]
        else:
            data.meta_info["temperature"] = self.config.rollout.temperature
            compute_entropy = True

        # perform recompute log_prob
        if self._is_lora:
            # Reset FSDP _is_root state to prevent AssertionError in lazy init
            # This is needed when using hybrid engine with LoRA adapters
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(self.actor.actor_module, FSDP):
                for module in self.actor.actor_module.modules():
                    if isinstance(module, FSDP) and hasattr(module, "_is_root"):
                        if module != self.actor.actor_module:  # Non-root modules
                            module._is_root = False

            # allow overrides via meta_info, fallback to "actor_lora"
            adapter_name = data.meta_info.pop(
                "adapter_name", self.active_adapter_name
            )  # default adapter is actor_lora
            logger.info(f"Adapter name for compute_log_prob: {adapter_name}")
            with (
                self.ulysses_sharding_manager,
                _use_adapter(self.actor.actor_module, adapter_name),
            ):
                data = self.ulysses_sharding_manager.preprocess_data(data)
                output, entropys = self.actor.compute_log_prob(
                    data=data, calculate_entropy=compute_entropy
                )

                # format according wether logprob update or logprob secret computation
                if "logprob_secret" in data.meta_info:
                    tensor_dict = {"log_probs": output}
                    if compute_entropy:
                        tensor_dict["entropys"] = entropys
                else:
                    tensor_dict = {"old_log_probs": output, "entropys": entropys}

                output = DataProto.from_dict(
                    tensors=tensor_dict,
                    meta_info={"temperature": data.meta_info["temperature"]},
                )
                output = self.ulysses_sharding_manager.postprocess_data(output)
        else:
            with self.ulysses_sharding_manager:
                data = self.ulysses_sharding_manager.preprocess_data(data)
                output, entropys = self.actor.compute_log_prob(
                    data=data, calculate_entropy=compute_entropy
                )

                # format according wether logprob update or logprob secret computation
                if "logprob_secret" in data.meta_info:
                    tensor_dict = {"log_probs": output}
                    if compute_entropy:
                        tensor_dict["entropys"] = entropys
                else:
                    tensor_dict = {"old_log_probs": output, "entropys": entropys}

                output = DataProto.from_dict(
                    tensors=tensor_dict,
                    meta_info={"temperature": data.meta_info["temperature"]},
                )
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during compute_log_prob", logger=logger
            )

        torch.cuda.empty_cache()

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # use ema for elicitation reward
            if "logprob_secret" in data.meta_info:
                data.meta_info["adapter_name"] = "ema"
                elicit_lp = True

            # use base model
            else:
                # if _is_lora, actor without lora applied is the ref
                data.meta_info["adapter_name"] = None
                elicit_lp = False

            data = self.compute_log_prob(data)

            # this old_log_probs is in fact ref_log_prob
            if not elicit_lp:
                data = DataProto.from_dict(
                    tensors={"ref_log_prob": data.batch["old_log_probs"]}
                )
            return data

        assert self._is_ref

        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        if "logprob_secret" in data.meta_info:
            data.meta_info["temperature"] = data.meta_info["logprob_secret"][
                "temperature"
            ]
            compute_entropy = data.meta_info["logprob_secret"]["calculate_entropy"]
        else:
            data.meta_info["temperature"] = self.config.rollout.temperature
            compute_entropy = False

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, entropys = self.ref_policy.compute_log_prob(
                data=data, calculate_entropy=compute_entropy
            )
            if "logprob_secret" in data.meta_info:
                tensor_dict = {"log_probs": output}
                if compute_entropy:
                    tensor_dict["entropys"] = entropys
            else:
                tensor_dict = {"ref_log_prob": output}

            output = DataProto.from_dict(tensors=tensor_dict)
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        torch.cuda.empty_cache()

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        from delta_belief_rl.utils.lora_adapters import layered_summon_lora_params

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        torch.distributed.barrier()

        if self._is_lora and hasattr(
            getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"
        ):
            lora_save_path = os.path.join(
                local_path, self.config.model.get("lora_path")
            )
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            adapters_to_save = list(peft_model.peft_config.keys())

            for name in adapters_to_save:
                if name not in peft_model.peft_config:
                    logger.warning(
                        f"Adapter '{name}' not found in peft_config, skipping"
                    )
                    continue

                if hasattr(peft_model, "set_adapter"):
                    peft_model.set_adapter(name)

                lora_params = layered_summon_lora_params(
                    self.actor_module_fsdp, adapter_name=name
                )

                if dist.get_rank() == 0:
                    save_dir = os.path.join(lora_save_path, name)
                    os.makedirs(save_dir, exist_ok=True)
                    cfg = peft_model.peft_config.get(name)
                    cfg = asdict(cfg)
                    cfg["task_type"] = cfg["task_type"].value
                    cfg["peft_type"] = cfg["peft_type"].value
                    cfg["target_modules"] = list(cfg["target_modules"])
                    try:
                        save_file(
                            lora_params,
                            os.path.join(save_dir, "adapter_model.safetensors"),
                        )
                        with open(
                            os.path.join(save_dir, "adapter_config.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(cfg, f, ensure_ascii=False, indent=4)
                        log_with_rank(
                            f"[rank-{self.rank}]: Saved adapter config with r={cfg.get('r')}, lora_alpha={cfg.get('lora_alpha')}, target_modules={len(cfg.get('target_modules', []))} modules",
                            rank=dist.get_rank(),
                            logger=logger,
                            log_only_rank_0=True,
                        )
                    except Exception as e:
                        log_with_rank(
                            f"Save LoRA Adapter Error ({e})",
                            rank=dist.get_rank(),
                            logger=logger,
                            log_only_rank_0=True,
                        )

                dist.barrier()
                log_with_rank(
                    f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                    rank=dist.get_rank(),
                    logger=logger,
                    log_only_rank_0=True,
                )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        # only support save and load ckpt for actor or standalone rollout
        assert self._is_actor, (
            f"Checkpoint loading is only supported for Actor _is_actor={self._is_actor}"
        )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )

        # Load LoRA adapters if present
        if self._is_lora and hasattr(
            getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"
        ):
            if self.config.model.get("skip_lora_adapter_files_on_resume", False):
                log_with_rank(
                    f"[rank-{self.rank}]: skip_lora_adapter_files_on_resume=True; skipping adapter safetensors load from checkpoint directory",
                    rank=dist.get_rank(),
                    logger=logger,
                    log_only_rank_0=True,
                )
            else:
                lora_load_path = os.path.join(
                    local_path, self.config.model.get("lora_path", "lora_adapters")
                )
                if os.path.exists(lora_load_path):
                    peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
                    adapters_to_load = list(peft_model.peft_config.keys())

                    log_with_rank(
                        f"[rank-{self.rank}]: Loading {len(adapters_to_load)} LoRA adapter(s) from: {lora_load_path}",
                        rank=dist.get_rank(),
                        logger=logger,
                        log_only_rank_0=True,
                    )

                    loaded_count = 0
                    for adapter_name in adapters_to_load:
                        adapter_dir = os.path.join(lora_load_path, adapter_name)
                        adapter_file = os.path.join(
                            adapter_dir, "adapter_model.safetensors"
                        )

                        if os.path.exists(adapter_file):
                            try:
                                log_with_rank(
                                    f"[rank-{self.rank}]: Loading adapter '{adapter_name}' from: {adapter_dir}",
                                    rank=dist.get_rank(),
                                    logger=logger,
                                    log_only_rank_0=True,
                                )

                                # Load LoRA adapter weights for this specific adapter
                                lora_state_dict = load_file(adapter_file)

                                # Set the LoRA weights into the model for this adapter
                                set_peft_model_state_dict(
                                    peft_model,
                                    lora_state_dict,
                                    adapter_name=adapter_name,
                                )
                                loaded_count += 1

                                log_with_rank(
                                    f"[rank-{self.rank}]: Successfully loaded adapter '{adapter_name}'",
                                    rank=dist.get_rank(),
                                    logger=logger,
                                    log_only_rank_0=True,
                                )
                            except Exception as e:
                                log_with_rank(
                                    f"Load LoRA Adapter '{adapter_name}' Error ({e})",
                                    rank=dist.get_rank(),
                                    logger=logger,
                                    log_only_rank_0=True,
                                )
                                raise
                        else:
                            log_with_rank(
                                f"[rank-{self.rank}]: Adapter '{adapter_name}' not found at: {adapter_file}, skipping",
                                rank=dist.get_rank(),
                                logger=logger,
                                log_only_rank_0=True,
                            )

                    log_with_rank(
                        f"[rank-{self.rank}]: Loaded {loaded_count}/{len(adapters_to_load)} LoRA adapters from checkpoint",
                        rank=dist.get_rank(),
                        logger=logger,
                        log_only_rank_0=True,
                    )
                else:
                    log_with_rank(
                        f"[rank-{self.rank}]: No LoRA adapters folder found at: {lora_load_path}, skipping LoRA loading",
                        rank=dist.get_rank(),
                        logger=logger,
                        log_only_rank_0=True,
                    )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_rollout_fanout(self, is_validate: bool):
        if is_validate:
            return self.config.rollout.val_kwargs.n
        else:
            return self.config.rollout.n


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=fsdp_size
        )

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= (
            torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        )
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size()
                // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size()
                // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = (
                self.config.forward_micro_batch_size
            )

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert (
                self.config.ppo_mini_batch_size
                % self.config.ppo_micro_batch_size_per_gpu
                == 0
            ), (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert (
                self.config.ppo_mini_batch_size
                // self.config.ppo_micro_batch_size_per_gpu
                > 0
            ), (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_to_local(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(
            tokenizer_path,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        self.processor = hf_processor(
            tokenizer_path,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        critic_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=critic_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            use_remove_padding = config.model.get("use_remove_padding", False)
            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(
                    model=critic_module,
                    ulysses_sp_size=self.ulysses_sequence_parallel_size,
                )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module, config=self.config.model.fsdp_config.wrap_policy
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # NOTE: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(
            critic_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        logger.info(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = (
            self._build_critic_model_optimizer(self.config)
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage(
                "After offload critic model during init", logger=logger
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage(
                "After offload critic optimizer during init", logger=logger
            )

        self.critic = DataParallelPPOCritic(
            config=self.config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor
            if self.processor is not None
            else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.critic_optimizer, device_id=torch.cuda.current_device()
            )

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["perf/mfu/critic"] = (
                estimated_flops
                * self.config.ppo_epochs
                / promised_flops
                / self.world_size
            )

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=fsdp_size
        )

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )
            self.tokenizer = hf_tokenizer(
                local_path,
                trust_remote_code=config.model.get("trust_remote_code", False),
            )

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if (
                config.model.get("use_remove_padding", False)
                or self.ulysses_sequence_parallel_size > 1
            ):
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(
                    model=reward_module,
                    ulysses_sp_size=self.ulysses_sequence_parallel_size,
                )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=reward_module, config=self.config.model.fsdp_config
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh,
        )

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import (
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )

        from verl.utils.ulysses import (
            gather_outpus_and_unpad,
            ulysses_pad_and_slice_inputs,
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = (
                        ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(
                    reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                ).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(
            attention_mask, dtype=scores.dtype
        )  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][
                -response_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(
                prompt_with_chat_template, return_tensors="pt", add_special_tokens=False
            )
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
        }

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = (
                    self.config.forward_max_token_len_per_gpu
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, indices = rearrange_micro_batches(
                    batch=rm_data.batch, max_token_len=max_token_len
                )
            else:
                micro_batches = rm_data.batch.split(
                    self.config.micro_batch_size_per_gpu
                )
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), (
                    f"{len(indices)} vs. {scores.size()}"
                )
                revert_indices = torch.tensor(
                    get_reverse_idx(indices), dtype=torch.long
                )
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError(
            "AsyncActorRolloutRefWorker does not support generate_sequences"
        )

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(
                f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}"
            )
        return self.rollout.execute_method(method, *args, **kwargs)
