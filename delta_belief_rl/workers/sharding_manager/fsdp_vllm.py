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

from collections import OrderedDict
import inspect
import logging
import os
import time
from dataclasses import asdict

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.model import check_exclude_modules, check_target_modules
from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.torch_functional import check_device_is_available
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge

from .base import BaseShardingManager

# Override the verl sleep level, for lora sleep level 2 does not work
VLLM_SLEEP_LEVEL = 1

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPVLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        load_format: str = "dummy_hf",
        layered_summon: bool = True,
        seed: int = 42,
        adapter_name: str = None,
    ):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer intialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        self.model_runner = (
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            if self.inference_engine
            else None
        )

        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon
        self.seed = seed
        self.adapter_name = adapter_name

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(
                gen_dp_rank + self.seed + 1000
            )  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = "dummy" not in load_format
        logger.info(
            f"FSDPVLLMShardingManager initialized with base_sync_done={self.base_sync_done}, load_format={self.load_format}, adapter_name={self.adapter_name}"
        )
        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        logger.info(
            f"[DEBUG ENTER] Starting __enter__ for adapter '{self.adapter_name}'"
        )
        logger.info(
            f"[DEBUG ENTER] base_sync_done={self.base_sync_done}, layered_summon={self.layered_summon}"
        )

        def __collect_lora_params(adapter_name: str) -> OrderedDict:
            """
            collect lora params or full params if base model is not ready in vllm
            work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
            """
            from peft.utils.save_and_load import get_peft_model_state_dict

            lora_params = OrderedDict()
            peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
            # select only the requested adapter
            if hasattr(peft_model, "set_adapter"):
                peft_model.set_adapter(
                    adapter_name
                )  # no-op on newer versions if already active
                logger.info(
                    f"Set active adapter to '{adapter_name}' for parameter collection"
                )

            if fsdp_version(self.module) > 0:
                if self.layered_summon:
                    from delta_belief_rl.utils.lora_adapters import lora_params_all

                    if not self.base_sync_done:
                        raise ValueError(
                            "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                            "rollout.load_format=safetensors"
                        )
                    lora_params = lora_params_all(
                        self.module, adapter_name=adapter_name
                    )
                else:
                    with FSDP.summon_full_params(self.module, writeback=False):
                        if self.base_sync_done:
                            lora_params = get_peft_model_state_dict(
                                peft_model, adapter_name=adapter_name
                            )
                            lora_params = {
                                name: param.full_tensor().detach().cpu()
                                if hasattr(param, "full_tensor")
                                else param.detach().cpu()
                                for name, param in lora_params.items()
                            }
                        else:
                            model = peft_model.base_model.model
                            orig_dev = (
                                "cpu"
                                if "cpu" in str(next(model.parameters()).device)
                                else get_device_name()
                            )
                            model = model.to("cpu")
                            for name, param in model.state_dict().items():
                                if any(x in name for x in ["_flat_param", "lora_"]):
                                    continue
                                name = name.replace(
                                    "_fsdp_wrapped_module.", ""
                                ).replace(".base_layer", "")
                                lora_params[name] = (
                                    param.full_tensor().detach().cpu()
                                    if hasattr(param, "full_tensor")
                                    else param.detach().cpu()
                                )
                            model = model.to(orig_dev)
                    get_torch_device().empty_cache()
            else:
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(
                        peft_model, adapter_name=adapter_name
                    )
                else:
                    model = peft_model.base_model.model
                    orig_dev = (
                        "cpu"
                        if "cpu" in str(next(model.parameters()).device)
                        else get_device_name()
                    )
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(
                            ".base_layer", ""
                        )
                        lora_params[name] = param.detach().cpu()
                    model = model.to(orig_dev)

            # sanity checks
            if self.base_sync_done:
                assert len(lora_params) > 0, (
                    f"[LoRA sanity] collected zero LoRA tensors for adapter '{adapter_name}'"
                )
                for _k, _v in lora_params.items():
                    assert isinstance(_v, torch.Tensor), (
                        f"[LoRA sanity] {_k} not a tensor"
                    )
                    assert not _v.is_cuda, f"[LoRA sanity] {_k} must be CPU; got CUDA"
                    assert _v.isfinite().all().item(), (
                        f"[LoRA sanity] {_k} contains NaN/Inf"
                    )

            return lora_params

        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        get_torch_device().empty_cache()

        log_gpu_memory_usage(
            "Before state_dict() in sharding manager memory", logger=logger
        )
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_config = None
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):
            assert self.adapter_name in peft_model.peft_config, (
                f"Adapter '{self.adapter_name}' not found in peft_config keys={list(peft_model.peft_config.keys())}"
            )
            peft_config = peft_model.peft_config.get(self.adapter_name)
            logger.info(
                f"[DEBUG ENTER] Using adapter '{self.adapter_name}' with config: {peft_config}"
            )
            logger.info("[DEBUG ENTER] About to collect LoRA parameters...")
            params = __collect_lora_params(adapter_name=self.adapter_name)
            logger.info(f"[DEBUG ENTER] Collected {len(params)} LoRA parameter tensors")
            # Log first few param names to verify they're LoRA params
            param_names = list(params.keys())[:5]
            logger.info(f"[DEBUG ENTER] First few param names: {param_names}")

            # CRITICAL: Strip PEFT prefixes from parameter names for vLLM compatibility
            # vLLM expects clean names like "model.layers.0.self_attn.q_proj.lora_A.weight"
            # but PEFT returns "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            cleaned_params = OrderedDict()
            for name, tensor in params.items():
                # Remove PEFT wrapper prefixes
                clean_name = name.replace("base_model.model.", "").replace(
                    "base_model.", ""
                )
                cleaned_params[clean_name] = tensor
            params = cleaned_params
            logger.info(
                f"[DEBUG ENTER] After cleaning, first few names: {list(params.keys())[:5]}"
            )

            # Check if B matrices are actually zero (as they should be for proper initialization)
            b_matrix_stats = []
            for name, tensor in list(params.items())[:6]:  # Check first 3 A/B pairs
                if "lora_B" in name:
                    b_abs_max = tensor.abs().max().item()
                    b_abs_mean = tensor.abs().mean().item()
                    b_matrix_stats.append(
                        f"{name}: max={b_abs_max:.6f}, mean={b_abs_mean:.6f}"
                    )
            logger.info(
                f"[DEBUG ENTER] Sample lora_B matrix stats (should be ~0): {b_matrix_stats}"
            )
        else:
            logger.info(
                "[DEBUG ENTER] No PEFT config found, collecting full model params"
            )
            params = self.module.state_dict()

        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage(
            "After state_dict() in sharding manager memory", logger=logger
        )

        # vllm need to set _set_allocator_settings to False
        logger.debug("fsdp vllm sharding_manager _set_allocator_settings to False")
        set_expandable_segments(False)

        if self.rollout_config.free_cache_engine:
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=["weights"])
            else:
                self.inference_engine.wake_up()

        # update model params
        self.update_params(params, peft_config=peft_config)
        log_gpu_memory_usage(
            "After sync model weights in sharding manager", logger=logger
        )
        del params

        get_torch_device().empty_cache()

        if (
            self.rollout_config.free_cache_engine
            and "tags" in inspect.signature(self.inference_engine.wake_up).parameters
        ):
            self.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage(
            "After del state_dict and empty_cache in sharding manager", logger=logger
        )

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self):
        if self.rollout_config.free_cache_engine:
            self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # _set_allocator_settings to True is required by fsdp2 to avoid oom
        logger.debug("fsdp vllm sharding_manager _set_allocator_settings to True")
        set_expandable_segments(True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        """Update model parameters in the vLLM inference engine.

        Synchronizes parameters from the FSDP training model to the vLLM inference
        engine, handling both full model parameters and LoRA adapters with proper
        device placement and memory management.

        Args:
            updated_params (dict): Dictionary of parameter names to tensor values.
            peft_config (optional): PEFT configuration for LoRA adapters.
        """
        logger.info(
            f"[DEBUG UPDATE_PARAMS] Called with {len(updated_params)} params, peft_config={'Yes' if peft_config else 'No'}"
        )
        model = self.model_runner.model
        if peft_config:
            logger.info(
                f"[DEBUG UPDATE_PARAMS] Preparing to sync LoRA adapter '{self.adapter_name}' to vLLM... with base_sync_done={self.base_sync_done}"
            )
            if self.base_sync_done:
                logger.info(
                    f"[DEBUG UPDATE_PARAMS] Syncing LoRA adapter '{self.adapter_name}' to vLLM..."
                )
                lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                logger.info(
                    f"[DEBUG UPDATE_PARAMS] [FSDPvLLM] LoRA adapter '{self.adapter_name}' assigned int_id={lora_int_id} for vLLM"
                )

                # Log some param info for debugging
                param_names = list(updated_params.keys())[:3]
                param_shapes = [updated_params[k].shape for k in param_names]
                logger.info(
                    f"[DEBUG UPDATE_PARAMS] Sample params: {list(zip(param_names, param_shapes))}"
                )

                lora_reqest = TensorLoRARequest(
                    lora_name=self.adapter_name,
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                logger.info("[DEBUG UPDATE_PARAMS] About to call add_lora...")
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(
                    f"[DEBUG UPDATE_PARAMS] vLLM load weights, loaded_params: {len(updated_params)}"
                )

                # Validate that the adapter was actually loaded
                loaded_ids = list(self.inference_engine.llm_engine.list_loras())
                logger.info(f"[DEBUG UPDATE_PARAMS] Loaded IDs from vLLM: {loaded_ids}")
                if lora_int_id not in loaded_ids:
                    raise RuntimeError(
                        f"LoRA adapter with int_id={lora_int_id} was not loaded into vLLM! "
                        f"Available IDs: {loaded_ids}"
                    )

                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                logger.info(
                    f"[FSDPvLLM] Successfully loaded LoRA adapter '{self.adapter_name}' with int_id={lora_int_id}"
                )
                logger.info(f"[FSDPvLLM] All loaded adapters in vLLM: {loaded_ids}")
                logger.warning("[DEBUG] Finished syncing LoRA adapter to vLLM")
                return
            else:

                def replace_lora_wrapper(k):
                    """Replace LoRA parameter keys with base layer equivalents.

                    Transforms LoRA parameter names to their corresponding base layer
                    names for proper weight loading in vLLM when base model sync is not done.

                    Args:
                        k (str): Original parameter key name.

                    Returns:
                        str: Transformed parameter key for base layer.
                    """
                    stacked_params = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                    if k.endswith(".weight"):
                        module_k = k[: -len(".weight")]
                        if check_exclude_modules(peft_config, module_k):
                            return k
                        elif any(
                            [module_k.endswith(s) for s in stacked_params]
                        ) or check_target_modules(peft_config, module_k):
                            return f"{module_k}.base_layer.weight"
                    if k.endswith(".bias"):
                        module_k = k[: -len(".bias")]
                        if check_exclude_modules(peft_config, module_k):
                            return k
                        elif any(
                            [module_k.endswith(s) for s in stacked_params]
                        ) or check_target_modules(peft_config, module_k):
                            return f"{module_k}.base_layer.bias"
                    return k

                updated_params = {
                    replace_lora_wrapper(k): v for k, v in updated_params.items()
                }

        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(model)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(
            (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor()
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in updated_params.items()
            )
        )
        self.base_sync_done = True
        logger.info(
            f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}"
        )


class VLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        inference_engine: LLM,
        device_mesh: DeviceMesh = None,
        seed: int = 42,
    ):
        self.inference_engine = inference_engine
        self.model_runner = (
            inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            if inference_engine
            else None
        )
        self.device_mesh = device_mesh
        self.seed = seed

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(
                gen_dp_rank + self.seed + 1000
            )  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def __enter__(self):
        get_torch_device().empty_cache()

        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def __exit__(self):
        get_torch_device().empty_cache()
        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
