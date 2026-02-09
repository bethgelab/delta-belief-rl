"""
Replica of verl/scripts/legacy_model_merger.py
"""

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
This script is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.

To merge FSDP checkpoints:
```sh
python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

To merge Megatron checkpoints:
```sh
python scripts/legacy_model_merger.py merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

For more details, please refer to documentation:
https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
"""

import argparse
import os
import re
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch.distributed._tensor import Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    PretrainedConfig,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from tqdm import tqdm

from verl.utils import hf_processor, hf_tokenizer


@dataclass
class ModelMergerConfig:
    operation: str  # 'merge' or 'test'
    backend: str
    local_dir: str
    hf_model_config_path: str
    target_dir: Optional[str] = "tmp"
    hf_upload_path: Optional[str] = None
    private: bool = False
    test_hf_dir: Optional[str] = None
    tie_word_embedding: bool = False
    is_value_model: bool = False
    hf_model_path: Optional[str] = None
    merge_lora: bool = False
    lora_alpha: Optional[float] = None
    lora_adapter_config: Optional[str] = None
    hf_upload: bool = field(init=False)

    def __post_init__(self):
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False


class BaseModelMerger(ABC):
    def __init__(self, config: ModelMergerConfig):
        self.config = config
        self.hf_model_config_path = config.hf_model_config_path

        if config.hf_model_path:
            print(
                "Warning: --hf_model_path is deprecated and will be removed in a future version. Currently verl will save huggingface model configuration files into checkpoint directories. Therefore, there is no need to provide --hf_model_path. "
            )
            self.hf_model_config_path = config.hf_model_path

        self.model_config = AutoConfig.from_pretrained(self.hf_model_config_path)

    def get_transformers_auto_model_class(self):
        if "ForTokenClassification" in self.model_config.architectures[0]:
            return AutoModelForTokenClassification
        elif "ForCausalLM" in self.model_config.architectures[0]:
            return AutoModelForCausalLM
        elif "ForConditionalGeneration" in self.model_config.architectures[0]:
            return AutoModelForVision2Seq

        raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    def patch_model_generation_config(self, model):
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(self.hf_model_config_path)
            except OSError:
                print(
                    f"Warning: Generation config file not found in {self.hf_model_config_path}, using a generation config created from the model config."
                )
        return model

    def _normalize_state_dict_keys_in_place(self, state_dict: dict[str, torch.Tensor]) -> None:
        for name in list(state_dict.keys()):
            key = (
                name.replace("base_model.model.", "")
                .replace(".base_layer.weight", ".weight")
                .replace(".base_layer.bias", ".bias")
            )
            if key != name:
                state_dict[key] = state_dict.pop(name)

    def _infer_lora_alpha_from_adapter_config(self) -> Optional[float]:
        import json

        if self.config.lora_adapter_config:
            adapter_config_path = Path(self.config.lora_adapter_config)
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            alpha = cfg.get("lora_alpha")
            return float(alpha) if alpha is not None else None

        local_dir = Path(self.config.local_dir)
        candidates = list(local_dir.glob("lora_adapters/**/adapter_config.json"))
        if not candidates:
            candidates = list(local_dir.glob("lora_adapter/adapter_config.json"))
        if not candidates:
            return None
        if len(candidates) > 1:
            shown = "\n".join(str(p) for p in candidates[:10])
            raise ValueError(
                "Multiple LoRA adapter configs found; please pass --lora_adapter_config to disambiguate.\n"
                f"Candidates (first 10):\n{shown}"
            )

        with open(candidates[0], "r", encoding="utf-8") as f:
            cfg = json.load(f)
        alpha = cfg.get("lora_alpha")
        return float(alpha) if alpha is not None else None

    def _get_lora_alpha_or_raise(self) -> float:
        if self.config.lora_alpha is not None:
            return float(self.config.lora_alpha)
        inferred = self._infer_lora_alpha_from_adapter_config()
        if inferred is None:
            raise ValueError(
                "LoRA merge requested but lora_alpha is unknown. Provide --lora_alpha or --lora_adapter_config "
                "(or ensure local_dir contains lora_adapters/**/adapter_config.json with lora_alpha)."
            )
        return inferred

    def merge_lora_into_base_weights(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Merge LoRA weights into base weights in-place.

        Expects LoRA tensors to be present in the state_dict (e.g. *.lora_A.* and *.lora_B.*).
        Uses scaling = lora_alpha / r.

        Returns:
            True if any LoRA weights were merged, else False.
        """

        lora_keys = [k for k in state_dict.keys() if "lora_" in k]
        if not lora_keys:
            return False

        alpha = self._get_lora_alpha_or_raise()

        groups: dict[str, dict[str, object]] = {}

        def _add(prefix: str, kind: str, key: str) -> None:
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][kind] = state_dict[key]
            groups[prefix][f"{kind}_key"] = key

        for key in list(state_dict.keys()):
            if ".lora_A." in key:
                prefix = key.split(".lora_A.", 1)[0]
                _add(prefix, "A", key)
            elif ".lora_B." in key:
                prefix = key.split(".lora_B.", 1)[0]
                _add(prefix, "B", key)
            elif ".lora_embedding_A." in key:
                prefix = key.split(".lora_embedding_A.", 1)[0]
                _add(prefix, "A", key)
            elif ".lora_embedding_B." in key:
                prefix = key.split(".lora_embedding_B.", 1)[0]
                _add(prefix, "B", key)

        merged_any = False
        for prefix, g in sorted(groups.items()):
            if "A" not in g or "B" not in g:
                raise ValueError(f"Incomplete LoRA tensors for '{prefix}': found keys {list(g.keys())}")

            A = g["A"]
            B = g["B"]
            if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
                raise TypeError(f"Unexpected LoRA tensor types for '{prefix}': {type(A)} / {type(B)}")

            if A.dim() != 2 or B.dim() != 2:
                raise ValueError(
                    f"Only 2D LoRA tensors are supported for merging. '{prefix}' got shapes {tuple(A.shape)} / {tuple(B.shape)}"
                )

            r = int(A.shape[0])
            scaling = alpha / float(r)

            base_key_candidates = [f"{prefix}.base_layer.weight", f"{prefix}.weight"]
            base_key = next((k for k in base_key_candidates if k in state_dict), None)
            if base_key is None:
                raise KeyError(
                    f"Could not find base weight for LoRA module '{prefix}'. Tried: {base_key_candidates}"
                )

            W = state_dict[base_key]
            delta = (B.to(torch.float32) @ A.to(torch.float32))
            if delta.shape != W.shape:
                if delta.T.shape == W.shape:
                    delta = delta.T
                else:
                    raise ValueError(
                        f"Delta shape mismatch for '{prefix}': base {tuple(W.shape)}, delta {tuple(delta.shape)}"
                    )

            merged = W.to(torch.float32) + (delta * scaling)
            state_dict[base_key] = merged.to(dtype=W.dtype)

            state_dict.pop(g["A_key"])  # type: ignore[arg-type]
            state_dict.pop(g["B_key"])  # type: ignore[arg-type]
            merged_any = True

        remaining = [k for k in state_dict.keys() if "lora_" in k]
        if remaining:
            shown = "\n".join(remaining[:20])
            raise ValueError(
                "Some LoRA-related keys remain after merge (unsupported format). First 20:\n" + shown
            )

        return merged_any

    def save_lora_adapter(self, state_dict: dict[str, torch.Tensor]):
        """
        Save lora adapter to safetensors.

        Returns:
            lora_path: str, the path to the lora adapter. None if no lora adapter found.

        Note:
            This function change the 'state_dict' in place.
        """
        lora_params_names = [name for name in state_dict.keys() if "lora_" in name]

        if len(lora_params_names) == 0:
            return None

        import json
        from typing import OrderedDict

        import peft
        from safetensors.torch import save_file

        lora_params = OrderedDict()
        target_modules = set()
        lora_key = None

        for name in lora_params_names:
            lora_key = name.replace(".default.weight", ".weight")
            target_modules.add(lora_key.split(".")[-3])
            lora_params[lora_key] = state_dict.pop(name)

        lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
        peft_dict = {
            "r": lora_rank,
            "lora_alpha": 64,  # lora_alpha is not set. An error should be raised to inform the user to set it manually.
            "target_modules": list(target_modules),
        }
        peft_config = peft.LoraConfig(**peft_dict).to_dict()
        peft_config["task_type"] = peft_config["task_type"].value if peft_config["task_type"] else None
        peft_config["peft_type"] = peft_config["peft_type"].value if peft_config["peft_type"] else None
        peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_path = os.path.join(self.config.target_dir, "lora_adapter")
        os.makedirs(lora_path, exist_ok=True)
        with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)
        save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))

        return lora_path

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(self.model_config, torch_dtype=torch.bfloat16)
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        if self.config.merge_lora:
            merged = self.merge_lora_into_base_weights(state_dict)
            if not merged:
                raise ValueError(
                    "--merge_lora was set but no LoRA tensors were found in the checkpoint state_dict. "
                    "This likely means LoRA weights are stored separately under lora_adapters/ and are not present in the FSDP shard state_dict."
                )
            print("Merged LoRA weights into base model weights")
        else:
            lora_path = self.save_lora_adapter(state_dict)
            if lora_path:
                print(f"Saving lora adapter to {lora_path}")

        self._normalize_state_dict_keys_in_place(state_dict)

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path)
        tokenizer = hf_tokenizer(self.hf_model_config_path)
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)

    def upload_to_huggingface(self):
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")

    @abstractmethod
    def merge_and_save(self):
        raise NotImplementedError("Subclasses should implement this method")


class FSDPModelMerger(BaseModelMerger):
    def _get_world_size(self) -> int:
        """Extracts the FSDP world_size from checkpoint filenames (e.g., 'model_world_size_8_rank_0.pt')."""
        for filename in os.listdir(self.config.local_dir):
            match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
            if match:
                return int(match.group(1))
        raise FileNotFoundError(
            f"Could not determine world size. No file matching 'model_world_size_(\\d+)_rank_0.pt' found in {self.config.local_dir}"
        )

    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        return torch.load(
            Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt",
            map_location="cpu",
            weights_only=False,
        )

    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        """
        Retrieves sharding information (device_mesh, mesh_dim_names) from a DTensor in the state_dict.
        If no DTensor is found, infers a simple FSDP mesh based on world_size.
        """
        pivot_key = sorted(list(state_dict.keys()))[0]
        weight = state_dict[pivot_key]

        if isinstance(weight, DTensor):
            # get sharding info
            device_mesh = weight.device_mesh
            mesh = device_mesh.mesh
            mesh_dim_names = device_mesh.mesh_dim_names
        else:
            # for non-DTensor
            mesh = np.array([world_size], dtype=np.int64)
            mesh_dim_names = ("fsdp",)

        return mesh, mesh_dim_names

    def _calculate_shard_configuration(
        self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]
    ) -> tuple[int, tuple[int, ...]]:
        """Calculates the total number of shards and the shape of the device mesh."""
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        if "tp" in mesh_dim_names:
            # TODO: "tp" is not supported yet due to the above assert
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        return total_shards, mesh_shape

    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        """Merges a list of tensors based on their DTensor placement"""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")

    def _load_and_merge_state_dicts(
        self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        # Merge state dicts from all shards
        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                # add tensor shard in order of rank to state_dict[key]
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.bfloat16())

                    placements = tuple(tensor.placements)
                    # replicated placement at dp dimension can be discarded
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.bfloat16())

        del model_state_dict_lst

        # Merge tensors
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                # merge shards
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    # 2-D list, FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def merge_and_save(self):
        world_size = self._get_world_size()
        rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)

        mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

        total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
        print(f"Processing model shards with {total_shards} {mesh_shape} in total")

        merged_state_dict = self._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._test_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _test_state_dict(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()

        hf_model = auto_model_class.from_pretrained(self.config.test_hf_dir, torch_dtype=torch.bfloat16)
        hf_state_dict = hf_model.state_dict()
        del hf_model

        hf_model_keys = set(hf_state_dict.keys())
        collected_keys = set(state_dict.keys())

        missing_keys = hf_model_keys - collected_keys
        assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

        extra_keys = collected_keys - hf_model_keys
        assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

        for key in hf_model_keys:
            hf_shape = hf_state_dict[key].shape
            collected_shape = state_dict[key].shape
            assert hf_shape == collected_shape, (
                f"Shape mismatch for key '{key}': original {hf_shape} vs collected {collected_shape}"
            )

            hf_dtype = hf_state_dict[key].dtype
            collected_dtype = state_dict[key].dtype
            assert hf_dtype == collected_dtype, (
                f"Dtype mismatch for key '{key}': original {hf_dtype} vs collected {collected_dtype}"
            )

            torch.testing.assert_close(hf_state_dict[key], state_dict[key], atol=1e-6, rtol=1e-6)

        print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")


def main():
    parser = argparse.ArgumentParser(description="verl model merger")
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    base_op_parser = argparse.ArgumentParser(add_help=False)
    base_op_parser.add_argument(
        "--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model"
    )
    base_op_parser.add_argument("--local_dir", type=str, required=True, help="Path to the saved model checkpoints")
    base_op_parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="(Deprecated) Path to the original Hugging Face model for config.",
    )
    base_op_parser.add_argument(
        "--tie-word-embedding",
        action="store_true",
        help="Whether to tie word embedding weights (currently only Megatron supported)",
    )
    base_op_parser.add_argument(
        "--is-value-model",
        action="store_true",
        help="Whether the model is a value model (currently only Megatron supported)",
    )

    merge_parser = subparsers.add_parser("merge", parents=[base_op_parser], help="Merge model checkpoints and save.")
    merge_parser.add_argument(
        "--target_dir", default="tmp", type=str, help="Directory to save the merged huggingface model"
    )
    merge_parser.add_argument(
        "--hf_upload_path", default=None, type=str, help="Hugging Face repository ID to upload the model"
    )
    merge_parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights into base weights and save/upload a single merged HF model.",
    )
    merge_parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha to use for merging (scaling = alpha/r). If omitted, tries to infer from lora_adapters/**/adapter_config.json.",
    )
    merge_parser.add_argument(
        "--lora_adapter_config",
        type=str,
        default=None,
        help="Path to a PEFT adapter_config.json to infer lora_alpha for merging.",
    )
    merge_parser.add_argument(
        "--private", action="store_true", help="Whether to upload the model to a private Hugging Face repository"
    )

    test_parser = subparsers.add_parser(
        "test", parents=[base_op_parser], help="Test merged model against a reference Hugging Face model"
    )
    test_parser.add_argument(
        "--test_hf_dir", type=str, required=True, help="Path to the reference Hugging Face model directory for testing"
    )

    args = parser.parse_args()

    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_path": args.hf_model_path,
        "hf_model_config_path": args.local_dir,
    }

    if args.operation == "merge":
        config = ModelMergerConfig(
            **common_config_args,
            target_dir=args.target_dir,
            hf_upload_path=args.hf_upload_path,
            merge_lora=args.merge_lora,
            lora_alpha=args.lora_alpha,
            lora_adapter_config=args.lora_adapter_config,
            private=args.private,
            test_hf_dir=None,
        )
        os.makedirs(config.target_dir, exist_ok=True)
    elif args.operation == "test":
        config = ModelMergerConfig(
            **common_config_args,
            test_hf_dir=args.test_hf_dir,
            # the following args are not used by test operation
            target_dir=None,
            hf_upload_path=None,
            private=False,
        )
    else:
        raise NotImplementedError(f"Unknown operation: {args.operation}")

    if config.backend == "fsdp":
        merger = FSDPModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()


if __name__ == "__main__":
    main()

