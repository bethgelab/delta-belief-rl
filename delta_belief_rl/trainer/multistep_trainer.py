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
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import sys
import json
import os
import time
import uuid
import copy
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss

from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from delta_belief_rl.utils.format import MTAlgoConfig

from delta_belief_rl.llm_agent.generation import *
from delta_belief_rl.llm_agent.belief import *
from delta_belief_rl.trainer.metrics_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_pass_k_low_variance,
    compute_success_rate,
    process_validation_metrics,
    reduce_metrics,
)
from delta_belief_rl.utils.format import repeat, make_json_serializable
from delta_belief_rl.utils.syntax import JUDGE_EXTRAS_KEYS, JUDGE_METRICS_KEYS


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    JudgeRollout = 7


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_TURN = "grpo_turn"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    MULTI_TURN_REINFORCE = "multi_turn_reinforce"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Special handling: If process_on_nodes contains [0], this indicates a CPU-only
        pool (e.g., for API-based workers). In this case, use_gpu=False and we create
        a single CPU-only node placement group.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models

            # Check if this is a CPU-only pool (GPU count is 0)
            if len(process_on_nodes) > 0 and process_on_nodes[0] == 0:
                # CPU-only pool: create a single node with no GPU requirements
                use_gpu = False
                # Create a new list to avoid modifying the original
                process_on_nodes = [1]  # 1 CPU-only node
                print(
                    f"[DEBUG ResourcePoolManager] Converting {resource_pool_name} to CPU-only: process_on_nodes={process_on_nodes}, use_gpu={use_gpu}"
                )
            else:
                use_gpu = True

            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=use_gpu,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster.

        Skips GPU validation for CPU-only pools (where GPU count is 0).
        """
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied (exclude CPU-only pools with 0 GPUs)
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
                if n_gpus > 0
            ]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)

            # Skip GPU validation for CPU-only pools
            if num_gpus == 0:
                continue

            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """

    reward_result = reward_fn(data, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]
    turn_rewards = reward_result["turn_rewards"]
    eog_tensor = reward_result["eog_tensor"]
    reward_extra_infos_dict = reward_result["reward_extra_info"]

    return reward_tensor, turn_rewards, eog_tensor, reward_extra_infos_dict


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    token_level_rewards = data.batch["token_level_rewards"]  # apply to all rewards
    # make a copy for logging purpose
    token_level_rewards_no_kl = token_level_rewards.clone().detach()
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    beta = kl_ctrl.value

    token_level_rewards = token_level_rewards - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    data.batch["token_level_rewards_no_kl"] = token_level_rewards_no_kl

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }

    return data, metrics


def compute_response_mask(
    data: DataProto, metrics: dict, multi_turn=True, state_masking=True
):
    if multi_turn and state_masking:
        response_length = data.batch["responses"].shape[-1]
        response_mask = data.batch["attention_mask"][:, -response_length:]
        assert response_length == data.batch["loss_mask"].shape[-1], (
            f"response_length {response_length} != loss_mask.shape[-1] {data.batch['loss_mask'].shape[-1]}"
        )

        data.batch["response_mask"] = data.batch["loss_mask"]

        metrics.update(
            {
                "state_tokens(total)": data.batch["response_mask"].sum().item(),
                "state_tokens/coverage": (
                    data.batch["response_mask"].sum() / response_mask.sum()
                ).item(),
            }
        )
        return data, metrics
    else:
        responses = data.batch["responses"]
        response_length = responses.size(1)
        attention_mask = data.batch["attention_mask"]
        data.batch["response_mask"] = attention_mask[:, -response_length:]
        return data, {}


def compute_advantage(data: DataProto, config=None):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    # prepare response group
    if config.adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=config.gamma,
            lam=config.lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif config.adv_estimator == AdvantageEstimator.GRPO:
        # we sum over last dim, so can pass directly as is
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["reward_tensor"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=config.norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif config.adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = (
            core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
            )
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif config.adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif config.adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
            config=config,
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif config.adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif config.adv_estimator == AdvantageEstimator.MULTI_TURN_REINFORCE:
        from delta_belief_rl.trainer.ppo.core_algos import compute_multi_turn_reinforce

        advantages, returns = compute_multi_turn_reinforce(
            token_level_rewards=data.batch[
                "token_level_rewards"
            ],  # turn_rewards + logprob_rewards + eog_rewards with shape (batch_size, seq_len)
            eog_rewards=data.batch[
                "eog_scores"
            ],  # eog_rewards only with shape (batch_size)
            response_mask=data.batch["response_mask"],
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif config.adv_estimator == AdvantageEstimator.GRPO_TURN:
        from delta_belief_rl.trainer.ppo.core_algos import compute_grpo_turn_advantage

        advantages, returns = compute_grpo_turn_advantage(
            token_level_rewards=data.batch[
                "reward_tensor"
            ],  # turn_rewards only with shape (batch_size, seq_len)
            response_mask=data.batch["response_mask"],  # same as loss_mask
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=config.norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayMultiStepTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer_actor,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer_actor = tokenizer_actor
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Check if the run is only for validation - propagate to the workers
        self.val_only = self.config.trainer.get("val_only", False)

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = (
            Role.RefPolicy in role_worker_mapping and not self.val_only
        )
        self.use_rm = Role.RewardModel in role_worker_mapping and not self.val_only
        self.use_judge = Role.JudgeRollout in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                config.algorithm.kl_ctrl
            )

        self.use_critic = False
        if (
            self.config.algorithm.adv_estimator == AdvantageEstimator.GAE
            and not self.config.multi_turn.logprob_reward.enabled
            and not self.val_only
        ):
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.MULTI_TURN_REINFORCE,
            AdvantageEstimator.GRPO_TURN,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError(
                f"Advantage estimator {self.config.algorithm.adv_estimator} is not supported"
            )

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        is_api_actor = config.actor_rollout_ref.ngpus == 0

        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )

        # Skip GPU-based validations if using API actor
        if not is_api_actor:
            assert real_train_batch_size % config.actor_rollout_ref.ngpus == 0, (
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
            )

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # Skip for API actors as they don't use GPUs
            if not is_api_actor:
                check_mutually_exclusive(
                    config.actor_rollout_ref.actor.ppo_micro_batch_size,
                    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                    "actor_rollout_ref.actor",
                )

            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz and not is_api_actor:
            assert (
                config.data.train_batch_size
                >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if (
            config.algorithm.use_kl_in_reward
            and config.actor_rollout_ref.actor.use_kl_loss
        ):
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        # Skip for API actors as they don't use FSDP
        if (
            not is_api_actor
            and config.actor_rollout_ref.actor.strategy == "fsdp"
            and (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
                > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1)
                > 1
            )
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from delta_belief_rl.utils.dataset.rl_dataset import RLHFDataset

        dataset_cls = RLHFDataset

        if self.val_only:
            print("Validation only mode: skipping train dataloader creation.")
            self.train_dataloader = None
        else:
            self.train_dataset = dataset_cls(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer_actor,
                processor=self.processor,
                config=self.config.data,
                sample_count=self.config.data.sample_count.train,
            )

            # use sampler for better ckpt resume
            if self.config.data.shuffle:
                train_dataloader_generator = torch.Generator()
                train_dataloader_generator.manual_seed(self.config.get("seed", 42))
                sampler = RandomSampler(
                    data_source=self.train_dataset, generator=train_dataloader_generator
                )
            else:
                sampler = SequentialSampler(data_source=self.train_dataset)

            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.data.get(
                    "gen_batch_size", self.config.data.train_batch_size
                ),
                num_workers=4,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )

            assert len(self.train_dataloader) >= 1

            # inject total_training_steps to actor/critic optim_config. This is hacky.
            total_training_steps = (
                len(self.train_dataloader) * self.config.trainer.total_epochs
            )

            if self.config.trainer.total_training_steps is not None:
                total_training_steps = self.config.trainer.total_training_steps

            self.total_training_steps = total_training_steps
            print(f"Total training steps: {self.total_training_steps}")

            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                    total_training_steps
                )
                self.config.critic.optim.total_training_steps = total_training_steps

            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer_actor,
            processor=self.processor,
            config=self.config.data,
            sample_count=self.config.data.sample_count.val,
        )
        # consider the design of single controller with a large val dataset in multi-modal scenarios
        # may lead to oom issues
        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.val_dataloader) >= 1

        print(f"Val dataloader size: {len(self.val_dataloader)}")

    def _finished_at(self, len_game: int, game_finished: bool):
        finished_at = -1
        if len_game < self.config.multi_turn.max_turns.val and game_finished:
            finished_at = len_game
        return finished_at

    def _save_validation_samples(
        self,
        history,
        secrets,
        game_status,
        scores,
        reward_extra_infos_dict,
        questions_cot=None,
        responses_cot=None,
        log_probs=None,
        log_prob_diffs=None,
        val_data_dir=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        # Extract actor and judge names from the config
        actor_name = (
            self.config.actor_rollout_ref.model.path.split("/")[-1]
            if self.config.actor_rollout_ref.model.path
            else "actor"
        )
        if self.config.judge_rollout.model.path:
            judge_name = self.config.judge_rollout.model.path.split("/")[-1]
        elif self.config.judge_rollout.model.api_model_name:
            judge_name = self.config.judge_rollout.model.api_model_name.split("/")[-1]
        else:
            judge_name = "judge"

        # Create a directory for validation data
        val_path = os.path.join(val_data_dir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(val_path, exist_ok=True)
        filename = os.path.join(val_path, "validation.jsonl")

        # Clear empty messages from history
        history = [[msg for msg in h if msg != ""] for h in history]

        # Extract game info from history and meta-info
        questions = [h[2::2] for h in history]
        questions = [
            [msg["content"] if isinstance(msg, dict) else msg for msg in q]
            for q in questions
        ]
        game_status = [bool(g) for g in game_status]
        finished_at = [
            self._finished_at(len(q), g) for q, g in zip(questions, game_status)
        ]

        n = len(secrets)
        base_data = {
            "secret": secrets,
            "actor": [actor_name] * n,
            "judge": [judge_name] * n,
            "game_status": game_status,
            "finished_at": finished_at,
            "questions_cot": questions_cot,
            "responses_cot": responses_cot,
            "score": scores,
            "log_prob_reward": log_probs,
            "log_prob_diff": log_prob_diffs,  # corresponds to elicit_reward
            "history": history,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items() if v is not None}

                # Remove 0 entries from end of lists
                filter_keys = ["log_prob_reward", "log_prob_diff"]
                for key in filter_keys:
                    if key in entry and isinstance(entry[key], list):
                        entry[key] = [x for x in entry[key] if x != 0]

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        secrets,
        game_status,
        scores,
        reward_extra_infos_dict,
        log_probs=None,
        log_prob_diffs=None,
        dump_path=None,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "secret": secrets,
            "input": inputs,
            "output": outputs,
            "score": scores,
            "game_status": game_status,
            "step": [self.global_steps] * n,
            "log_prob_reward": log_probs,
            "log_prob_diff": log_prob_diffs,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items() if v is not None}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use seed from config for deterministic shuffling
        rng = np.random.RandomState(self.config.get("seed", 42))
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _estimate_beliefs(self, multi_step_config: GenMultiEnvConfig):
        print("[DEBUG] belief run")

        multi_step_config.max_turns = self.config.multi_turn.max_turns.val

        multi_step_manager = LLMGenerationManager(
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer_actor=self.tokenizer_actor,  # actor
            judge_rollout_wg=self.judge_rollout_wg,
            config=multi_step_config,
            meta_info={
                "eos_token_id": self.tokenizer_actor.eos_token_id,
                "pad_token_id": self.tokenizer_actor.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            },
        )

        elicitation_type = set(self.config.multi_turn.belief.method)

        multi_step_manager.initialize_rollout_state()

        for test_data in self.val_dataloader:
            timing_raw = {}
            raw_prompt = test_data.pop("raw_prompt")
            gt = test_data.pop("golden_answers")
            game_status = test_data.pop("game_status", None)

            with _timer("step", timing_raw):
                with _timer("gen", timing_raw):
                    multi_step_manager.timing_raw = timing_raw
                    if "logprob" in elicitation_type:
                        filename = "belief_logprob.jsonl"
                        history = [copy.deepcopy(d) for d in raw_prompt.tolist()]

                        # Initialize belief results
                        belief_results = {}
                        for idx, gt_secret in enumerate(gt):
                            belief_results[str(idx)] = {
                                "ground_truth": gt_secret,
                                "messages": history[idx],
                                "logprob": [],
                                "diff_logprob": [],
                                "game_status": game_status[idx]
                                if game_status is not None
                                else None,
                            }

                        # Compute beliefs step-by-step
                        max_length = max([len(hist) for hist in history])
                        max_steps = (max_length + 1) // 2
                        active = [1 for _ in range(len(history))]

                        for step in tqdm(
                            range(max_steps), desc="Belief Estimation Steps"
                        ):
                            end = (step + 1) * 2
                            messages = []
                            gt_active = []
                            active_indices = []

                            for idx, (hist, gt_val) in enumerate(zip(history, gt)):
                                if len(hist) >= end:
                                    messages.append(hist[:end])
                                    gt_active.append(gt_val)
                                    active_indices.append(idx)
                                else:
                                    active[idx] = 0

                            if not messages:
                                break

                            # Call the new belief_log_prob method
                            logprob_scores = multi_step_manager.belief_log_prob(
                                messages=messages, gt=np.array(gt_active), model="actor"
                            )

                            # Store results
                            for i, idx in enumerate(active_indices):
                                lp = float(logprob_scores[i])
                                belief_results[str(idx)]["logprob"].append(lp)
                                if step != 0:
                                    prev_lp = belief_results[str(idx)]["logprob"][-2]
                                    belief_results[str(idx)]["diff_logprob"].append(
                                        lp - prev_lp
                                    )

                    if "confidence" in elicitation_type:
                        filename = "belief_confidence.jsonl"
                        from delta_belief_rl.llm_agent.belief import BeliefManager

                        belief_manager_conf = BeliefManager(
                            actor_rollout_wg=multi_step_manager.actor_rollout_wg,
                            tokenizer_actor=multi_step_manager.tokenizer_actor,
                            judge_rollout_wg=multi_step_manager.judge_rollout_wg,
                            config=multi_step_manager.config,
                            meta_info=multi_step_manager.meta_info,
                        )
                        belief_results = belief_manager_conf.belief_confidence(
                            raw_prompt=raw_prompt,
                            gts=gt,
                            game_status=game_status,
                        )
                    if "accuracy" in elicitation_type:
                        filename = "belief_accuracy.jsonl"
                        from delta_belief_rl.llm_agent.belief import BeliefManager

                        belief_manager_acc = BeliefManager(
                            actor_rollout_wg=multi_step_manager.actor_rollout_wg,
                            tokenizer_actor=multi_step_manager.tokenizer_actor,
                            judge_rollout_wg=multi_step_manager.judge_rollout_wg,
                            config=multi_step_manager.config,
                            meta_info=multi_step_manager.meta_info,
                        )
                        belief_results = belief_manager_acc.belief_accuracy(
                            raw_prompt=raw_prompt,
                            gts=gt,
                            game_status=game_status,
                        )
        multi_step_manager.shutdown_rollout_state()
        belief_data_dir = self.config.multi_turn.belief.get("data_dir", None)

        if belief_data_dir:
            os.makedirs(belief_data_dir, exist_ok=True)

            with open(os.path.join(belief_data_dir, filename), "w") as f:
                for key, value in make_json_serializable(belief_results).items():
                    flat_entry = {
                        "ground_truth": value.get("ground_truth"),
                        "log_prob_reward": value.get("logprob", []),
                        "diff_logprob": value.get("diff_logprob", []),
                        "game_status": value.get("game_status"),
                        "messages": value.get("messages"),
                        "entropy": value.get("entropy", []),
                        "vocab": value.get("vocab", []),
                        "confidence": value.get("confidence", []),
                    }

                    # Remove None values
                    flat_entry = {
                        k: v
                        for k, v in flat_entry.items()
                        if v is not None and (not isinstance(v, list) or len(v) > 0)
                    }
                    f.write(json.dumps(flat_entry, ensure_ascii=False) + "\n")

    def _validate(self, multi_step_manager: LLMGenerationManager):
        multi_step_manager.meta_info["validate"] = True
        multi_step_manager.config.max_turns = self.config.multi_turn.max_turns.val

        data_source_lst = []
        group_by_gt = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_logprobs = []
        sample_logprob_diffs = []
        sample_active_masks = []
        sample_histories = []

        judge_metrics: Dict[str, int] = {key: 0 for key in JUDGE_METRICS_KEYS}
        judge_extras: Dict[str, int] = {key: 0 for key in JUDGE_EXTRAS_KEYS}

        for test_data in self.val_dataloader:
            timing_raw = {}
            non_tensor_batch = {}
            raw_prompt = test_data.pop("raw_prompt")
            gt = test_data.pop("golden_answers")  # numpy.ndarray
            extra_info = test_data.pop("extra_info", None)

            scenario = None
            if extra_info is not None:
                scenario = np.array(
                    [
                        info.get("scenario") if isinstance(info, dict) else None
                        for info in extra_info
                    ],
                    dtype=object,
                )

            with _timer("step", timing_raw):
                with _timer("gen", timing_raw):
                    multi_step_manager.timing_raw = timing_raw
                    if multi_step_manager.judge_rollout_wg is not None:
                        multi_step_manager.initialize_rollout_state()
                        test_gen_output = multi_step_manager.run_game(
                            raw_prompt=raw_prompt,
                            gt=gt,
                            n=self.config.multi_turn.val.n,
                            scenario=scenario,
                        )
                        multi_step_manager.shutdown_rollout_state()

            # grouping is by the gt
            grouping_gt_batch = np.repeat(gt, self.config.multi_turn.val.n, axis=0)

            # decode the input ids to text
            input_ids = test_gen_output.batch["prompts"]
            sample_inputs.extend(
                self.tokenizer_actor.batch_decode(input_ids, skip_special_tokens=True)
            )
            output_ids = test_gen_output.batch["responses"]
            output_texts = [
                self.tokenizer_actor.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_gen_output

            # Extract and repeat non-tensor fields
            non_tensor_batch = {
                "reward_model": test_data.pop("reward_model"),
                "data_source": test_data.pop("data_source"),
                "scenario": scenario,
            }
            non_tensor_batch = repeat(
                non_tensor_batch,
                repeat_times=self.config.multi_turn.val.n,
                interleave=True,
            )

            # Attach history to non_tensor_batch
            non_tensor_batch["history"] = test_gen_output.non_tensor_batch.get(
                "history", None
            )
            test_batch.non_tensor_batch = non_tensor_batch

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            if multi_step_manager.config.logprob_reward.enabled:
                sample_logprobs.extend(test_batch.meta_info["logprob"])
                sample_logprob_diffs.extend(test_batch.meta_info["logprob_diff"])

            # Collect active_mask for success rate calculation
            sample_active_masks.extend(test_batch.meta_info["active_mask"])

            # Keep per-sample histories aligned with scores across batches
            history_batch = test_batch.non_tensor_batch.get("history", None)
            if history_batch is None:
                sample_histories.extend([[] for _ in range(reward_tensor.shape[0])])
            else:
                sample_histories.extend(history_batch)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # add number of question as extra reward info
            reward_extra_infos_dict["num_questions"].extend(
                test_batch.meta_info["turns_stats"]
            )  # (bsz) where bsz = n * original_bsz

            for metric_name, metric_val in test_batch.meta_info[
                "judge_metrics"
            ].items():
                if metric_name in judge_metrics:
                    judge_metrics[metric_name] += metric_val
                else:
                    judge_metrics[metric_name] = metric_val
            for metric_name, metric_val in test_batch.meta_info["judge_extras"].items():
                if metric_name in judge_extras:
                    judge_extras[metric_name] += metric_val
                else:
                    judge_extras[metric_name] = metric_val

            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )
            group_by_gt.extend(grouping_gt_batch)

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            secrets = np.array(group_by_gt)
            game_completed = [
                1 - game for game in sample_active_masks
            ]  # flip the status, 1 -> completed across all batches

            if self.val_only:
                self._save_validation_samples(
                    history=sample_histories,
                    secrets=secrets,
                    game_status=game_completed,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    questions_cot=test_batch.meta_info.get("questions_cot", None),
                    responses_cot=test_batch.meta_info.get("responses_cot", None),
                    log_probs=(sample_logprobs if len(sample_logprobs) > 0 else None),
                    log_prob_diffs=(
                        sample_logprob_diffs if len(sample_logprob_diffs) > 0 else None
                    ),
                    val_data_dir=val_data_dir,
                )
            else:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    secrets=secrets,
                    scores=sample_scores,
                    game_status=game_completed,
                    log_probs=(sample_logprobs if len(sample_logprobs) > 0 else None),
                    log_prob_diffs=(
                        sample_logprob_diffs if len(sample_logprob_diffs) > 0 else None
                    ),
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources, group_by_gt, reward_extra_infos_dict
        )
        metric_dict = {}

        # Calculate success rate
        val_k = self.config.multi_turn.val.n
        success_rate = 1.0 - (np.sum(sample_active_masks) / len(sample_active_masks))
        metric_dict["val-core/success_rate"] = success_rate
        success_mask = 1.0 - np.array(sample_active_masks)
        success_rate_mean, success_rate_std = compute_success_rate(
            success_mask=success_mask,
            k=val_k,
        )
        metric_dict[f"val-core/success_rate/mean@{val_k}"] = success_rate_mean
        metric_dict[f"val-core/success_rate/std@{val_k}"] = success_rate_std
        metric_dict["val-core/invalid_total"] = np.sum(
            reward_extra_infos_dict["invalid_count"]
        )
        metric_dict["val-core/repeated_total"] = np.sum(
            reward_extra_infos_dict["repeated_count"]
        )

        # Compute low variance pass@k
        pass_at_k = compute_pass_k_low_variance(
            success_mask=success_mask,
            n=val_k,
        )

        for k, k_val in enumerate(pass_at_k):
            metric_dict[f"pass@k/{k + 1}"] = k_val

        if "wandb" in self.config.trainer.logger:
            import wandb

            data = [[k, k_val] for k, k_val in enumerate(pass_at_k, start=1)]
            table = wandb.Table(data=data, columns=["k", "pass@k"])

            wandb.log({"pass@k/pass@k_table": table}, step=self.global_steps)

        # Calculate judge metrics & extras
        n_obs = sum(list(judge_metrics.values()))
        for metric_name, metric_val in judge_metrics.items():
            metric_dict[f"val-core/judge/{metric_name}"] = (
                metric_val / n_obs if n_obs > 0 else 0
            )
        for metric_name, metric_val in judge_extras.items():
            metric_dict[f"val-core/judge/extra/{metric_name}"] = (
                metric_val / n_obs if n_obs > 0 else 0
            )

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [
                        int(name.split("@")[-1].split("/")[0])
                        for name in metric2val.keys()
                    ]
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "maj", "best"]
                        )
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            role = self.config.actor_rollout_ref.role

            if (
                self.config.multi_turn.logprob_sampling.enabled
                or self.config.multi_turn.logprob_reward.enabled
            ):
                assert role in ["actor_rollout", "actor_rollout_ref"], (
                    f"Logprob computation requires 'actor_rollout_ref' or 'actor_rollout' role, but got '{role}'."
                )

            if self.config.multi_turn.debug:
                print("[DEBUG] using hybrid engine, with role for actor as: ", role)
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=role,  # "actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool][role] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # create a judge
        if self.config.multi_turn.enable and self.use_judge:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.JudgeRollout
            )
            judge_rollout_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.JudgeRollout],
                config=self.config.judge_rollout,
            )
            self.resource_pool_to_cls[resource_pool]["judge_rollout"] = (
                judge_rollout_cls
            )

        # initialize WorkerGroup
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}
        if (
            OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout")
            is not None
        ):
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )

        from delta_belief_rl.utils.watchdog import kill_if_hangs

        with kill_if_hangs(seconds=300):
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                # if judge then assign directly to the resource pool
                if class_dict.get("judge_rollout", None) is not None:
                    wg_dict = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=class_dict["judge_rollout"],
                        **wg_kwargs,
                    )

                # if actor and ref are colocated, we can use colocated worker class
                else:
                    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                    wg_dict = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=worker_dict_cls,
                        **wg_kwargs,
                    )

                # # Spawn workers
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())

                all_wg.update(spawn_wg)

                # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
                self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        else:
            self.ref_policy_wg = None

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        if self.config.multi_turn.enable:
            if self.use_judge:
                self.judge_rollout_wg = all_wg["judge_rollout"]
                try:
                    self.judge_rollout_wg.init_model()
                except Exception as e:
                    # exit code
                    print(f"judge_rollout_wg.init_model failed: {e}")
                    print("exiting code")
                    raise
                    sys.exit(1)
            else:
                # specify this for later code to use
                self.judge_rollout_wg = None

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[role]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = self.config.trainer.get("checkpoint_step", None)
            if global_step_folder is None:
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), (
                    "resume ckpt must be str type"
                )
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")

        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # load dataloader,
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        self._load_checkpoint()

        if (
            self.config.algorithm.adv_estimator == "grpo"
            or self.config.algorithm.adv_estimator == "reinforce_plus_plus_baseline"
        ):
            reward_attribution_level = "trajectory"
        elif (
            self.config.algorithm.adv_estimator == "reinforce_plus_plus"
            or self.config.algorithm.adv_estimator == "multi_turn_reinforce"
            or self.config.algorithm.adv_estimator == "grpo_turn"
        ):
            reward_attribution_level = "token"
        else:
            raise ValueError(
                f"Unsupported adv_estimator: {self.config.algorithm.adv_estimator}"
            )

        # Set up the multi_step_manager
        multi_step_config = GenMultiEnvConfig(
            max_turns=self.config.multi_turn.max_turns.train,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_obs_length=self.config.judge_rollout.rollout.prompt_length,
            actor_cot=self.config.actor_rollout_ref.actor.cot,
            actor_thinking=self.config.actor_rollout_ref.actor.thinking,
            judge_thinking=self.config.judge_rollout.thinking,
            logprob_reward=LogProbRewardConfig(
                enabled=self.config.multi_turn.logprob_reward.enabled,
                agg_method=self.config.multi_turn.logprob_reward.agg,
                base_model=self.config.multi_turn.logprob_reward.base_model,
                step_model=self.config.multi_turn.logprob_reward.step_model,
                normalised=self.config.multi_turn.logprob_reward.normalised,
                methods=set(self.config.multi_turn.logprob_reward.methods),
                clipping={
                    "min": self.config.multi_turn.logprob_reward.clip_min,
                    "max": self.config.multi_turn.logprob_reward.clip_max,
                },
                tau=self.config.multi_turn.logprob_reward.tau,
                level=reward_attribution_level,
            ),
            logprob_sampling=LogprobSamplingConfig(
                enabled=self.config.multi_turn.logprob_sampling.enabled,
                best_n=self.config.multi_turn.logprob_sampling.best_n,
                worst_n=self.config.multi_turn.logprob_sampling.worst_n,
                p_best=self.config.multi_turn.logprob_sampling.p_best,
            ),
            verify_judge=VerifyJudgeConfig(
                enabled=self.config.multi_turn.verify_judge.enabled,
                methods=set(self.config.multi_turn.verify_judge.methods),
                false_positive_behavior=self.config.multi_turn.verify_judge.false_positive_behavior,
                short_circuit=self.config.multi_turn.verify_judge.short_circuit,
            ),
            repeated_prompt=self.config.judge_rollout.repeated_prompt,
            debug=self.config.multi_turn.debug,
            env=self.config.multi_turn.env,
        )

        multi_step_manager = LLMGenerationManager(
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer_actor=self.tokenizer_actor,
            judge_rollout_wg=self.judge_rollout_wg,
            config=multi_step_config,
            meta_info={
                "eos_token_id": self.tokenizer_actor.eos_token_id,
                "pad_token_id": self.tokenizer_actor.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.do_sample,
                "validate": False,
            },
        )

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate(multi_step_manager)
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.val_only:
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        cfg_algo = MTAlgoConfig(
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            adv_estimator=self.config.algorithm.adv_estimator,
            use_kl_in_reward=self.config.algorithm.use_kl_in_reward,
            kl_penalty=self.config.algorithm.kl_penalty,
            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            ),
            norm_adv_in_mtr=self.config.algorithm.get("norm_adv_in_mtr", False),
            clip_adv_in_mtr=self.config.algorithm.get("clip_adv_in_mtr", False),
            clip_adv_min=self.config.algorithm.get("clip_adv_min", -5.0),
            clip_adv_max=self.config.algorithm.get("clip_adv_max", 5.0),
            only_propagate_eog_in_mtr=self.config.algorithm.get(
                "only_propagate_eog_in_mtr", True
            ),
        )

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        timing_raw = defaultdict(float)

        if self.config.algorithm.filter_groups.enabled:
            print(
                f"Running with DAPO filter_groups enabled. {self.config.algorithm.filter_groups.metric=}, {self.config.algorithm.filter_groups.max_num_gen_batches=}"
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                num_gen_batches += 1

                non_tensor_batch = {}
                raw_prompt = batch_dict.pop("raw_prompt")
                gt = batch_dict.pop("golden_answers")
                extra_info = batch_dict.pop("extra_info", None)

                scenario = None
                if extra_info is not None:
                    scenario = np.array(
                        [
                            info.get("scenario") if isinstance(info, dict) else None
                            for info in extra_info
                        ],
                        dtype=object,
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # change to training mode
                        multi_step_manager.meta_info["validate"] = False
                        multi_step_manager.config.max_turns = (
                            self.config.multi_turn.max_turns.train
                        )
                        multi_step_manager.initialize_rollout_state()
                        gen_batch_output = multi_step_manager.run_game(
                            raw_prompt=raw_prompt,
                            gt=gt,
                            n=self.config.multi_turn.train.n,
                            scenario=scenario,
                        )
                        multi_step_manager.shutdown_rollout_state()

                    # Extract and repeat non-tensor fields
                    non_tensor_batch = {
                        "reward_model": batch_dict.pop("reward_model"),
                        "data_source": batch_dict.pop("data_source"),
                        "uid": np.array(
                            [str(uuid.uuid4()) for _ in range(len(gt))], dtype=object
                        ),
                    }

                    # repeat to align with repeated responses in rollout
                    non_tensor_batch = repeat(
                        non_tensor_batch,
                        repeat_times=self.config.multi_turn.train.n,
                    )
                    # Attach history to non_tensor_batch
                    non_tensor_batch["history"] = gen_batch_output.non_tensor_batch.get(
                        "history", None
                    )
                    new_batch = gen_batch_output
                    new_batch.non_tensor_batch = non_tensor_batch

                    judge_metrics = new_batch.meta_info["judge_metrics"]
                    n_obs = sum(list(judge_metrics.values()))
                    for metric_name, metric_val in judge_metrics.items():
                        metrics[f"judge/{metric_name}"] = (
                            metric_val / n_obs if n_obs > 0 else 0
                        )
                    judge_extras = new_batch.meta_info["judge_extras"]
                    for metric_name, metric_val in judge_extras.items():
                        metrics[f"judge/extra/{metric_name}"] = (
                            metric_val / n_obs if n_obs > 0 else 0
                        )

                    with _timer("reward", timing_raw):
                        (
                            reward_tensor,
                            turn_rewards,
                            eog_tensor,
                            reward_extra_infos_dict,
                        ) = compute_reward(new_batch, self.reward_fn)

                        # reward tensor is now bsz, n_turns
                        new_batch.batch["reward_tensor"] = (
                            reward_tensor  # (bsz, n_turns)
                        )
                        new_batch.batch["eog_tensor"] = eog_tensor  # (bsz,)
                        new_batch.batch["turn_rewards"] = turn_rewards  # (bsz, n_turns)

                        # remove history from non_tensor_batch
                        #  NOTE: all non_tensor_batch have to be np.arrray for DataProto chunking, as it's
                        new_batch.non_tensor_batch.pop("history")

                    # DAPO
                    if not self.config.algorithm.filter_groups.enabled:
                        batch = new_batch
                    else:
                        # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"]
                                .sum(dim=-1)
                                .numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"]
                                .sum(dim=-1)
                                .numpy()
                            )
                        else:
                            raise ValueError(
                                f"Unsupported filter metric: {metric_name}"
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                            strict=True,
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(
                            new_batch.non_tensor_batch["uid"]
                        ):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = (
                            new_batch
                            if batch is None
                            else DataProto.concat([batch, new_batch])
                        )

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = (
                                self.config.algorithm.filter_groups.max_num_gen_batches
                            )
                            if (
                                max_num_gen_batches <= 0
                                or num_gen_batches < max_num_gen_batches
                            ):
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = (
                                    self.global_steps >= self.total_training_steps
                                )
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = (
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n
                            )
                            batch = batch[:traj_bsz]

                    batch, metrics = compute_response_mask(
                        batch,
                        metrics,
                        self.config.multi_turn.enable,
                        self.config.actor_rollout_ref.actor.state_masking,
                    )

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy_loss": entropy_loss.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                            else:
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
                                )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # NOTE: currently do not support kl in reward for multi-step

                        batch = compute_advantage(batch, config=cfg_algo)
                        # sanity check the computational graph
                        assert not batch.batch["advantages"].requires_grad, (
                            "advantages require grad, there is a bug in advantage computation"
                        )
                        assert batch.batch["advantages"].grad_fn is None, (
                            "advantages have grad_fn, there is a bug in advantage computation"
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    if "ema" in self.config.actor_rollout_ref.model.adapters:
                        with _timer("update_ema", timing_raw):
                            if (
                                self.global_steps
                                < self.config.actor_rollout_ref.model.ema_warmup_steps
                            ):
                                beta = 0.0
                            else:
                                beta = self.config.actor_rollout_ref.model.ema_beta
                            ema_metrics = self.actor_rollout_wg.update_ema(beta)
                            metrics.update(ema_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if (
                        rollout_data_dir is not None
                        and self.config.trainer.rollout_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.rollout_freq == 0
                        )
                    ):
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer_actor.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer_actor.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = batch.batch["reward_tensor"].sum(-1).cpu().tolist()
                            game_completed = [
                                1 - game for game in batch.meta_info["active_mask"]
                            ]  # flip the status, 1 -> completed
                            if self.config.multi_turn.logprob_reward.enabled:
                                sample_logprobs = batch.meta_info["logprob"]
                                sample_logprob_diffs = batch.meta_info["logprob_diff"]
                            else:
                                sample_logprobs = []
                                sample_logprob_diffs = []

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                secrets=np.repeat(
                                    gt, self.config.multi_turn.train.n, axis=0
                                ),
                                scores=scores,
                                game_status=game_completed,
                                log_probs=(
                                    sample_logprobs
                                    if len(sample_logprobs) > 0
                                    else None
                                ),
                                log_prob_diffs=(
                                    sample_logprob_diffs
                                    if len(sample_logprob_diffs) > 0
                                    else None
                                ),
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate(multi_step_manager)
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(
                    compute_data_metrics(
                        batch=batch,
                        use_critic=self.use_critic,
                        extra_info=reward_extra_infos_dict,
                    )
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                # dapo
                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
