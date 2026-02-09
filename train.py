"""
Borrowed from verl.trainer.main_ppo.py
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from delta_belief_rl.trainer.multistep_trainer import RayMultiStepTrainer

import ray
import hydra
import os
import random
from verl import DataProto
import torch
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
from delta_belief_rl.env.twenty_questions import reward
from omegaconf import DictConfig, OmegaConf, open_dict


class RewardManager:
    """The reward manager."""

    __SUPPORTED_ALGORITHMS = {
        "grpo",
        "reinforce_plus_plus",
        "multi_turn_reinforce",
        "reinforce_plus_plus_baseline",
        "grpo_turn",
    }

    def __init__(
        self,
        tokenizer,
        num_examine,
        algorithm: str,
        compute_score: Callable[
            [str, int, List[Dict[reward.TQRewardSignal, float]]],
            Tuple[npt.NDArray[np.float64], float],
        ]
        | None = None,
        reward_signals: Dict[reward.TQRewardSignal, float] | DictConfig | None = None,
        debug: bool = False,
        max_turns: int = 20,
        w1: float = 1.0,
    ) -> None:
        assert algorithm in self.__SUPPORTED_ALGORITHMS, (
            f"The reward manager only supports algorithms {self.__SUPPORTED_ALGORITHMS}"
        )

        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.algorithm = algorithm
        self.compute_score = compute_score or self._default_compute_score
        self.debug = debug
        self.max_turns = max_turns
        self.w1 = w1  # weight for reward scaling

        if type(reward_signals) is DictConfig:
            self.reward_signals = OmegaConf.to_container(reward_signals, resolve=True)
        else:
            self.reward_signals = reward_signals

    def _default_compute_score(
        self,
        data_source: str,
        game_status: int,
        history: List[Dict[reward.TQRewardSignal, float]],
    ) -> Tuple[npt.NDArray[np.float64], float]:
        if data_source in ["tw_q"]:
            res = reward.traj_reward_fn(
                game_status,
                history,
                reward_signals=self.reward_signals,
                debug=self.debug,
            )
            return res
        elif data_source in ["guess_my_city"]:
            from delta_belief_rl.env.guess_my_city import reward as gmc_reward

            res = gmc_reward.traj_reward_fn(
                game_status,
                history,
                reward_signals=self.reward_signals,
                debug=self.debug,
            )
            return res
        elif data_source in ["customer_service"]:
            from delta_belief_rl.env.customer_service import reward as cs_reward

            res = cs_reward.traj_reward_fn(
                game_status,
                history,
                reward_signals=self.reward_signals,
                debug=self.debug,
            )
            return res
        elif data_source in ["murder_mystery"]:
            from delta_belief_rl.env.murder_mystery import reward as mm_reward

            res = mm_reward.traj_reward_fn(
                game_status,
                history,
                reward_signals=self.reward_signals,
                debug=self.debug,
            )
            return res
        else:
            raise NotImplementedError(
                f"Reward function is not implemented for {data_source=}"
            )

    def _apply_rewards_to_tokens(
        self,
        turn_rewards: npt.NDArray[np.float64],  # n_turns
        eog_reward: float,
        response_mask: torch.Tensor,  # seq_len
    ) -> torch.Tensor:
        """
        The token-level reward distribution logic for the Multi-Turn Reinforce algorithm.

        Args:
            turn_rewards (npt.NDArray[np.float64]): the per-turn rewards.
            eog_reward (float): the end-of-game reward.
            response_mask (torch.Tensor): the tensor to populate with token-level rewards; values are assumed to be zero.

        Returns:
            torch.Tensor: a tensor populated with token-level rewards; same shape as `response_mask`.
        """

        rewards = torch.full_like(
            response_mask, float("nan"), dtype=torch.float32
        )  # seq_len

        turn_rev_i = 0
        last_mask = 0
        for token_i in reversed(range(response_mask.shape[0])):
            if response_mask[token_i] == 0:
                last_mask = 0
                continue

            # Accumulate the *intended* value in a local variable (avoid NaN += 0)
            val_token_i = 0.0

            # New turn
            if last_mask == 0:
                turn_rev_i += 1

            # Summary of the application of the end-of-game reward, depending on the algorithm:
            # - grpo: the last token of the last turn gets the reward.
            # - reinforce_plus_plus: the last token of every turn gets the reward.
            # - multi_turn_reinforce: all tokens of the last turn get the reward.
            if (
                turn_rev_i == 1
                and (
                    (
                        (
                            self.algorithm == "grpo"
                            or self.algorithm == "reinforce_plus_plus_baseline"
                        )
                        and last_mask == 0
                    )
                    or self.algorithm == "multi_turn_reinforce"
                )
            ) or (self.algorithm == "reinforce_plus_plus" and last_mask == 0):
                val_token_i += float(eog_reward)

            # Make sure the `response_mask` matches the number of turns given by `turn_rewards`
            assert len(turn_rewards) >= turn_rev_i, (
                "The number of turns observed in the `response_mask` does not match the number of turns considered by `turn_rewards`"
            )

            # Summary of the application of per-turn rewards, depending on the algorithm:
            # - grpo: the last token of each turn gets the reward of that turn.
            # - reinforce_plus_plus: the last token of each turn gets the reward of that turn.
            # - multi_turn_reinforce: all tokens of each turn get the reward of that turn.
            if last_mask == 0 or self.algorithm == "multi_turn_reinforce":
                val_token_i += float(turn_rewards[-turn_rev_i])

                # Write the final value (even if it's exactly 0.0)
                rewards[token_i] = val_token_i

            last_mask = 1

        # perform assertions based on loss type
        masked = response_mask.bool()
        if self.algorithm == "multi_turn_reinforce":
            missing = torch.isnan(rewards) & masked
            assert missing.sum().item() == 0, (
                f"Not all masked tokens were assigned for {self.algorithm}. "
                f"Missing indices: {missing.nonzero(as_tuple=True)[0].tolist()}"
            )
            # (Optional) also ensure nothing outside mask was written
            assert torch.all(torch.isnan(rewards[~masked])), (
                "Values written outside response_mask"
            )

        elif (
            self.algorithm == "grpo"
            or self.algorithm == "reinforce_plus_plus_baseline"
            or self.algorithm == "reinforce_plus_plus"
        ):
            # In GRPO, only the last token of each turn gets a value.
            # Count turns (runs of True) in the mask:
            # A turn start is True preceded by False (or start of sequence).
            left_shifted_false = torch.cat(
                [torch.tensor([True], device=masked.device), ~masked[:-1]]
            )
            turn_starts = masked & left_shifted_false
            n_turns = int(turn_starts.sum().item())

            num_written = int((~torch.isnan(rewards)).sum().item())
            assert num_written == n_turns, (
                f"{self.algorithm}: number of assigned tokens ({num_written}) must equal number of turns ({n_turns})."
            )
            # Ensure assigned tokens are a subset of the mask
            assert torch.all((~torch.isnan(rewards)) <= masked), (
                "GRPO: wrote outside mask"
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # convert to 0.0
        rewards = torch.nan_to_num(rewards, nan=0.0)

        return rewards

    def __call__(self, data: DataProto, return_dict=True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # new reward_tensor of (bsz, max_turns)
        reward_tensor = torch.zeros(
            (data.batch["responses"].shape[0], self.max_turns), dtype=torch.float32
        )  # for now hardcoded max turns to 20
        eog_tensor = torch.zeros(
            (data.batch["responses"].shape[0],), dtype=torch.float32
        )
        invalid_count_bsz, repeated_count_bsz = [], []  # total counts in the batch
        reward_extra_info = defaultdict(list)
        turn_scores_reward = torch.zeros_like(reward_tensor)

        # get history of all the games
        history = data.non_tensor_batch["history"]
        assert len(history) == len(data), (
            f"history length {len(history)} != data length {len(data)}"
        )

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[
                -valid_response_length:
            ]  # sequences are padded to the left

            # decode
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch["data_source"]

            # multi turn additional info
            game_rounds = data_item.meta_info.get("turns_stats", None)
            game_rounds = game_rounds[i] if game_rounds else None
            game_status = data_item.meta_info.get("active_mask", None)
            game_status = (
                game_status[i] if game_status is not None else 1
            )  # default: not finished

            # calculate per turn scores and eog score
            per_turn_scores, eog_score, invalid_count, repeated_count = (
                self.compute_score(
                    data_source=data_source,
                    game_status=game_status,
                    history=history[i],
                )
            )

            # convert to torch tensors
            turn_scores = torch.as_tensor(
                per_turn_scores, device=reward_tensor.device, dtype=reward_tensor.dtype
            )
            eog_score_tensor = torch.tensor(
                eog_score, device=reward_tensor.device, dtype=reward_tensor.dtype
            )

            # apply per batch element reward score to tokens
            reward_tensor[i, : turn_scores.numel()] = turn_scores
            if self.algorithm == "grpo":
                reward_tensor[i, len(per_turn_scores) - 1] += (
                    eog_score_tensor  # last turn gets eog score
                )

            elif self.algorithm == "grpo_turn":
                reward_tensor[i, : len(per_turn_scores)] += (
                    eog_score_tensor  # all turns get eog score
                )

            if "elicit_reward" in data_item.batch:
                # the elictaiton reward is alrady normalised
                elicit_reward = data_item.batch["elicit_reward"]  # (max_turns,)
                reward_tensor[i] += self.w1 * elicit_reward

            eog_tensor[i] = eog_score_tensor
            turn_scores_reward[i, : turn_scores.numel()] = (
                turn_scores  # turn_scores might be less than max_turns
            )
            invalid_count_bsz.append(invalid_count)
            repeated_count_bsz.append(repeated_count)

            # Get data_source from data_item if available, otherwise use a default value
            data_source = data_item.non_tensor_batch.get("data_source", "default")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if (
                already_print_data_sources[data_source] < self.num_examine
                and self.debug
            ):
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[game_rounds]", game_rounds)
                print("[game_status]", game_status)
                print("[ground_truth]", ground_truth)
                print("[score]", per_turn_scores)

        reward_extra_info["invalid_count"] = invalid_count_bsz
        reward_extra_info["repeated_count"] = repeated_count_bsz

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "turn_rewards": turn_scores_reward,  # bsz, max_turns
                "eog_tensor": eog_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


def get_custom_reward_fn(config):
    import importlib.util
    import os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    if spec is None:
        raise RuntimeError(f"Failed to create module spec from '{file_path}'")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not function_name:
        raise ValueError("Function name not specified in custom_reward_function config")

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{file_path}'."
        )

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(
    version_base=None,
    config_path="delta_belief_rl/config",
    config_name="base_multiturn",
)
def main(config):
    os.makedirs("log", exist_ok=True)
    run_ppo(config)


def run_ppo(config) -> None:
    # check if ray client address is set
    if "RAY_CLIENT_ADDRESS" not in os.environ:
        raise ValueError("RAY_CLIENT_ADDRESS is not set")

    ray.init(
        address=os.environ["RAY_CLIENT_ADDRESS"],
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "ROCR_VISIBLE_DEVICES": "",  # clears in workers
                "HIP_VISIBLE_DEVICES": "",  # keep empty on NVIDIA
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            }
        },
        namespace="default",
        _node_ip_address=os.environ.get("RAY_NODE_IP_ADDRESS"),
    )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        from verl.utils.fs import copy_to_local
        from omegaconf import OmegaConf

        from pprint import pprint
        import torch

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Set global random seeds for reproducibility
        config.seed = config.get("seed", 42)
        if config.seed in ("", None):
            config.seed = 42
        else:
            config.seed = int(config.seed)
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # For additional reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # download the checkpoint from hdfs
        local_path_actor = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer

        tokenizer_actor = hf_tokenizer(local_path_actor)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            from delta_belief_rl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                JudgeRolloutWorker,
            )
            from delta_belief_rl.workers.api_workers import (
                APIJudgeRolloutWorker,
                APIActorRolloutWorker,
            )
            from verl.single_controller.ray import RayWorkerGroup

            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        from delta_belief_rl.trainer.multistep_trainer import ResourcePoolManager, Role

        # Detect if API actor should be used
        use_api_actor = (
            config.actor_rollout_ref.model.get("api_model_name", None) is not None
        )

        # Assert that API actor is only used for validation-only mode
        if use_api_actor:
            assert config.trainer.val_only, (
                "API-based Actor Rollout Worker can only be used in validation-only mode. "
                "Set 'trainer.val_only=true' or remove 'actor_rollout_ref.model.api_model_name' for training."
            )

        role_worker_mapping = {}
        resource_pool_spec = {}
        mapping = {}

        # Configure Actor Rollout Worker
        if use_api_actor:
            print("[INFO] Using API-based Actor Rollout Worker")
            # API actor uses CPU only - set GPU count to 0
            resource_pool_spec["actor_pool"] = [0]
            role_worker_mapping[Role.ActorRollout] = ray.remote(num_gpus=0)(
                APIActorRolloutWorker
            )
            mapping[Role.ActorRollout] = "actor_pool"
            # Pass tokenizer path through config for API worker to load
            with open_dict(config):
                config.actor_rollout_ref.model.tokenizer_path = (
                    config.actor_rollout_ref.model.path
                )
                # Store original path if needed, then clear for API mode
                if not hasattr(config.actor_rollout_ref.model, "local_model_path"):
                    config.actor_rollout_ref.model.local_model_path = (
                        config.actor_rollout_ref.model.path
                    )
        else:
            print("[INFO] Using local GPU-based Actor Rollout Worker")
            resource_pool_spec["actor_pool"] = [config.actor_rollout_ref.ngpus]
            role_worker_mapping[Role.ActorRollout] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.ActorRollout] = "actor_pool"

        global_pool_id = (
            "actor_pool"
            if config.actor_rollout_ref.ref.resource_pool_actor
            else "judge_pool"
        )

        if config.judge_rollout.enable:
            # Detect if API judge should be used
            use_api_judge = (
                config.judge_rollout.model.get("api_model_name", None) is not None
            )

            if use_api_judge:
                print("[INFO] Using API-based Judge Rollout Worker")

                # API judge uses CPU only - set GPU count to 0
                # ResourcePoolManager will automatically create a CPU-only pool
                resource_pool_spec["judge_pool"] = [0]
                role_worker_mapping[Role.JudgeRollout] = ray.remote(num_gpus=0)(
                    APIJudgeRolloutWorker
                )
                mapping[Role.JudgeRollout] = "judge_pool"

                # Pass tokenizer path through config for API worker to load
                with open_dict(config):
                    config.judge_rollout.model.tokenizer_path = (
                        config.actor_rollout_ref.model.path
                    )
                    config.judge_rollout.model.path = None  # Clear local model path
            else:
                print("[INFO] Using local GPU-based Judge Rollout Worker")
                resource_pool_spec["judge_pool"] = [config.judge_rollout.ngpus]
                role_worker_mapping[Role.JudgeRollout] = ray.remote(JudgeRolloutWorker)
                mapping[Role.JudgeRollout] = "judge_pool"

        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        if config.reward_model.enable:
            raise NotImplementedError

        reward_manager_cls = RewardManager
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer_actor,
            num_examine=0,
            algorithm=config.algorithm.adv_estimator,
            reward_signals=config.multi_turn.reward.signals,
            debug=config.multi_turn.debug,
            max_turns=config.multi_turn.max_turns.train,
            w1=config.multi_turn.reward.weight_elicit,
        )
        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer_actor,
            num_examine=1,
            algorithm=config.algorithm.adv_estimator,
            reward_signals=config.multi_turn.reward.signals,
            debug=config.multi_turn.debug,
            max_turns=config.multi_turn.max_turns.val,
            w1=config.multi_turn.reward.weight_elicit,
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        if config.multi_turn.debug:
            print(f"[DEBUG] resource_pool_manager: {resource_pool_manager}")
            print(f"[DEBUG] role_worker_mapping: {role_worker_mapping}")

        trainer = RayMultiStepTrainer(
            config=config,
            tokenizer_actor=tokenizer_actor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
