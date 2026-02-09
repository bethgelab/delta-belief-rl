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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def msum(x, m, dim=None):
    return (x * m).sum(dim=dim)


def mcount(m, dim=None, eps=1e-8):
    return m.sum(dim=dim).clamp_min(eps)


def mmean(x, m, dim=None, eps=1e-8):
    return msum(x, m, dim=dim) / mcount(m, dim=dim, eps=eps)


def mvar(x, m, dim=-1, eps=1e-8):
    """
    x: tensor of any shape (B, L)
    m: mask same shape as x (bool or 0/1) #B, L
    dim: dimension to reduce over
    """
    m = m.to(dtype=x.dtype)
    count = m.sum(dim=dim, keepdim=True).clamp_min(eps)  # keep reduced dim (,B)
    mu = (x * m).sum(
        dim=dim, keepdim=True
    ) / count  # shape matches x with keepdim=True # (B, 1)
    var = ((x - mu) ** 2) * m  # (B, L)
    out = var.sum(dim=dim, keepdim=False) / count.squeeze(dim)  # (,B)
    return out


def compute_data_metrics(
    batch: DataProto, use_critic: bool = False, extra_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    resp_mask = batch.batch["response_mask"].bool()
    resp_length = resp_mask.sum(-1).float()  # (B,)
    max_response_length = batch.batch["responses"].shape[-1]  # (L,)
    num_questions = batch.meta_info["turns_stats"]

    # rewards (bsz, n_turns) all rewards combined
    mean_turn_reward = batch.batch["reward_tensor"].mean(dim=-1)  # (bsz)
    std_turn_reward = torch.std(batch.batch["reward_tensor"], dim=-1)  # (bsz)

    # turn_rewards  (bsz, n_turns)
    if "turn_rewards" in batch.batch:
        mean_turn_rewards = batch.batch["turn_rewards"].mean(dim=-1)  # (bsz)
        std_turn_rewards = torch.std(batch.batch["turn_rewards"], dim=-1)  # (bsz)

    # elicitation reward (bsz, n_turns)
    if "elicit_reward" in batch.batch:
        mean_turn_elicit = batch.batch["elicit_reward"].mean(dim=-1)  # (bsz)
        std_turn_elicit = torch.std(batch.batch["elicit_reward"], dim=-1)  # (bsz)

    # Global per-token aggregates (true masked mean/std over all active tokens)
    # Sequence-level view (every sequece contributes equally)
    adv_token_mean_per_seq = mmean(batch.batch["advantages"], resp_mask, dim=-1)  # (B,)
    adv_token_std_per_seq = torch.sqrt(
        mvar(batch.batch["advantages"], resp_mask, dim=-1)
    )  # (B,)
    ret_token_mean_per_seq = mmean(batch.batch["returns"], resp_mask, dim=-1)  # (B,)
    ret_token_std_per_seq = torch.sqrt(
        mvar(batch.batch["returns"], resp_mask, dim=-1)
    )  # (B,)

    metrics = {
        # eog
        "critic/eog/mean": torch.mean(batch.batch["eog_tensor"]).detach().item(),
        "critic/eog/max": torch.max(batch.batch["eog_tensor"]).detach().item(),
        "critic/eog/min": torch.min(batch.batch["eog_tensor"]).detach().item(),
        "critic/eog/std": torch.std(batch.batch["eog_tensor"]).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(mean_turn_reward).detach().item(),
        "critic/rewards/max": torch.max(batch.batch["reward_tensor"]).detach().item(),
        "critic/rewards/min": torch.min(batch.batch["reward_tensor"]).detach().item(),
        "critic/rewards/std": torch.mean(std_turn_reward).detach().item(),
        # adv
        "critic/advantages/mean_per_seq_mean": torch.mean(adv_token_mean_per_seq)
        .detach()
        .item(),
        "critic/advantages/token_max": torch.max(batch.batch["advantages"][resp_mask])
        .detach()
        .item(),
        "critic/advantages/token_min": torch.min(batch.batch["advantages"][resp_mask])
        .detach()
        .item(),
        "critic/advantages/mean_per_seq_std": torch.mean(adv_token_std_per_seq)
        .detach()
        .item(),
        # adv global view (token level)
        "critic/advantages/token_mean_global": mmean(
            batch.batch["advantages"], resp_mask
        ).item(),
        "critic/advantages/token_std_global": torch.mean(
            torch.sqrt(mvar(batch.batch["advantages"], resp_mask))
        ).item(),
        # returns
        "critic/returns/mean_per_seq_mean": torch.mean(ret_token_mean_per_seq)
        .detach()
        .item(),
        "critic/returns/token_max": torch.max(batch.batch["returns"][resp_mask])
        .detach()
        .item(),
        "critic/returns/token_min": torch.min(batch.batch["returns"][resp_mask])
        .detach()
        .item(),
        "critic/returns/mean_per_seq_std": torch.mean(ret_token_std_per_seq)
        .detach()
        .item(),
        # returns global view (token level)
        "critic/returns/token_mean_global": mmean(
            batch.batch["returns"], resp_mask
        ).item(),
        "critic/returns/token_std_global": torch.mean(
            torch.sqrt(mvar(batch.batch["returns"], resp_mask))
        ).item(),
        **(
            {
                # elicit scores
                "critic/elicit_scores/mean": torch.mean(mean_turn_elicit)
                .detach()
                .item(),
                "critic/elicit_scores/max": torch.max(batch.batch["elicit_reward"])
                .detach()
                .item(),
                "critic/elicit_scores/min": torch.min(batch.batch["elicit_reward"])
                .detach()
                .item(),
                "critic/elicit_scores/std": torch.mean(std_turn_elicit).detach().item(),
            }
            if "elicit_reward" in batch.batch
            else {}
        ),
        **(
            {
                # turn rewards
                "critic/turn_rewards/mean": torch.mean(mean_turn_rewards)
                .detach()
                .item(),
                "critic/turn_rewards/max": torch.max(batch.batch["turn_rewards"])
                .detach()
                .item(),
                "critic/turn_rewards/min": torch.min(batch.batch["turn_rewards"])
                .detach()
                .item(),
                "critic/turn_rewards/std": torch.mean(std_turn_rewards).detach().item(),
            }
            if "turn_rewards" in batch.batch
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(resp_length).detach().item(),
        "response_length/max": torch.max(resp_length).detach().item(),
        "response_length/min": torch.min(resp_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(resp_length, max_response_length).float()
        )
        .detach()
        .item(),
        # number of questions
        "num_questions/mean": np.mean(num_questions),
        "num_questions/max": np.max(num_questions),
        "num_questions/min": np.min(num_questions),
        # successful games
        "train/mean_successful": 1.0
        - (
            np.sum(batch.meta_info["active_mask"]) / len(batch.meta_info["active_mask"])
        ),
        **(
            {  # invalid/repeated counts
                "train/total_invalid": np.sum(extra_info["invalid_count"]),
                "train/mean_invalid": np.mean(extra_info["invalid_count"]),
                "train/total_repeated": np.sum(extra_info["repeated_count"]),
                "train/mean_repeated": np.mean(extra_info["repeated_count"]),
            }
            if extra_info is not None
            else {}
        ),
    }
    return metrics


def compute_timing_metrics(
    batch: DataProto, timing_raw: Dict[str, float]
) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name]
            * 1000
            / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(
    batch: DataProto, timing_raw: Dict[str, float], n_gpus: int
) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
) -> list[tuple[float, float]]:
    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        data=var_vals, subset_size=n, reduce_fns=[np.max, np.min]
                    )
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = (
                        bon_mean,
                        bon_std,
                    )
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = (
                        won_mean,
                        won_std,
                    )
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [
                            {"val": val, "pred": pred}
                            for val, pred in zip(var_vals, var2vals["pred"])
                        ]
                        [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                            data=vote_data,
                            subset_size=n,
                            reduce_fns=[
                                partial(calc_maj_val, vote_key="pred", val_key="val")
                            ],
                        )
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = (
                            maj_n_mean,
                            maj_n_std,
                        )

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][
                        metric_name
                    ].append(metric_val)

    data_src2var2metric2val = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(
                    prompt_vals
                )

    return data_src2var2metric2val


def compute_success_rate(success_mask: np.ndarray, k: int) -> Tuple[float, float]:
    """Compute mean and std pass@1 success rate.

    Args:
        success_mask (np.ndarray): Boolean array indicating success for each sample.
        k (int): Number of samples per instance.

    Returns:
        float: Mean and std pass@1 metric.
    """
    if isinstance(success_mask, list):
        success_mask = np.array(success_mask)

    n_instances = len(success_mask) // k
    success_mask_reshaped = success_mask.reshape(n_instances, k)
    mean_pass_at_1_per_iter = np.mean(success_mask_reshaped, axis=0).astype(float)
    mean_pass_at_1 = np.mean(mean_pass_at_1_per_iter)
    std_pass_at_1 = np.std(mean_pass_at_1_per_iter)

    return mean_pass_at_1, std_pass_at_1


def compute_pass_k_low_variance(
    success_mask: np.ndarray,
    n: int,
) -> List[float]:
    """
    Low-variance pass@k (Chen et al.) for all k = 1..n.

    success_mask:
        flattened array of shape (bsz * n)
        samples are contiguous per instance
    n:
        number of samples per instance

    returns:
        list of pass@k values for k = 1..n
    """
    success_mask = np.asarray(success_mask, dtype=bool)
    assert success_mask.size % n == 0

    bsz = success_mask.size // n
    success = success_mask.reshape(bsz, n)

    # c_i = number of correct samples per instance
    c = success.sum(axis=1)
    m = n - c  # incorrect samples

    pass_at_k = np.zeros(n)
    fail = np.ones(bsz)

    for k in range(1, n + 1):
        j = k - 1
        valid = m > j

        # multiplicative update of failure probability
        fail = np.where(
            valid,
            fail * (m - j) / (n - j),
            0.0,
        )

        pass_at_k[k - 1] = (1.0 - fail).mean()

    return pass_at_k.tolist()


def compute_pass_at_k(
    success_mask: np.ndarray, max_k: int, n_bootstrap: int = 1000
) -> tuple[List[float], List[float]]:
    """Compute pass@k metrics for k in [1, max_k].

    Args:
        success_mask (np.ndarray): Boolean array indicating success for each sample.
        max_k (int): Maximum k value.
        n_bootstrap (int): Number of bootstrap samples.

    Returns:
        tuple[List[float], List[float]]: Mean and std pass@k metrics for k in [1, max_k].
    """
    if isinstance(success_mask, list):
        success_mask = np.array(success_mask)

    n_instances = len(success_mask) // max_k
    success_mask_reshaped = success_mask.reshape(n_instances, max_k)

    pass_at_k_bootstrap = np.zeros((n_bootstrap, max_k))

    for b in range(n_bootstrap):
        # Bootstrap instances once
        bootstrap_idxs = np.random.choice(n_instances, size=n_instances, replace=True)
        bootstrap_mask = success_mask_reshaped[bootstrap_idxs, :]

        # Compute pass@k for all k values
        for k in range(1, max_k + 1):
            bootstrap_mask_k = np.zeros((n_instances, k), dtype=bool)
            for i in range(n_instances):
                k_idxs = np.random.choice(max_k, size=k, replace=False)
                bootstrap_mask_k[i] = bootstrap_mask[i, k_idxs]

            pass_at_k = 1.0 - np.prod(1.0 - bootstrap_mask_k, axis=1)
            pass_at_k_bootstrap[b, k - 1] = np.mean(pass_at_k)

    means = np.mean(pass_at_k_bootstrap, axis=0).tolist()
    stds = np.std(pass_at_k_bootstrap, axis=0).tolist()

    return means, stds
