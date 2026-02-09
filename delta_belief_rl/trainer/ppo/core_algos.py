import torch
import numpy as np
from collections import defaultdict
from enum import Enum
from typing import Any, Optional
from verl.trainer.config import AlgoConfig
import verl.utils.torch_functional as verl_F

"""
Based on verl/verl/trainer/ppo/core_algos.py
"""


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_TURN = "grpo_turn"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    MULTI_TURN_REINFORCE = "multi_turn_reinforce"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


@register_adv_est(AdvantageEstimator.MULTI_TURN_REINFORCE)
def compute_multi_turn_reinforce(
    token_level_rewards: torch.Tensor,
    eog_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Multi-turn REINFORCE++.

    Args:
        token_level_rewards `(torch.Tensor)`: the token-level rewards, which are turn_rewards
            + eog_rewards + logprob_rewards
        - shape: (bs, response_length)
        eog_rewards `(torch.Tensor)`: the end-of-game rewards
        - shape: (bs)
        response_mask `(torch.Tensor)`: mask for valid response tokens
        - shape: (bs, response_length)
        config (AlgoConfig): the algorithm config

    Returns:
        advantages: `(torch.Tensor)`
        - shape: (bs, response_length)
        Returns: `(torch.Tensor)`
        - shape: (bs, response_length)
    """

    assert token_level_rewards.shape == response_mask.shape, (
        f"token_level_rewards and response_mask must have the same shape, got {token_level_rewards.shape}, {response_mask.shape}"
    )
    assert eog_rewards.dim() == 1 and eog_rewards.shape[0] == response_mask.shape[0], (
        f"eog_rewards must be of shape (bs,), got {eog_rewards.shape}"
    )
    assert config is not None

    gamma = getattr(config, "gamma", 0.99)
    only_propagate_eog_in_mtr = getattr(config, "only_propagate_eog_in_mtr", True)

    # each batch might have different number of turns, so update game for that given batch
    T = response_mask.shape[1]
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        returns = torch.full_like(token_level_rewards, float("nan"))
        running_return = eog_rewards.clone().to(
            device=device, dtype=dtype
        )  # shape (bs,)
        last_turn_mask = torch.zeros_like(running_return)  # shape (bs,)

        for t in reversed(range(T)):
            # No intra-turn discount; masked tokens get zero credit
            # eog rewards are already added to the tokens of the last turn; use the last_turn_mask to
            # not add them again
            returns[:, t] = (
                token_level_rewards[:, t] + running_return * last_turn_mask
            ) * response_mask[:, t]

            # Update running_return when we cross a 0 -> 1 boundary (start of a new turn)
            if t < T - 1:
                start_boundary = (response_mask[:, t] == 0) & (
                    response_mask[:, t + 1] == 1
                )
                if start_boundary.any():
                    # update running return to discounted reward of future turn
                    if only_propagate_eog_in_mtr:
                        running_return[start_boundary] = (
                            running_return[start_boundary] * gamma
                        )
                    else:
                        running_return[start_boundary] = (
                            returns[start_boundary, t + 1] * gamma
                        )
                    last_turn_mask[start_boundary] = 1.0

        if config.norm_adv_in_mtr:
            advantages = verl_F.masked_whiten(returns, response_mask)
            advantages = advantages * response_mask
            if config.clip_adv_in_mtr:
                advantages = torch.clamp(
                    advantages, min=config.clip_adv_min, max=config.clip_adv_max
                )
        else:
            advantages = returns

        assert not torch.isnan(returns[response_mask.bool()]).any(), (
            "NaNs found in active positions of returns"
        )
        assert (returns[~response_mask.bool()] != 0.0).sum().item() == 0.0, (
            "No non-zero values found in inactive positions of returns"
        )

    return advantages, returns


@register_adv_est(AdvantageEstimator.GRPO_TURN)
def compute_grpo_turn_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on turn level

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, n_turns)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length) dtype bool or int (0/1)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """

    # Detect start of a turn
    shifted = torch.roll(response_mask, 1, dims=1)
    shifted[:, 0] = 0  # first column can't be a continuation
    turn_start = (response_mask == 1) & (
        shifted == 0
    )  # 0,0,1,0,0,1,0,0,1, where 1 indicates start of a turn (bsz, resp_len)

    # Cumulative sum → turn number for each position
    turn_index = torch.cumsum(turn_start.int(), dim=1)  # bsz, response_len
    # turn_index: 0 0 1 1 1 2 2 2 3 3 3 ...

    turn_scores_all = torch.zeros_like(
        token_level_rewards, device=token_level_rewards.device
    )  # bsz, n_turns

    with torch.no_grad():
        bsz = token_level_rewards.shape[0]
        # compute turn-level grpo scores
        for turn in range(token_level_rewards.shape[1]):
            id2score = defaultdict(list)
            id2mean = {}
            id2std = {}
            for i in range(bsz):
                id2score[index[i]].append(token_level_rewards[i, turn])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    scores_tensor = torch.stack(id2score[idx])
                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                if norm_adv_by_std_in_grpo:
                    turn_scores_all[i, turn] = (
                        token_level_rewards[i, turn] - id2mean[index[i]]
                    ) / (id2std[index[i]] + epsilon)
                else:
                    turn_scores_all[i, turn] = (
                        token_level_rewards[i, turn] - id2mean[index[i]]
                    )

        # Convert 1-based turn index → 0-based
        idx = (turn_index - 1).clamp(min=0)

        # Gather scores per token
        advantages = torch.gather(turn_scores_all, 1, idx)
        advantages = advantages * response_mask  # zero out pads

    return advantages, advantages
