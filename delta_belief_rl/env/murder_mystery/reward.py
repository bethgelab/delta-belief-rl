from typing import Dict, List, Literal, Tuple
import numpy as np
import numpy.typing as npt

# Minimal reward schema mirroring 20q style
TQRewardSignal = Literal[
    "win",
    "invalid_action",
    "repeated_action",
    "traj_len",
    "action_len",
    "all_valid",
]

_default_reward_signals: Dict[TQRewardSignal, float] = {
    "win": 1.0,
    "invalid_action": -1.0,
    "repeated_action": -0.1,
    "traj_len": -0.01,
    "action_len": -0.0,
    "all_valid": 0.0,
}


def per_turn_reward_fn(
    action: str,
    obs: str,
    reward_signals: Dict[TQRewardSignal, float] | None = None,
) -> Tuple[float, bool, bool]:
    """Fake per-turn reward: always returns 0 and marks actions valid."""
    _ = action, obs  # unused
    signals = (
        _default_reward_signals
        if reward_signals is None
        else _default_reward_signals | reward_signals
    )
    # For a fake function, we ignore content and keep things valid.
    reward_penalty = 0.0 + signals.get("all_valid", 0.0)
    return reward_penalty, False, False


def traj_reward_fn(
    game_status: int,
    history: List[dict],
    reward_signals: Dict[TQRewardSignal, float] | None = None,
    debug: bool = False,
) -> Tuple[npt.NDArray[np.float64], float, int, int]:
    """Fake trajectory reward matching 20qs signature.

    Returns per-turn rewards (zeros), an end-of-game reward (win if game_status==0),
    and zero invalid/repeated counts.
    """
    _ = history, debug  # unused
    signals = (
        _default_reward_signals
        if reward_signals is None
        else _default_reward_signals | reward_signals
    )

    # pretend there was exactly one turn
    per_turn_rewards = np.array([0.0], dtype=float)
    end_of_game_reward = signals.get("win", 1.0) if game_status == 0 else 0.0
    invalid_count = 0
    repeated_count = 0
    return per_turn_rewards, end_of_game_reward, invalid_count, repeated_count
