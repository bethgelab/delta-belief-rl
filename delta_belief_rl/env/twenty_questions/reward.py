from typing import List, Literal, Dict, Tuple
import numpy as np
import numpy.typing as npt


"""
Reward signals for the Twenty Questions game.
- `win`: Reward for winning the game (i.e., guessing the secret word).
- `invalid_action`: Penalty for invalid actions (e.g., deviating from the task).
- `repeated_action`: Penalty for repeated actions.
- `traj_len`: Penalty for the length of the trajectory (i.e., number of rounds).
- `action_len`: Penalty for the length of the action (i.e., number of tokens).
"""
TQRewardSignal = Literal[
    "win", "invalid_action", "repeated_action", "traj_len", "action_len", "all_valid"
]

ActionLengthMode = Literal["linear", "uniform"]

_default_reward_signals: Dict[TQRewardSignal, float] = {
    "win": 1.0,
    "invalid_action": -1.0,  # -10.0,
    "repeated_action": -0.05,  # -0.5,
    "traj_len": -0.05,  # -0.01,
    "action_len": -0.0,
    "all_valid": 0.1,  # bonus for all valid actions in the trajectory
}


def per_turn_reward_fn(
    action: str,
    obs: str,
    reward_signals: Dict[TQRewardSignal, float] | None = None,
) -> float:
    """
    The reward function for the Twenty Questions game.

    This function combines the token-wise, action-wise, and trajectory-wise rewards.
    It should be called at the action level, and it will return rewards for each token.

    Args:
        action (str): The action taken by the actor (i.e., a question or guess).
        obs (str): The observation after taking the action (i.e., the judge's response).
        reward_signals (Dict[TQRewardSignal, float]): The reward signals to use for the game.

    Returns:
        reward_penalty (float): the reward_penality for a given turn
    """

    action = action.strip().lower()
    obs = obs.strip().lower()

    if reward_signals is None:
        signals = _default_reward_signals
    else:
        signals = _default_reward_signals | reward_signals

    reward_penalty = 0.0
    invalid, repeated = False, False

    if "not asked a valid" in obs.lower() or action.count("?") > 1:
        reward_penalty += signals["invalid_action"]
        invalid = True

    if "already asked" in obs.lower() or "repeated" in obs.lower():
        reward_penalty += signals["repeated_action"]
        repeated = True

    if not invalid and not repeated:
        reward_penalty += signals["all_valid"]

    return reward_penalty, invalid, repeated


def traj_reward_fn(
    game_status: int,
    history: List[dict],
    reward_signals: Dict[TQRewardSignal, float] | None = None,
    debug: bool = False,
) -> Tuple[npt.NDArray[np.float64], float, int, int]:
    """
    The trajectory-wise reward function for the Twenty Questions game.

    Args:
        game_status (int): The status of the game, 0 for finished, 1 for ongoing.
        history (List[dict]): The conversation history of the game, containing actions and
          observations with dict format as 'role' and 'content'
        reward_signals (Dict[TQRewardSignal, float]): The reward signals to use for the game.

    Returns:
        Tuple[np.array[float], float, int, int]:
        - The per-turn rewards, where the size is equal to the amount
            of turns taken in the game
        - the end-of-game reward, consisting of the 0/1 reward and
            the trajectory-length penalty.
        - the number of invalid actions in the trajectory
        - the number of repeated actions in the trajectory
    """
    # Note: if game_status is Boolean, the assertion will not be triggered
    assert isinstance(game_status, int), "(TRAJ-REWARD-FN) Game status must be int 0/1"

    if reward_signals is None:
        signals = _default_reward_signals
    else:
        signals = _default_reward_signals | reward_signals

    if type(history) is str:
        history = [line.strip() for line in history.split("\n") if line.strip()]

    # Parse actions & observations from the conversation history
    action_list = []
    obs_list = []

    # get the action and obs from the history
    # skip initial system prompt and user prompt
    for hist_dict in history[2:]:
        # if the history is a dict, then it is a valid action and obs
        if isinstance(hist_dict, dict):
            if hist_dict["role"] == "assistant":
                content = hist_dict["content"]
                if debug and ("\n" in content and content.rsplit("\n", 1)[-1].strip()):
                    print(f"[WARNING] Contains multiple sentences: '{content}'")
                    print(f"[WARNING] history: {history}")
                action_list.append(content.replace("\n", " "))
            elif hist_dict.get("role") == "user":
                obs_list.append(hist_dict.get("content", ""))
        # otherwise exit the loop
        else:
            break

    assert len(action_list) == len(obs_list), (
        "(TRAJ-REWARD-FN) Number of parsed actions and observations must match"
    )

    n_turns = len(action_list)

    if n_turns == 0:
        print(
            "[WARN] (TRAJ-REWARD-FN) No actions in the trajectory, setting reward to zero."
        )
        return np.array([0.0]), 0.0

    per_turn_rewards = np.zeros(n_turns, dtype=float)
    invalid_count, repeated_count = 0, 0
    for i in range(n_turns):
        per_turn_rewards[i], invalid, repeated = per_turn_reward_fn(
            action=action_list[i],
            obs=obs_list[i],
            reward_signals=signals,
        )
        invalid_count += invalid  # bool is subclass of int
        repeated_count += repeated

    end_of_game_reward = 0.0
    if game_status == 0:
        end_of_game_reward += signals["win"]
    if n_turns > 1:  # ignore in single-turn games
        end_of_game_reward += signals["traj_len"] * n_turns

    return per_turn_rewards, end_of_game_reward, invalid_count, repeated_count
