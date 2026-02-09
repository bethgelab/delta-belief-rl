import numpy as np
import torch
from verl.protocol import DataProto
from verl.trainer.config import AlgoConfig
from dataclasses import dataclass


def episode_centered_diff(lp_diff: torch.Tensor) -> torch.Tensor:
    """
    Center elicitation differences per episode over valid turns.

    Args:
        lp_diff: (B, T) tensor, with NaN for invalid/padded turns.

    Returns:
        centered: (B, T) tensor, NaNs preserved.
    """
    # Mask for valid entries
    mask = ~torch.isnan(lp_diff)  # (B, T) bool

    # Replace NaNs with 0 so they don't affect sums
    safe_diff = torch.where(mask, lp_diff, torch.zeros_like(lp_diff))

    # Sum over valid turns per episode
    sums = safe_diff.sum(dim=1, keepdim=True)  # (B, 1)

    # Count of valid turns per episode
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)  # (B, 1)

    # Per-episode mean
    means = sums / counts

    # Centered diffs (NaNs stay NaN)
    centered = torch.where(mask, lp_diff - means, lp_diff)

    return centered


def repeat(batch: dict, repeat_times=2, interleave=False):
    """
    Repeat the batch data a specified number of times.

    Args:
        repeat_times (int): Number of times to repeat the data.
        interleave (bool): Whether to interleave the repeated data.

    Returns:
        DataProto: A new DataProto with repeated data.
    """
    repeated_batch = {}
    for key, val in batch.items():
        if not interleave:
            repeated_batch[key] = np.repeat(val, repeat_times, axis=0)
        else:
            repeated_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

    return repeated_batch


def nanstd(x: torch.Tensor, dim=None, keepdim=False, unbiased=True):
    """
    Compute std ignoring NaN values.

    Args:
        x: input tensor
        dim: dimension along which to compute
        keepdim: whether to keep reduced dimension
        unbiased: whether to apply Bessel's correction (N-1 in denominator)

    Returns:
        Tensor of std values
    """
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))

    count = mask.sum(dim=dim, keepdim=True).clamp(min=1)  # avoid div by zero
    mean = masked_x.sum(dim=dim, keepdim=True) / count

    sq_diff = torch.where(
        mask, (x - mean) ** 2, torch.tensor(0.0, device=x.device, dtype=x.dtype)
    )
    if unbiased:
        denom = (count - 1).clamp(min=1)
    else:
        denom = count
    var = sq_diff.sum(dim=dim, keepdim=True) / denom

    std = torch.sqrt(var)
    if not keepdim:
        std = std.squeeze(dim)
    return std


def make_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # convert tensor to list
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj  # assume it's already JSON-serializable


# modified from verl.protocol.pad_dataproto_to_divisor
def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    length = len(data)
    pad_size = (-length) % size_divisor
    if pad_size == 0:
        return data, 0

    padding_protos = []
    remaining_pad = pad_size
    while remaining_pad > 0:
        take_size = min(remaining_pad, len(data))
        padding_protos.append(data[:take_size])
        remaining_pad -= take_size
    data_padded = DataProto.concat([data] + padding_protos)

    return data_padded, pad_size


# here need to account that the output can be larger than the input
# due to vllm rollout n>1
def unpad_dataproto(data: "DataProto", pad_size, rollout_fanout: int = 1):
    if pad_size != 0:
        data = data[: -(pad_size * rollout_fanout)]
    return data


def ratio_report(
    episode_level_scores: torch.Tensor,
    token_level_scores: torch.Tensor,
    loss_mask: torch.Tensor,  # [B, L] bool/int mask (1 = compare here)
    threshold: float,
    require_same_indices: bool = False,
):
    """
    Compute element-wise token/episode ratios only where loss_mask == 1 and at least
    one of the tensors has a value (non-zero). Returns per-batch counts above/below
    the threshold and the ratio tensor (NaN where no comparison was made).
    Episode values may be zero; zeros are handled explicitly (no global eps bias).
    """
    assert episode_level_scores.shape == token_level_scores.shape, "shape mismatch"
    assert episode_level_scores.ndim == 2, "Expect (bsz, seq_len)"
    assert loss_mask.shape == episode_level_scores.shape, "mask shape mismatch"

    m = loss_mask.bool()
    E = episode_level_scores
    T = token_level_scores
    where = m & ((E != 0) | (T != 0))
    r = torch.full_like(T, float("nan"))
    r[where] = T[where] / (E[where] + 1e-12)  # signed ratio; eps only to avoid /0

    # Use log-ratio for symmetry (distance from 1.0 is |log r|)
    lr = torch.full_like(T, float("nan"))
    lr[where] = torch.log(torch.abs(T[where]) + 1e-12) - torch.log(
        torch.abs(E[where]) + 1e-12
    )

    def pct(x):
        return 100.0 * x

    band = (r[where].abs() >= 1 / 1.25) & (
        r[where].abs() <= 1.25
    )  # within 0.8..1.25 in magnitude
    print(
        "ratio | mean:", torch.nanmean(r).item(), " median:", torch.nanmedian(r).item()
    )
    print("log|ratio| | mean abs:", torch.nanmean(lr.abs()).item())
    print("share within 0.8..1.25:", pct(torch.nanmean(band.float())).item(), "%")

    # a few percentiles (of |log ratio|):
    vals = lr[where].abs().flatten().nan_to_num()
    for q in [0.5, 0.75, 0.9]:
        print(f"|log ratio| p{int(q * 100)}:", torch.quantile(vals, q).item())

    where = m & ((E != 0) | (T != 0))
    mean_ep = E[where].mean().item()
    mean_tok = T[where].mean().item()
    print("mean episode:", mean_ep, " mean token:", mean_tok)

    r = T[where] / (E[where] + 1e-12)
    print("share T > E:", (r > 1).float().mean().item())
    print("share T < E:", (r < 1).float().mean().item())

    lr = torch.log(torch.abs(T[where]) + 1e-12) - torch.log(torch.abs(E[where]) + 1e-12)
    print("median log-ratio:", torch.median(lr).item())


@dataclass
class MTAlgoConfig(AlgoConfig):
    """Extended Configuration for the multi-turn algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        norm_adv_in_mtr (bool): Whether to normalize advantage in multi-turn reinforce.
    """

    norm_adv_in_mtr: bool = (
        False  # whether to normalize advantage in multi-turn reinforce
    )
    only_propagate_eog_in_mtr: bool = (
        True  # whether to only propagate eog reward in multi-turn reinforce
    )
    clip_adv_in_mtr: bool = False  # whether to clip advantage in multi-turn reinforce
    clip_adv_min: float = -3.0  # min value to clip advantage
    clip_adv_max: float = 3.0  # max value to clip advantage
