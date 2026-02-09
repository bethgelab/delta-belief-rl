from contextlib import contextmanager
import torch
from peft.tuners.lora.layer import LoraLayer
from collections import OrderedDict
from contextlib import nullcontext
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from collections import defaultdict


def _get_active_adapter(model):
    return getattr(model, "active_adapter", None)


def _set_adapter(model, name_or_none):
    model.set_adapter(name_or_none)


def _peft_under_fsdp(module):
    # Use wrapped module if present (FSDP)
    return getattr(module, "_fsdp_wrapped_module", module)


@contextmanager
def _use_adapter(model, name_or_none):
    m = _peft_under_fsdp(model)
    # return base model if no adapter name given
    if name_or_none is None:
        if not hasattr(m, "disable_adapter"):
            raise RuntimeError(
                "This PEFT version cannot disable adapters without set_adapter(None/[]). "
                "Please upgrade peft (>=0.11) to use disable_adapter() safely."
            )
        with m.disable_adapter():
            yield
        return

    # Named adapter case, asssumes single active adapter present
    prev = _get_active_adapter(m)
    _set_adapter(m, name_or_none)
    try:
        yield
    finally:
        _set_adapter(m, prev)


@contextmanager
def _maybe_summon_full_params_if_fsdp(module):
    # Enter FSDP full-params context only if needed
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if hasattr(module, "_is_fsdp_module") or isinstance(module, FSDP):
            with FSDP.summon_full_params(module, writeback=False):
                yield
            return
    except Exception:
        pass
    yield


@torch.no_grad()
def _adapter_signature(peft_model, adapter: str):
    """Cheap fingerprints for the adapter: per-tensor (sum, sumsq), and global counts."""
    sums, sumsqs = [], []
    n_el = 0
    for mod in peft_model.modules():
        if isinstance(mod, LoraLayer):
            A = mod.lora_A[adapter].weight
            B = mod.lora_B[adapter].weight
            for T in (A, B):
                t = T.detach()
                sums.append(t.sum().item())
                sumsqs.append((t * t).sum().item())
                n_el += t.numel()
            # (optional) embeddings
            if hasattr(mod, "lora_embedding_A") and adapter in mod.lora_embedding_A:
                EA = mod.lora_embedding_A[adapter].weight
                EB = mod.lora_embedding_B[adapter].weight
                for T in (EA, EB):
                    t = T.detach()
                    sums.append(t.sum().item())
                    sumsqs.append((t * t).sum().item())
                    n_el += t.numel()
    return {
        "n_elem": n_el,
        "sum": float(sum(sums)),
        "sumsq": float(sum(sumsqs)),
    }


@torch.no_grad()
def _adapter_distance(peft_model, src: str, dst: str):
    """
    Returns global distances between two adapters (A/B and embeddings if present).
    """
    metrics = defaultdict(float)
    eps = 1e-12
    layer_cnt = 0
    changed_layers = 0

    for mod in peft_model.modules():
        if not isinstance(mod, LoraLayer):
            continue
        layer_cnt += 1

        A_src, B_src = mod.lora_A[src].weight.detach(), mod.lora_B[src].weight.detach()
        A_dst, B_dst = mod.lora_A[dst].weight.detach(), mod.lora_B[dst].weight.detach()

        for tag, X, Y in (("A", A_src, A_dst), ("B", B_src, B_dst)):
            D = X - Y
            # absolute metrics
            metrics[f"{tag}/l1"] += D.abs().sum().item()
            metrics[f"{tag}/l2"] += torch.linalg.norm(D).item()
            metrics[f"{tag}/linf"] = max(metrics[f"{tag}/linf"], D.abs().max().item())

            # relative L2
            denom = torch.linalg.norm(X).item() + eps
            metrics[f"{tag}/rel_l2"] += torch.linalg.norm(D).item() / denom

            # cosine similarity (treat flattened vectors)
            dot = (X.flatten().double() @ Y.flatten().double()).item()
            nx = torch.linalg.norm(X).double().item() + eps
            ny = torch.linalg.norm(Y).double().item() + eps
            cos = dot / (nx * ny)
            metrics[f"{tag}/cos"] += float(cos)

            if D.abs().max().item() > 0:
                changed_layers += 1

        # (optional) embeddings
        if (
            hasattr(mod, "lora_embedding_A")
            and src in mod.lora_embedding_A
            and dst in mod.lora_embedding_A
        ):
            EA_src = mod.lora_embedding_A[src].weight.detach()
            EB_src = mod.lora_embedding_B[src].weight.detach()
            EA_dst = mod.lora_embedding_A[dst].weight.detach()
            EB_dst = mod.lora_embedding_B[dst].weight.detach()
            for tag, X, Y in (("EA", EA_src, EA_dst), ("EB", EB_src, EB_dst)):
                D = X - Y
                metrics[f"{tag}/l1"] += D.abs().sum().item()
                metrics[f"{tag}/l2"] += torch.linalg.norm(D).item()
                metrics[f"{tag}/linf"] = max(
                    metrics[f"{tag}/linf"], D.abs().max().item()
                )
                denom = torch.linalg.norm(X).item() + eps
                metrics[f"{tag}/rel_l2"] += torch.linalg.norm(D).item() / denom
                dot = (X.flatten().double() @ Y.flatten().double()).item()
                nx = torch.linalg.norm(X).double().item() + eps
                ny = torch.linalg.norm(Y).double().item() + eps
                metrics[f"{tag}/cos"] += float(dot / (nx * ny))
                if D.abs().max().item() > 0:
                    changed_layers += 1

    # normalize “averaged” metrics by layer count where it makes sense
    if layer_cnt > 0:
        for k in list(metrics.keys()):
            if k.endswith(("/rel_l2", "/cos")):
                metrics[k] /= layer_cnt
    metrics["changed_layers_ratio"] = changed_layers / max(layer_cnt, 1)
    return dict(metrics)


def ema_update_adapter(peft_model, src_name: str, dst_name: str, beta: float):
    """
    dst := beta * dst + (1 - beta) * src   (in-place EMA on LoRA tensors)
    """
    assert 0.0 <= beta <= 1.0, "EMA beta must be in [0,1]"
    # sanity: both adapters must exist in this PEFT model
    pcfg = getattr(peft_model, "peft_config", {})
    if isinstance(pcfg, dict):
        assert src_name in pcfg, f"Missing src adapter '{src_name}'"
        assert dst_name in pcfg, f"Missing dst adapter '{dst_name}'"

    with torch.no_grad():
        for module in peft_model.modules():
            if hasattr(module, "lora_A") and src_name in module.lora_A:
                # A matrices
                A_src = module.lora_A[src_name].weight
                A_dst = module.lora_A[dst_name].weight
                A_dst.mul_(beta).add_(A_src, alpha=(1 - beta))

                # B matrices
                B_src = module.lora_B[src_name].weight
                B_dst = module.lora_B[dst_name].weight
                B_dst.mul_(beta).add_(B_src, alpha=(1 - beta))


def layered_summon_lora_params(fsdp_module, adapter_name: str) -> OrderedDict:
    """
    Collect LoRA tensors for a specific adapter from an FSDP-wrapped PEFT model.
    Returns CPU, contiguous tensors keyed the same way PEFT emits them.

    NOTE:
    - Do NOT pass `state_dict=` to PEFT; it confuses adapter scoping.
    - Always pass `adapter_name=...`.
    - We optionally filter by submodule prefixes to keep memory peak low.
    """
    from peft.utils.save_and_load import get_peft_model_state_dict

    def _prefix_children(module, prefix_root):
        # yield immediate children whose qualified name starts with prefix_root
        for name, submodule in module.named_modules():
            if name.startswith(prefix_root) and "." not in name[len(prefix_root) :]:
                yield name, submodule

    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    # some PEFT versions need the adapter active to materialize correct keys
    if hasattr(peft_model, "set_adapter"):
        try:
            peft_model.set_adapter(adapter_name)
        except Exception:
            pass

    lora_params = OrderedDict()

    # Common prefixes seen with different wrappers
    prefix_list = [
        # FSDP1
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # FSDP2 / plain
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]

    for root_prefix in prefix_list:
        for qual_name, submodule in _prefix_children(fsdp_module, root_prefix):
            # skip container nodes
            if qual_name.endswith(".model") or qual_name.endswith(".layers"):
                continue

            ctx = (
                FSDP.summon_full_params(submodule, writeback=False)
                if isinstance(submodule, FSDP) or hasattr(submodule, "_is_fsdp_module")
                else nullcontext()
            )
            with ctx:
                # Get the FULL adapter dict from the PEFT model (for this adapter)
                full_adapter_sd = get_peft_model_state_dict(
                    peft_model, adapter_name=adapter_name
                )

                # Filter keys that belong under this submodule prefix.
                # Normalize the emitted PEFT keys to the same root seen in named_modules().
                # We normalize the left part used above:
                norm_prefix = qual_name.replace("_fsdp_wrapped_module.", "")
                sub_sd = {
                    k: v
                    for k, v in full_adapter_sd.items()
                    if k.startswith(norm_prefix)
                }

                if not sub_sd:
                    continue

                # Ensure CPU + contiguous (required by most tensor consumers, e.g. vLLM)
                for k, v in sub_sd.items():
                    if hasattr(v, "full_tensor"):  # FSDP flat param
                        v = v.full_tensor()
                    lora_params[k] = v.detach().cpu().contiguous()

    return lora_params


def lora_params_all(fsdp_module, adapter_name: str) -> OrderedDict:
    """
    Correct adapter must be set before
    Here only extract the adapter weights
    """
    from peft.utils.save_and_load import get_peft_model_state_dict

    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    with (
        FSDP.summon_full_params(fsdp_module, writeback=False)
        if hasattr(fsdp_module, "_is_fsdp_module")
        else nullcontext()
    ):
        sd = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
    return OrderedDict((k, v.detach().cpu().contiguous()) for k, v in sd.items())
