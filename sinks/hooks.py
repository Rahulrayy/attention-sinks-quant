"""Per-module forward hooks that reduce to scalars *inside* the hook.

Trap §9.2 — do NOT set ``output_attentions=True`` and read the result. HF
accumulates every layer's probabilities into one output tuple, so peak memory is
``L x B x H x T x T x 2`` bytes: 5.9 GB for a 22L/32H model at T=2048. It also
silently disables SDPA/FlashAttention for a 2-3x slowdown.

The fix used here is that a forward hook may REPLACE a module's output. Each
attention hook reads the probabilities, folds them into running statistics, and
returns the output tuple with the probability tensor swapped for ``None``. The
model then stores ``None`` in its ``all_attentions`` tuple, so peak memory holds
one attention tensor rather than L of them, and the statistics are already
reduced to floats by the time the tensor is freed.

Trap §9.3 — never stash a raw activation tensor on an accumulator. Running
moments update in-hook; the tensor is dropped immediately. If you find yourself
writing ``self.acc.append(tensor)``, stop.

Module discovery is structural, not by class name: these hooks must attach to
the *vendored* ``modeling_qwen3.py`` inside the QwQZh/gated_attention subfolders
as well as to stock GPT-2, and those do not share a naming convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .metrics import head_entropy, received_attention, residual_inf_norm, sink_mass


@dataclass
class StreamingStats:
    """Online moments and running max. Holds floats. Never holds a tensor.

    Moments merge with the Pebay batch formulas rather than accumulating raw
    power sums: activations here reach magnitudes of ~1e3, and a fourth power
    sum over millions of elements loses too much precision in the subtraction
    that recovers the centred moment.
    """

    count: int = 0
    running_max: float = float("-inf")
    mean: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        """Fold one batch of values in. The caller drops the tensor afterwards."""
        x = tensor.detach().to(torch.float64).flatten()
        n_b = x.numel()
        if n_b == 0:
            return

        self.running_max = max(self.running_max, x.abs().max().item())

        mean_b = x.mean().item()
        d = x - mean_b
        m2_b = (d**2).sum().item()
        m3_b = (d**3).sum().item()
        m4_b = (d**4).sum().item()

        n_a = self.count
        if n_a == 0:
            self.count, self.mean, self.m2, self.m3, self.m4 = n_b, mean_b, m2_b, m3_b, m4_b
            return

        n = n_a + n_b
        delta = mean_b - self.mean
        m2_a, m3_a, m4_a = self.m2, self.m3, self.m4

        self.mean += delta * n_b / n
        self.m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
        self.m3 = (
            m3_a + m3_b
            + delta**3 * n_a * n_b * (n_a - n_b) / n**2
            + 3 * delta * (n_a * m2_b - n_b * m2_a) / n
        )
        self.m4 = (
            m4_a + m4_b
            + delta**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / n**3
            + 6 * delta**2 * (n_a**2 * m2_b + n_b**2 * m2_a) / n**2
            + 4 * delta * (n_a * m3_b - n_b * m3_a) / n
        )
        self.count = n

    def kurtosis(self) -> float:
        """Excess (Fisher) kurtosis — Bondarenko's outlier proxy.

        0 for a Gaussian, so the numbers are directly comparable to Quantizable
        Transformers rather than off by 3.
        """
        if self.count < 2 or self.m2 <= 0:
            return 0.0
        return self.count * self.m4 / self.m2**2 - 3.0

    def variance(self) -> float:
        return self.m2 / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "max_abs": self.running_max if self.count else 0.0,
            "mean": self.mean,
            "variance": self.variance(),
            "kurtosis": self.kurtosis(),
        }


@dataclass
class AttentionRecord:
    """Running per-head sink mass and entropy for one attention module."""

    name: str
    sink_mass_sum: torch.Tensor | None = None      # (H,) — a statistic, not an activation
    entropy_sum: torch.Tensor | None = None        # (H,)
    received_sum: torch.Tensor | None = None       # (T,)
    batches: int = 0

    def update(self, probs: torch.Tensor) -> None:
        s = sink_mass(probs).mean(dim=0).detach().cpu().to(torch.float64)
        h = head_entropy(probs).mean(dim=0).detach().cpu().to(torch.float64)
        r = received_attention(probs).mean(dim=0).detach().cpu().to(torch.float64)

        if self.batches == 0:
            self.sink_mass_sum, self.entropy_sum, self.received_sum = s, h, r
        else:
            self.sink_mass_sum += s
            self.entropy_sum += h
            self.received_sum += r
        self.batches += 1

    def as_dict(self) -> dict:
        if not self.batches:
            return {"name": self.name, "batches": 0}
        s = self.sink_mass_sum / self.batches
        return {
            "name": self.name,
            "batches": self.batches,
            "sink_mass_per_head": s.tolist(),
            "entropy_per_head": (self.entropy_sum / self.batches).tolist(),
            "received_attention": (self.received_sum / self.batches).tolist(),
            "frac_heads_sinking": float((s > 0.5).to(torch.float64).mean().item()),
        }


@dataclass
class ResidualRecord:
    """Running per-token inf-norm and activation moments for one block."""

    name: str
    max_norm: torch.Tensor | None = None           # (T,)
    stats: StreamingStats = field(default_factory=StreamingStats)
    channel_max: torch.Tensor | None = None        # (C,)

    def update(self, hidden: torch.Tensor) -> None:
        norms = residual_inf_norm(hidden).amax(dim=0).detach().cpu().to(torch.float64)
        self.max_norm = norms if self.max_norm is None else torch.maximum(self.max_norm, norms)

        ch = hidden.detach().abs().amax(dim=(0, 1)).cpu().to(torch.float64)
        self.channel_max = ch if self.channel_max is None else torch.maximum(self.channel_max, ch)

        self.stats.update(hidden)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "per_token_inf_norm": self.max_norm.tolist() if self.max_norm is not None else [],
            "channel_max": self.channel_max.tolist() if self.channel_max is not None else [],
            **self.stats.as_dict(),
        }


@dataclass
class HookHandles:
    """Owns registered handles so a measurement run can always clean up."""

    handles: list = field(default_factory=list)

    def add(self, handle) -> None:
        self.handles.append(handle)

    def remove_all(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> HookHandles:
        return self

    def __exit__(self, *exc) -> None:
        self.remove_all()


# --- structural discovery ----------------------------------------------------

def _as_tuple(output):
    return output if isinstance(output, tuple) else (output,)


def _find_attention_probs(output) -> tuple[int, torch.Tensor] | None:
    """Locate the (B, H, T, T) probability tensor in a module output.

    Identified by shape rather than by tuple position: GPT-2 and the vendored
    Qwen3 modelling code put it in different slots, and a positional assumption
    would silently read the wrong tensor on one of them.
    """
    for i, item in enumerate(_as_tuple(output)):
        if isinstance(item, torch.Tensor) and item.dim() == 4 and item.shape[-1] == item.shape[-2]:
            return i, item
    return None


def _find_hidden_states(output) -> torch.Tensor | None:
    """Locate the (B, T, C) residual-stream tensor in a block output."""
    for item in _as_tuple(output):
        if isinstance(item, torch.Tensor) and item.dim() == 3:
            return item
    return None


def _looks_like_attention(module) -> bool:
    """Structural test: a module holding q/k/v projections, or GPT-2's fused c_attn."""
    children = dict(module.named_children())
    return ("q_proj" in children and "k_proj" in children) or "c_attn" in children


def _looks_like_block(module) -> bool:
    """A decoder block: contains an attention submodule somewhere beneath it."""
    return any(_looks_like_attention(m) for m in module.children())


# --- hook attachment ---------------------------------------------------------

def attach_attention_hooks(
    model,
    records: dict[str, AttentionRecord],
    *,
    null_out_probs: bool = True,
) -> HookHandles:
    """Hook each attention module; accumulate sink mass, entropy, received mass.

    Requires the model to be built with ``attn_implementation="eager"`` and run
    with ``output_attentions=True`` — SDPA never materialises the probabilities,
    so there would be nothing to read.

    With ``null_out_probs`` the hook returns the output tuple with the
    probability tensor replaced by ``None``, which is what keeps peak memory at
    one attention tensor instead of L. Turn it off only to debug the hook
    itself, and expect the memory blowup described in trap §9.2 if you do.
    """
    handles = HookHandles()

    for name, module in model.named_modules():
        if not _looks_like_attention(module):
            continue
        records.setdefault(name, AttentionRecord(name))

        def hook(mod, args, output, _name=name):
            found = _find_attention_probs(output)
            if found is None:
                return None
            idx, probs = found
            records[_name].update(probs)
            if not null_out_probs or not isinstance(output, tuple):
                return None
            replaced = list(output)
            replaced[idx] = None
            return tuple(replaced)

        handles.add(module.register_forward_hook(hook))

    if not handles.handles:
        raise RuntimeError(
            "no attention modules found. Discovery is structural (q_proj/k_proj "
            "or c_attn); if the vendored modelling code names its projections "
            "differently, extend _looks_like_attention rather than guessing at "
            "class names."
        )
    return handles


def attach_residual_hooks(model, records: dict[str, ResidualRecord]) -> HookHandles:
    """Hook each decoder block's output; accumulate per-token inf-norms."""
    handles = HookHandles()

    for name, module in model.named_modules():
        if not _looks_like_block(module):
            continue
        records.setdefault(name, ResidualRecord(name))

        def hook(mod, args, output, _name=name):
            hidden = _find_hidden_states(output)
            if hidden is not None:
                records[_name].update(hidden)
            return None

        handles.add(module.register_forward_hook(hook))

    if not handles.handles:
        raise RuntimeError("no decoder blocks found; extend _looks_like_block")
    return handles
