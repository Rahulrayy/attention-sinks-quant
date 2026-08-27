"""Metric definitions. Locked before any data was seen — plan §6.

Changing a definition here after seeing results invalidates the audit framing.
If one genuinely has to change, record why in LIMITATIONS.md.

Every function here takes tensors and returns tensors or floats. None of them
load a model, read a config, or touch the filesystem.
"""

from __future__ import annotations

import torch

# Guards a log(0) inside the entropy sum. Attention probabilities are already
# non-negative, so this only ever affects exactly-zero entries.
_EPS = 1e-12


def sink_mass(
    attn_probs: torch.Tensor,
    *,
    sink_position: int = 0,
    exclude_self: bool = True,
) -> torch.Tensor:
    """S(l,h): mean over query positions q > 0 of attention weight on ``sink_position``.

    ``attn_probs`` is (B, H, T, T), softmax-normalised over the last (key) axis.
    Returns (B, H) — the FULL per-head distribution, deliberately not reduced
    further. The distribution is bimodal; a mean over heads hides exactly the
    structure this project is about. Callers report the distribution plus the
    fraction of heads with S > 0.5.

    Query position 0 is excluded by default: under a causal mask it can only
    attend to itself, so its "sink mass" is identically 1 and would bias every
    head upward by 1/T for reasons that have nothing to do with the phenomenon.
    """
    if attn_probs.dim() != 4:
        raise ValueError(f"expected (B, H, T, T), got {tuple(attn_probs.shape)}")

    to_sink = attn_probs[..., sink_position]          # (B, H, T)
    if exclude_self:
        to_sink = to_sink[..., sink_position + 1 :]
    if to_sink.numel() == 0:
        raise ValueError("no query positions left after excluding the sink position")
    return to_sink.mean(dim=-1)


def fraction_heads_sinking(sink_masses: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of heads with S > threshold. Reported alongside the distribution."""
    return (sink_masses > threshold).float().mean().item()


def head_entropy(attn_probs: torch.Tensor, *, exclude_first_query: bool = True) -> torch.Tensor:
    """H(l,h): mean Shannon entropy (nats) of the attention distribution over queries.

    Separates genuine sinking from merely diffuse attention. A head that puts
    0.9 on position 0 and a head that spreads uniformly can both look
    "unfocused" by some measures; they have very different entropies.

    Returns (B, H).
    """
    if attn_probs.dim() != 4:
        raise ValueError(f"expected (B, H, T, T), got {tuple(attn_probs.shape)}")

    p = attn_probs[:, :, 1:, :] if exclude_first_query else attn_probs
    ent = -(p * (p + _EPS).log()).sum(dim=-1)          # (B, H, T')
    return ent.mean(dim=-1)


def residual_inf_norm(hidden_states: torch.Tensor) -> torch.Tensor:
    """m(t,l) = max_c |x^(l)_{t,c}| for one layer. (B, T, C) -> (B, T)."""
    if hidden_states.dim() != 3:
        raise ValueError(f"expected (B, T, C), got {tuple(hidden_states.shape)}")
    return hidden_states.detach().abs().amax(dim=-1)


def aggregate_inf_norm(per_layer_norms: torch.Tensor) -> torch.Tensor:
    """M(t) = max_l m(t,l). Stack of (L, B, T) -> (B, T)."""
    if per_layer_norms.dim() != 3:
        raise ValueError(f"expected (L, B, T), got {tuple(per_layer_norms.shape)}")
    return per_layer_norms.amax(dim=0)


def outlier_channels(hidden_states: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
    """Channel c is flagged iff max_t |x_tc| > threshold * median_c(max_t |x_tc|).

    (B, T, C) -> bool (C,). The median is taken across channels, so the
    threshold is relative to a typical channel rather than to an absolute
    activation scale that varies by model and layer.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"expected (B, T, C), got {tuple(hidden_states.shape)}")

    per_channel_max = hidden_states.detach().abs().amax(dim=(0, 1)).to(torch.float32)
    median = per_channel_max.median()
    return per_channel_max > threshold * median


def aggregate_outlier_channels(
    per_layer_channel_max: torch.Tensor, threshold: float = 100.0
) -> torch.Tensor:
    """Model-wide outlier channels from per-layer maxima. (L, C) -> bool (C,).

    The channel-axis analogue of ``aggregate_inf_norm``, and reduced the same
    way: max over layers first, then the ``outlier_channels`` rule on the
    result. A residual-stream channel that is extreme anywhere in the network is
    extreme in the residual stream, because the residual stream is one object
    the blocks read and write in turn.

    This exists because the fp16 exception it feeds is applied to the whole
    model at once: ``ExceptionSpec.channel_indices`` is a single list of feature
    indices, not one list per layer. Some reduction across layers is therefore
    forced, and the choice must be stated rather than left to whoever calls it.

    The alternative — flag per layer, then union — is NOT the same set, and it
    is the more permissive one: a channel that is mild against its own layer's
    median can still be flagged there while being unremarkable against the
    network-wide median. Reducing first keeps one median for one mask, which is
    what the single-mask exception actually implements. ``union`` below computes
    the other set so the difference can be measured instead of argued.
    """
    if per_layer_channel_max.dim() != 2:
        raise ValueError(f"expected (L, C), got {tuple(per_layer_channel_max.shape)}")
    if per_layer_channel_max.numel() == 0:
        raise ValueError("no per-layer channel maxima to reduce")

    per_channel_max = per_layer_channel_max.detach().abs().amax(dim=0).to(torch.float32)
    median = per_channel_max.median()
    return per_channel_max > threshold * median


def union_outlier_channels(
    per_layer_channel_max: torch.Tensor, threshold: float = 100.0
) -> torch.Tensor:
    """Flag per layer against that layer's own median, then union. (L, C) -> (C,).

    The reduction ``aggregate_outlier_channels`` does not use. Kept so the gap
    between the two definitions is a number a caller can print, rather than a
    judgement call buried in whichever one got written first.
    """
    if per_layer_channel_max.dim() != 2:
        raise ValueError(f"expected (L, C), got {tuple(per_layer_channel_max.shape)}")
    if per_layer_channel_max.numel() == 0:
        raise ValueError("no per-layer channel maxima to reduce")

    m = per_layer_channel_max.detach().abs().to(torch.float32)
    medians = m.median(dim=1, keepdim=True).values
    return (m > threshold * medians).any(dim=0)


def excess_kurtosis(x: torch.Tensor) -> float:
    """Per-layer activation kurtosis — Bondarenko's outlier proxy.

    Excess (Fisher) kurtosis: 0 for a Gaussian. Included in this form so the
    numbers are directly comparable to Quantizable Transformers rather than
    off by 3.
    """
    xf = x.detach().to(torch.float32).flatten()
    centred = xf - xf.mean()
    var = centred.pow(2).mean()
    if var < _EPS:
        return 0.0
    return (centred.pow(4).mean() / var.pow(2) - 3.0).item()


def received_attention(attn_probs: torch.Tensor, *, exclude_first_query: bool = True) -> torch.Tensor:
    """Attention mass each token RECEIVES, averaged over heads and queries.

    (B, H, T, T) -> (B, T). This is the quantity the magnitude-based detector is
    validated against: a token flagged by residual-stream magnitude alone must
    also turn out to be one that heads actually attend to.
    """
    if attn_probs.dim() != 4:
        raise ValueError(f"expected (B, H, T, T), got {tuple(attn_probs.shape)}")

    p = attn_probs[:, :, 1:, :] if exclude_first_query else attn_probs
    return p.mean(dim=(1, 2))
