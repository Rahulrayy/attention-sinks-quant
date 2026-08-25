"""Magnitude-based multi-level sink detection.

    sink(t) = True  iff  M(t) > tau * median_t(M(t))

Why not ``t == 0``: CushionCache reports three sink levels in QwQ-32B and six
in Qwen3-14B. A position_0 rule silently misses most of them on exactly the
model family this project is built on, and every attribution number downstream
would then be computed against an incomplete sink set.

Side benefit: this needs only residual-stream norms, no attention probabilities,
which is also the way around the output_attentions memory blowup (trap §9.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


class DetectorValidationError(RuntimeError):
    """Raised when flagged tokens do NOT receive elevated attention mass.

    This is a hard gate, not a warning. If the detector and the attention map
    disagree, the entire attribution chain — and therefore D_sink — is broken.
    Day 2 does not pass while this can be raised.
    """


@dataclass
class DetectionResult:
    tau: float
    mask: torch.Tensor                     # bool (B, T)
    n_flagged: int
    positions: list[int] = field(default_factory=list)   # union over the batch

    def as_dict(self) -> dict:
        return {
            "tau": self.tau,
            "n_flagged": self.n_flagged,
            "positions": self.positions,
            "flagged_fraction": float(self.mask.float().mean().item()),
        }


def detect_sinks(aggregate_norms: torch.Tensor, tau: float) -> torch.Tensor:
    """Flag tokens whose aggregate inf-norm exceeds tau x the per-sequence median.

    ``aggregate_norms`` is M(t) with shape (B, T). The median is taken per
    sequence, not over the whole batch: activation scale drifts between
    sequences, and a batch-wide median would let one high-magnitude sequence
    suppress detection in the others.
    """
    if aggregate_norms.dim() != 2:
        raise ValueError(f"expected (B, T), got {tuple(aggregate_norms.shape)}")
    if tau <= 1.0:
        raise ValueError(f"tau must exceed 1.0 to mean anything, got {tau}")

    m = aggregate_norms.detach().to(torch.float32)
    median = m.median(dim=-1, keepdim=True).values
    return m > tau * median


def layer_relative_norms(per_layer_norms: torch.Tensor) -> torch.Tensor:
    """(L, B, T) -> (B, T): each layer's norms divided by that layer's own median,
    then maxed over layers.

    This is the PRIMARY detector input, decided empirically on 2026-08-20 rather
    than by argument — see LIMITATIONS.md §11.

    The aggregate form M(t) = max_l m(t,l) maxes over layers BEFORE the median is
    taken, so the denominator is set by whichever layer has the largest typical
    activation. That compresses every ratio toward 1. Measured on real models the
    compression is severe: GPT-2 small put its position-0 sink at only 14.2x the
    median, Qwen3-0.6B-Base at 19.3x, both far under the tau=100 the plan
    originally proposed. Normalising within each layer first puts the same Qwen
    sink at 1154x, and the detector then validates against attention across the
    whole tau range from 5 to 100 instead of a narrow 5-10 window.

    Wider stable range is the point. A detector that only agrees with attention
    inside a narrow band of tau has been fitted to one model, not validated.
    """
    if per_layer_norms.dim() != 3:
        raise ValueError(f"expected (L, B, T), got {tuple(per_layer_norms.shape)}")
    median = per_layer_norms.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
    return (per_layer_norms / median).amax(dim=0)


def detect_sinks_layerwise(per_layer_norms: torch.Tensor, tau: float) -> torch.Tensor:
    """Flag tokens whose layer-relative norm exceeds tau in ANY layer.

    Unlike ``detect_sinks`` the threshold is absolute, because the input is
    already expressed as a ratio to each layer's own median.
    """
    if tau <= 1.0:
        raise ValueError(f"tau must exceed 1.0 to mean anything, got {tau}")
    return layer_relative_norms(per_layer_norms) > tau


def sweep_tau(aggregate_norms: torch.Tensor, taus: list[float]) -> dict[float, DetectionResult]:
    """Run the detector at every tau. Report sensitivity; never hard-code one value.

    The sweep IS the result. If the flagged set is stable across tau the finding
    is robust; if it collapses between 20 and 50 that instability is itself
    worth reporting, and quietly picking a tau would have hidden it.
    """
    out: dict[float, DetectionResult] = {}
    for tau in taus:
        mask = detect_sinks(aggregate_norms, tau)
        positions = torch.nonzero(mask.any(dim=0), as_tuple=False).flatten().tolist()
        out[tau] = DetectionResult(
            tau=tau, mask=mask, n_flagged=int(mask.sum().item()), positions=positions
        )
    return out


def sweep_tau_layerwise(
    per_layer_norms: torch.Tensor, taus: list[float]
) -> dict[float, DetectionResult]:
    """Layer-relative sweep. The primary detector; see layer_relative_norms."""
    ratios = layer_relative_norms(per_layer_norms)
    out: dict[float, DetectionResult] = {}
    for tau in taus:
        mask = ratios > tau
        positions = torch.nonzero(mask.any(dim=0), as_tuple=False).flatten().tolist()
        out[tau] = DetectionResult(
            tau=tau,
            mask=mask,
            n_flagged=int(mask.sum().item()),
            positions=positions,
        )
    return out


def validate_against_attention(
    mask: torch.Tensor,
    received: torch.Tensor,
    *,
    attention_percentile: int = 95,
    min_position_0_recall: float = 1.0,
    min_agreement: float = 1.0,
) -> dict:
    """Confirm flagged tokens actually receive elevated attention mass.

    ``mask`` is the detector output (B, T); ``received`` is per-token received
    attention (B, T) from ``metrics.received_attention``.

    Two independent checks, both of which must hold:

    1. Position 0 is recovered in at least ``min_position_0_recall`` of
       sequences. Whatever else the detector finds, it must not miss the one
       sink position the entire literature agrees on.
    2. At least ``min_agreement`` of flagged tokens sit above the
       ``attention_percentile`` of the received-attention distribution. A
       magnitude outlier that no head attends to is not a sink, and treating it
       as one would put junk into the fp16 exception list.

    Returns a report dict on success. Raises DetectorValidationError otherwise —
    loudly, never warn-and-continue.
    """
    if mask.shape != received.shape:
        raise ValueError(f"shape mismatch: mask {tuple(mask.shape)} vs received {tuple(received.shape)}")
    if not mask.any():
        raise DetectorValidationError(
            "detector flagged no tokens at all — tau is too high, or the "
            "residual-stream norms were never populated"
        )

    pos0_recall = float(mask[:, 0].float().mean().item())
    if pos0_recall < min_position_0_recall:
        raise DetectorValidationError(
            f"position 0 recovered in only {pos0_recall:.1%} of sequences "
            f"(need {min_position_0_recall:.1%}). The detector is missing the "
            "canonical sink; do not proceed to attribution."
        )

    r = received.detach().to(torch.float32)
    threshold = torch.quantile(r.flatten(), attention_percentile / 100.0)
    flagged = r[mask]
    agreement = float((flagged > threshold).float().mean().item())

    if agreement < min_agreement:
        bad = torch.nonzero(mask & (r <= threshold), as_tuple=False).tolist()
        raise DetectorValidationError(
            f"only {agreement:.1%} of flagged tokens exceed the p{attention_percentile} "
            f"received-attention threshold (need {min_agreement:.1%}). "
            f"Magnitude and attention disagree at (batch, pos): {bad[:20]}"
            f"{' ...' if len(bad) > 20 else ''}. The attribution chain is broken."
        )

    return {
        "position_0_recall": pos0_recall,
        "attention_agreement": agreement,
        "attention_threshold": float(threshold.item()),
        "n_flagged": int(mask.sum().item()),
    }
