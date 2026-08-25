"""Wrap nn.Linear to toggle weight/activation fake-quant, with an fp16 exception list.

The exception list is how D_sink is measured: quantize everything, then re-run
holding specific token positions (or channels) in fp16 and diff the damage.

    D_sink = dppl(exception=none) - dppl(exception=detected_sinks)

Note that ``position_0`` is KVQuant's published method. It is a CONTROL here,
not a contribution, and the README says so explicitly.

Convention, stated rather than buried: the LM head is left unquantized by
default, as is standard in this literature. It is a single large matmul whose
quantization damage is unrelated to the attention-sink mechanism, and including
it would add a constant offset to every arm of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fakequant import Granularity, apply_grid, quantize_dequantize, scale_from_amax

DEFAULT_SKIP = ("lm_head",)


def _is_conv1d(module: nn.Module) -> bool:
    """Detect HF's Conv1D, which GPT-2 uses instead of nn.Linear.

    Conv1D stores its weight TRANSPOSED — (in_features, out_features) — and
    computes ``x @ W + b`` rather than ``x @ W.T + b``. Two consequences, both
    of which are silent if missed:

      * a patcher that only looks for nn.Linear leaves GPT-2 entirely
        unquantized and reports zero damage everywhere;
      * per-channel weight quantization reduces over every axis but the first,
        which on a transposed weight yields one scale per INPUT channel. That
        does not crash. It just produces wrong numbers for the one checkpoint in
        the roster whose job is to be the pre-QK-Norm contrast.

    Duck-typed rather than imported so this module keeps working without
    transformers installed.
    """
    return (
        hasattr(module, "nf")
        and hasattr(module, "weight")
        and getattr(module.weight, "dim", lambda: 0)() == 2
    )


@dataclass
class ExceptionSpec:
    """Which activation entries stay in fp16.

    token_positions -> indices along the sequence axis (position_0, detected_sinks)
    channel_indices -> indices along the feature axis (outlier_channels)

    Empty on both axes means the ``none`` arm: full damage, the reference cell.
    """

    kind: str = "none"
    token_positions: list[int] = field(default_factory=list)
    channel_indices: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.token_positions and not self.channel_indices

    def entry_mask(self, x: torch.Tensor) -> torch.Tensor | None:
        """Bool mask over ``x``, True where an entry is held in fp16.

        Needed for scale derivation, not just for pasting values back: exempted
        entries must be excluded from the observed range, otherwise the tokens
        that ARE quantized still share a scale dragged out by the outlier and
        the exception buys nothing. See compute_scale_zero_point.
        """
        if self.is_empty:
            return None
        mask = torch.zeros_like(x, dtype=torch.bool)
        if self.token_positions and x.dim() >= 2:
            idx = torch.as_tensor(self.token_positions, device=x.device)
            idx = idx[idx < x.shape[-2]]
            if idx.numel():
                mask[..., idx, :] = True
        if self.channel_indices:
            idx = torch.as_tensor(self.channel_indices, device=x.device)
            idx = idx[idx < x.shape[-1]]
            if idx.numel():
                mask[..., idx] = True
        return mask

    def apply(self, x_quant: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """Restore the exempted entries to their original unquantized values."""
        if self.is_empty:
            return x_quant
        out = x_quant.clone()
        if self.token_positions and x_orig.dim() >= 2:
            idx = torch.as_tensor(self.token_positions, device=x_orig.device)
            idx = idx[idx < x_orig.shape[-2]]
            if idx.numel():
                out[..., idx, :] = x_orig[..., idx, :]
        if self.channel_indices:
            idx = torch.as_tensor(self.channel_indices, device=x_orig.device)
            idx = idx[idx < x_orig.shape[-1]]
            if idx.numel():
                out[..., idx] = x_orig[..., idx]
        return out


class QuantLinear(nn.Module):
    """Drop-in nn.Linear wrapper. Quant toggles without re-patching the model.

    Weight quantization is cached: weights are frozen during evaluation, so
    re-deriving the same grid on every forward pass would burn time for an
    identical result. ``invalidate_weight_cache`` exists for the case where
    that assumption stops holding.

    Activation scales are dynamic when no calibrated range is supplied. Static
    per-tensor scales come from quant.calibrate; per-token scales are inherently
    dynamic, since the whole point is that each token gets its own.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        w_bits: int | None = 8,
        a_bits: int | None = 8,
        w_granularity: Granularity = "per_channel",
        a_granularity: Granularity = "per_token",
        exceptions: ExceptionSpec | None = None,
        static_amax: float | None = None,
        transposed: bool = False,
    ):
        super().__init__()
        self.base = base
        self.transposed = transposed
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_granularity = w_granularity
        self.a_granularity = a_granularity
        self.exceptions = exceptions or ExceptionSpec()
        self.enabled = True
        self._w_cache: torch.Tensor | None = None

        # Static-calibration state. `static_amax` is the range observed during a
        # calibration pass; `observing` turns this layer into a pass-through that
        # only records what it sees.
        self.static_amax = static_amax
        self.observing = False
        self.observed_amax = 0.0

    def invalidate_weight_cache(self) -> None:
        self._w_cache = None

    def _weight(self) -> torch.Tensor:
        if not self.enabled or self.w_bits is None:
            return self.base.weight
        if self._w_cache is None:
            w = self.base.weight
            if self.transposed and self.w_granularity == "per_channel":
                # Conv1D weight is (in, out). Transpose so per_channel reduces
                # to one scale per OUTPUT channel, then transpose back.
                self._w_cache = quantize_dequantize(
                    w.t().contiguous(), self.w_bits, self.w_granularity
                ).t()
            else:
                self._w_cache = quantize_dequantize(w, self.w_bits, self.w_granularity)
        return self._w_cache

    def _activations(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations, excluding fp16-exempt entries from the range.

        The exclusion is the whole mechanism. Under per_tensor, dropping the
        sink token from the range lets every other token share a tight scale —
        that recovered headroom IS D_sink. Under per_token each row already has
        its own scale, so the exclusion changes nothing for the other rows, and
        D_sink should collapse toward zero. That contrast between the two
        granularities is the result this project exists to measure, so it must
        come out of the mechanism rather than being assumed.
        """
        if self.observing:
            # Pass-through: record the range this layer sees, excluding entries
            # that will be held in fp16. The exclusion has to happen HERE too,
            # not only at eval time — a range calibrated with the outlier still
            # in it would hand the quantized tokens the dragged-out scale that
            # the exception exists to avoid, and D_sink would come out flat.
            mask = self.exceptions.entry_mask(x)
            observed = x if mask is None else x.masked_fill(mask, 0.0)
            self.observed_amax = max(
                self.observed_amax, observed.detach().abs().amax().item()
            )
            return x

        if not self.enabled or self.a_bits is None:
            return x

        mask = self.exceptions.entry_mask(x)

        if self.a_granularity == "per_tensor" and self.static_amax is not None:
            scale, zp, qmin, qmax = scale_from_amax(self.static_amax, self.a_bits)
            xq = apply_grid(x, scale, zp, qmin, qmax)
        else:
            source = x if mask is None else x.masked_fill(mask, 0.0)
            xq = quantize_dequantize(
                x, self.a_bits, self.a_granularity, scale_source=source
            )
        return self.exceptions.apply(xq, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = self._activations(x)
        w = self._weight()
        if self.transposed:
            out = torch.addmm(self.base.bias, xq.view(-1, xq.size(-1)), w)
            return out.view(*xq.shape[:-1], w.shape[-1])
        return F.linear(xq, w, self.base.bias)

    def extra_repr(self) -> str:
        return (
            f"w{self.w_bits}/{self.w_granularity}, a{self.a_bits}/{self.a_granularity}, "
            f"exc={self.exceptions.kind}, enabled={self.enabled}"
        )


def patch_model(
    model: nn.Module,
    *,
    w_bits: int | None = 8,
    a_bits: int | None = 8,
    w_granularity: Granularity = "per_channel",
    a_granularity: Granularity = "per_token",
    exceptions: ExceptionSpec | None = None,
    skip: tuple[str, ...] = DEFAULT_SKIP,
):
    """Replace every nn.Linear with QuantLinear. Returns ``(restore, patched_names)``.

    ``restore`` puts the original modules back, so one loaded model can walk the
    whole grid without being reloaded from disk between configurations, which
    matters when the checkpoint is 2 GB and the grid has 120 cells.
    """
    replaced: list[tuple[nn.Module, str, nn.Linear]] = []
    names: list[str] = []

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            is_conv1d = _is_conv1d(child)
            if not isinstance(child, nn.Linear) and not is_conv1d:
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(pat in full for pat in skip):
                continue
            setattr(
                module,
                child_name,
                QuantLinear(
                    child,
                    w_bits=w_bits,
                    a_bits=a_bits,
                    w_granularity=w_granularity,
                    a_granularity=a_granularity,
                    exceptions=exceptions,
                    transposed=is_conv1d,
                ),
            )
            replaced.append((module, child_name, child))
            names.append(full)

    if not replaced:
        raise RuntimeError(
            "patch_model replaced nothing. Either the model exposes no nn.Linear "
            "or Conv1D layers via named_children, or every candidate matched "
            f"skip={skip}. A silently unpatched model reports zero quantization "
            "damage everywhere, which is exactly the no-op failure that "
            "tests/test_fakequant.py exists to catch."
        )

    def restore() -> None:
        for parent, child_name, original in replaced:
            setattr(parent, child_name, original)

    return restore, names


def set_quant_enabled(model: nn.Module, enabled: bool) -> int:
    """Toggle every QuantLinear in a patched model. Returns how many were hit.

    Used to take the fp16 reference measurement without unpatching, so the
    quantized and reference runs go through byte-identical code paths.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.enabled = enabled
            n += 1
    return n


def set_observing(model: nn.Module, observing: bool) -> int:
    """Put every QuantLinear into (or out of) calibration pass-through mode."""
    n = 0
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.observing = observing
            n += 1
    return n


def commit_observed_ranges(model: nn.Module) -> dict[str, float]:
    """Freeze each layer's observed range as its static scale.

    Returns the committed ranges so they can be written into the run JSON. A
    layer that observed nothing keeps ``static_amax = None`` and falls back to
    dynamic scaling rather than silently quantizing against a zero range.
    """
    committed: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear) and module.observed_amax > 0.0:
            module.static_amax = module.observed_amax
            committed[name] = module.observed_amax
    return committed


def resolve_fp16_exceptions(
    kind: str,
    *,
    sink_mask: torch.Tensor | None = None,
    outlier_mask: torch.Tensor | None = None,
) -> ExceptionSpec:
    """Build an ExceptionSpec for one cell of the configs/quant.yaml grid.

    kind is one of: none, position_0, detected_sinks, outlier_channels.
    """
    if kind == "none":
        return ExceptionSpec(kind)

    if kind == "position_0":
        return ExceptionSpec(kind, token_positions=[0])

    if kind == "detected_sinks":
        if sink_mask is None:
            raise ValueError("detected_sinks needs a sink_mask from sinks.detector")
        positions = torch.nonzero(sink_mask.any(dim=0), as_tuple=False).flatten().tolist()
        if not positions:
            raise ValueError(
                "sink_mask flagged nothing, so this arm would be identical to the "
                "none arm and D_sink would come out a spurious zero"
            )
        return ExceptionSpec(kind, token_positions=positions)

    if kind == "outlier_channels":
        if outlier_mask is None:
            raise ValueError("outlier_channels needs an outlier_mask from sinks.metrics")
        channels = torch.nonzero(outlier_mask, as_tuple=False).flatten().tolist()
        if not channels:
            raise ValueError("outlier_mask flagged no channels")
        return ExceptionSpec(kind, channel_indices=channels)

    raise ValueError(f"unknown fp16 exception kind: {kind!r}")
