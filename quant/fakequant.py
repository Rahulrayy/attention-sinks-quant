"""Fake quantization: simulated round-trip error in fp16.

NOT int8 inference. No bitsandbytes, no CUTLASS, no Ada requirement. Measuring
real quantized speedup is a different project (LIMITATIONS.md §3).

    q(x)  = clamp(round(x / s) + z, qmin, qmax)
    x_hat = s * (q - z)

Granularity is the axis the whole project turns on:

  per_tensor  one scale for the entire tensor. A single outlier drags the scale
              out and crushes everything else. This is where the classic 2023
              effect lives.
  per_token   one scale per token, i.e. per row of the activation. Outliers are
              confined to the rows that carry them. The MODERN baseline, and
              the arm that makes this project's central question askable.
  per_channel one scale per output channel of a weight matrix.

Note that per_token and per_channel reduce over the same axis for a 2-D tensor.
That is correct, not a copy-paste bug: an activation is (tokens, features) and
a weight is (out_features, in_features), so "one scale per row" means per-token
in the first case and per-output-channel in the second. They diverge for 3-D
activations, which is why they are kept as separate names.
"""

from __future__ import annotations

from typing import Literal

import torch

Granularity = Literal["per_tensor", "per_token", "per_channel"]

# Scales below this are treated as zero-valued tensors rather than divided by.
_EPS = 1e-12


def _reduce_dims(x: torch.Tensor, granularity: Granularity) -> tuple[int, ...]:
    """Which axes collapse when the scale is computed.

    per_tensor  -> every axis
    per_token   -> the feature axis, leaving one scale per token
    per_channel -> everything except axis 0, leaving one scale per out-channel
    """
    if granularity == "per_tensor":
        return tuple(range(x.dim()))
    if granularity == "per_token":
        return (-1,)
    if granularity == "per_channel":
        return tuple(range(1, x.dim()))
    raise ValueError(f"unknown granularity: {granularity!r}")


def compute_scale_zero_point(
    x: torch.Tensor,
    bits: int,
    granularity: Granularity,
    *,
    symmetric: bool = True,
    scale_source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Derive scale (and zero point) from the observed range along the reduce axes.

    Returns ``(scale, zero_point, qmin, qmax)``. ``scale`` and ``zero_point``
    broadcast against ``x``. Both are float32 regardless of the input dtype —
    deriving a scale in bf16 quantizes the quantizer, which is its own silent
    no-op failure mode.

    ``scale_source`` lets the range be derived from a DIFFERENT tensor than the
    one being quantized, which is what makes a mixed-precision exception list
    mean anything. Holding a sink token in fp16 is only worth doing because that
    token is then excluded from the range, letting every remaining token share a
    tight scale. Derive the scale from the full tensor and merely paste the fp16
    values back afterwards, and the surviving tokens still carry the dragged-out
    scale — the intervention would measure almost nothing under per_tensor,
    which is precisely the arm where the classic effect is supposed to live.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    src = x if scale_source is None else scale_source
    if src.shape != x.shape:
        raise ValueError(
            f"scale_source shape {tuple(src.shape)} must match x {tuple(x.shape)}"
        )

    xf = src.detach().to(torch.float32)
    dims = _reduce_dims(xf, granularity)

    if symmetric:
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        amax = xf.abs().amax(dim=dims, keepdim=True)
        scale = amax / qmax
        zero_point = torch.zeros_like(scale)
    else:
        qmin, qmax = 0, 2**bits - 1
        xmin = xf.amin(dim=dims, keepdim=True).clamp(max=0.0)
        xmax = xf.amax(dim=dims, keepdim=True).clamp(min=0.0)
        scale = (xmax - xmin) / (qmax - qmin)
        zero_point = torch.round(qmin - xmin / scale.clamp(min=_EPS))

    # A degenerate range means the tensor is constant along those axes; a scale
    # of exactly 0 would produce inf/nan rather than the correct passthrough.
    scale = scale.clamp(min=_EPS)
    return scale, zero_point, qmin, qmax


def qrange(bits: int, symmetric: bool = True) -> tuple[int, int]:
    """Integer grid bounds for a given bit width."""
    if symmetric:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def scale_from_amax(amax: float | torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build a symmetric scale from a PRE-OBSERVED absolute maximum.

    This is the static-calibration path: the range comes from a calibration pass
    rather than from the tensor being quantized. Per-tensor activation quant in
    the 2023 literature is static, and letting it re-derive its range from each
    evaluation batch would quietly hand it per-batch adaptivity that a deployed
    per-tensor scheme does not have — flattering exactly the arm this project is
    supposed to be auditing.
    """
    qmin, qmax = qrange(bits, symmetric=True)
    amax_t = torch.as_tensor(amax, dtype=torch.float32)
    scale = (amax_t / qmax).clamp(min=_EPS)
    return scale, torch.zeros_like(scale), qmin, qmax


def apply_grid(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """Round-trip ``x`` onto an integer grid defined by an existing scale."""
    xf = x.detach().to(torch.float32)
    scale = scale.to(xf.device)
    zero_point = zero_point.to(xf.device)
    q = torch.clamp(torch.round(xf / scale) + zero_point, qmin, qmax)
    return ((q - zero_point) * scale).to(x.dtype)


def quantize_dequantize(
    x: torch.Tensor,
    bits: int,
    granularity: Granularity,
    *,
    symmetric: bool = True,
    scale_source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Round-trip ``x`` through a ``bits``-bit grid at the given granularity.

    Returns a tensor of the same shape and dtype as ``x``, holding only values
    representable on that grid. The arithmetic runs in float32 and is cast back
    at the end.

    Pass ``scale_source`` to derive the range from a masked copy of ``x`` — see
    ``compute_scale_zero_point``. Values outside the derived range are clamped
    onto the grid rather than expanding it, which is the intended behaviour: an
    excluded outlier is meant to be handled by the fp16 exception list, not by
    quietly widening the scale for everyone else.
    """
    scale, zero_point, qmin, qmax = compute_scale_zero_point(
        x, bits, granularity, symmetric=symmetric, scale_source=scale_source
    )
    return apply_grid(x, scale, zero_point, qmin, qmax)
