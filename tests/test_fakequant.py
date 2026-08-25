"""HARD GATE — plan §9 trap 1 / §11 Day 3. Nothing downstream is valid until
every test in this file passes.

The failure this exists to catch: a fake-quant that silently does nothing. If
W8/per-channel AND A8/per-tensor both come out lossless, the quantizer is a
no-op, every Delta-ppl is zero-plus-noise, and D_sink -- the headline metric --
is measuring rounding in the evaluation harness rather than in the model.

The signature of a correct implementation is ASYMMETRIC:
    W8  per-channel  -> near lossless      (this is the control)
    A8  per-tensor   -> visibly degrades
    A4  per-tensor   -> degrades badly
A quantizer that passes the first assertion and fails the other two is not
"partly working", it is broken in the specific way that invalidates the project.
"""

import pytest

torch = pytest.importorskip("torch")

from quant.fakequant import quantize_dequantize  # noqa: E402


def rel_err(x, x_hat):
    return ((x - x_hat).norm() / x.norm()).item()


@pytest.fixture
def weights():
    """Well-behaved, roughly per-channel-homogeneous — like real weight matrices."""
    torch.manual_seed(0)
    return torch.randn(512, 512)


@pytest.fixture
def activations_with_outliers():
    """Activations carrying massive activations, as measured in §6.

    A few channels blown up by ~100x, on a few token positions. This is the
    regime the whole project is about; a quantizer that is only tested on
    Gaussian noise will look fine and then lie on real data.
    """
    torch.manual_seed(0)
    x = torch.randn(128, 512)
    x[0, [7, 42, 311]] *= 100.0     # position 0 = the sink
    x[64, [7, 42]] *= 60.0          # a mid-sequence secondary sink
    return x


# --- the three load-bearing assertions --------------------------------------

def test_w8_per_channel_is_near_lossless(weights):
    err = rel_err(weights, quantize_dequantize(weights, bits=8, granularity="per_channel"))
    assert err < 0.01, f"W8 per-channel should be near-lossless, got rel err {err:.4f}"


def test_a8_per_tensor_visibly_degrades(activations_with_outliers):
    x = activations_with_outliers
    err = rel_err(x, quantize_dequantize(x, bits=8, granularity="per_tensor"))
    assert err > 0.01, (
        f"A8 per-tensor rel err {err:.4f} is suspiciously low. With 100x outliers "
        "present the shared scale must be dragged out and the non-outlier values "
        "crushed. This near-certainly means the quantizer is a no-op."
    )


def test_a4_per_tensor_degrades_badly(activations_with_outliers):
    x = activations_with_outliers
    err = rel_err(x, quantize_dequantize(x, bits=4, granularity="per_tensor"))
    assert err > 0.10, f"A4 per-tensor should be severely damaged, got {err:.4f}"


# --- the axis the project turns on ------------------------------------------

def test_per_token_beats_per_tensor_on_outlier_activations(activations_with_outliers):
    """Per-token scaling isolates the outlier rows, so it MUST do better.

    If this does not hold, the modern-baseline arm is misconfigured and the
    central research question cannot be asked.
    """
    x = activations_with_outliers
    per_tensor = rel_err(x, quantize_dequantize(x, bits=8, granularity="per_tensor"))
    per_token = rel_err(x, quantize_dequantize(x, bits=8, granularity="per_token"))
    assert per_token < per_tensor, (
        f"per_token ({per_token:.4f}) should beat per_tensor ({per_tensor:.4f}) "
        "when outliers are confined to a few token positions"
    )


# --- generic sanity ----------------------------------------------------------

def test_output_is_actually_on_the_grid(weights):
    """Round-tripped values must land on a discrete grid, not pass through."""
    x_hat = quantize_dequantize(weights, bits=4, granularity="per_tensor")
    assert x_hat.unique().numel() <= 2 ** 4, "more distinct values than the grid allows"


def test_lower_bits_never_help(activations_with_outliers):
    x = activations_with_outliers
    errs = [rel_err(x, quantize_dequantize(x, bits=b, granularity="per_tensor"))
            for b in (8, 6, 4)]
    assert errs == sorted(errs), f"error must be monotone in bit width, got {errs}"


def test_is_not_the_identity_function(weights):
    """The blunt version of the no-op check."""
    x_hat = quantize_dequantize(weights, bits=4, granularity="per_tensor")
    assert not torch.equal(weights, x_hat), "quantizer returned its input unchanged"
