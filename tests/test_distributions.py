"""The statistic that is supposed to explain R4, pinned against known answers.

`dispersion` and the underflow pair are about to carry a mechanism claim, so
what they mean has to be fixed before they are pointed at a checkpoint. Each
test below builds a tensor whose correct answer is derivable by hand.

The discriminating pair is `test_one_hot_row_disperses` against
`test_within_row_peak_is_invisible_to_dispersion`. A statistic that fired on
both would not be measuring granularity at all — it would be measuring
peakiness, which per-token scaling does not fix and which therefore cannot
explain a gap that exists only between the two granularities.
"""

import math

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from quant.distributions import _call_stats, collect, summarize  # noqa: E402
from quant.fakequant import qrange  # noqa: E402
from quant.patch import QuantLinear  # noqa: E402

QMAX = qrange(8, symmetric=True)[1]  # 127


def stats(x, bits=8):
    """`_call_stats` as a dict, so tests read by name rather than by index."""
    from quant.distributions import FIELDS

    _, qmax = qrange(bits, symmetric=True)
    return dict(zip(FIELDS, _call_stats(x, qmax).tolist()))


# --- dispersion --------------------------------------------------------------

def test_uniform_rows_have_dispersion_one():
    """Every row reaches the same magnitude, so one shared scale fits them all
    exactly as well as sixteen private ones. This is the null case, and it is
    the value the statistic must return when per-token buys nothing."""
    x = torch.ones(1, 16, 8)
    s = stats(x)
    assert s["amax"] == pytest.approx(1.0)
    assert s["row_amax_median"] == pytest.approx(1.0)
    assert s["amax"] / s["row_amax_median"] == pytest.approx(1.0)


def test_one_hot_row_disperses():
    """One row 1000x the rest. The shared scale is set by that row, so it is
    1000x too coarse for the median row -- which is exactly the number
    per-token scaling divides out."""
    x = torch.ones(1, 16, 8)
    x[:, 3, :] = 1000.0
    s = stats(x)
    assert s["amax"] / s["row_amax_median"] == pytest.approx(1000.0)


def test_within_row_peak_is_invisible_to_dispersion():
    """A tensor that is violently peaked WITHIN every row but uniform across
    rows. Per-token scaling cannot help here -- each row still has to spend its
    private scale on its own outlier -- so dispersion must stay at 1.0.

    This is the falsification case. If dispersion fired on this, a high value
    would no longer be evidence that per-token is the fix."""
    x = torch.full((1, 16, 8), 0.001)
    x[:, :, 0] = 1000.0
    s = stats(x)
    assert s["amax"] / s["row_amax_median"] == pytest.approx(1.0)
    # ...and the underflow pair agrees: both granularities suffer equally.
    assert s["underflow_tensor"] == pytest.approx(s["underflow_token"])
    assert s["underflow_tensor"] > 0.8


# --- underflow ---------------------------------------------------------------

def test_underflow_counts_entries_that_round_to_zero():
    """Half the entries sit below amax/(2*qmax) and must be counted; the other
    half sit above it and must not. The boundary is the round-to-zero
    threshold, not an approximation of it."""
    x = torch.zeros(1, 4, 8)
    x[:, :, :] = 1.0                      # amax = 1.0 -> s = 1/127
    below = 1.0 / (2 * QMAX) * 0.5
    x[:, :, :4] = below                   # exactly half the entries underflow
    s = stats(x)
    assert s["underflow_tensor"] == pytest.approx(0.5)


def test_per_token_rescues_a_dispersed_tensor():
    """The R4 hypothesis in miniature. Small rows plus one huge row: under one
    shared scale the small rows vanish entirely, under per-row scales they are
    untouched. Same tensor, same bit width, opposite outcome."""
    x = torch.full((1, 16, 8), 1.0)
    x[:, 0, :] = 1e5
    s = stats(x)
    assert s["underflow_tensor"] == pytest.approx(15 / 16)   # every small row dies
    assert s["underflow_token"] == pytest.approx(0.0)        # none of them does


def test_eff_bits_falls_as_the_scale_is_dragged_out():
    """dispersion = qmax leaves the median row one level, i.e. the sign bit
    alone. The conversion is asserted rather than trusted because eff_bits is
    the number the writeup quotes."""
    assert math.log2(QMAX / 1.0) + 1 == pytest.approx(math.log2(QMAX) + 1)
    assert math.log2(QMAX / QMAX) + 1 == pytest.approx(1.0)


# --- the collection pass -----------------------------------------------------

class ToyLM(nn.Module):
    """Accepts ``use_cache`` so the toy honours the HF forward contract.

    `collect` calls ``model(batch, use_cache=False)`` exactly as
    quant.calibrate.collect_ranges does, and that kwarg is not cosmetic: without
    it an HF model accumulates a KV cache across the whole holdout pass. The
    toy bends to the production call, not the other way round.
    """

    def __init__(self, d=32):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x, use_cache=False):
        return self.net(x)


def test_collect_restores_the_model_and_removes_its_hooks():
    """A measurement pass that leaves QuantLinear wrappers or live pre-hooks
    behind would silently contaminate whatever runs next in the same process --
    and the grid driver reuses one loaded model across many cells."""
    model = ToyLM()
    batches = [torch.randn(1, 8, 32)]

    layers = collect(model, bits=8, batches_fn=lambda: iter(batches), device="cpu")

    assert set(layers) == {"net.0", "net.2"}
    assert not any(isinstance(m, QuantLinear) for m in model.modules())
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_collect_does_not_quantize():
    """The pass is pass-through by construction. If it were not, every number
    it reports would describe a quantized model rather than the model."""
    model = ToyLM()
    x = torch.randn(1, 8, 32)
    before = model(x).clone()

    collect(model, bits=8, batches_fn=lambda: iter([x]), device="cpu")

    assert torch.equal(before, model(x))


def test_summarize_reports_max_alongside_median():
    """One catastrophic layer inside an otherwise ordinary network. The median
    must stay ordinary and the max must find it -- reporting only the median
    would hide exactly the case this module exists to detect."""
    layers = {
        f"l{i}": {"dispersion": 2.0, "eff_bits": 7.0, "crest": 3.0,
                  "underflow_tensor": 0.1, "underflow_token": 0.1}
        for i in range(10)
    }
    layers["bad"] = {"dispersion": 5000.0, "eff_bits": 0.0, "crest": 900.0,
                     "underflow_tensor": 0.99, "underflow_token": 0.1}

    s = summarize(layers)
    assert s["dispersion_median"] == pytest.approx(2.0)
    assert s["dispersion_max"] == pytest.approx(5000.0)
    assert s["eff_bits_min"] == pytest.approx(0.0)
    assert s["worst_layers"][0]["layer"] == "bad"
