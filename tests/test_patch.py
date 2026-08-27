"""The fp16-exception mechanism that produces D_sink.

The tests that matter here are the last group. D_sink is the headline metric,
and it is only meaningful if holding a sink token in fp16 actually buys the
REMAINING tokens a tighter scale. A version of this that merely pastes fp16
values back over an already-derived scale would report near-zero D_sink under
per_tensor and the project would conclude "no effect" from a bug rather than
from the data.

The expected contrast, which these tests pin:
  per_tensor -> exempting the sink helps every other token   (D_sink > 0)
  per_token  -> every row already had its own scale          (D_sink == 0)
"""

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from quant.fakequant import quantize_dequantize as qdq  # noqa: E402
from quant.patch import (  # noqa: E402
    ExceptionSpec,
    QuantLinear,
    patch_model,
    resolve_fp16_exceptions,
    set_quant_enabled,
)


def toy_model(d=64):
    """No cross-token mixing, so per-token effects stay isolated and readable."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))


def outlier_input(b=1, t=16, d=64, scale=100.0):
    """A massive activation at position 0, as measured in §6."""
    torch.manual_seed(0)
    x = torch.randn(b, t, d)
    x[:, 0, :] *= scale
    return x


def mse(a, b):
    return (a - b).pow(2).mean().item()


# --- patching mechanics ------------------------------------------------------

def test_patch_replaces_linears_and_restore_reverses():
    model = toy_model()
    before = [type(m) for m in model.modules()]
    restore, names = patch_model(model)
    assert len(names) == 2
    assert any(isinstance(m, QuantLinear) for m in model.modules())
    restore()
    assert [type(m) for m in model.modules()] == before


def test_skip_list_is_respected():
    model = nn.Sequential(nn.Linear(8, 8))
    model.add_module("lm_head", nn.Linear(8, 8))
    _, names = patch_model(model)
    assert not any("lm_head" in n for n in names)


def test_patching_nothing_raises():
    """A silently unpatched model reports zero damage everywhere."""
    with pytest.raises(RuntimeError, match="replaced nothing"):
        patch_model(nn.Sequential(nn.ReLU()))


def test_quantization_actually_changes_the_output():
    model, x = toy_model(), outlier_input()
    ref = model(x)
    patch_model(model, a_bits=4, a_granularity="per_tensor")
    assert not torch.allclose(ref, model(x)), "patched model is a no-op"


def test_set_quant_enabled_round_trips():
    model, x = toy_model(), outlier_input()
    ref = model(x)
    patch_model(model, a_bits=4, a_granularity="per_tensor")
    assert set_quant_enabled(model, False) == 2
    torch.testing.assert_close(model(x), ref)
    set_quant_enabled(model, True)
    assert not torch.allclose(model(x), ref)


# --- exception specs ---------------------------------------------------------

def test_entry_mask_covers_token_rows():
    spec = ExceptionSpec("position_0", token_positions=[0])
    mask = spec.entry_mask(torch.zeros(1, 4, 8))
    assert mask[0, 0].all() and not mask[0, 1:].any()


def test_entry_mask_covers_channels():
    spec = ExceptionSpec("outlier_channels", channel_indices=[3])
    mask = spec.entry_mask(torch.zeros(1, 4, 8))
    assert mask[..., 3].all() and mask.sum() == 4


def test_apply_restores_exact_values():
    spec = ExceptionSpec("position_0", token_positions=[0])
    orig = torch.randn(1, 4, 8)
    restored = spec.apply(torch.zeros_like(orig), orig)
    torch.testing.assert_close(restored[:, 0], orig[:, 0])
    assert restored[:, 1:].abs().sum() == 0


def test_out_of_range_indices_are_dropped_not_fatal():
    """A sink detected at position 900 must not crash a length-16 eval batch."""
    spec = ExceptionSpec("detected_sinks", token_positions=[0, 900])
    assert spec.entry_mask(torch.zeros(1, 16, 8))[0, 0].all()


def test_resolve_each_grid_arm():
    assert resolve_fp16_exceptions("none").is_empty
    assert resolve_fp16_exceptions("position_0").token_positions == [0]

    sink_mask = torch.zeros(2, 16, dtype=torch.bool)
    sink_mask[:, [0, 7]] = True
    assert resolve_fp16_exceptions("detected_sinks", sink_mask=sink_mask).token_positions == [0, 7]

    outlier = torch.zeros(8, dtype=torch.bool)
    outlier[[2, 5]] = True
    assert resolve_fp16_exceptions("outlier_channels", outlier_mask=outlier).channel_indices == [2, 5]


def test_empty_sink_mask_is_rejected():
    """Silently equalling the none arm would make D_sink a spurious zero."""
    with pytest.raises(ValueError, match="spurious zero"):
        resolve_fp16_exceptions("detected_sinks", sink_mask=torch.zeros(2, 16, dtype=torch.bool))


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown fp16 exception"):
        resolve_fp16_exceptions("keep_everything")


# --- the mechanism behind D_sink ---------------------------------------------

def test_excluding_the_outlier_from_the_range_helps_other_tokens():
    """At the tensor level: this is the effect D_sink is supposed to capture."""
    x = outlier_input().squeeze(0)                   # (16, 64), row 0 is the sink
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0] = True

    naive = qdq(x, 8, "per_tensor")
    excluded = qdq(x, 8, "per_tensor", scale_source=x.masked_fill(mask, 0.0))

    err_naive = (x[1:] - naive[1:]).norm().item()
    err_excluded = (x[1:] - excluded[1:]).norm().item()
    assert err_excluded < err_naive / 10, (
        f"excluding the sink from the range should shrink error on the other "
        f"tokens by an order of magnitude; got {err_naive:.4f} -> {err_excluded:.4f}"
    )


def test_per_tensor_d_sink_is_positive():
    """Holding position 0 in fp16 must measurably reduce damage under per_tensor."""
    x = outlier_input()
    ref = toy_model()(x)

    model = toy_model()
    patch_model(model, a_bits=8, a_granularity="per_tensor",
                exceptions=resolve_fp16_exceptions("none"))
    damage_none = mse(ref, model(x))

    model = toy_model()
    patch_model(model, a_bits=8, a_granularity="per_tensor",
                exceptions=resolve_fp16_exceptions("position_0"))
    damage_p0 = mse(ref, model(x))

    assert damage_p0 < damage_none, f"D_sink not positive: {damage_none} -> {damage_p0}"


def test_per_token_leaves_non_sink_tokens_untouched():
    """The null-hypothesis arm (trap §9.7), pinned as a mechanism.

    Under per_token every row already has its own scale, so exempting position 0
    cannot help any other row. The toy model has no cross-token mixing, so this
    must hold EXACTLY. If it ever stops holding, per-token scaling has silently
    stopped being per-token.
    """
    x = outlier_input()

    model = toy_model()
    patch_model(model, a_bits=8, a_granularity="per_token",
                exceptions=resolve_fp16_exceptions("none"))
    out_none = model(x)

    model = toy_model()
    patch_model(model, a_bits=8, a_granularity="per_token",
                exceptions=resolve_fp16_exceptions("position_0"))
    out_p0 = model(x)

    torch.testing.assert_close(out_none[:, 1:], out_p0[:, 1:], rtol=0, atol=0)


def test_per_tensor_benefits_more_than_per_token():
    """The central comparison of the whole project, in miniature.

    If per_token showed the same D_sink as per_tensor, there would be no
    research question left to ask.
    """
    x = outlier_input()
    ref = toy_model()(x)

    def damage(granularity, arm):
        model = toy_model()
        patch_model(model, a_bits=8, a_granularity=granularity,
                    exceptions=resolve_fp16_exceptions(arm))
        return mse(ref, model(x))

    d_sink_tensor = damage("per_tensor", "none") - damage("per_tensor", "position_0")
    d_sink_token = damage("per_token", "none") - damage("per_token", "position_0")
    assert d_sink_tensor > d_sink_token


# --- the channel mask must not be applied to an axis it did not measure ------
#
# One ExceptionSpec is handed to every patched module, and a decoder's modules
# do not share an input width. A residual-stream channel index means nothing on
# the MLP intermediate, and the old bounds-clip quietly accepted it: the indices
# are all in range there, so 56 of a 1B arm's 196 modules got an exemption
# chosen by coincidence. That is an arm that reports a number while measuring
# nothing, which is the failure mode this repo exists to catch.

def test_channel_indices_apply_at_the_width_they_were_measured_at():
    spec = ExceptionSpec("outlier_channels", channel_indices=[3], channel_width=8)
    mask = spec.entry_mask(torch.zeros(2, 4, 8))
    assert mask is not None and mask[..., 3].all()
    assert mask.sum() == 2 * 4          # one channel, every token


def test_a_wider_tensor_is_left_alone_rather_than_clipped_into():
    """The rejected case. Every index is in range on a 64-wide input, which is
    exactly why a bounds check passes it and a width check does not."""
    spec = ExceptionSpec("outlier_channels", channel_indices=[3], channel_width=8)
    wide = torch.zeros(2, 4, 64)
    assert spec.entry_mask(wide).sum() == 0

    x_orig = torch.arange(2 * 4 * 64, dtype=torch.float32).reshape(2, 4, 64)
    x_quant = torch.zeros_like(x_orig)
    assert torch.equal(spec.apply(x_quant, x_orig), x_quant)   # nothing restored


def test_a_spec_with_no_declared_width_keeps_the_old_behaviour():
    """`None` means "the caller vouches for these indices". Kept so a hand-built
    spec in a test or a notebook is not silently disarmed."""
    spec = ExceptionSpec("outlier_channels", channel_indices=[3])
    assert spec.entry_mask(torch.zeros(2, 4, 64)).sum() == 2 * 4


def test_resolve_records_the_width_from_the_mask_it_was_given():
    """Nobody has to remember to pass the width: the mask's length is it."""
    outlier = torch.zeros(1024, dtype=torch.bool)
    outlier[[7, 900]] = True
    spec = resolve_fp16_exceptions("outlier_channels", outlier_mask=outlier)
    assert spec.channel_indices == [7, 900]
    assert spec.channel_width == 1024


def test_the_exemption_still_buys_a_tighter_scale_at_the_right_width():
    """The mechanism test, on the feature axis. A channel carrying a huge value
    drags a per-tensor scale; exempting it must reduce the error on the OTHER
    channels, or the arm is cosmetic."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 64)
    x[..., 5] = 500.0                                   # the outlier channel
    outlier = torch.zeros(64, dtype=torch.bool)
    outlier[5] = True
    spec = resolve_fp16_exceptions("outlier_channels", outlier_mask=outlier)

    others = [c for c in range(64) if c != 5]
    mask = spec.entry_mask(x)
    plain = qdq(x, 8, "per_tensor")
    kept = spec.apply(qdq(x, 8, "per_tensor", scale_source=x.masked_fill(mask, 0.0)), x)

    err_plain = (plain - x)[..., others].abs().mean().item()
    err_kept = (kept - x)[..., others].abs().mean().item()
    assert err_kept < err_plain / 10, (
        f"exempting the outlier channel should shrink error on the other 63 by "
        f"an order of magnitude; got {err_plain:.4f} -> {err_kept:.4f}"
    )
