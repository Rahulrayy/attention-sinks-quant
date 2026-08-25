"""Bootstrap machinery, and the pairing invariants it depends on.

The tests that matter here are the ones about VARIANCE SOURCES. This project
audits papers that report point estimates without error bars; the ways an error
bar can be fake are therefore load-bearing knowledge, not trivia.
"""

import pytest

np = pytest.importorskip("numpy")

from analysis.stats import (  # noqa: E402
    Interval,
    paired_bootstrap,
    paired_deltas,
    sequence_bootstrap,
)


def test_positive_effect_does_not_cross_zero():
    ci = paired_bootstrap([0.20, 0.22, 0.19, 0.24, 0.21])
    assert ci.point == pytest.approx(0.212, abs=1e-6)
    assert not ci.crosses_zero


def test_noise_around_zero_crosses_zero():
    rng = np.random.default_rng(0)
    ci = paired_bootstrap(rng.normal(0, 0.01, 40))
    assert ci.crosses_zero


def test_identical_arms_give_a_degenerate_interval():
    """THE per-token failure mode (C18), pinned.

    Five identical deltas produce an interval of width zero. That is not
    precision — it means the randomness source does not apply to this arm. The
    test exists so the property is visible rather than mistaken for a result.
    """
    ci = paired_bootstrap([0.5] * 5)
    assert ci.width == 0.0
    assert not ci.crosses_zero


def test_single_seed_is_reported_as_n_one():
    """No variance information; must not fabricate an interval."""
    ci = paired_bootstrap([0.3])
    assert ci.n == 1 and ci.n_resamples == 0 and ci.width == 0.0


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no deltas"):
        paired_bootstrap([])


def test_non_finite_is_rejected():
    """A NaN would silently poison the mean of every resample."""
    with pytest.raises(ValueError, match="non-finite"):
        paired_bootstrap([0.1, float("nan"), 0.3])


def test_format_flags_zero_crossing_in_text():
    assert "crosses zero" in Interval(0.0, -0.1, 0.1, 5, 100).format()
    assert "crosses zero" not in Interval(0.5, 0.4, 0.6, 5, 100).format()


# --- pairing invariants ------------------------------------------------------

def test_paired_deltas_matches_on_seed():
    assert paired_deltas({0: 1.0, 1: 2.0}, {0: 0.5, 1: 0.5}) == [0.5, 1.5]


def test_mismatched_seeds_are_rejected():
    """Silently intersecting would give an unpaired comparison wearing a paired
    label — exactly the mistake trap §9.8 warns about."""
    with pytest.raises(ValueError, match="different seeds"):
        paired_deltas({0: 1.0, 1: 2.0}, {0: 0.5, 2: 0.5})


def test_sequence_bootstrap_pairs_elementwise():
    none = [1.0, 1.2, 0.9, 1.1]
    exempt = [0.5, 0.7, 0.4, 0.6]
    ci = sequence_bootstrap(none, exempt)
    assert ci.point == pytest.approx(0.5, abs=1e-9)
    assert not ci.crosses_zero


def test_sequence_bootstrap_rejects_length_mismatch():
    with pytest.raises(ValueError, match="different numbers of sequences"):
        sequence_bootstrap([1.0, 2.0], [1.0])


def test_sequence_bootstrap_gives_width_where_draws_give_none():
    """The C18 fix, as a property: sequences vary even when draws do not."""
    rng = np.random.default_rng(1)
    none = rng.normal(1.0, 0.15, 32)
    exempt = none - 0.02 + rng.normal(0, 0.05, 32)
    ci = sequence_bootstrap(none, exempt)
    assert ci.width > 0.0, "sequence bootstrap must produce a non-degenerate CI"
