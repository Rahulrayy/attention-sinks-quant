"""Paired bootstrap over per-seed deltas.

Trap §9.8 — seed variance at 17M params is ugly, and Track-A calibration-draw
variance is not much better. NEVER compare unpaired means: the between-seed
spread swamps the between-arm effect and everything comes back null for the
wrong reason. Bootstrap the per-seed *delta* instead.

Report CI width alongside every point estimate. If a CI crosses zero, say so in
the prose, not only in the figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    n: int
    n_resamples: int

    @property
    def crosses_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def format(self, unit: str = "", places: int = 4) -> str:
        """Point estimate with CI, flagged when it crosses zero.

        The flag is deliberately in the TEXT and not only in a figure: a reader
        skimming a table should not have to check whether an error bar happens
        to touch the axis.
        """
        s = f"{self.point:+.{places}f}{unit} [{self.lo:+.{places}f}, {self.hi:+.{places}f}]"
        return s + ("  (CI crosses zero)" if self.crosses_zero else "")

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "ci_width": self.width,
            "crosses_zero": self.crosses_zero,
            "n_seeds": self.n,
        }


def paired_bootstrap(
    deltas,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over already-paired per-seed differences.

    ``deltas[i]`` is the difference measured on seed i — the SAME calibration
    draw and the SAME held-out slice for both arms. Pairing must happen before
    this function is called; it cannot be recovered here.

    With few seeds (5 is the plan's minimum) the percentile interval is wide and
    lumpy. That is honest: it reflects how little information 5 draws carry, and
    smoothing it would be the failure this project audits others for.
    """
    d = np.asarray(deltas, dtype=np.float64).ravel()
    if d.size == 0:
        raise ValueError("no deltas to bootstrap")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"deltas contain non-finite values: {d}")

    if d.size == 1:
        # A single seed carries no variance information. Return a degenerate
        # interval rather than a fake one — and the caller can see n=1.
        v = float(d[0])
        return Interval(v, v, v, 1, 0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(d.mean()), float(lo), float(hi), int(d.size), n_resamples)


def paired_deltas(a_by_seed: dict[int, float], b_by_seed: dict[int, float]) -> list[float]:
    """Build per-seed differences a - b, keyed on seed.

    Raises when the two arms were not measured on the same seeds. Silently
    intersecting them would produce an unpaired comparison wearing a paired
    label, which is the specific mistake trap §9.8 warns about.
    """
    if set(a_by_seed) != set(b_by_seed):
        only_a = sorted(set(a_by_seed) - set(b_by_seed))
        only_b = sorted(set(b_by_seed) - set(a_by_seed))
        raise ValueError(
            f"arms were measured on different seeds; cannot pair. "
            f"only in a: {only_a}, only in b: {only_b}"
        )
    return [a_by_seed[s] - b_by_seed[s] for s in sorted(a_by_seed)]


def crosses_zero(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    return lo <= 0.0 <= hi


def sequence_bootstrap(
    none_per_seq,
    exempt_per_seq,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap over held-out SEQUENCES rather than calibration draws.

    Why this exists: the calibration draw is not a randomness source for every
    arm. Per-token activation scaling is dynamic and never reads the calibration
    set, so five disjoint draws produce five identical numbers and a CI of width
    zero — on the arm the project is actually about. Sequences vary for every
    arm, so bootstrapping them keeps the two granularities comparable.

    Both arrays index the SAME held-out sequences in the same order, so the
    difference is paired sequence-by-sequence.
    """
    a = np.asarray(none_per_seq, dtype=np.float64).ravel()
    b = np.asarray(exempt_per_seq, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"arms scored different numbers of sequences ({a.size} vs {b.size}); "
            "they cannot be paired"
        )
    return paired_bootstrap(a - b, n_resamples=n_resamples, alpha=alpha, seed=seed)


def describe_variance_sources(df, group_cols, value_col="delta_ppl") -> "list[dict]":
    """Report per-group spread across calibration draws.

    Used to make the zero-variance property of the dynamic arms VISIBLE in the
    output rather than leaving a reader to infer it from suspiciously tight
    intervals.
    """
    out = []
    for key, sub in df.groupby(group_cols):
        vals = np.asarray(sub[value_col], dtype=np.float64)
        out.append(
            {
                **dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                "n": len(vals),
                "std_across_draws": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "calibration_is_a_variance_source": bool(
                    len(vals) > 1 and vals.std(ddof=1) > 0
                ),
            }
        )
    return out
