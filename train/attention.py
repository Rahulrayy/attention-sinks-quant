"""ALL arm-specific code lives here. Nothing arm-specific goes in model.py.

Design rule §7.3: if arm logic leaks into model.py, the ablation is no longer
clean and the three arms stop being comparable.

Trap §9.5 — every arm, INCLUDING baseline, takes the naive attention path.
softmax1 and gating cannot use fused SDPA (you are modifying the softmax, so
you materialise B x H x T x T). If baseline used fused kernels and the variants
did not, this would benchmark kernel implementations rather than architectures.
Consequence: wall-clock comparisons between arms are meaningless. Report steps
and tokens, never seconds.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Off-by-one softmax:  exp(x_i) / (1 + sum_j exp(x_j)).

    Equivalent to appending a fixed zero logit and dropping its output slot,
    which is how this is implemented (numerically stable via the same max-shift
    trick as a standard softmax).

    METHODOLOGICAL NOTE (plan §5) — this CANNOT be retrofitted onto pretrained
    weights. Let s = sum(exp) / (1 + sum(exp)) < 1. Then

        softmax1(x) = s * softmax(x)

    so the output is not "the same attention with less sink"; it is the original
    convex combination uniformly scaled down. A retrofitted model degrades for
    reasons that say nothing about the sink mechanism. Several blog posts got
    this wrong. tests/test_softmax1.py pins the identity.
    """
    zero = x.new_zeros(*x.shape[:-1], 1) if dim in (-1, x.dim() - 1) else None
    if zero is None:
        raise NotImplementedError("softmax1 currently assumes dim=-1")
    extended = torch.cat([x, zero], dim=-1)
    return F.softmax(extended, dim=-1)[..., :-1]


class OutputGate(nn.Module):
    """G1: query-dependent sigmoid gate on the SDPA output (Qiu et al., 2025).

    elementwise -> one gate scalar per (head, head_dim) channel   [the published best]
    headwise    -> one gate scalar per head                        [coarser variant]

    Mirrors the config flags on QwQZh/gated_attention:
    elementwise_attn_output_gate / headwise_attn_output_gate.
    """

    def __init__(self, d_model: int, n_head: int, gate_type: str = "elementwise"):
        super().__init__()
        if gate_type not in ("elementwise", "headwise"):
            raise ValueError(f"unknown gate_type: {gate_type}")
        self.gate_type = gate_type
        self.n_head = n_head
        out_features = d_model if gate_type == "elementwise" else n_head
        self.proj = nn.Linear(d_model, out_features, bias=False)

    def forward(self, attn_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """attn_out: (B, T, d_model) post-SDPA. x: (B, T, d_model) block input."""
        gate = torch.sigmoid(self.proj(x))
        if self.gate_type == "headwise":
            gate = gate.repeat_interleave(attn_out.size(-1) // self.n_head, dim=-1)
        return attn_out * gate


class CausalSelfAttention(nn.Module):
    """Naive (materialised) causal attention, shared by all three arms."""

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
