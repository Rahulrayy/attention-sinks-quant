"""Pins the scaling identity from plan §5.

This is the test that stops the most common misreading of softmax1: that it is
"softmax with the sink removed" and can therefore be swapped into pretrained
weights. It cannot. It is the same convex combination, uniformly scaled down.
"""

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from train.attention import softmax1  # noqa: E402


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(4, 6, 32, 32) * 3.0


def test_matches_scaled_softmax(logits):
    """softmax1(x) == s * softmax(x)  with  s = sum(exp) / (1 + sum(exp))."""
    exp_sum = logits.exp().sum(dim=-1, keepdim=True)
    s = exp_sum / (1.0 + exp_sum)
    expected = s * F.softmax(logits, dim=-1)
    torch.testing.assert_close(softmax1(logits), expected, rtol=1e-5, atol=1e-6)


def test_scale_factor_is_strictly_less_than_one(logits):
    """s < 1 always — this is exactly why retrofitting degrades the model."""
    assert (softmax1(logits).sum(dim=-1) < 1.0).all()


def test_relative_weights_are_unchanged(logits):
    """The ratio between any two attended positions is identical to softmax.

    The sink is not redistributed. It is uniformly discounted. If this test
    ever fails, the implementation has stopped being softmax1.
    """
    a, b = softmax1(logits), F.softmax(logits, dim=-1)
    ratio = (a / b)
    assert torch.allclose(ratio, ratio[..., :1].expand_as(ratio), rtol=1e-5, atol=1e-6)


def test_can_attend_to_nothing():
    """With all logits very negative the output goes to ~0, which is the point:
    softmax1 lets a head do nothing, where softmax forces it to pick someone."""
    x = torch.full((1, 1, 4, 8), -30.0)
    assert softmax1(x).sum(dim=-1).max().item() < 1e-6


def test_numerical_stability_large_logits():
    x = torch.tensor([[[[1000.0, 999.0, -1000.0]]]])
    out = softmax1(x)
    assert torch.isfinite(out).all()
