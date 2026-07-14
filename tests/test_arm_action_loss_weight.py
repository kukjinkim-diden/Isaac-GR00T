"""Unit tests for weight_arm_action_loss (per-group action loss weighting).

The shared flow head splits its gradient across all action dims, so the many
finger dims drown out the few arm dims and the arm degrades. Up-weighting the
leading arm dims rebalances it. These pin the weighting math (CPU-only).
"""
import torch

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import weight_arm_action_loss


def test_disabled_when_dim_zero_or_weight_one():
    loss = torch.ones(2, 4, 27)
    assert torch.equal(weight_arm_action_loss(loss, arm_action_dim=0, arm_weight=3.0), loss)
    assert torch.equal(weight_arm_action_loss(loss, arm_action_dim=7, arm_weight=1.0), loss)


def test_scales_only_arm_dims():
    loss = torch.ones(2, 4, 27)
    out = weight_arm_action_loss(loss, arm_action_dim=7, arm_weight=3.0)
    assert torch.all(out[..., :7] == 3.0)   # arm dims scaled
    assert torch.all(out[..., 7:] == 1.0)   # finger dims untouched
    assert out.shape == loss.shape


def test_is_autograd_safe_and_out_of_place():
    loss = torch.ones(1, 2, 10, requires_grad=True)
    out = weight_arm_action_loss(loss, arm_action_dim=3, arm_weight=5.0)
    out.sum().backward()                     # must not raise (no in-place on grad tensor)
    # gradient flows W to arm dims, 1 to the rest
    assert torch.all(loss.grad[..., :3] == 5.0)
    assert torch.all(loss.grad[..., 3:] == 1.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
