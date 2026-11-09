"""The deterministic pool preserves adaptive bins and their gradients."""
import math
import unittest

import torch
from torch.nn import functional as F

from deepprobe_vlm.modeling import (SpatialPoolProjector, _adaptive_bin_weights,
                                    _pool_with_bin_weights)


def original_projector_forward(projector, states):
    """Reference the former CUDA adaptive-pooling implementation on CPU."""
    side = math.isqrt(states.shape[1])
    weights = torch.sigmoid(projector.gate(states)).float().clamp_min(1e-6)
    values = (states.float() * weights).transpose(1, 2).reshape(states.shape[0], -1, side, side)
    weights = weights.transpose(1, 2).reshape(states.shape[0], 1, side, side)
    pooled = F.adaptive_avg_pool2d(values, projector.grid) / F.adaptive_avg_pool2d(weights, projector.grid)
    pooled = pooled.flatten(2).transpose(1, 2).to(projector.projection[0].weight.dtype)
    return projector.projection(pooled)


class SpatialPoolProjectorTest(unittest.TestCase):
    def test_forward_and_all_gradients_match_adaptive_pool(self):
        # 7 -> 3 and 27 -> 4 have overlapping bins; 8 -> 4 is divisible.
        for side, grid in ((7, 3), (8, 4), (27, 4)):
            with self.subTest(side=side, grid=grid):
                torch.manual_seed(17 + side)
                projector = SpatialPoolProjector(vision_dim=5, language_dim=7, grid=grid)
                states = torch.randn(2, side * side, 5, requires_grad=True)
                actual = projector(states)
                expected = original_projector_forward(projector, states)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

                upstream = torch.randn_like(actual)
                leaves = (states, *projector.parameters())
                actual_grads = torch.autograd.grad(actual, leaves, grad_outputs=upstream)
                expected_grads = torch.autograd.grad(expected, leaves, grad_outputs=upstream)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)

    def test_pool_remains_fp32_under_bf16_autocast(self):
        for side, grid in ((7, 3), (8, 4), (27, 4)):
            with self.subTest(side=side, grid=grid):
                torch.manual_seed(29 + side)
                values = torch.randn(2, 5, side, side, requires_grad=True)
                bins = _adaptive_bin_weights(side, grid, values)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    actual = _pool_with_bin_weights(values, bins)
                    expected = F.adaptive_avg_pool2d(values, grid)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
                upstream = torch.randn_like(actual)
                actual_grad = torch.autograd.grad(actual, values, grad_outputs=upstream)[0]
                expected_grad = torch.autograd.grad(expected, values, grad_outputs=upstream)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)

    def test_nondivisible_boundary_contributes_to_overlapping_bins(self):
        values = torch.zeros(1, 1, 7, 7)
        values[0, 0, 2, 2] = 1
        bins = _adaptive_bin_weights(7, 3, values)
        pooled = _pool_with_bin_weights(values, bins)
        self.assertEqual(int((pooled > 0).sum()), 4)
        torch.testing.assert_close(pooled, F.adaptive_avg_pool2d(values, 3))


if __name__ == "__main__":
    unittest.main()
