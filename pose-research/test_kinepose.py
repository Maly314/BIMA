import unittest

import torch

from kinepose_losses import kinepose_loss, limb_field_targets
from kinepose_model import KinePose, decode_coordinates, parameter_report


class KinePoseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_full_forward_contract(self):
        model = KinePose(width=16, graph_depth=2, variant="full")
        output = model(torch.randn(2, 3, 256, 192))
        self.assertEqual(tuple(output.heatmaps.shape), (2, 17, 64, 48))
        self.assertEqual(tuple(output.offsets.shape), (2, 17, 2, 64, 48))
        self.assertEqual(tuple(output.limb_fields.shape), (2, 24, 64, 48))
        self.assertEqual(tuple(output.log_variance.shape), (2, 17))
        self.assertEqual(tuple(decode_coordinates(output).shape), (2, 17, 2))

    def test_graph_branch_receives_gradient(self):
        model = KinePose(width=16, graph_depth=1, variant="full")
        model.refine_scale.data.fill_(0.1)
        images = torch.randn(2, 3, 256, 192)
        keypoints = torch.rand(2, 17, 2) * torch.tensor([48.0, 64.0])
        visible = torch.ones(2, 17, dtype=torch.bool)
        loss, _ = kinepose_loss(model(images), keypoints, visible)
        loss.backward()
        graph_gradients = [parameter.grad for parameter in model.graph.parameters()]
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in graph_gradients))

    def test_limb_fields_are_localized(self):
        keypoints = torch.zeros(1, 17, 2)
        visible = torch.zeros(1, 17, dtype=torch.bool)
        keypoints[0, 5] = torch.tensor([10.0, 10.0])
        keypoints[0, 6] = torch.tensor([30.0, 10.0])
        visible[0, 5] = visible[0, 6] = True
        fields, masks = limb_field_targets(keypoints, visible)
        self.assertGreater(int(masks[0, 0].sum()), 20)
        self.assertAlmostEqual(float(fields[0, 0, 10, 20]), 1.0, places=5)
        self.assertAlmostEqual(float(fields[0, 1, 10, 20]), 0.0, places=5)

    def test_small_model_budget(self):
        report = parameter_report(KinePose(width=32, graph_depth=3, variant="full"))
        self.assertLess(report["parameters"], 2_100_000)

    def test_plain_control_skips_graph_refinement(self):
        model = KinePose(width=16, graph_depth=1, variant="plain")
        output = model(torch.randn(1, 3, 256, 192))
        self.assertTrue(torch.equal(output.heatmaps, output.coarse_heatmaps))

    def test_sharp_hybrid_forward_contract(self):
        model = KinePose(width=16, graph_depth=2, variant="hybrid_sharp")
        output = model(torch.randn(2, 3, 256, 192))
        self.assertEqual(tuple(output.heatmaps.shape), (2, 17, 64, 48))

    def test_anchor_hybrid_backpropagates_to_coordinate_delta(self):
        model = KinePose(width=16, graph_depth=1, variant="hybrid_anchor")
        images = torch.randn(2, 3, 256, 192)
        keypoints = torch.rand(2, 17, 2) * torch.tensor([48.0, 64.0])
        visible = torch.ones(2, 17, dtype=torch.bool)
        loss, _ = kinepose_loss(model(images), keypoints, visible)
        loss.backward()
        gradient = model.coordinate_delta[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_fast_hybrid_has_equal_capacity_and_larger_residual_scale(self):
        plain = KinePose(width=16, graph_depth=1, variant="plain")
        fast = KinePose(width=16, graph_depth=1, variant="hybrid_fast")
        self.assertEqual(parameter_report(plain), parameter_report(fast))
        self.assertAlmostEqual(float(fast.stage1[0].layer_scale.detach().mean()), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
