import unittest

import torch

from kinepose_losses import kinepose_loss
from kinepose_model import parameter_report
from kineres_model import KineResPose


class KineResPoseTests(unittest.TestCase):
    def test_forward_and_equal_capacity_control(self):
        plain = KineResPose(graph_depth=2, decoder_channels=64, variant="plain")
        graph = KineResPose(graph_depth=2, decoder_channels=64, variant="graph")
        self.assertEqual(parameter_report(plain), parameter_report(graph))
        output = graph(torch.randn(2, 3, 256, 192))
        self.assertEqual(tuple(output.heatmaps.shape), (2, 17, 64, 48))

    def test_graph_gradient_from_initial_state(self):
        model = KineResPose(graph_depth=1, decoder_channels=64, variant="graph")
        images = torch.randn(2, 3, 256, 192)
        keypoints = torch.rand(2, 17, 2) * torch.tensor([48.0, 64.0])
        visible = torch.ones(2, 17, dtype=torch.bool)
        loss, _ = kinepose_loss(model(images), keypoints, visible)
        loss.backward()
        gradients = [parameter.grad for parameter in model.graph.parameters()]
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
