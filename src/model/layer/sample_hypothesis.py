"""SampleHypothesis layer implementation"""
import torch
from torch import nn


class SampleHypothesis(nn.Module):
    def __init__(self, sample_coord):
        """
        :param sample_coord: tensor of shape (43, 3) containing the 42 vertices of a icosahedron centered around (0, 0, 0) and 1 more vertex for the center (0, 0, 0) 
        """
        super().__init__()
        self.sample_delta = torch.from_numpy(sample_coord).to(torch.float)

    def forward(self, mesh_coords):
        """
        :param mesh_coords: tensor of shape (N, 3), where N=2466 <= number of vertices
        :return sample_points_per_vertices: [N * S, 3], where N * S = 2466 * 43 = 106038
        """
        device = mesh_coords.get_device()
        center_points = torch.unsqueeze(mesh_coords, axis=1)
        center_points = torch.tile(center_points, [1, 43, 1])

        delta = torch.unsqueeze(self.sample_delta, 0).to(device)

        sample_points_per_vertices = torch.add(center_points, delta)
        outputs = torch.reshape(sample_points_per_vertices, [-1, 3])
        return outputs

