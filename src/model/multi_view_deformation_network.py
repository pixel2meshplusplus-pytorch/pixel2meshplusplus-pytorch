"""MultiViewDeformationNetwork network implementation"""
from torch import nn

from src.model.layer.sample_hypothesis import SampleHypothesis
from src.model.layer.cross_view_perceptual_feature_pooling import LocalGraphProjection
from src.model.layer.deformation_reasoning import DeformationReasoning


class MultiViewDeformationNetwork(nn.Module):
    def __init__(self, stage2_feat_dim, sample_coord, sample_adj):
        """
        :param stage2_feat_dim: number of features after projection = 339
        :param sample_coord: tensor of shape (43, 3) containing the 42 vertices of a icosahedron centered around (0, 0, 0) and 1 more vertex for the center (0, 0, 0) 
        :param sample_adj: tensor of shape (K, 43, 43)
        """
        super().__init__()

        # sample hypothesis points
        self.sample_1 = SampleHypothesis(sample_coord)
        # projection block
        self.projection_1 = LocalGraphProjection()
        # # DRB
        self.dr_1 = DeformationReasoning(stage2_feat_dim=stage2_feat_dim,
                                        sample_adj=sample_adj,
                                        sample_coord=sample_coord)

        # sample hypothesis points
        self.sample_2 = SampleHypothesis(sample_coord)
        # projection block
        self.projection_2 = LocalGraphProjection()
        # # DRB
        self.dr_2 = DeformationReasoning(stage2_feat_dim=stage2_feat_dim,
                                        sample_adj=sample_adj,
                                        sample_coord=sample_coord)
                        

    def forward(self, mesh_coords, camera, img_feat):
        """
        :param mesh_coords: tensor of shape (N, 3), where N=2466 <= number of vertices
        :param camera: tensor of shape (N, 5), where N = 3 <= number of cameras
        :param img_feat: list of 3 tensors with shapes: (3, 224, 224, 16),  (3, 112, 112, 32) and (3, 56, 56, 64)
        :return tensor of shape (N, 3) <= next mesh coordinates
        """
        blk_sample_1 = self.sample_1(mesh_coords)
        blk_proj_feat_1 = self.projection_1(blk_sample_1, camera, img_feat)
        blk_out_1 = self.dr_1(mesh_coords, blk_proj_feat_1)

        blk_sample_2 = self.sample_2(blk_out_1)
        blk_proj_feat_2 = self.projection_2(blk_sample_2, camera, img_feat)
        blk_out_2 = self.dr_2(blk_out_1, blk_proj_feat_2)

        return {
            "pred_coord": [blk_out_1.unsqueeze(0), blk_out_2.unsqueeze(0)],
            "pred_coord_before_deform": [mesh_coords.unsqueeze(0), blk_out_1.unsqueeze(0)]
        }

        # return {blk_out_1, blk_out_2}

