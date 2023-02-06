"""Taken and adapted from Pixel2Mesh Pytorch implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layer.chamfer_wrapper import ChamferDist
from src.utils.mesh_to_point_cloud import sample_point_cloud


class P2MLoss(nn.Module):
    def __init__(self, lape_idx, edges, faces):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        self.laplace_idx = lape_idx#[torch.as_tensor(l, dtype=torch.long) for l in lape_idx]
        self.edges = edges #[torch.as_tensor(e, dtype=torch.long) for e in edges]
        self.faces = faces #[torch.as_tensor(f, dtype=torch.long) for f in faces]
        

        ###############################################
        # Our code: hard coded option values
        self.normal = 0.5
        self.edge = 359
        self.laplace = 1500
        self.move = 50
        self.constant = 1.
        self.chamfer = [3000., 3000., 3000.]
        self.chamfer_opposite = 0.55
        self.reconst = 0.
        ################################################


    def edge_regularization(self, pred, edges):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """
        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss = self.l2_loss(input1, input2) * input1.size(-1) if block_idx > 0 else 0
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """
        
        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss = 0., 0., 0., 0., 0.
        lap_const = [0.2, 1., 1.]

        gt_coord, gt_normal, gt_images = targets["points"].float(), targets["normals"].float(), targets["images"]
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]
        image_loss = 0.

        # if outputs["reconst"] is not None and self.reconst != 0:
        #     image_loss = self.image_loss(gt_images, outputs["reconst"])

        for i in range(len(self.laplace_idx)):
            ####################################################
            # Our contribution
            sample_pt = sample_point_cloud(pred_coord[i][0], self.faces[i], 4000).unsqueeze(0)
            sample_pred = torch.cat([pred_coord[i], sample_pt], 1)
            # print(gt_coord[:,:5,:].shape, pred_coord[i].shape)
            # dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord[:,:5,:], pred_coord[i][:,:5,:])
            dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, sample_pred)
           
            ####################################################
            chamfer_loss += self.chamfer[i] * (torch.mean(dist1) +
                                                               self.chamfer_opposite * torch.mean(dist2))

            normal_loss += self.normal_loss(gt_normal, idx2, pred_coord[i], self.edges[i])
            edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
            lap, move = self.laplace_regularization(pred_coord_before_deform[i], pred_coord[i], i)
            lap_loss += lap_const[i] * lap
            move_loss += lap_const[i] * move

        # loss = torch.Tensor(1).cuda()
        # loss.requires_grad = True
        loss = 0
        loss += chamfer_loss
        loss += self.laplace * lap_loss
        loss += self.move * move_loss
        loss += self.edge * edge_loss
        loss += self.normal * normal_loss

        loss = loss * self.constant

        return loss#, {
        #     "loss": loss,
        #     "loss_chamfer": chamfer_loss,
        #     "loss_edge": edge_loss,
        #     "loss_laplace": lap_loss,
        #     "loss_move": move_loss,
        #     "loss_normal": normal_loss,
        # }
