
"""LocalGraphProjection layer implementation"""
import torch
from torch import nn
import numpy as np

from src.utils.tools import camera_trans, camera_trans_inv, reduce_std

class LocalGraphProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.view_number = 3

    def forward(self, inputs, camera, img_feat):
        """
        :param inputs: tensor of shape (N * S, 3), where N * S = 2466 * 43 = 106038
        :param camera: tensor of shape (N, 5), where N = 3 <= number of cameras
        :param img_feat: list of 3 tensors with shapes: (3, 224, 224, 16),  (3, 112, 112, 32) and (3, 56, 56, 64)
        :return tensor of shape (N * S, K), where K = 339 = 3 + 112 + 112 + 112
        """
        coord = inputs
        device = inputs.get_device()
        out1_list = []
        out2_list = []
        out3_list = []

        for i in range(self.view_number):
            point_origin = camera_trans_inv(camera[0], inputs)
            point_crrent = camera_trans(camera[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]

            h = 248.0 * torch.divide(-Y, -Z) + 112.0
            w = 248.0 * torch.divide(X, -Z) + 112.0

            # Replace NaNs with 0's
            h = torch.nan_to_num(h)
            w = torch.nan_to_num(w)

            n = torch.full(h.shape, i, dtype=torch.int32).to(device)

            x = h / (224.0 / 224)
            y = w / (224.0 / 224)
            x = torch.clamp(x, min=0, max=223)
            y = torch.clamp(y, min=0, max=223)
            out1 = self.bi_linear_sample(img_feat[0], n, x.to(device), y.to(device))
            out1_list.append(out1)

            x = h / (224.0 / 112)
            y = w / (224.0 / 112)
            x = torch.clamp(x, min=0, max=111)
            y = torch.clamp(y, min=0, max=111)
            out2 = self.bi_linear_sample(img_feat[1], n, x.to(device), y.to(device))
            out2_list.append(out2)

            x = h / (224.0 / 56)
            y = w / (224.0 / 56)
            x = torch.clamp(x, min=0, max=55)
            y = torch.clamp(y, min=0, max=55)
            out3 = self.bi_linear_sample(img_feat[2], n, x.to(device), y.to(device))
            out3_list.append(out3)
        # ----
        all_out1 = torch.stack(out1_list, 0).to(device)
        all_out2 = torch.stack(out2_list, 0).to(device)
        all_out3 = torch.stack(out3_list, 0).to(device)

        # 3*N*[16+32+64] -> 3*N*F
        image_feature = torch.concat([all_out1, all_out2, all_out3], 2).to(device)

        image_feature_max, _ = torch.max(image_feature, dim=0)
        image_feature_mean = torch.mean(image_feature, dim=0).to(device)
        image_feature_std = reduce_std(image_feature, axis=0).to(device) # use other reduce_std?
    
        outputs = torch.concat([coord, image_feature_max.to(device), image_feature_mean, image_feature_std], 1).to(device)
        return outputs

    def bi_linear_sample(self, img_feat, n, x, y):
        x1 = torch.floor(x)
        x2 = torch.ceil(x)
        y1 = torch.floor(y)
        y2 = torch.ceil(y)

        x1int = x1.type(torch.int32)
        x2int = x2.type(torch.int32)
        y1int = y1.type(torch.int32)
        y2int = y2.type(torch.int32)

        ind11 = torch.stack([n, x1int, y1int], 1)
        Q11 = gather_nd(img_feat, ind11)

        ind12 = torch.stack([n, x1int, y2int], 1)
        Q12 = gather_nd(img_feat, ind12)

        ind21 = torch.stack([n, x2int, y1int], 1)
        Q21 = gather_nd(img_feat, ind21)

        ind22 = torch.stack([n, x2int, y2int], 1)
        Q22 = gather_nd(img_feat, ind22)

        weights = torch.multiply(torch.subtract(x2, x), torch.subtract(y2, y))
        Q11 = torch.multiply(torch.unsqueeze(weights, 1), Q11)
        weights = torch.multiply(torch.subtract(x, x1), torch.subtract(y2, y))
        Q21 = torch.multiply(torch.unsqueeze(weights, 1), Q21)
        weights = torch.multiply(torch.subtract(x2, x), torch.subtract(y, y1))
        Q12 = torch.multiply(torch.unsqueeze(weights, 1), Q12)
        weights = torch.multiply(torch.subtract(x, x1), torch.subtract(y, y1))
        Q22 = torch.multiply(torch.unsqueeze(weights, 1), Q22)
        
        outputs = torch.sum(torch.stack([Q11, Q21, Q12, Q22]), dim=0)

        return outputs
    

# Pytorch implementation of tensorflow.gather_nd(params, indices)
# copied from: https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/38
def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    orig_shape = list(indices.shape)
    num_samples = np.int32(np.prod(orig_shape[:-1]))
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]
    return output.reshape(out_shape).contiguous()



