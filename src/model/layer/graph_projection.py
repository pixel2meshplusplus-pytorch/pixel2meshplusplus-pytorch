"""Graph projection for Pixel2Mesh"""
import torch
from torch import nn
from src.utils.tools import *

class GraphProjection(nn.Module):
    """graph projection"""
    def __init__(self):
        super().__init__()
        self.view_number=3 #no of different views of camera

    def forward(self, inputs, img_feat, cameras):
        coord = torch.squeeze(inputs)
        out1_list = []
        out2_list = []
        out3_list = []
        out4_list = []
        point_origin = camera_trans_inv(cameras[0], inputs)

        for i in range(self.view_number):
            point_crrent = camera_trans(cameras[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]
            h = 248.0 * torch.div(-Y, -Z) + 112.0 #divide
            w = 248.0 * torch.div(X, -Z) + 112.0

            # Replace NaNs with 0's
            h = torch.nan_to_num(h)
            w = torch.nan_to_num(w)

            h = torch.clamp(h, min=0, max=223)
            w = torch.clamp(w, min=0, max=223)

            n = torch.full(h.shape, i, dtype=torch.float32).to(h.get_device())
            indices = torch.stack([n, h, w], 1).to(h.get_device())

            idx = (indices / (224.0 / 56.0)).to(dtype=torch.int32)
            img_feat0 = img_feat[0].permute((0, 2, 3, 1))
            out1 = gather_nd(img_feat0, idx)
            out1_list.append(out1)

            idx = (indices / (224.0 / 28.0)).to(dtype=torch.int32)
            img_feat1 = img_feat[1].permute((0, 2, 3, 1))
            out2 = gather_nd(img_feat1, idx)
            out2_list.append(out2)

            idx = (indices / (224.0 / 14.0)).to(dtype=torch.int32)
            img_feat2 = img_feat[2].permute((0, 2, 3, 1))
            out3 = gather_nd(img_feat2, idx)
            out3_list.append(out3)

            idx = (indices / (224.0 / 7.00)).to( dtype=torch.int32)
            img_feat3 = img_feat[3].permute((0, 2, 3, 1))
            out4 = gather_nd(img_feat3, idx)
            out4_list.append(out4)

        all_out1 = torch.stack(out1_list, 0)
        all_out2 = torch.stack(out2_list, 0)
        all_out3 = torch.stack(out3_list, 0)
        all_out4 = torch.stack(out4_list, 0)

        # 3*N*[64+128+256+512] -> 3*N*F
        image_feature = torch.cat([all_out1, all_out2, all_out3, all_out4], 2)
        # 3*N*F -> N*F
        # image_feature = tf.reshape(tf.transpose(image_feature, [1, 0, 2]), [-1, FLAGS.feat_dim * 3])

        #image_feature = tf.reduce_max(image_feature, axis=0)
        image_feature_max , _= torch.max(input=image_feature, dim=0)
        image_feature_mean = torch.mean(input=image_feature, dim=0)
        image_feature_std = reduce_std(image_feature, axis=0)
        
        outputs = torch.concat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
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
