"""Multiview Pixel2Mesh"""
import torch
from torch import nn

from src.model.layer.graph_convolution import GraphConvolution
from src.model.layer.graph_pooling import GraphPooling
from src.model.layer.graph_projection import GraphProjection

class MVP2M(nn.Module):
    def __init__(self, supports, pool_idxs, feat_dim, hidden_dim, coord_dim):
        """
        :param supports: list of 3 elements, each is a another list of 2 sparse tensor:
          - first main element
            - first  Sparse tensor indices shape: (156, 2), values shape: (156)
            - second Sparse tensor indices shape: (1080, 2), values shape: (1080)

          - second main element
            - first  Sparse tensor indices shape: (618, 2), values shape: (618)
            - second Sparse tensor indices shape: (4314, 2), values shape: (4314)

          - third main element
            - first  Sparse tensor indices shape: (2466, 2), values shape: (2466)
            - second Sparse tensor indices shape: (17250, 2), values shape: (17250)

        :param pool_idxs: list of 2 elements:
          - first element shape (462, 2)
          - second element shape (1848, 2)

        :param feat_dim: number = 2883 
        :param hidden_dim: number = 192 
        :param coord_dim: number of channels per image, default is 3
        """

        super().__init__()
        layers=[]

        # Cameras should be passed to the forward pass, not in the init of the model
        layers.append(GraphProjection())
        layers.append(GraphConvolution(feat_dim, hidden_dim, supports[0]))

        for _ in range(12):
            layers.append(GraphConvolution(hidden_dim, hidden_dim, supports[0]))
        layers.append(GraphConvolution(hidden_dim, coord_dim, supports[0], act=lambda x: x))

        #2nd block
        layers.append(GraphProjection())
        layers.append(GraphPooling(pool_idxs[0]))
        layers.append(GraphConvolution(feat_dim + hidden_dim, hidden_dim, supports[1]))

        for _ in range(12):
            layers.append(GraphConvolution(hidden_dim, hidden_dim, supports[1]))
        layers.append(GraphConvolution(hidden_dim, coord_dim, supports[1], act=lambda x: x))

        #3rd block
        layers.append(GraphProjection())
        layers.append(GraphPooling(pool_idxs[1]))
        layers.append(GraphConvolution(feat_dim + hidden_dim, hidden_dim, supports[2]))

        for _ in range(13):
            layers.append(GraphConvolution(hidden_dim, hidden_dim, supports[2]))
        layers.append(GraphConvolution(hidden_dim, coord_dim,supports[2], act=lambda x: x))

        self.layers = nn.ModuleList(layers)

        self.unpooling0 = GraphPooling(pool_idxs[0])
        self.unpooling1 = GraphPooling(pool_idxs[1])

    def forward(self, inputs, img_feat, cameras):
        """
        :param inputs: tensor of shape (156, 3)
        :param camera: tensor of shape (N, 5), where N = 3 <= number of cameras
        :param img_feat: list of 4 tensors with shapes: (3, 64, 56, 56), (3, 128, 28, 28), (3, 256, 14, 14), (3, 512, 7, 7)
        """
        eltwise = [3, 5, 7, 9, 11, 13,
                   19, 21, 23, 25, 27, 29,
                   35, 37, 39, 41, 43, 45]
        concat = [15, 31]

        x = []
        x.append(inputs)

        for idx, layer in enumerate(self.layers[:48]):
            if isinstance(layer, GraphProjection):
                hidden = layer(
                    inputs=x[-1],
                    img_feat=img_feat,
                    cameras=cameras)

            else:
                hidden = layer(inputs=x[-1])
            if idx in eltwise:
                hidden = (hidden + x[-2]) * 0.5
            if idx in concat:
                hidden = torch.cat([hidden, x[-2]], 1)
            x.append(hidden)

        output1 = x[15]
        output1_2 = self.unpooling0(output1)

        output2 = x[31]
        output2_2 = self.unpooling1(output2)

        output3 = x[48]

        return {
            "pred_coord": [output1.unsqueeze(0), output2.unsqueeze(0), output3.unsqueeze(0)],
            "pred_coord_before_deform": [inputs.unsqueeze(0), output1_2.unsqueeze(0), output2_2.unsqueeze(0)]
        }
        

