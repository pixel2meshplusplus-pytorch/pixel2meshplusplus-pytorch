
"""3D CNN network implementation"""
import torch
from torch import nn


class LocalGConv(nn.Module):
    def __init__(self, input_dim, output_dim, sample_adj, dropout=0, act=nn.ReLU, bias=True):
        """
        :param input_dim: number of input features
        :param output_dim: number of output features
        :param sample_adj: tensor of shape (K, 43, 43)
        """
        super(LocalGConv, self).__init__()

        self.act = act()
        self.dropout = nn.Dropout(p=dropout)
        self.k = len(sample_adj)
        temp=[]
        for i in range(self.k):
            temp.append(torch.from_numpy(sample_adj[i]).to(torch.float))

        self.support = temp
        self.local_graph_vert = 43
        self.output_dim = output_dim

        self.weights = nn.Parameter(torch.zeros((self.k, input_dim, output_dim), dtype=torch.float))
        for i in range(self.k):
            nn.init.xavier_uniform_(self.weights[i])
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros([output_dim], dtype=torch.float))
            

    def forward(self, inputs):
        """
        :param inputs: tensor of shape (N, S, input_dim)
        :return: tensor of shape (N, S, output_dim)
        """
        device = inputs.get_device()
        x = inputs  # N, S, VF
        x = self.dropout(x)
  
        output = torch.zeros((x.shape[0], x.shape[1], self.output_dim)).to(device)

        for i in range(self.k):
            pre_sup = torch.einsum('ijk,kl->ijl', x, self.weights[i].to(device))
            support = torch.einsum('ij,kjl->kil', self.support[i].to(device), pre_sup)
            output += support
        
        if self.bias is not None:
            output += self.bias.to(device)

        return self.act(output)


class DeformationReasoning(nn.Module):
    def __init__(self, stage2_feat_dim, sample_coord, sample_adj):
        """
        :param stage2_feat_dim: number of features after projection = 339
        :param sample_coord: tensor of shape (43, 3) containing the 42 vertices of a icosahedron centered around (0, 0, 0) and 1 more vertex for the center (0, 0, 0) 
        :param sample_adj: tensor of shape (2, 43, 43)
        """
        super().__init__()
        self.s = 43
        self.delta_coord = torch.from_numpy(sample_coord).to(torch.float)

        self.f = stage2_feat_dim
        self.hidden_dim = 192

        self.local_conv1 = LocalGConv(input_dim=stage2_feat_dim, output_dim=self.hidden_dim, sample_adj=sample_adj)
        self.local_conv2 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, sample_adj=sample_adj)
        self.local_conv3 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, sample_adj=sample_adj)
        self.local_conv4 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, sample_adj=sample_adj)
        self.local_conv5 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, sample_adj=sample_adj)
        self.local_conv6 = LocalGConv(input_dim=self.hidden_dim, output_dim=1, sample_adj=sample_adj)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, mesh_coords, projected_features):
        """
        :param mesh_coords: tensor of shape (N, 3), where N=2466 <= number of vertices
        :param projected_features: tensor of shape (N * S, K), where K = 339 = 3 + 112 + 112 + 112
        :return tensor of shape (N, 3) <= next mesh coordinates
        """
        device = mesh_coords.get_device()
        x = projected_features  # NS, F
        x = torch.reshape(x, [-1, self.s, self.f])  # N, S, F

        x1 = self.local_conv1(x)
        x2 = self.local_conv2(x1)
        x3 = torch.add(self.local_conv3(x2), x1)
        x4 = self.local_conv4(x3)
        x5 = torch.add(self.local_conv5(x4), x3)
        x6 = self.local_conv6(x5)  # N, S, 1

        score = self.softmax(x6)  # N, S, 1

        delta_coord = torch.tile(score,[1,1,3]).to(device) *  torch.tile(self.delta_coord.to(device),[score.shape[0],1,1])
        next_coord = torch.sum(delta_coord, dim=1)
        next_coord += mesh_coords

        return next_coord



