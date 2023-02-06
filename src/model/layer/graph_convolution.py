"""Graph convolution for Pixel2Mesh"""
import torch
from torch import nn
import numpy as np
import torch_sparse


class GraphConvolution(nn.Module):
    """Same Ol' graph convolution """
    def __init__(self, input_dim, output_dim, support, act=nn.ReLU()):
        super().__init__()
        self.output_dim = output_dim

        self.act= act
        temp=list()
        for i in range(len(support)):
            temp.append([(np.array(support[i][0]).T).tolist(),support[i][1],support[i][2]])

        self.support=temp
        self.weights= nn.Parameter(torch.zeros((len(self.support), input_dim, output_dim), dtype=torch.float32))
        for i in range(len(self.support)):
            nn.init.xavier_uniform_(self.weights[i])
        self.bias = nn.Parameter(torch.zeros([output_dim], dtype=torch.float32))

    def forward(self, inputs):
        x=inputs
        #supports = list()
        device = inputs.get_device()
        output = torch.zeros((x.shape[0], self.output_dim)).to(device)

        for i in range(len(self.support)):
            n = self.support[i][2][0]
            pre_sup= torch.matmul(x, self.weights[i]).to(device) 
            # support = torch_sparse.spmm(torch.tensor(self.support[i][0]).to(device), torch.tensor(self.support[i][1]).to(device), n, n, pre_sup).to(device)
            spr_mat=torch.sparse_coo_tensor(indices=self.support[i][0],values=self.support[i][1],size=self.support[i][2], dtype=torch.float32)
            support = torch.sparse.mm(spr_mat.to(device), pre_sup).to(device)
            #supports.append(support)
            output += support
        #output=torch.add(supports) #sum(supports)
        return self.act(output)




