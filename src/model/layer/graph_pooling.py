"""Graph pooling for Pixel2Mesh"""
import torch
from torch import nn
from src.utils.tools import gather_nd_torch

class GraphPooling(nn.Module):
    """Graph Pooling 
    param pool_idx [*,2]
    
    """
    def __init__(self, pool_idx) -> None:
        super().__init__()
        self.size=pool_idx.shape[0] #462,2
        self.pool_idx=torch.from_numpy(pool_idx).to(torch.int64)

    def forward(self, inputs):

        X=inputs.unsqueeze(0).expand(self.size,-1,-1) # 156,3075-> 1,156,3075-> 462,156,3075
        temp= gather_nd_torch(X, self.pool_idx.unsqueeze(2), batch_dim=1)# 462,156,3075,idx 462,2,3075, -> 462, 2, 3075
        add_feat = (1 / 2.0) * torch.sum(input=temp, dim=1,keepdim=True) # 462 , 3075
        outputs = torch.cat([inputs.squeeze(), add_feat.squeeze()], 0) # 156,3075 + 462,3075 = 618, 3075

        return outputs
