import torch
import torch.nn as nn
import math
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # Xavier 初始化，保证参数方差稳定
        std = (2.0 / (in_features + out_features)) ** 0.5

        # 正态分布初始化，数据截断在 [-3sigma, 3sigma]，防止权重出现极端值
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 einsum 方便处理张量维度变化
        # x 最后一维为 i，权重维度为 o * i，输出结果的最后一维为 o
        return torch.einsum('...i, oi -> ...o', x, self.weight)
    

class Embedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((num_embedding, embedding_dim), **factory_kwargs))

        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 相当于用 token_ids 查表，每个 id 对应一个嵌入向量
        return self.weight[token_ids]
    

def silu(in_features):
    return in_features * torch.sigmoid(in_features)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        signal = self.w3(x)
        return self.w2(gate * signal)