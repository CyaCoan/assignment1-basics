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
    

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)

    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Q: (batch_size, ..., n, d_k)
    K: (batch_size, ..., m, d_k)
    V: (batch_size, ..., m, d_v)
    mask: (..., n, m) 或者是可以广播到该形状的布尔张量 (True 表示关注, False 表示屏蔽)
    """
    d_k = Q.size(-1)
    scores = torch.einsum('...nk, ...mk -> ...nm', Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    probs = softmax(scores, dim=-1)

    output = torch.einsum('...nm, ...mk -> ...nk', probs, V)

    return output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()

        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)

        t = torch.arange(max_seq_len, device=device).float()

        # freqs_matrix 的形状为 (max_seq_len, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos.unsqueeze(1)
            sin.unsqueeze(1)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x_float = x.to(torch.float32)

        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        output = (x_float / rms) * self.weight

        return output.to(in_dtype)