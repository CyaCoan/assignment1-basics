import torch
import torch.nn as nn
from .nn import Linear, Embedding, SwiGLU, RMSNorm, CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
        use_rms_norm: bool = True,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu"
    ):
        
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        self.attn = CausalSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            max_seq_len=max_seq_len, 
            theta=theta,
            device=device, 
            dtype=dtype
        )

        if use_rms_norm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            # 如果禁用 Norm，使用 Identity 占位，它直接返回输入，不改变任何东西
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            # 标准 FFN: x -> Linear -> SiLU -> Linear -> out
            # 注意: 为了公平对比，通常 SiLU FFN 的 d_ff 应该是 4 * d_model
            # 这里我们使用传入的 d_ff
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        
        # Pre-norm (Llama 默认, 也是作业基准)
        # 公式: x = x + Sublayer(Norm(x))
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        
        # Post-norm (原始 Transformer, Ablation 2)
        # 公式: x = Norm(x + Sublayer(x))
        elif self.norm_mode == "post":
            # 注意: Post-norm 通常很难训练，需要 Warmup
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
            
        return x