import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    # m 维度 (b, s, 1)
    m = torch.max(logits, dim=-1, keepdim=True).values

    # 通过 targets 查找 logits 对应 id 上的分量
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 减去 logits 分量的最大值，这样 logits 分量最大为 0 ，避免指数运算时数据爆炸
    shifted_logits = logits - m

    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    loss = log_sum_exp - target_logits

    return torch.mean(loss)