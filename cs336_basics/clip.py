import torch
from collections.abc import Iterable

def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        return
    
    total_norm = 0.0

    for p in params_with_grad:
        param_norm = torch.norm(p.grad.detach(), p=2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    eps = 1e-6

    if total_norm > max_norm:
        coeff = max_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.detach().mul_(coeff)