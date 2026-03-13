import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 要从数据集里选取长度为 context_length 的序列
    # 且要将该序列向后偏移 1 位的结果作为目标序列
    # 因此确定最大的序列起始位置，避免选取时超出范围
    max_idx = len(dataset) - context_length - 1

    # 随机产生 batch_size 个起始位置
    # np.random.randint 在 [0, max_idx] 之间产生随机整数
    ix = torch.randint(0, max_idx + 1, (batch_size,))

    # 根据索引提取输入和目标
    x_stack = [dataset[i : i + context_length] for i in ix]
    y_stack = [dataset[i + 1 : i + context_length + 1] for i in ix]

    # 转换为 PyTorch 张量并移动到指定设备
    # 注意：dataset 通常是 int32 或 int64，转为 torch 后通常使用 torch.long (int64)
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()

    return x, y