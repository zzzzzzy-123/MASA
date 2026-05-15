import torch
from torch import nn

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred):
        # 1. 展平张量
        gold = gold.reshape(-1)
        pred = pred.reshape(-1)

        # 2. 沿着整个 Batch 求均值
        gold_mean = torch.mean(gold)
        pred_mean = torch.mean(pred)

        # 3. 计算协方差 (除以 N)
        covariance = torch.mean((gold - gold_mean) * (pred - pred_mean))

        # 4. 计算方差 (unbiased=False 也是除以 N)
        gold_var = torch.var(gold, unbiased=False)
        pred_var = torch.var(pred, unbiased=False)

        # 5. 计算 CCC
        ccc = (2. * covariance) / (
                gold_var + pred_var + (gold_mean - pred_mean) ** 2 + 1e-8)

        # 6. CCC 越大越好，所以 Loss 取 1 - CCC
        ccc_loss = 1. - ccc

        return ccc_loss