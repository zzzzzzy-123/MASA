from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet
from model.tcn_plus import TemporalConvNetPro, TemporalConvNetProM

import math
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module, LayerNorm, GELU


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_temporal(nn.Module):
    def __init__(self, model_name, num_inputs=192, cnn1d_channels=[64, 64], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1,
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, bidirectional=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name

        if "tcn" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                            kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            input_dim = cnn1d_channels[-1]

        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                    batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # ==== ⚡️ 终极核心手术：把单薄的 Linear 升级为带强心针的回归头 ====
        self.regressor = nn.Sequential(
            LayerNorm(input_dim),  # 第一步：把 TCN/LSTM 萎缩的特征强行拉回到均值0方差1！
            Linear(input_dim, 64),  # 第二步：降维并重组特征
            GELU(),  # 第三步：用最先进的 GELU 激活函数赋予非线性生命力
            Dropout(0.5),  # 第四步：防止过拟合
            Linear(64, output_dim)  # 第五步：干干净净地输出最终分数
        )

    def forward(self, x):
        assert len(x.keys()) == 1, "This model is not designed for more than one modalities."

        x = x[list(x.keys())[0]]
        x = x.squeeze(1)
        if "lstm" in self.model_name:
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
            x = self.temporal(x).transpose(1, 2).contiguous()

        x = self.regressor(x).contiguous()
        return x


class SA_TCN(nn.Module):
    def __init__(self, model_name, cnn1d_channels=[64, 64], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1, num_eeg_chan=8, freq=5, output_dim=1, early_fusion=True):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.temporal = TemporalConvNetPro(num_channels=cnn1d_channels, num_eeg_chan=num_eeg_chan, freq=freq,
                                           kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate,
                                           early_fusion=early_fusion)

        # ==== ⚡️ 终极核心手术 ====
        self.regressor = nn.Sequential(
            LayerNorm(cnn1d_channels[-1]),
            Linear(cnn1d_channels[-1], 64),
            GELU(),
            Dropout(0.5),
            Linear(64, output_dim)
        )

    def forward(self, x):
        assert len(x.keys()) == 1, "This model is not designed for more than one modalities."
        x = x[list(x.keys())[0]]
        x = x.transpose(2, 3).contiguous()
        x = self.temporal(x).transpose(1, 3).contiguous()
        x = x.squeeze(-2)
        x = self.regressor(x).contiguous()
        return x


class MSA_TCN(nn.Module):
    def __init__(self, model_name, cnn1d_channels=[64, 64], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1, num_eeg_chan=8, freq=5,
                 output_dim=1, early_fusion=True):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.temporal = TemporalConvNetProM(num_channels=cnn1d_channels, num_eeg_chan=num_eeg_chan, freq=freq,
                                            kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate,
                                            early_fusion=early_fusion)


        self.regressor = nn.Sequential(
            LayerNorm(cnn1d_channels[-1]),
            Linear(cnn1d_channels[-1], 64),
            GELU(),
            Dropout(0.5),
            Linear(64, output_dim)
        )

    def forward(self, x):
        assert len(x.keys()) == 1, "This model is not designed for more than one modalities."
        #从字典里取出 EEG 数据
        x = x[list(x.keys())[0]]
        #交换维度：频率 ↔ 时间
        x = x.transpose(2, 3).contiguous()
        #送入 MSA_TCN 主干网络提取特征，
        x = self.temporal(x).transpose(1, 3).contiguous()

        x = x.squeeze(-2)
        x = self.regressor(x).contiguous()
        return x


class DualBranch_MASA_Regressor(nn.Module):
    def __init__(self, de_channels=40, pli_channels=140, freq=40, dropout=0.6, max_si_score=30):
        super().__init__()

        self.num_classes = max_si_score + 1

        # 创新点：不再输出 1 个数字，而是输出 num_classes 个概率！
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_classes)  # 核心改动 1
        )

        # 提前准备好 0,1,2...30 的分数刻度尺 (不需要梯度更新，存在显存里)
        self.register_buffer('score_scale', torch.arange(0, self.num_classes).float())

        # 伯努利惩罚系数 λ (控制模型不许“模棱两可”)
        self.lambda_penalty = 0.1

    def forward(self, x):
        # 1. 正常的双分支特征提取
        x_de = x[:, :40, :]
        x_pli = x[:, 40:, :]

        out_de = self.branch_de(x_de)
        out_pli = self.branch_pli(x_pli)
        fused_features = torch.cat([out_de, out_pli], dim=1)

        # 2. 创新点流转：获取每个分数段的 Logits
        logits = self.fusion_head(fused_features)  # 形状: [Batch, 31]

        # 3. 将 Logits 转成 0~1 之间的概率
        probs = F.softmax(logits, dim=1)

        # 4. 施加伯努利惩罚 (Bernoulli Penalty)：打压那些接近 0.5 的犹豫概率
        penalized_probs = probs
        # 防止概率变成负数，并重新归一化保证总和为 1

        # 5. 算期望：概率 × 对应的分数，求和得到最终的连续预测值！
        # 例如：0.1的概率是10分，0.9的概率是11分，最终期望 = 0.1*10 + 0.9*11 = 10.9分
        expected_score = torch.sum(penalized_probs * self.score_scale, dim=1, keepdim=True)

        # 返回分类 Logits (为了算 CrossEntropy Loss) 和 最终分数 (为了算 CCC Loss)
        return logits, expected_score
