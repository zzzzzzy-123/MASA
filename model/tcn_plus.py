import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()#批次, 通道, 高度(频率/电极), 宽度(时间)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

#时序残差块
class TemporalBlockPro(nn.Module):
    #dilation扩张率
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlockPro, self).__init__()
        # 第一层 2D 扩张卷积
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=(0, padding), dilation=(1, dilation)))#权重归一化
        self.chomp1 = Chomp2d(padding) # 切未来
        self.relu1 = nn.PReLU()# 激活
        self.dropout1 = nn.Dropout(dropout)# 防止过拟合
        # 第一层 2D 扩张卷积
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=(0, padding), dilation=(1, dilation)))
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入输出通道不一样，用 1x1 卷积调整
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.PReLU()
        self.init_weights()
    #权重初始化
    def init_weights(self):
        #让权重服从正态分布
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        net = self.net(x) # 主分支：两层卷积
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(net + res)# 相加+激活

#SA-TCN
class TemporalConvNetPro(nn.Module):
    #early_fusion是否做空间融合
    def __init__(self, num_channels, num_eeg_chan=32, freq=6, kernel_size=2, dropout=0.2, early_fusion=True):
        super(TemporalConvNetPro, self).__init__()
        self.early_fusion = early_fusion
        if early_fusion:
            self.fusion_layer = weight_norm(nn.Conv2d(
                in_channels=num_channels[0], out_channels=num_channels[0],
                kernel_size=(num_eeg_chan, 1), stride=(1, 1)
            ))
        else:
            self.fusion_layer = nn.Identity()#跳过

        self.space_aware_temporal_layer = nn.Sequential(
            weight_norm(nn.Conv2d(
                in_channels=1, out_channels=num_channels[0],
                kernel_size=(freq, kernel_size), stride=(freq, 1),
                dilation=(1, 2), padding=(0, ((kernel_size - 1) * 2)))),
            Chomp2d((kernel_size - 1) * 2),# 切掉未来
            nn.PReLU(),         # 激活
            nn.Dropout(dropout),# 防止过拟合

            self.fusion_layer   # 空间融合

        )
        layers = []
        num_levels = len(num_channels) - 1
        for i in range(num_levels):
            dilation_size = 2 ** (i+2)
            in_channels = num_channels[i] if i == 0 else num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlockPro(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size - 1) * dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)#把所有层打包成一个整体
        self.init_weights()

    def init_weights(self):
        ## 给第一层卷积（时频卷积）赋初始权重
        self.space_aware_temporal_layer[0].weight.data.normal_(0, 0.01)
        if self.early_fusion:
            self.fusion_layer.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.space_aware_temporal_layer(x)
        return self.network(x)

#空间感知分支
class SpaceAwareTemporalBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, num_eeg_chan=32, freq=6, kernel_size=2, dropout=0.2, early_fusion=True):
        super(SpaceAwareTemporalBlock, self).__init__()
        self.early_fusion = early_fusion
        if early_fusion:
            self.fusion_layer = weight_norm(nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=(num_eeg_chan, 1), stride=(1, 1)
            ))
        else:
            self.fusion_layer = nn.Identity()

        self.space_aware_temporal_layer = nn.Sequential(
            weight_norm(nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(freq, kernel_size), stride=(freq, 1),
                dilation=(1, 2), padding=(0, ((kernel_size - 1) * 2)))),
            Chomp2d((kernel_size - 1) * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            self.fusion_layer
        )
        self.init_weights()

    def forward(self, x):
        return self.space_aware_temporal_layer(x)

    def init_weights(self):
        self.space_aware_temporal_layer[0].weight.data.normal_(0, 0.01)
        if self.early_fusion:
            self.fusion_layer.weight.data.normal_(0, 0.01)

#MSA_TCN
class TemporalConvNetProM(nn.Module):
    def __init__(self, num_channels, num_eeg_chan=32, freq=6, kernel_size=[2, 4, 6], dropout=0.2, early_fusion=True):
        super(TemporalConvNetProM, self).__init__()
        self.early_fusion = early_fusion
        self.sa_tcn_1 = SpaceAwareTemporalBlock(
            out_channels=num_channels[0], num_eeg_chan=num_eeg_chan,
            freq=freq, kernel_size=kernel_size[0], dropout=dropout, early_fusion=early_fusion)

        self.sa_tcn_2 = SpaceAwareTemporalBlock(
            out_channels=num_channels[0], num_eeg_chan=num_eeg_chan,
            freq=freq, kernel_size=kernel_size[1], dropout=dropout, early_fusion=early_fusion)

        self.sa_tcn_3 = SpaceAwareTemporalBlock(
            out_channels=num_channels[0], num_eeg_chan=num_eeg_chan,
            freq=freq, kernel_size=kernel_size[2], dropout=dropout, early_fusion=early_fusion)

        layers = []
        num_levels = len(num_channels) - 1
        for i in range(num_levels):
            dilation_size = 2 ** (i+2)
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlockPro(in_channels, out_channels, kernel_size[1], stride=1, dilation=dilation_size,
                                     padding=int((kernel_size[1] - 1) * dilation_size), dropout=dropout)]

        self.OneByOneConv = weight_norm(nn.Conv2d(
                in_channels=3*num_channels[0], out_channels=num_channels[0],
                kernel_size=(1, 1), stride=(1, 1)
            ))
        self.OneByOneConv.weight.data.normal_(0, 0.01)
        self.pure_temporal_layers = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.sa_tcn_1(x)
        x2 = self.sa_tcn_2(x)
        x3 = self.sa_tcn_3(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.OneByOneConv(x)
        return self.pure_temporal_layers(x)


# ==== 终极灵活版 MASA_TCN_Regressor ====

class MASA_TCN_Regressor(nn.Module):
    # 修正版：完美适配通道与频段的乘积，加入不确定性混合输出头！
    def __init__(self, num_channels_list, num_eeg_chan=8, freq=5, kernel_sizes=[2, 4, 6], dropout=0.3, max_si_score=30):
        #护法修复：在参数列表末尾加上了 max_si_score=30
        super(MASA_TCN_Regressor, self).__init__()

        # 1. 自动计算总输入特征数 (8通道 * 每通道频段)
        total_in_features = num_eeg_chan * freq

        self.feature_gate = nn.Parameter(torch.ones(total_in_features, 1))
        #设定压缩后的目标频段数 (每个通道只保留 2 个高密度特征)
        self.reduced_freq = min(freq, 2)
        bottleneck_out_channels = num_eeg_chan * self.reduced_freq

        # ==============================================================
        # 机关一：深度卷积瓶颈层 (Depthwise Bottleneck)
        # ==============================================================
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=total_in_features,
                      out_channels=bottleneck_out_channels,
                      kernel_size=1,
                      groups=num_eeg_chan),
            nn.BatchNorm1d(bottleneck_out_channels),
            nn.GELU()
        )

        # ==============================================================
        # 机关二：特征级随机丢弃
        # ==============================================================
        self.spatial_dropout = nn.Dropout1d(p=min(dropout, 0.1))

        # 1. 实例化 TCN 主干
        self.tcn = TemporalConvNetProM(
            num_channels=num_channels_list,
            num_eeg_chan=num_eeg_chan,
            freq=self.reduced_freq,
            kernel_size=kernel_sizes,
            dropout=dropout,
            early_fusion=True
        )

        # ==============================================================
        #创新点三挂载：不确定性感知回归头！
        # ==============================================================
        self.num_classes = max_si_score

        final_out_channels = num_channels_list[-1]

        #核心改变：最后一层输出 num_classes 个分类的 Logits，而不是 1 个数字
        self.num_thresholds = max_si_score

        self.ordinal_head = nn.Sequential(
            nn.Linear(final_out_channels, final_out_channels // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_out_channels // 2, self.num_classes)  # <--- 改成输出分类 Logits
        )

        self.regressor = self.ordinal_head

        self.score_regressor = nn.Sequential(
            nn.Linear(final_out_channels, final_out_channels // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_out_channels // 2, 1)
        )

        self.risk_head = nn.Sequential(
            nn.Linear(final_out_channels, final_out_channels // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_out_channels // 2, 2)
        )

        # 提前准备好 0~30 的分数刻度尺 (存在显存中，不参与梯度更新)
        self.register_buffer('score_scale', torch.arange(0, self.num_classes).float())
        # 伯努利惩罚系数 λ (强迫模型必须有明确的态度，不许模棱两可)
        self.lambda_penalty = 0.1

    def forward(self, x):
        # 假设 x 形状: [Batch, 1, 40或96, Seq_len] -> [Batch, 40或96, Seq_len]
        if x.dim() == 4:
            x = x.squeeze(1)

        #启动机关零：特征门控
        gate = torch.sigmoid(self.feature_gate)
        x = x * gate

        #启动瓶颈压缩
        x = self.bottleneck(x)

        #启动特征丢弃
        x = self.spatial_dropout(x)

        # 还原回 TCN 喜欢的四维形状
        x = x.unsqueeze(1)

        tcn_out = self.tcn(x)

        # ==============================================================
        #机关三：全局平均池化 (GAP)
        # ==============================================================
        tcn_features = tcn_out[:, :, 0, :]  # [Batch, Hidden, Seq_len]
        global_features = torch.mean(tcn_features, dim=-1)  # 拍扁成: [Batch, Hidden]

        # ==============================================================
        #计算混合输出：Logits + 连续期望分数
        # ==============================================================
        # 1. 获取分类 Logits (用于算交叉熵)
        logits = self.regressor(global_features)  # 形状: [Batch, 31]

        # 2. Direct regression score. The trainer optimizes this continuous output.
        direct_score = self.score_regressor(global_features)
        risk_logits = self.risk_head(global_features)

        # 必须同时返回 logits 和 expected_score 喂给 trainer!
        return direct_score, logits, risk_logits
