import torch.nn as nn
import torch.nn.functional as F

class SECONDFPN(nn.Module):
    """
    SECONDFPN (Feature Pyramid Network for SECOND) 颈部网络。
    用于融合来自主干网络（如SECOND）的多尺度特征。
    """
    def __init__(self, in_channels, out_channels, num_outs, upsample_strides=[1, 2, 4]):
        """
        Args:
            in_channels (list[int]): 输入特征图的通道数列表，例如 [64, 128, 256]
            out_channels (int): 输出特征图的统一通道数
            num_outs (int): 期望输出的特征图数量
            upsample_strides (list[int]): 上采样步长列表
        """
        super(SECONDFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # 输入特征图数量
        self.num_outs = num_outs
        
        # 1. 使用1x1卷积调整输入特征图的通道数
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = nn.Conv2d(
                in_channels[i], out_channels, kernel_size=1, stride=1, padding=0
            )
            self.lateral_convs.append(l_conv)
        
        # 2. 使用3x3卷积进一步融合特征（防止混叠效应）
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            fpn_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.fpn_convs.append(fpn_conv)
        
        # 3. 上采样设置
        self.upsample_strides = upsample_strides
        
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): 来自主干网络的多尺度特征图列表
        Returns:
            outs (list[Tensor]): 融合后的多尺度特征图列表
        """
        # 1. 调整输入特征图的通道数
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # 2. 自顶向下路径（从深层特征到浅层特征）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 上采样
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
        
        # 3. 构建输出特征图
        outs = []
        for i in range(used_backbone_levels):
            outs.append(self.fpn_convs[i](laterals[i]))
        
        # 4. 如果需要额外输出，在最深层特征图上进行下采样
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        
        return tuple(outs)  # 返回元组便于后续处理