import torch.nn as nn
import torch.nn.functional as F

class SECONDFPN(nn.Module):
    """
    根据提供的模型结构重新设计的SECONDFPN网络
    包含deblocks模块，用于处理多尺度特征
    """
    def __init__(self, in_channels, out_channels, upsample_strides):
        """
        Args:
            in_channels (list[int]): 输入特征图的通道数列表，例如 [64, 128, 256, 512]
            out_channels (list[int]): 输出特征图的通道数列表，例如 [64, 64, 64, 64]
            upsample_strides (list[float]): 上采样步长列表，例如 [0.25, 0.5, 1, 2]
        """
        super(SECONDFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_ins = len(in_channels)
        
        # 验证参数长度一致
        assert len(in_channels) == len(out_channels) == len(upsample_strides), \
            f"参数长度不一致: in_channels({len(in_channels)}), out_channels({len(out_channels)}), upsample_strides({len(upsample_strides)})"
        
        # 创建deblocks模块列表
        self.deblocks = nn.ModuleList()
        
        for i in range(self.num_ins):
            in_ch = in_channels[i]
            out_ch = out_channels[i]
            stride = upsample_strides[i]
            
            # 根据上采样步长选择卷积类型
            if stride < 1.0:
                # 下采样，使用Conv2d
                # 计算实际的下采样步长
                conv_stride = int(1.0 / stride)
                conv = nn.Conv2d(in_ch, out_ch, 
                                 kernel_size=conv_stride, 
                                 stride=conv_stride, 
                                 bias=False)
            elif stride == 1.0:
                # 保持原尺寸，使用ConvTranspose2d
                conv = nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=1,
                                         stride=1,
                                         bias=False)
            else:  # stride > 1.0
                # 上采样，使用ConvTranspose2d
                conv_stride = int(stride)
                conv = nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=conv_stride,
                                         stride=conv_stride,
                                         bias=False)
            
            # 构建Sequential模块：Conv -> BatchNorm -> ReLU
            deblock = nn.Sequential(
                conv,
                nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            
            self.deblocks.append(deblock)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs (list[Tensor]): 输入特征图列表
        Returns:
            outs (list[Tensor]): 输出特征图列表
        """
        # 验证输入数量
        assert len(inputs) == self.num_ins, \
            f"输入特征图数量({len(inputs)})与模型定义({self.num_ins})不匹配"
        
        # 对每个输入特征图应用对应的deblock
        outs = []
        for i, deblock in enumerate(self.deblocks):
            out = deblock(inputs[i])
            outs.append(out)
        
        return outs