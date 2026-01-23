import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class SECONDFPNWithConcat(nn.Module):
    """
    SECONDFPN的包装器，接受4个独立的4维张量作为输入，
    并将输出在通道维度上进行concat连接
    """
    def __init__(self, in_channels, out_channels, upsample_strides):
        """
        Args:
            in_channels (list[int]): 输入特征图的通道数列表
            out_channels (list[int]): 输出特征图的通道数列表
            upsample_strides (list[float]): 上采样步长列表
        """
        super(SECONDFPNWithConcat, self).__init__()
        
        # 创建原始的SECONDFPN模型
        self.second_fpn = SECONDFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_strides=upsample_strides
        )
        
        # 保存参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_ins = len(in_channels)
        
        # 计算concat后的总通道数
        self.concat_channels = sum(out_channels)
    
    def forward(self, input0, input1, input2, input3):
        """
        前向传播
        Args:
            input0, input1, input2, input3: 四个4维张量输入
        Returns:
            concat_output: 在通道维度上concat后的输出
        """
        # 将输入组合成列表
        inputs = [input0, input1, input2, input3]
        
        # 验证输入数量
        assert len(inputs) == self.num_ins, \
            f"输入张量数量({len(inputs)})与模型定义({self.num_ins})不匹配"
        
        # 通过SECONDFPN处理
        outputs = self.second_fpn(inputs)
        
        # 验证所有输出具有相同尺寸
        output_shapes = [out.shape[2:] for out in outputs]
        if not all(shape == output_shapes[0] for shape in output_shapes):
            raise ValueError(f"输出特征图尺寸不一致: {output_shapes}")
        
        # 在通道维度上进行concat连接
        concat_output = torch.cat(outputs, dim=1)
        
        return concat_output

def test_second_fpn_with_concat():
    """测试带concat的SECONDFPN模型"""
    # 使用用户提供的配置参数
    in_channels = [64, 128, 256, 512]
    out_channels = [64, 64, 64, 64]
    upsample_strides = [0.25, 0.5, 1, 2]
    
    print("=== SECONDFPN 模型测试（带concat版）===")
    print(f"输入通道数: {in_channels}")
    print(f"输出通道数: {out_channels}")
    print(f"上采样步长: {upsample_strides}")
    print(f"Concat后总通道数: {sum(out_channels)}")
    
    # 创建模型实例
    model = SECONDFPNWithConcat(
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_strides=upsample_strides
    )
    
    # 打印模型结构
    print("\n=== 模型结构 ===")
    print(model)
    
    # 1. 创建模拟输入（4个独立的4维张量）
    batch_size = 2
    target_size = 32
    inputs = []
    
    for i, stride in enumerate(upsample_strides):
        if stride < 1.0:
            input_size = int(target_size / stride)
        elif stride == 1.0:
            input_size = target_size
        else:  # stride > 1.0
            input_size = int(target_size / stride)
        
        input_tensor = torch.randn(batch_size, in_channels[i], input_size, input_size)
        inputs.append(input_tensor)
        print(f"输入 {i}: {input_tensor.shape}")
    
    # 2. 前向传播测试
    model.eval()
    with torch.no_grad():
        # 使用4个独立的输入
        concat_output = model(inputs[0], inputs[1], inputs[2], inputs[3])
    
    print("\n=== 前向传播结果 ===")
    print(f"Concat输出形状: {concat_output.shape}")
    print(f"  批次大小: {concat_output.shape[0]}")
    print(f"  通道数: {concat_output.shape[1]} (sum of {out_channels})")
    print(f"  高度: {concat_output.shape[2]}")
    print(f"  宽度: {concat_output.shape[3]}")
    
    # 3. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== 模型参数量 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 导出为ONNX格式
    try:
        # 为ONNX导出创建示例输入
        dummy_inputs = []
        for i, stride in enumerate(upsample_strides):
            if stride < 1.0:
                input_size = int(32 / stride)
            elif stride == 1.0:
                input_size = 32
            else:  # stride > 1.0
                input_size = int(32 / stride)
            
            dummy_input = torch.randn(1, in_channels[i], input_size, input_size)
            dummy_inputs.append(dummy_input)
        
        # 导出ONNX模型
        onnx_path = "second_fpn_with_concat_no_verify.onnx"
        
        # 设置模型为评估模式
        model.eval()
        
        torch.onnx.export(
            model,
            tuple(dummy_inputs),  # 四个独立的输入
            onnx_path,
            input_names=[f'input_{i}' for i in range(len(dummy_inputs))],
            output_names=['concat_output'],
            opset_version=11,
            dynamic_axes={
                **{f'input_{i}': {0: 'batch_size'} for i in range(len(dummy_inputs))},
                'concat_output': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"\n=== ONNX导出成功 ===")
        print(f"模型已保存至: {onnx_path}")
        
    except Exception as e:
        print(f"\n=== ONNX导出失败 ===")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

def test_various_concat_dims():
    """测试不同的concat维度"""
    print("\n" + "="*50)
    print("测试不同的concat维度")
    print("="*50)
    
    # 测试配置
    config = {
        'in_channels': [64, 128, 256, 512],
        'out_channels': [64, 64, 64, 64],
        'upsample_strides': [0.25, 0.5, 1, 2]
    }
    
    # 创建模型
    model = SECONDFPNWithConcat(**config)
    model.eval()
    
    # 创建测试输入
    batch_size = 1
    target_size = 32
    inputs = []
    
    for i, stride in enumerate(config['upsample_strides']):
        if stride < 1.0:
            input_size = int(target_size / stride)
        elif stride == 1.0:
            input_size = target_size
        else:  # stride > 1.0
            input_size = int(target_size / stride)
        
        input_tensor = torch.randn(batch_size, config['in_channels'][i], input_size, input_size)
        inputs.append(input_tensor)
    
    # 测试通道维度concat
    with torch.no_grad():
        concat_output = model(inputs[0], inputs[1], inputs[2], inputs[3])
    
    print(f"通道维度concat输出形状: {concat_output.shape}")
    print(f"  期望通道数: {sum(config['out_channels'])} = {config['out_channels']}")
    
    # 获取中间输出
    with torch.no_grad():
        intermediate_outputs = model.second_fpn(inputs)
    
    print(f"\n中间输出形状:")
    for i, output in enumerate(intermediate_outputs):
        print(f"  输出 {i}: {output.shape}")
    
    # 手动验证concat
    manual_concat = torch.cat(intermediate_outputs, dim=1)
    print(f"\n手动concat输出形状: {manual_concat.shape}")
    
    # 验证结果一致
    diff = torch.abs(concat_output - manual_concat).max().item()
    if diff < 1e-6:
        print("  Concat结果验证通过")
    else:
        print(f"  Concat结果不一致，最大差异: {diff}")

if __name__ == "__main__":
    # 运行主测试
    test_second_fpn_with_concat()
    
    # 运行不同concat维度测试
    test_various_concat_dims()
