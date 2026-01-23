import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from secondfpn2 import SECONDFPN

def test_second_fpn():
    """测试重新设计的SECONDFPN模型"""
    # 使用用户提供的配置参数
    in_channels = [64, 128, 256, 512]
    out_channels = [64, 64, 64, 64]
    upsample_strides = [0.25, 0.5, 1, 2]
    
    print("=== SECONDFPN 模型测试（deblocks版）===")
    print(f"输入通道数: {in_channels}")
    print(f"输出通道数: {out_channels}")
    print(f"上采样步长: {upsample_strides}")
    
    # 创建模型实例
    model = SECONDFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_strides=upsample_strides
    )
    
    # 打印模型结构
    print("\n=== 模型结构 ===")
    print(model)
    
    # 1. 创建模拟输入
    batch_size = 2
    # 根据上采样步长推测输入特征图尺寸
    # 假设最终输出尺寸为32x32
    target_size = 32
    inputs = []
    for i, stride in enumerate(upsample_strides):
        if stride < 1.0:
            # 下采样，输入尺寸较大
            input_size = int(target_size / stride)
        elif stride == 1.0:
            # 保持原尺寸
            input_size = target_size
        else:  # stride > 1.0
            # 上采样，输入尺寸较小
            input_size = int(target_size / stride)
        
        input_tensor = torch.randn(batch_size, in_channels[i], input_size, input_size)
        inputs.append(input_tensor)
        print(f"输入 {i}: {input_tensor.shape}")
    
    # 2. 前向传播测试
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    print("\n=== 前向传播结果 ===")
    for i, output in enumerate(outputs):
        print(f"输出 {i}: {output.shape}")
        print(f"  预期输出通道: {out_channels[i]}")
    
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
        onnx_path = "second_fpn_deblocks.onnx"
        
        # 定义包装函数来处理多个输入
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
            
            def forward(self, input0, input1, input2, input3):
                inputs = [input0, input1, input2, input3]
                outputs = self.model(inputs)
                return tuple(outputs)
        
        # 使用包装后的模型
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        torch.onnx.export(
            wrapped_model,
            tuple(dummy_inputs),
            onnx_path,
            input_names=[f'input_{i}' for i in range(len(dummy_inputs))],
            output_names=[f'output_{i}' for i in range(len(dummy_inputs))],
            opset_version=11,
            dynamic_axes={
                **{f'input_{i}': {0: 'batch_size'} for i in range(len(dummy_inputs))},
                **{f'output_{i}': {0: 'batch_size'} for i in range(len(dummy_inputs))}
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

def test_manual_config():
    """手动测试特定配置"""
    print("\n" + "="*50)
    print("手动测试特定配置")
    print("="*50)
    
    # 测试配置：与提供的模型结构一致
    config = {
        'in_channels': [64, 128, 256, 512],
        'out_channels': [64, 64, 64, 64],
        'upsample_strides': [0.25, 0.5, 1, 2]
    }
    
    print(f"in_channels: {config['in_channels']}")
    print(f"out_channels: {config['out_channels']}")
    print(f"upsample_strides: {config['upsample_strides']}")
    
    # 创建模型
    model = SECONDFPN(**config)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    # 创建测试输入
    batch_size = 1
    # 根据提供的结构推测输入尺寸
    # 假设最终输出为32x32
    target_size = 32
    inputs = []
    
    # 第一个输入：stride=0.25，下采样4倍，输入尺寸应为128x128
    inputs.append(torch.randn(batch_size, 64, 128, 128))
    # 第二个输入：stride=0.5，下采样2倍，输入尺寸应为64x64
    inputs.append(torch.randn(batch_size, 128, 64, 64))
    # 第三个输入：stride=1，不变，输入尺寸应为32x32
    inputs.append(torch.randn(batch_size, 256, 32, 32))
    # 第四个输入：stride=2，上采样2倍，输入尺寸应为16x16
    inputs.append(torch.randn(batch_size, 512, 16, 16))
    
    print("\n输入特征图形状:")
    for i, inp in enumerate(inputs):
        print(f"  输入 {i}: {inp.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    print("\n输出特征图形状:")
    for i, output in enumerate(outputs):
        print(f"  输出 {i}: {output.shape}")
    
    # 验证所有输出是否具有相同尺寸
    output_shapes = [out.shape[2:] for out in outputs]
    if all(shape == output_shapes[0] for shape in output_shapes):
        print(f"\n所有输出特征图尺寸一致: {output_shapes[0]}")
    else:
        print(f"\n输出特征图尺寸不一致: {output_shapes}")

if __name__ == "__main__":
    # 运行主测试
    test_second_fpn()
    
    # 运行手动配置测试
    # test_manual_config()
