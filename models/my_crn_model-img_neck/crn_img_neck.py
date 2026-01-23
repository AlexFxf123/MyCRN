import torch
import torch.nn as nn
import torch.nn.functional as F
from secondfpn import SECONDFPN

def test_second_fpn():
    """测试SECONDFPN模型的各项功能"""
    # 设置参数（模拟常见的SECOND主干网络输出）
    in_channels = [64, 128, 256]   # 不同层级的通道数
    out_channels = 256             # FPN输出通道数
    batch_size = 2                 # 批大小
    input_shapes = [(32, 32), (16, 16), (8, 8)]  # 特征图空间尺寸 (H, W)
    
    # 创建模型实例
    model = SECONDFPN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_outs=3  # 输出3个特征图
    )
    
    print("=== SECONDFPN 模型测试 ===")
    print(f"输入通道数: {in_channels}")
    print(f"输出通道数: {out_channels}")
    print(f"输入特征图形状: {[f'(batch, {c}, {h}, {w})' for (h, w), c in zip(input_shapes, in_channels)]}")
    
    # 1. 创建模拟输入
    inputs = []
    for i, (h, w) in enumerate(input_shapes):
        input_tensor = torch.randn(batch_size, in_channels[i], h, w)
        inputs.append(input_tensor)
        print(f"输入 {i}: {input_tensor.shape}")
    
    # 2. 前向传播测试
    model.eval()
    with torch.no_grad():
        # 修改：传递单个参数（列表）
        outputs = model(inputs)
    
    print("\n=== 前向传播结果 ===")
    for i, output in enumerate(outputs):
        print(f"输出 {i}: {output.shape}")
    
    # 3. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== 模型参数量 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 导出为ONNX格式
    try:
        # 为ONNX导出创建示例输入
        dummy_inputs = [torch.randn(1, in_channels[i], h, w) 
                       for i, (h, w) in enumerate(input_shapes)]
        
        # 导出ONNX模型
        onnx_path = "second_fpn_fixed.onnx"
        
        # 关键修改：定义一个包装函数来处理多个输入
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
            
            def forward(self, input0, input1, input2):
                # 将多个输入打包成列表
                inputs = [input0, input1, input2]
                outputs = self.model(inputs)
                return outputs
        
        # 使用包装后的模型
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        torch.onnx.export(
            wrapped_model,
            tuple(dummy_inputs),  # 传递多个输入张量
            onnx_path,
            input_names=[f'input_{i}' for i in range(len(dummy_inputs))],
            output_names=[f'output_{i}' for i in range(len(dummy_inputs))],
            opset_version=11,  # 使用较新的算子集
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

if __name__ == "__main__":
    test_second_fpn()
