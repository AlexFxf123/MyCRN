import torch
import torch.nn as nn
from torchvision.models import resnet18

### 出版代码，后验证有问题 ###
# 加载预训练的ResNet-18模型，这个模型和mmcv定义的不一致
# crn_img_backbone = resnet18(pretrained=True)
# 删除最后两层（全局平均池化层和全连接层）
# crn_img_backbone = nn.Sequential(*list(crn_img_backbone.children())[:-2])
# 打印修改后的模型结构
# print(crn_img_backbone)
### 验证代码 ###
# 自定义ResNet18模型代码
from resnet18 import ResNet18
crn_img_backbone = ResNet18(num_classes=10)


# 创建一个随机输入张量（模拟图像批次）
dummy_input = torch.randn(1, 3, 224, 224)

# 前向传播
with torch.no_grad():
    features = crn_img_backbone(dummy_input)

# 计算模型参数总数
total_params = sum(p.numel() for p in crn_img_backbone.parameters())

print(f"输入形状: {dummy_input.shape}")
for i, feat in enumerate(features):
        print(f"输出特征图{i+1}形状: {feat.shape}")
print(f'模型参数总数: {total_params}')

# 导出为onnx格式
torch.onnx.export(
    crn_img_backbone,                           # 要导出的模型
    dummy_input,                                # 模型的输入张量
    "crn_img_backbone2.onnx",                   # 导出文件名
    export_params=True,                         # 是否导出训练好的参数
    opset_version=11,                           # ONNX算子集版本
)