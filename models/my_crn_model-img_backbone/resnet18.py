# resnet18的模型代码
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """搭建BasicBlock模块"""
    expansion = 1           # 不做扩展
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()      # 调用父类 nn.Module的构造函数

        # 使用BN层是不需要使用bias的，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # BN层, BN层放在conv层和relu层中间使用
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:         # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
        """神经网络的前向传播函数:
            它接受一个输入张量X，然后通过一些卷积层和批量归一化层来计算输出张量Y。
            如果存在下采样层，它将对输入张量进行下采样以使其与输出张量的尺寸相同。
            最后，输出张量Y和输入张量X的恒等映射相加并通过ReLU激活函数进行激活。"""


class ResNet18(nn.Module):
    # num_classes是训练集的分类个数
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.in_channels = 64   # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.residual_layer(64, 2)
        self.layer2 = self.residual_layer(128, 2, stride=2)
        self.layer3 = self.residual_layer(256, 2, stride=2)
        self.layer4 = self.residual_layer(512, 2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    # 创建残差网络层
    def residual_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride !=1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )
        
        layers = []
        # layers列表保存某个layer_x组块里for循环生成的所有层
        # 添加每一个layer_x组块里的第一层，第一层决定此组块是否需要下采样(后续层不需要)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        # 非关键字参数的特征是一个星号*加上参数名，比如*number，定义后，number可以接收任意数量的参数，并将它们储存在一个tuple中
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)             # x = self.layer1(x)
        x2 = self.layer2(x1)            # x = self.layer2(x)
        x3 = self.layer3(x2)            # x = self.layer3(x)
        x4 = self.layer4(x3)            # x = self.layer4(x)
        # x = self.avgpool(x4)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x1, x2, x3, x4           # return x
    
# 测试ResNet18模型
if __name__ == "__main__":
    model = ResNet18(num_classes=1000)
    print(model)

    # 创建一个随机输入张量（模拟图像批次）
    dummy_input = torch.randn(1, 3, 224, 224)

    # 前向传播
    with torch.no_grad():
        features = model(dummy_input)

    # 计算模型参数总数
    total_params = sum(p.numel() for p in model.parameters())

    print(f"输入形状: {dummy_input.shape}")
    for i, feat in enumerate(features):
        print(f"输出特征图{i+1}形状: {feat.shape}")
    print(f'模型参数总数: {total_params}')   