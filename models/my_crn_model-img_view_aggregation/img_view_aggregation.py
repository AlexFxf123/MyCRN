import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast


class ViewAggregation(nn.Module):
    """
    Aggregate frustum view features transformed by depth distribution / radar occupancy
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ViewAggregation, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x)
        x = self.out_conv(x)
        return x
    
if __name__ == '__main__':

    output_channels = 80  # 输出通道数为80
    model = ViewAggregation(output_channels*2,
                            output_channels*2,
                            output_channels)

    ##### 测试view_aggregation_net代码，查看模型结构 #######
    dummy_input = torch.randn(1, output_channels*2, 32, 32)  
    print("\n=== view_aggregation_net 模型结构 ===")
    print(model)
    with torch.no_grad():
        output = model(dummy_input) 
    print(f"view_aggregation_net 输出形状: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f'view_aggregation_net 模型参数总数: {total_params}\n')
    # 导出为onnx格式
    torch.onnx.export(
        model,               # 要导出的模型
        dummy_input,                             # 模型的输入张量
        "view_aggregation_net_new.onnx",         # 导出文件名
        export_params=True,                      # 是否导出训练好的参数
        opset_version=11,                        # ONNX算子集版本
    )