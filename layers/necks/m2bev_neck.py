"""M2BevNeck: 2D→BEV 视图变换后的 3D 卷积处理 (Fast-BEV).
参考: https://github.com/Sense-GVT/Fast-BEV
"""
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class ResModule2D(nn.Module):
    """残差模块: Conv3x3 → BN → ReLU → Conv3x3 → BN → + → ReLU"""

    def __init__(self, n_channels, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.conv0 = ConvModule(n_channels, n_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        self.conv1 = ConvModule(n_channels, n_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        return self.act(x + identity)


class M2BevNeck(nn.Module):
    """2D→BEV 转换后的 3D 卷积处理。

    将 Z 轴折叠到通道维度后，用 2D 卷积处理 BEV 平面。

    Args:
        in_channels: 输入通道数 (单层 Z 的通道数)
        out_channels: 输出通道数
        num_layers: 下采样残差块数量
        stride: 下采样步长
        is_transpose: 是否交换 X/Y 轴 (适配 Anchor3DHead 的 anchor 顺序)
        fuse: 通道融合配置 dict(in_channels, out_channels) 或 None
    """

    def __init__(self, in_channels, out_channels, num_layers=2, stride=2,
                 is_transpose=True, fuse=None, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.is_transpose = is_transpose

        # Z 轴折叠后的通道数: 假设 n_voxels[2]=4, 所以 Z*C
        # fuse 可选: 1x1 卷积降维
        self.fuse = None
        if fuse is not None:
            self.fuse = nn.Conv2d(fuse['in_channels'], fuse['out_channels'], kernel_size=1)

        # 对齐原版 M2BevNeck 架构:
        #   ResModule2D(in_channels) → Conv(in→out, stride=2) →
        #   [ResModule2D(out_channels) → Conv(out→out, stride=1)] × num_layers
        in_c = fuse['out_channels'] if fuse else in_channels
        model = []
        model.append(ResModule2D(in_c, norm_cfg))
        model.append(nn.Conv2d(in_c, out_channels, 3, stride=stride, padding=1))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))
            model.append(nn.BatchNorm2d(out_channels))
            model.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """x: [N, C, X, Y, Z] 或 [N, C, X, Y]"""
        if x.dim() == 5:
            N, C, X, Y, Z = x.shape
            # 折叠 Z 轴到通道: [N, C, X, Y, Z] → [N, X, Y, Z*C] → [N, Z*C, X, Y]
            x = x.permute(0, 2, 3, 4, 1).reshape(N, X, Y, Z * C).permute(0, 3, 1, 2)
        # x: [N, Z*C, X, Y] 或 [N, C, X, Y]
        if self.fuse is not None:
            x = self.fuse(x)
        x = self.model(x)
        if self.is_transpose:
            x = x.transpose(-1, -2)
        return [x]


def get_points(n_voxels, voxel_size, origin, device='cpu'):
    """生成 BEV 体素网格点坐标 (LiDAR 坐标系)."""
    points = torch.stack(torch.meshgrid(
        torch.arange(n_voxels[0], device=device),
        torch.arange(n_voxels[1], device=device),
        torch.arange(n_voxels[2], device=device),
    ))
    new_origin = origin.to(device) - n_voxels.float().to(device) / 2.0 * voxel_size.to(device)
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_inplace(features, points, projection, img_shape):
    """将 2D 图像特征反投影到 3D 体素空间 (in-place 覆盖模式)。

    Args:
        features: [N_cam, C, H, W] 多相机特征图
        points: [3, vx, vy, vz] 体素网格点
        projection: [N_cam, 3, 4] 投影矩阵 P = intrinsic @ extrinsic[:3]
        img_shape: (H, W) 特征图尺寸
    Returns:
        volume: [C, vx, vy, vz] 3D 体素特征
    """
    device = features.device
    n_cam, C, H, W = features.shape
    vx, vy, vz = points.shape[1:]

    # 展平体素点: [3, vx*vy*vz]
    pts_flat = points.reshape(3, -1)  # [3, N]
    ones = torch.ones(1, pts_flat.shape[1], device=device)
    pts_homo = torch.cat([pts_flat, ones], dim=0).unsqueeze(0)  # [1, 4, N]
    pts_homo = pts_homo.expand(n_cam, -1, -1)  # [N_cam, 4, N]

    # 投影到图像: [N_cam, 3, N]
    proj = projection[:, :3, :]  # [N_cam, 3, 4]
    pts_2d = torch.bmm(proj, pts_homo)  # [N_cam, 3, N]

    x = (pts_2d[:, 0] / (pts_2d[:, 2] + 1e-6)).round().long()
    y = (pts_2d[:, 1] / (pts_2d[:, 2] + 1e-6)).round().long()
    z = pts_2d[:, 2]

    valid = (x >= 0) & (y >= 0) & (x < W) & (y < H) & (z > 0)

    volume = torch.zeros(C, vx * vy * vz, device=device, dtype=features.dtype)
    for i in range(n_cam):
        mask = valid[i]
        volume[:, mask] = features[i, :, y[i, mask], x[i, mask]]
    return volume.reshape(C, vx, vy, vz)
