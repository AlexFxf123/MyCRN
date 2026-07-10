"""
纯LSS图像→BEV骨干网络。

使用标准 Lift-Splat-Shoot 流水线（无RVT雷达视图变换）：
1. Backbone + Neck → 2D图像特征
2. DepthNet → 深度分布 + 上下文特征
3. 深度加权构建特征体
4. get_geometry_collapsed → 3D坐标（z轴折叠，高效）
5. voxel_pooling_v2 → BEV特征图
"""

import torch
import torch.nn as nn

from mmdet3d.models import build_neck
from mmdet.models import build_backbone

from ops.voxel_pooling_v2 import voxel_pooling
from .rvt_lss_fpn import DepthNet


class LSSBEVBackbone(nn.Module):
    """纯LSS图像→BEV骨干网络，不含RVT雷达视图变换。

    Args:
        x_bound (list): X轴范围和分辨率 [min, max, step]。
        y_bound (list): Y轴范围和分辨率 [min, max, step]。
        z_bound (list): Z轴范围和分辨率 [min, max, step]。
        d_bound (list): 深度范围和分辨率 [min, max, step]。
        final_dim (tuple): 输入图像最终尺寸 (H, W)。
        downsample_factor (int): 特征图相对于输入的下采样因子。
        output_channels (int): 输出BEV特征通道数。
        img_backbone_conf (dict): 图像骨干网络配置。
        img_neck_conf (dict): 图像颈部网络配置。
        depth_net_conf (dict): 深度网络配置。
        camera_aware (bool): 是否使用相机参数调制深度估计。默认 False。
    """

    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, depth_net_conf, camera_aware=False):
        super().__init__()

        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.camera_aware = camera_aware

        # voxel grid buffers
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))

        # frustum
        self.register_buffer('frustum', self._create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        # backbone + neck
        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.img_neck.init_weights()
        self.img_backbone.init_weights()

        # depth net
        self.depth_net = DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            camera_aware=self.camera_aware,
        )

    # ── 工具方法 ────────────────────────────────────────────────

    def _create_frustum(self):
        """生成视锥体网格 (D, H_feat, W_feat, 4)。"""
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)
        return torch.stack((x_coords, y_coords, d_coords, paddings), -1)

    def get_cam_feats(self, imgs):
        """提取多相机图像特征。

        Args:
            imgs (Tensor): (B, S, N, 3, H, W) 输入图像。

        Returns:
            Tensor: (B, S, N, C_feat, H_feat, W_feat) 特征图。
        """
        B, S, N, C, H, W = imgs.shape
        imgs = imgs.flatten().view(B * S * N, C, H, W)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(B, S, N, img_feats.shape[1],
                                      img_feats.shape[2], img_feats.shape[3])
        return img_feats

    def get_geometry_collapsed(self, sensor2ego_mat, intrin_mat, ida_mat,
                               bda_mat, z_min=-5., z_max=3.):
        """将视锥体点从像素坐标变换到自车坐标系，z轴折叠（高效）。

        相比标准 get_geometry，此方法将高度维折叠到z=0平面，
        并返回z有效掩码用于过滤，大幅减少 voxel pooling 的点数。

        Args:
            sensor2ego_mat (Tensor): (B, N, 4, 4) 相机→自车矩阵。
            intrin_mat (Tensor): (B, N, 4, 4) 内参矩阵。
            ida_mat (Tensor): (B, N, 4, 4) IDA变换矩阵。
            bda_mat (Tensor, optional): (B, 4, 4) BDA变换矩阵。
            z_min (float): z轴最小值阈值。
            z_max (float): z轴最大值阈值。

        Returns:
            tuple[Tensor, Tensor]:
                points_out: (B, N, D, 1, W_feat, 3) 折叠后的3D坐标。
                points_valid_z: (B, N, D, H_feat, W_feat) z有效掩码。
        """
        B, N, _, _ = sensor2ego_mat.shape

        points = self.frustum
        ida_mat = ida_mat.view(B, N, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)).double()

        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)).double()
        points = combine.view(B, N, 1, 1, 1, 4, 4).matmul(points).float()

        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, N, 1, 1).view(
                B, N, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3]
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

        return points_out, points_valid_z

    def _forward_depth_net(self, feat, mats_dict):
        """前向DepthNet。

        Args:
            feat (Tensor): (B*N, C_feat, H_feat, W_feat) 图像特征。
            mats_dict (dict): 相机参数。

        Returns:
            Tensor: (B*N, depth_channels + output_channels, H_feat, W_feat)
        """
        return self.depth_net(feat, mats_dict)

    # ── 单帧前向 ────────────────────────────────────────────────

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict,
                              return_depth=False):
        """单帧LSS前向。

        Args:
            sweep_index (int): 当前帧索引。
            sweep_imgs (Tensor): (B, 1, N, 3, H, W) 当前帧图像。
            mats_dict (dict): 相机参数。
            return_depth (bool): 是否返回深度图。

        Returns:
            Tensor: (B, output_channels, H_bev, W_bev) BEV特征图。
                若 return_depth=True，返回 (feature_map, depth)。
        """
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        B, _, N, C, H, W = sweep_imgs.shape

        # 1. 图像特征
        img_feats = self.get_cam_feats(sweep_imgs)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img_backbone'].append(t1.elapsed_time(t2))

        source_features = img_feats[:, 0, ...]  # (B, N, C_feat, H_feat, W_feat)

        # 2. 深度估计
        depth_feature = self._forward_depth_net(
            source_features.reshape(B * N, source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )
        depth_occupancy = depth_feature[:, :self.depth_channels].softmax(1)
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['img_dep'].append(t2.elapsed_time(t3))

        # 3. 深度加权上下文
        image_feature = depth_feature[:, self.depth_channels:(
            self.depth_channels + self.output_channels)]
        img_feat_with_depth = depth_occupancy.unsqueeze(1) * image_feature.unsqueeze(2)

        # 4. 几何变换（z轴折叠）
        geom_xyz, geom_xyz_valid = self.get_geometry_collapsed(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None))

        # 用z有效掩码过滤并折叠高度维
        geom_xyz_valid = geom_xyz_valid.reshape(
            B * N, *geom_xyz_valid.shape[2:]).unsqueeze(1)
        img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3)
        # (B*N, C_out, D, 1, W_feat)

        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['img_transform'].append(t3.elapsed_time(t4))

        # 5. Voxel Pooling (高效V2版本)
        C_feat = img_feat_with_depth.shape[1]
        _, _, D, _, W_feat = img_feat_with_depth.shape

        img_feat_with_depth = img_feat_with_depth.view(
            B, N, C_feat, D, 1, W_feat)
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 1, 3, 4, 5, 2).contiguous()
        img_feat_with_depth = img_feat_with_depth.reshape(B, -1, C_feat)

        geom_xyz = geom_xyz.reshape(B, -1, 3)

        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz[..., 2] = 0

        feature_map = voxel_pooling(
            geom_xyz, img_feat_with_depth.contiguous(),
            self.voxel_num.to(geom_xyz.device))
        # (B, C_out, H_bev, W_bev)

        if self.times is not None:
            t5.record()
            torch.cuda.synchronize()
            self.times['img_pool'].append(t4.elapsed_time(t5))

        if return_depth:
            return feature_map.contiguous(), depth_feature[:, :self.depth_channels].softmax(1)
        return feature_map.contiguous()

    # ── 多帧前向 ────────────────────────────────────────────────

    def forward(self, sweep_imgs, mats_dict, times=None, return_depth=False):
        """前向函数。

        Args:
            sweep_imgs (Tensor): (B, num_sweeps, N, 3, H, W) 多帧图像。
            mats_dict (dict): 相机参数。
            times (dict, optional): 计时字典。
            return_depth (bool): 是否返回深度图。

        Returns:
            Tensor: (B, num_sweeps, output_channels, H_bev, W_bev) 多帧BEV特征。
                若 return_depth=True，返回 (bev_feats, depth, times)。
        """
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        B, num_sweeps, N, C, H, W = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0, sweep_imgs[:, 0:1, ...], mats_dict,
            return_depth=return_depth)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            if return_depth:
                return key_frame_res[0].unsqueeze(1), key_frame_res[1], self.times
            return key_frame_res.unsqueeze(1), self.times

        key_frame_feature = key_frame_res[0] if return_depth else key_frame_res
        ret_feature_list = [key_frame_feature]

        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index, sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict, return_depth=False)
                ret_feature_list.append(feature_map)

        if return_depth:
            return torch.stack(ret_feature_list, 1), key_frame_res[1], self.times
        return torch.stack(ret_feature_list, 1), self.times
