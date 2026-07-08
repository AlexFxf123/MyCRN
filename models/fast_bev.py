"""FastBEV 模型 — 纯视觉 BEV 检测器。
参考: https://github.com/Sense-GVT/Fast-BEV

架构: backbone(ResNet) → neck(FPN) → neck_fuse → 
      LSS view transform → neck_3d(M2BevNeck) → head(FreeAnchor3DHead)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np

from mmcv.cnn import ConvModule
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from layers.necks.m2bev_neck import M2BevNeck, get_points, backproject_inplace


class FPN(nn.Module):
    """简易 FPN 多尺度特征融合。"""

    def __init__(self, in_channels, out_channels, num_outs=4):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            l_conv = ConvModule(in_channels[i], out_channels, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        self.num_outs = num_outs

    def forward(self, inputs):
        """inputs: 多尺度特征列表 [P2, P3, P4, P5]"""
        laterals = [conv(inp) for conv, inp in zip(self.lateral_convs, inputs)]
        # Top-down 路径 (避免 in-place 操作破坏梯度)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear')
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        # Extra levels
        if len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs


class FastBEV(nn.Module):
    """Fast-BEV 纯视觉 BEV 检测器。

    Args:
        backbone: ResNet 配置
        neck: FPN 配置
        neck_fuse: 多尺度融合配置
        neck_3d: M2BevNeck 配置
        bbox_head: FreeAnchor3DHead 配置
        n_voxels: [[vx, vy, vz]] 体素数量
        voxel_size: [[vx, vy, vz]] 体素大小
        multi_scale_id: 用于 BEV 投影的多尺度层级
    """

    def __init__(self,
                 backbone,
                 neck,
                 neck_fuse,
                 neck_3d,
                 bbox_head,
                 n_voxels=[[200, 200, 4]],
                 voxel_size=[[0.5, 0.5, 1.5]],
                 multi_scale_id=[0],
                 with_cp=True,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        # 多尺度 ResNet backbone
        self.backbone = _MultiScaleResNet(depth=backbone['depth'])

        # FPN neck
        self.neck = FPN(**neck)

        # 多尺度特征融合
        self.multi_scale_id = multi_scale_id
        self.neck_fuse = nn.ModuleList()
        for i, msid in enumerate(multi_scale_id):
            in_c = neck['out_channels'] * (len(multi_scale_id) - msid)
            if isinstance(neck_fuse['in_channels'], list):
                fuse_in = neck_fuse['in_channels'][i]
            else:
                fuse_in = in_c
            self.neck_fuse.append(
                nn.Conv2d(fuse_in, neck_fuse['out_channels'][i] if isinstance(neck_fuse['out_channels'], list) else neck_fuse['out_channels'],
                          3, padding=1))

        # M2BevNeck
        self.neck_3d = M2BevNeck(**neck_3d)

        # 检测头
        from layers.heads.free_anchor3d_head import FreeAnchor3DHead
        self.bbox_head = FreeAnchor3DHead(**bbox_head)

        # BEV 配置
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size

        # 时间测量
        self.idx = 0
        self.times_dict = {}

    def extract_feat(self, img, mats_dict, mode='train'):
        """完整特征提取流程: backbone → FPN → LSS → M2BevNeck。

        Args:
            img: [B, N*seq, 3, H, W]
            mats_dict: 包含 intrin_mats, sensor2ego_mats 等
        Returns:
            bev_feat: [B, C_out, Y, X]
            depth: None
            mlvl_feats: FPN 输出的多尺度特征
        """
        B, T, C, H, W = img.shape
        seq = 1  # 简化: 单帧
        N = T // seq  # 相机数

        # 1. Backbone: [B*T, 3, H, W] -> 多尺度特征 [P2, P3, P4, P5]
        img_reshaped = img.view(B * T, C, H, W)
        mlvl_feats_in = self.backbone(img_reshaped)

        # 2. FPN
        mlvl_feats = self.neck(mlvl_feats_in)  # 每个尺度 64 通道

        # 3. 多尺度融合
        fused_feats = []
        for idx, msid in enumerate(self.multi_scale_id):
            base_feat = mlvl_feats[msid]
            for j in range(msid + 1, len(mlvl_feats)):
                up = F.interpolate(mlvl_feats[j], size=base_feat.shape[2:], mode='bilinear', align_corners=False)
                base_feat = torch.cat([base_feat, up], dim=1)
            base_feat = self.neck_fuse[idx](base_feat)
            fused_feats.append(base_feat)

        # 4. LSS 视图变换 (2D → 3D BEV Volume)
        volumes = []
        for lvl, feat in enumerate(fused_feats):
            # feat: [B*T, C, H_lvl, W_lvl]
            _, C_feat, H_feat, W_feat = feat.shape
            n_vox = self.n_voxels[lvl] if lvl < len(self.n_voxels) else self.n_voxels[0]
            v_size = self.voxel_size[lvl] if lvl < len(self.voxel_size) else self.voxel_size[0]

            # 投影矩阵: intrinsic @ extrinsic[:3]
            intrin = mats_dict['intrin_mats'][:, 0, :, :]  # [B, N_cam, 4, 4]
            # sensor2ego 的逆作为 extrinsic
            s2e = mats_dict['sensor2ego_mats'][:, 0, :, :]  # [B, N_cam, 4, 4]

            volume_list = []
            for b in range(B):
                # B 个样本独立处理
                b_feat = feat[b * N:(b + 1) * N]  # [N, C, H, W]
                projection = torch.bmm(intrin[b, :, :3, :3], s2e[b, :, :3, :4])  # [N, 3, 4]

                # 生成体素网格点 (在 feat 所在设备上)
                device = b_feat.device
                origin = torch.tensor([0, 0, 0], device=device, dtype=torch.float)
                v_size_t = torch.tensor(v_size, device=device, dtype=torch.float)
                n_vox_t = torch.tensor(n_vox, device=device, dtype=torch.long)
                points = get_points(n_vox_t, v_size_t, origin, device=device)

                # 反投影
                volume = backproject_inplace(b_feat, points, projection, (H_feat, W_feat))
                volume_list.append(volume)

            volumes.append(torch.stack(volume_list))  # [B, C, vx, vy, vz]

        # 5. 多尺度 volume 融合
        if len(volumes) == 1:
            bev_volume = volumes[0]
        else:
            # 将不同尺度的 volume 对齐到最大分辨率
            target_size = volumes[0].shape[2:]
            aligned = []
            for v in volumes:
                if v.shape[2:4] != target_size:
                    v = F.interpolate(v, size=target_size, mode='trilinear', align_corners=False)
                aligned.append(v)
            bev_volume = torch.cat(aligned, dim=1)

        # 6. M2BevNeck: [B, Z*C, X, Y] → [B, C_out, Y', X']
        bev_feat = self.neck_3d(bev_volume)
        if isinstance(bev_feat, list):
            bev_feat = bev_feat[0]

        return bev_feat, None, mlvl_feats

    def forward(self, sweep_imgs, mats_dict, is_train=False, **kwargs):
        """前向传播。

        Args:
            sweep_imgs: [B, T, N, 3, H, W]
            mats_dict: 相机参数
        Returns:
            if train: (cls_scores, bbox_preds, dir_cls_preds), depth_preds
            if test: (cls_scores, bbox_preds, dir_cls_preds)
        """
        B, T, N, C, H, W = sweep_imgs.shape
        # 合并 T 和 N: [B, T*N, 3, H, W]
        img = sweep_imgs.view(B, T * N, C, H, W)

        if is_train:
            bev_feat, depth, _ = self.extract_feat(img, mats_dict, 'train')
            cls_scores, bbox_preds, dir_cls_preds = self.bbox_head([bev_feat])
            # 返回一个 dummy depth 以兼容框架
            dummy_depth = sweep_imgs.new_zeros(B, 1, 1, 1)
            return (cls_scores, bbox_preds, dir_cls_preds), dummy_depth
        else:
            if self.idx < 100:
                self.times = None
            bev_feat, _, _ = self.extract_feat(img, mats_dict, 'test')
            cls_scores, bbox_preds, dir_cls_preds = self.bbox_head([bev_feat])
            if self.idx < 1002:
                self.idx += 1
            return cls_scores, bbox_preds, dir_cls_preds

    def get_targets(self, gt_boxes_3d, gt_labels_3d):
        """为兼容现有框架的占位方法。"""
        return {'gt_bboxes_3d': gt_boxes_3d, 'gt_labels_3d': gt_labels_3d}

    def loss(self, targets, preds):
        """兼容框架的 loss 接口。

        preds: (cls_scores, bbox_preds, dir_cls_preds)
        targets: dict
        """
        cls_scores, bbox_preds, dir_cls_preds = preds
        gt_bboxes_3d = targets['gt_bboxes_3d']
        gt_labels_3d = targets['gt_labels_3d']

        losses = self.bbox_head.loss(
            cls_scores, bbox_preds, dir_cls_preds,
            gt_bboxes_3d, gt_labels_3d, gt_bboxes_3d)

        # 兼容 CRN 框架的返回格式
        loss_det = losses.get('positive_bag_loss', cls_scores[0].new_zeros(1))
        loss_hm = cls_scores[0].new_zeros(1)
        loss_bbox = losses.get('negative_bag_loss', cls_scores[0].new_zeros(1))
        return loss_det, loss_hm, loss_bbox

    def get_bboxes(self, preds, img_metas):
        """将模型输出解码为边界框。
        preds: (cls_scores, bbox_preds, dir_cls_preds)
            每个元素是 list of [B, A*C, H, W] per level
        """
        cls_scores, bbox_preds, dir_cls_preds = preds
        return self.bbox_head.get_bboxes(cls_scores, bbox_preds, dir_cls_preds, img_metas)


class _MultiScaleResNet(nn.Module):
    """ResNet 输出多尺度特征 (C2, C3, C4, C5)."""
    def __init__(self, depth=18):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50
        nets = {18: resnet18, 34: resnet34, 50: resnet50}
        net = nets.get(depth, resnet18)(pretrained=True)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 64,  H/4
        self.layer2 = net.layer2  # 128, H/8
        self.layer3 = net.layer3  # 256, H/16
        self.layer4 = net.layer4  # 512, H/32

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
