"""FreeAnchor3DHead — 基于 FreeAnchor 的 3D 检测头。
参考: https://github.com/Sense-GVT/Fast-BEV

核心思想: 将每个 GT 的 top-k anchor 组成"包"，
最大化包的正确性 (Positive Bag Loss)，而非逐个 anchor 匹配。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def delta2bbox_3d(anchors, deltas):
    """将 delta 编码转换为 3D 边界框。
    anchors: [N, 9] (x, y, z, w, l, h, rot, vx, vy)
    deltas:  [N, 9] 编码的偏移量
    """
    PAS = torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       device=deltas.device)
    # 解码
    anchors_whl = anchors[:, 3:6].clamp(min=1e-5)
    ds = deltas * PAS
    x = ds[:, 0:1] * anchors_whl[:, 0:1] + anchors[:, 0:1]
    y = ds[:, 1:2] * anchors_whl[:, 1:2] + anchors[:, 1:2]
    z = ds[:, 2:3] * anchors_whl[:, 2:3] + anchors[:, 2:3]
    w = torch.exp(ds[:, 3:4]) * anchors_whl[:, 0:1]
    l = torch.exp(ds[:, 4:5]) * anchors_whl[:, 1:2]
    h = torch.exp(ds[:, 5:6]) * anchors_whl[:, 2:3]
    rot = ds[:, 6:7] + anchors[:, 6:7]
    vx = ds[:, 7:8] + anchors[:, 7:8] if deltas.shape[1] > 7 else anchors[:, 7:8]
    vy = ds[:, 8:9] + anchors[:, 8:9] if deltas.shape[1] > 8 else anchors[:, 8:9]
    return torch.cat([x, y, z, w, l, h, rot, vx, vy], dim=1)


def bbox3d_overlap(bboxes1, bboxes2):
    """计算两组 3D 边界框的 IoU (简化版)。
    使用 BEV (鸟瞰) 的 2D IoU * 高度 IoU 近似。
    """
    # BEV IoU
    b1_bev = bboxes1[:, [0, 1, 3, 4, 6]]  # x, y, w, l, rot
    b2_bev = bboxes2[:, [0, 1, 3, 4, 6]]
    # 简化为 axis-aligned 近似
    iou_bev = torch.zeros(len(bboxes1), len(bboxes2), device=bboxes1.device)
    # 高度 IoU
    z1_min = bboxes1[:, 2] - bboxes1[:, 5] / 2
    z1_max = bboxes1[:, 2] + bboxes1[:, 5] / 2
    z2_min = bboxes2[:, 2] - bboxes2[:, 5] / 2
    z2_max = bboxes2[:, 2] + bboxes2[:, 5] / 2
    inter_h = torch.min(z1_max[:, None], z2_max[None, :]) - torch.max(z1_min[:, None], z2_min[None, :])
    inter_h = inter_h.clamp(min=0)
    union_h = torch.max(z1_max[:, None], z2_max[None, :]) - torch.min(z1_min[:, None], z2_min[None, :])
    iou_h = inter_h / (union_h + 1e-6)
    # 简化 BEV IoU (用 min/max 矩形近似)
    for i in range(len(bboxes1)):
        x1_min, x1_max = bboxes1[i, 0] - bboxes1[i, 3] / 2, bboxes1[i, 0] + bboxes1[i, 3] / 2
        y1_min, y1_max = bboxes1[i, 1] - bboxes1[i, 4] / 2, bboxes1[i, 1] + bboxes1[i, 4] / 2
        for j in range(len(bboxes2)):
            x2_min, x2_max = bboxes2[j, 0] - bboxes2[j, 3] / 2, bboxes2[j, 0] + bboxes2[j, 3] / 2
            y2_min, y2_max = bboxes2[j, 1] - bboxes2[j, 4] / 2, bboxes2[j, 1] + bboxes2[j, 4] / 2
            ix = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            iy = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            ux = max(x1_max, x2_max) - min(x1_min, x2_min)
            uy = max(y1_max, y2_max) - min(y1_min, y2_min)
            iou_bev[i, j] = (ix * iy) / (ux * uy + 1e-6) if ux * uy > 0 else 0
    return iou_bev * iou_h


class FreeAnchor3DHead(nn.Module):
    """FreeAnchor 3D 检测头。

    Args:
        num_classes: 类别数 (nuScenes=10)
        in_channels: 输入特征通道数
        feat_channels: 特征通道数
        num_convs: 检测头中卷积层数 (Fast-BEV 设为 0)
        pre_anchor_topk: 每个 GT 选取的 top-k anchor 数
        bbox_thr: 边界框阈值
        gamma: FocalLoss gamma
        alpha: FocalLoss alpha
        anchor_generator: anchor 生成器配置
        bbox_coder: 边界框编码器
        loss_cls: 分类损失配置
        loss_bbox: 回归损失配置
        loss_dir: 方向损失配置
    """

    def __init__(self,
                 num_classes=10,
                 in_channels=192,
                 feat_channels=192,
                 num_convs=0,
                 is_transpose=True,
                 pre_anchor_topk=25,
                 bbox_thr=0.5,
                 gamma=2.0,
                 alpha=0.5,
                 anchor_generator=None,
                 bbox_coder=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_dir=None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.is_transpose = is_transpose
        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

        # anchor 生成器
        self.anchor_generator = AnchorGenerator(**anchor_generator)
        self.bbox_coder = bbox_coder

        # 检测头卷积 (num_convs 通常为 0)
        self.convs = nn.Sequential()
        for i in range(num_convs):
            self.convs.add_module(f'conv{i}', ConvModule(in_channels, feat_channels, 3, padding=1))

        # 分类、回归、方向分支
        num_anchors = self.anchor_generator.num_base_anchors
        self.conv_cls = nn.Conv2d(feat_channels, num_anchors * num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(feat_channels, num_anchors * self.bbox_coder['code_size'], 3, padding=1)
        self.conv_dir_cls = nn.Conv2d(feat_channels, num_anchors * 2, 3, padding=1)

    def forward(self, feats):
        """feats: list of [B, C, H, W]"""
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        if self.num_convs > 0:
            x = self.convs(x)
        if self.is_transpose:
            x = x.transpose(-1, -2)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def loss(self, cls_scores, bbox_preds, dir_cls_preds, gt_bboxes_3d, gt_labels_3d, img_metas):
        """FreeAnchor 损失。

        返回:
            dict: {'loss_cls':, 'loss_bbox':, 'loss_dir':}
        """
        batch_size = len(gt_bboxes_3d)
        device = cls_scores[0].device

        # 展平所有层级的预测
        all_cls = []
        all_reg = []
        all_dir = []
        all_anchors = []
        for cls, reg, dir_c in zip(cls_scores, bbox_preds, dir_cls_preds):
            B, A, H, W = cls.shape
            num_anchors = A // self.num_classes
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, self.bbox_coder['code_size'])
            dir_c = dir_c.permute(0, 2, 3, 1).reshape(B, -1, 2)
            # 生成 anchor
            anchors = self.anchor_generator.single_level_grid_anchors(
                (H, W), self.is_transpose, device).reshape(-1, self.bbox_coder['code_size'])
            anchors = anchors.unsqueeze(0).expand(B, -1, -1)

            all_cls.append(cls)
            all_reg.append(reg)
            all_dir.append(dir_c)
            all_anchors.append(anchors)

        cls_scores = torch.cat(all_cls, dim=1)  # [B, N, C]
        bbox_preds = torch.cat(all_reg, dim=1)  # [B, N, 9]
        dir_cls_preds = torch.cat(all_dir, dim=1)  # [B, N, 2]
        all_anchors = torch.cat(all_anchors, dim=1)  # [B, N, 9]

        # 解码边界框
        decoded_bboxes = delta2bbox_3d(all_anchors.view(-1, 9), bbox_preds.view(-1, 9))
        decoded_bboxes = decoded_bboxes.view(batch_size, -1, 9)
        valid_gts = [g for g in gt_bboxes_3d if g is not None and len(g) > 0]

        losses = {'loss_cls': cls_scores.new_zeros(1),
                  'loss_bbox': bbox_preds.new_zeros(1),
                  'loss_dir': dir_cls_preds.new_zeros(1)}

        if not valid_gts:
            return losses

        # 对每张图分别计算 FreeAnchor 损失
        for b in range(batch_size):
            if gt_bboxes_3d[b] is None or len(gt_bboxes_3d[b]) == 0:
                continue
            self._free_anchor_loss_single(
                b, cls_scores, bbox_preds, dir_cls_preds,
                decoded_bboxes, all_anchors,
                gt_bboxes_3d, gt_labels_3d, img_metas, losses)

        return losses

    def _free_anchor_loss_single(self, b, cls_scores, bbox_preds, dir_cls_preds,
                                 decoded_bboxes, all_anchors,
                                 gt_bboxes_3d, gt_labels_3d, img_metas, losses):
        device = cls_scores.device
        gt_boxes = gt_bboxes_3d[b].tensor.to(device)
        gt_labels = gt_labels_3d[b].to(device)
        num_gts = len(gt_boxes)
        num_anchors = cls_scores.shape[1]

        # Sigmoid 概率
        cls_prob = cls_scores[b].sigmoid()  # [N, C]
        box_prob = torch.ones(num_anchors, device=device)

        # 计算每个 GT 与每个 anchor 的 IoU
        ious = bbox3d_overlap(gt_boxes, decoded_bboxes[b])  # [num_gts, N]

        # object_box_prob: 将 IoU 映射到 [0, 1]
        object_box_prob = (ious - self.bbox_thr).clamp(min=0) / (1 - self.bbox_thr)

        # 为每个 GT 选取 top-k anchors
        _, topk_idxs = ious.topk(self.pre_anchor_topk, dim=1)  # [num_gts, topk]
        matched_cls_prob = torch.zeros(num_gts, self.pre_anchor_topk, device=device)
        matched_box_prob = torch.zeros(num_gts, self.pre_anchor_topk, device=device)

        for g in range(num_gts):
            matched_anchors = topk_idxs[g]
            matched_cls_prob[g] = cls_prob[matched_anchors, gt_labels[g]]
            matched_box_prob[g] = object_box_prob[g, matched_anchors]

        # Positive Bag Loss
        matched_prob = matched_cls_prob * matched_box_prob  # [num_gts, topk]
        weight = 1 / (1 - matched_prob + 1e-8)
        weight = weight / weight.sum(dim=1, keepdim=True)
        bag_prob = (weight * matched_prob).sum(dim=1)
        pos_loss = -torch.log(bag_prob + 1e-8).mean()
        losses['loss_cls'] += pos_loss * self.alpha

        # 回归损失 (仅对匹配的 anchor)
        matched_bbox_preds = bbox_preds[b][topk_idxs]  # [num_gts, topk, 9]
        matched_anchors = all_anchors[b][topk_idxs]
        matched_gt = gt_boxes.unsqueeze(1).expand(-1, self.pre_anchor_topk, -1)
        # SmoothL1
        diff = (matched_bbox_preds - matched_gt)
        loss_box = smooth_l1_loss(diff, beta=1/9)
        losses['loss_bbox'] += loss_box.mean() * 0.8

        # 方向损失
        dir_targets = get_direction_target(matched_anchors.reshape(-1, 9),
                                           matched_gt.reshape(-1, 9), dir_offset=0.7854)
        dir_preds = dir_cls_preds[b][topk_idxs].reshape(-1, 2)
        losses['loss_dir'] += F.cross_entropy(dir_preds, dir_targets) * 0.8

    def get_bboxes(self, cls_scores, bbox_preds, dir_cls_preds, img_metas, cfg=None):
        """获取最终检测结果。"""
        results = []
        batch_size = cls_scores[0].shape[0]
        for b in range(batch_size):
            scores = cls_scores[b].sigmoid()
            bboxes = bbox_preds[b]
            dir_cls = dir_cls_preds[b]
            result = self._get_bboxes_single(scores, bboxes, dir_cls, img_metas[b])
            results.append(result)
        return results

    def _get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds, img_meta):
        """单张图的检测结果解码。"""
        # 展平 + 阈值过滤
        cls_scores = cls_scores.view(-1, self.num_classes)
        bbox_preds = bbox_preds.view(-1, self.bbox_coder['code_size'])
        dir_cls_preds = dir_cls_preds.view(-1, 2)

        max_scores, _ = cls_scores.max(dim=1)
        keep = max_scores > 0.1
        cls_scores = cls_scores[keep]
        bbox_preds = bbox_preds[keep]
        dir_cls_preds = dir_cls_preds[keep]

        if len(cls_scores) == 0:
            return [LiDARInstance3DBoxes(torch.zeros(0, 9)), torch.zeros(0), torch.zeros(0)]

        # 解码
        # 需要知道对应的 anchor
        # 简化: 直接返回 bbox_preds 作为预测 (需配合 anchor 编码)
        boxes = LiDARInstance3DBoxes(bbox_preds, box_dim=9)
        scores, labels = cls_scores.max(dim=1)
        return [boxes, scores, labels]


def get_direction_target(anchors, reg_targets, dir_offset=0.7854):
    """计算方向分类目标。
    根据旋转角度区分类别 (前/后，±π/2)。
    """
    rot_gt = reg_targets[:, 6] + dir_offset
    dir_cls_targets = (rot_gt > 0).long()
    return dir_cls_targets


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 Loss。"""
    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


def multi_apply(func, *args, **kwargs):
    """对多个输入应用同一函数。"""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


from functools import partial


class AnchorGenerator(object):
    """简易 anchor 生成器。"""

    def __init__(self, ranges, sizes, custom_values, rotations, reshape_out=True):
        self.ranges = ranges  # [[x1, y1, z1, x2, y2, z2]]
        self.sizes = sizes  # [[w, l, h], ...]
        self.custom_values = custom_values
        self.rotations = rotations
        self.reshape_out = reshape_out
        self.num_base_anchors = len(sizes) * len(rotations)

    def single_level_grid_anchors(self, featmap_size, transpose=False, device='cuda'):
        """生成单层特征图的网格 anchors。"""
        H, W = featmap_size
        stride_x = (self.ranges[0][3] - self.ranges[0][0]) / W
        stride_y = (self.ranges[0][4] - self.ranges[0][1]) / H
        shift_x = torch.arange(0, W, device=device) * stride_x + stride_x / 2 + self.ranges[0][0]
        shift_y = torch.arange(0, H, device=device) * stride_y + stride_y / 2 + self.ranges[0][1]
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=0).t()

        base_anchors = []
        for size in self.sizes:
            for rot in self.rotations:
                base_anchors.append(size + self.custom_values + [rot] + [0, 0])
        base_anchors = torch.tensor(base_anchors, device=device)  # [num_base, 9]

        # 所有位置 x 所有 base anchors
        all_anchors = base_anchors.view(1, -1, 9) + shifts.view(-1, 1, 9)
        all_anchors[:, :, :2] += shifts[:, None, :]
        return all_anchors.reshape(-1, 9)
