"""FreeAnchor3DHead — 基于 FreeAnchor 的 3D 检测头。
参考: https://github.com/Sense-GVT/Fast-BEV

直接基于 mmdetection3d 的 FreeAnchor3DHead 实现，
使用 bbox_coder.encode/decode、bbox_overlaps_nearest_3d 等基础设施。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from torch.cuda.amp import autocast

from mmcv.cnn import ConvModule
from mmdet.core.bbox import build_bbox_coder
from mmdet3d.core import (box3d_multiclass_nms, limit_period,
                          xywhr2xyxyr)
from mmdet3d.core.bbox import LiDARInstance3DBoxes, bbox_overlaps_nearest_3d


class FreeAnchor3DHead(nn.Module):
    """FreeAnchor 3D 检测头 (对齐原版 Fast-BEV 实现)。"""

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
                 diff_rad_by_sin=True,
                 dir_offset=0.7854,
                 anchor_generator=None,
                 bbox_coder=None,
                 train_cfg=None,
                 test_cfg=None):
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
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        self.use_direction_classifier = True
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        self.anchor_generator = AnchorGenerator(**anchor_generator)

        bbox_coder_cfg = dict(type='DeltaXYZWLHRBBoxCoder', **bbox_coder)
        self.bbox_coder = build_bbox_coder(bbox_coder_cfg)
        self.box_code_size = self.bbox_coder.code_size
        self.num_anchors = self.anchor_generator.num_base_anchors

        self.convs = nn.Sequential()
        for i in range(num_convs):
            self.convs.add_module(f'conv{i}',
                ConvModule(in_channels, feat_channels, 3, padding=1))

        self.conv_cls = nn.Conv2d(feat_channels,
                                  self.num_anchors * num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(feat_channels,
                                  self.num_anchors * self.box_code_size, 3, padding=1)
        self.conv_dir_cls = nn.Conv2d(feat_channels,
                                      self.num_anchors * 2, 3, padding=1)

    def forward(self, feats):
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

    def get_anchors(self, featmap_sizes, img_metas=None, device='cuda'):
        """生成多级 anchors (对齐原版 Anchor3DHead.get_anchors)。"""
        if img_metas is not None:
            num_imgs = len(img_metas)
        else:
            num_imgs = 1
        mlvl_anchors = []
        for featmap_size in featmap_sizes:
            anchors = self.anchor_generator.single_level_grid_anchors(
                featmap_size, self.is_transpose, device)
            mlvl_anchors.append(anchors)
        return [mlvl_anchors for _ in range(num_imgs)]

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[..., 6:7])
        boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1)
        return boxes1, boxes2

    def loss(self, cls_scores, bbox_preds, dir_cls_preds,
             gt_bboxes, gt_labels, input_metas):
        """FreeAnchor 损失 (对齐原版实现)。"""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device

        # 1. 获取 anchors
        anchor_list = self.get_anchors(featmap_sizes, input_metas, device=device)
        anchors = [torch.cat(a) for a in anchor_list]  # [N_per_img, 9]

        # 2. 展平预测
        cls_scores = [
            cls.permute(0, 2, 3, 1).reshape(cls.size(0), -1, self.num_classes)
            for cls in cls_scores
        ]
        bbox_preds = [
            b.permute(0, 2, 3, 1).reshape(b.size(0), -1, self.box_code_size)
            for b in bbox_preds
        ]
        dir_cls_preds = [
            d.permute(0, 2, 3, 1).reshape(d.size(0), -1, 2)
            for d in dir_cls_preds
        ]
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        dir_cls_preds = torch.cat(dir_cls_preds, dim=1)

        cls_prob = torch.sigmoid(cls_scores)
        box_prob_list = []
        num_pos = 0
        positive_losses = []

        # 3. 逐样本计算
        for batch_id, (anchors_, gt_labels_, gt_bboxes_, cls_prob_,
                        bbox_preds_, dir_cls_preds_) in enumerate(
            zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds, dir_cls_preds)):

            if hasattr(gt_bboxes_, 'tensor'):
                gt_bboxes_ = gt_bboxes_.tensor.to(anchors_.device)
            else:
                gt_bboxes_ = gt_bboxes_.to(anchors_.device)
            num_anchors = anchors_.size(0)

            # ---- 3a. image_box_prob (仅用于 negative loss) ----
            with torch.no_grad():
                pred_boxes = self.bbox_coder.decode(anchors_, bbox_preds_)
                object_box_iou = bbox_overlaps_nearest_3d(gt_bboxes_, pred_boxes)

                t1 = self.bbox_thr
                t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)

                num_obj = gt_labels_.size(0)
                indices = torch.stack(
                    [torch.arange(num_obj, device=device), gt_labels_], dim=0)
                object_cls_box_prob = torch.sparse_coo_tensor(
                    indices, object_box_prob)

                # image_box_prob: P{a_j in A+}, shape: [N, C]
                box_cls_prob = torch.sparse.sum(
                    object_cls_box_prob, dim=0).to_dense()  # [C, N]

                nonzero_inds = torch.nonzero(box_cls_prob, as_tuple=False).t()
                if nonzero_inds.numel() == 0:
                    image_box_prob = torch.zeros(
                        num_anchors, self.num_classes,
                        device=device, dtype=object_box_prob.dtype)
                else:
                    # nonzero_inds: [2, K] where each column is (class_id, anchor_id)
                    classes, anchor_ids = nonzero_inds[0], nonzero_inds[1]
                    batch_indices = torch.arange(num_obj, device=device)
                    # For each (c, a) pair, get max over all GTs of class c
                    nonzero_box_prob = torch.where(
                        (gt_labels_.unsqueeze(dim=-1) == classes),
                        object_box_prob[:, anchor_ids],
                        torch.tensor(0., device=device)
                    ).max(dim=0).values  # [K]

                    image_box_prob = torch.zeros(
                        num_anchors, self.num_classes,
                        device=device, dtype=object_box_prob.dtype)
                    image_box_prob[anchor_ids, classes] = nonzero_box_prob
                box_prob_list.append(image_box_prob)

            # ---- 3b. Positive bag ----
            match_quality_matrix = bbox_overlaps_nearest_3d(gt_bboxes_, anchors_)
            _, matched = torch.topk(match_quality_matrix, self.pre_anchor_topk,
                                    dim=1, sorted=False)

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2,
                gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)
            ).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors_[matched]
            matched_object_targets = self.bbox_coder.encode(
                matched_anchors,
                gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors))

            # Direction loss
            loss_dir = None
            if self.use_direction_classifier:
                matched_dir_targets = get_direction_target(
                    matched_anchors, matched_object_targets,
                    self.dir_offset, one_hot=False)
                loss_dir = F.cross_entropy(
                    dir_cls_preds_[matched].transpose(-2, -1),
                    matched_dir_targets, reduction='none')

            # Bbox loss (with diff_rad_by_sin)
            if self.diff_rad_by_sin:
                pos_preds, matched_object_targets = self.add_sin_difference(
                    bbox_preds_[matched], matched_object_targets)
            else:
                pos_preds = bbox_preds_[matched]

            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                cw = pos_preds.new_tensor(code_weight)
                bbox_weights = cw.unsqueeze(0).unsqueeze(0)
            else:
                bbox_weights = None

            loss_bbox = smooth_l1_loss(
                pos_preds, matched_object_targets,
                beta=1/9, weight=bbox_weights).sum(-1)

            if loss_dir is not None:
                loss_bbox = loss_bbox + loss_dir

            matched_box_prob = torch.exp(-loss_bbox)
            num_pos += len(gt_bboxes_)
            positive_losses.append(
                self.positive_bag_loss(matched_cls_prob, matched_box_prob))

        # ---- 3c. Aggregate ----
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        if len(box_prob_list) > 0:
            box_prob = torch.stack(box_prob_list, dim=0)
        else:
            box_prob = cls_prob.new_zeros(0)

        negative_loss = self.negative_bag_loss(
            cls_prob, box_prob).sum() / max(1, num_pos * self.pre_anchor_topk)

        return {
            'positive_bag_loss': positive_loss,
            'negative_bag_loss': negative_loss,
        }

    def positive_bag_loss(self, matched_cls_prob, matched_box_prob):
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight = weight / weight.sum(dim=1, keepdim=True)
        bag_prob = (weight * matched_prob).sum(dim=1)
        bag_prob = bag_prob.clamp(0, 1)
        with autocast(enabled=False):
            return self.alpha * F.binary_cross_entropy(
                bag_prob.float(), torch.ones_like(bag_prob).float(), reduction='none')

    def negative_bag_loss(self, cls_prob, box_prob):
        if box_prob.shape[0] == 0:
            return cls_prob.new_zeros(1)
        prob = cls_prob * (1 - box_prob)
        prob = prob.clamp(0, 1)
        with autocast(enabled=False):
            neg_loss = prob.float() ** self.gamma * F.binary_cross_entropy(
                prob.float(), torch.zeros_like(prob).float(), reduction='none')
        return (1 - self.alpha) * neg_loss

    def get_bboxes(self, cls_scores, bbox_preds, dir_cls_preds,
                   input_metas, cfg=None):
        """获取检测结果 (对齐 Anchor3DHead.get_bboxes)。"""
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device

        mlvl_anchors = []
        for featmap_size in featmap_sizes:
            anchors = self.anchor_generator.single_level_grid_anchors(
                featmap_size, self.is_transpose, device)
            mlvl_anchors.append(anchors.reshape(-1, self.box_code_size))

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            dir_cls_pred_list = [dir_cls_preds[i][img_id].detach() for i in range(num_levels)]
            proposals = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                mlvl_anchors, input_metas[img_id], cfg)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds,
                          mlvl_anchors, input_meta, cfg=None):
        """单张图检测 (对齐 Anchor3DHead.get_bboxes_single)。"""
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_bboxes, mlvl_scores, mlvl_dir_scores = [], [], []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0.05)
        max_num = cfg.get('max_num', 500)

        # Build NMS config object (box3d_multiclass_nms needs attribute access)
        nms_cfg = type('NmsCfg', (), dict(use_rotate_nms=True, nms_thr=0.2))()
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        results = box3d_multiclass_nms(
            mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
            score_thr, max_num, nms_cfg, mlvl_dir_scores)

        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, 0, np.pi)
            bboxes[..., 6] = dir_rot + self.dir_offset + np.pi * dir_scores.to(bboxes.dtype)
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels


def get_direction_target(anchors, reg_targets, dir_offset=0, num_bins=2, one_hot=True):
    """方向分类目标 (对齐原版 train_mixins.py)。"""
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_targets = torch.zeros(*dir_cls_targets.shape, num_bins,
                                  dtype=anchors.dtype, device=dir_cls_targets.device)
        dir_targets.scatter_(2, dir_cls_targets.unsqueeze(dim=-1), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets


def smooth_l1_loss(pred, target, beta=1.0, weight=None):
    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    if weight is not None:
        loss = loss * weight
    return loss


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def multiclass_scale_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
                         score_thr, max_num, cfg, mlvl_dir_scores=None):
    """Scale-NMS: 不同类别使用不同 NMS 类型/阈值 (对齐原版 Fast-BEV)."""
    from mmcv.ops import nms_rotated

    num_classes = mlvl_scores.shape[1] - 1
    bboxes, scores, labels, dir_scores = [], [], [], []

    nms_type_list = cfg.get('nms_type_list', ['rotate'] * num_classes)
    nms_thr_list = cfg.get('nms_thr_list', [0.2] * num_classes)
    nms_radius_thr_list = cfg.get('nms_radius_thr_list', [1] * num_classes)
    nms_rescale_factor = cfg.get('nms_rescale_factor', [1.0] * num_classes)

    for i in range(num_classes):
        cls_mask = mlvl_scores[:, i] > score_thr
        if not cls_mask.any():
            continue

        _scores = mlvl_scores[cls_mask, i]
        _bboxes = mlvl_bboxes[cls_mask, :]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_mask, :]

        # Apply rescale
        rescale = nms_rescale_factor[i] if i < len(nms_rescale_factor) else 1.0
        _bboxes_for_nms = _bboxes_for_nms.clone()
        _bboxes_for_nms[:, 2:4] = _bboxes_for_nms[:, 2:4] * rescale

        nms_type = nms_type_list[i] if i < len(nms_type_list) else 'rotate'
        if nms_type == 'circle':
            radius = nms_radius_thr_list[i] if i < len(nms_radius_thr_list) else 1.0
            # Simple distance-based NMS
            _, order = _scores.sort(descending=True)
            centers = _bboxes_for_nms[:, :2]
            keep_mask = torch.ones(len(_scores), dtype=torch.bool, device=_bboxes.device)
            for idx in order:
                if not keep_mask[idx]:
                    continue
                dist = torch.norm(centers[idx].unsqueeze(0) - centers, dim=1)
                keep_mask[dist < radius] = False
                keep_mask[idx] = True
            selected = torch.where(keep_mask)[0]
        else:
            nms_thr = nms_thr_list[i] if i < len(nms_thr_list) else 0.2
            rotated = xywhr2xyxyr(_bboxes_for_nms)
            if torch.isnan(rotated).any() or torch.isinf(rotated).any():
                selected = torch.arange(len(_scores), device=_scores.device)
            else:
                try:
                    selected = nms_rotated(rotated, _scores, nms_thr)[0].long().reshape(-1)
                except RuntimeError:
                    selected = torch.arange(len(_scores), device=_scores.device)

        bboxes.append(_bboxes[selected])
        scores.append(_scores[selected])
        labels.append(_scores.new_full((len(selected),), i, dtype=torch.long))
        if mlvl_dir_scores is not None:
            dir_scores.append(mlvl_dir_scores[cls_mask][selected])

    if not bboxes:
        empty = mlvl_scores.new_zeros(0)
        ret = (mlvl_bboxes.new_zeros(0, mlvl_bboxes.size(-1)), empty, empty.long())
        if mlvl_dir_scores is not None:
            ret = ret + (empty,)
        return ret

    bboxes = [b.reshape(-1, mlvl_bboxes.size(-1)) for b in bboxes]
    bboxes = torch.cat(bboxes)[:max_num]
    scores = torch.cat([s.reshape(-1) for s in scores])[:max_num]
    labels = torch.cat([l.reshape(-1) for l in labels])[:max_num]
    if mlvl_dir_scores is not None:
        dir_scores = torch.cat([d.reshape(-1) for d in dir_scores])[:max_num]
        return bboxes, scores, labels, dir_scores
    return bboxes, scores, labels


class AnchorGenerator(object):
    """Anchor 生成器 (等效 AlignedAnchor3DRangeGenerator)。"""

    def __init__(self, ranges, sizes, custom_values, rotations, reshape_out=True):
        self.ranges = ranges
        self.sizes = sizes
        self.custom_values = custom_values
        self.rotations = rotations
        self.reshape_out = reshape_out
        self.num_base_anchors = len(sizes) * len(rotations)

    def single_level_grid_anchors(self, featmap_size, transpose=False, device='cuda'):
        H, W = featmap_size
        x_range = self.ranges[0]
        stride_x = (x_range[3] - x_range[0]) / W
        stride_y = (x_range[4] - x_range[1]) / H
        shift_x = torch.arange(W, device=device) * stride_x + stride_x / 2 + x_range[0]
        shift_y = torch.arange(H, device=device) * stride_y + stride_y / 2 + x_range[1]
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='xy')
        shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
        z = x_range[2]
        base_anchors = []
        for size in self.sizes:
            w, l, h = size
            for rot in self.rotations:
                base_anchors.append([0, 0, z, w, l, h, rot] + list(self.custom_values))
        base_anchors = torch.tensor(base_anchors, device=device, dtype=torch.float)
        anchors = base_anchors.unsqueeze(0) + torch.cat(
            [shifts, torch.zeros(shifts.shape[0], 7, device=device)], dim=1).unsqueeze(1)
        return anchors.reshape(-1, 9)
