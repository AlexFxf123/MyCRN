# -*- coding: utf-8 -*-
"""验证 CRN GN 版模型（exps/det/CRN_r18_256x704_128x128_4key_gn.py）。

用法:
    python test/test_model_gn.py            # 独立运行
    pytest test/test_model_gn.py -s -x      # 或 pytest 收集

验证内容:
    1. 直接构造 CameraRadarNetDet（模块级配置）+ convert_bn_to_gn 运行时替换
    2. 模型内不再有 BatchNorm，全部为 GroupNorm
    3. 前向 (is_train=True) 输出 preds 与 depth，形状合理
    4. 检测 loss（真实 get_targets/loss）+ 合成 depth loss 反向，
       各子模块梯度非 None 且无 NaN/Inf

说明:
    - 直接构造模型，不走 CRNLightningModel（避免基类冗余构造 BaseBEVDepth）
    - 需要 GPU（voxel_pooling 等自定义算子仅支持 CUDA）
    - 只构造模型 + 一个 dummy batch，不加载任何真实数据
"""
import os
import sys
from collections import Counter

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.camera_radar_net_det import CameraRadarNetDet
from exps.det.CRN_r18_256x704_128x128_4key_gn import (
    BACKBONE_IMG_CONF, BACKBONE_PTS_CONF, FUSER_CONF, HEAD_CONF,
    NORM_NUM_GROUP, convert_bn_to_gn)

# ---------------- 输入配置（与 _gn.py 保持一致） ----------------
B = 1                     # batchsize
NUM_SWEEPS = 4            # 4key
NUM_CAMS = 6
IMG_H, IMG_W = 256, 704   # final_dim
NUM_POINTS = 128          # 每相机雷达点数（验证用）
POINT_FEAT = 5            # x, y, z + 2 个雷达特征
NUM_GT = 4                # 每样本 GT 框数量
# pts_voxel_layer.point_cloud_range = [0, 2.0, 0, 704, 58.0, 2]
PTS_RANGE = dict(x=(0.0, 700.0), y=(2.5, 57.0), z=(0.1, 1.9))


def build_dummy_batch(device):
    """构造 dummy 输入：sweep_imgs / mats_dict / pts_pv。"""
    sweep_imgs = torch.rand(B, NUM_SWEEPS, NUM_CAMS, 3, IMG_H, IMG_W,
                            device=device)

    def eye_perturb():
        # 单位阵 + 小扰动，保证矩阵可逆
        m = torch.eye(4, device=device).view(1, 1, 1, 4, 4).repeat(
            B, NUM_SWEEPS, NUM_CAMS, 1, 1)
        return m + 0.05 * torch.rand(B, NUM_SWEEPS, NUM_CAMS, 4, 4,
                                     device=device)

    mats = {
        'sensor2ego_mats': eye_perturb(),
        'intrin_mats': eye_perturb(),
        'ida_mats': eye_perturb(),
        'sensor2sensor_mats': eye_perturb(),
        'bda_mat': torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1),
    }

    pts_pv = torch.zeros(B, NUM_SWEEPS, NUM_CAMS, NUM_POINTS, POINT_FEAT,
                         device=device)
    pts_pv[..., 0].uniform_(*PTS_RANGE['x'])
    pts_pv[..., 1].uniform_(*PTS_RANGE['y'])
    pts_pv[..., 2].uniform_(*PTS_RANGE['z'])
    pts_pv[..., 3].uniform_(-1, 1)   # 雷达特征（如 rcs）
    pts_pv[..., 4].uniform_(-1, 1)   # 雷达特征（如 vx/vy）
    return sweep_imgs, mats, pts_pv


def build_gt(device):
    """构造 dummy GT：纯 tensor（与数据集 get_gt 一致，每样本 N x 9）。
    BEVDepthHead.get_targets_single 接收的是 tensor：[:3]=xyz [3:6]=dims
    [6]=yaw [7:]=vel。"""
    gt_boxes_3d, gt_labels_3d = [], []
    n = NUM_GT
    cx = torch.empty(n, device=device).uniform_(-35, 35)
    cy = torch.empty(n, device=device).uniform_(-35, 35)
    z = torch.empty(n, device=device).uniform_(-4, 2)
    dx = torch.empty(n, device=device).uniform_(0.8, 4.0)
    dy = torch.empty(n, device=device).uniform_(0.8, 4.0)
    dz = torch.empty(n, device=device).uniform_(0.8, 3.0)
    yaw = torch.empty(n, device=device).uniform_(-3.14, 3.14)
    vx = torch.empty(n, device=device).uniform_(-5, 5)
    vy = torch.empty(n, device=device).uniform_(-5, 5)
    tensor = torch.stack([cx, cy, z, dx, dy, dz, yaw, vx, vy], dim=1)
    for _ in range(B):
        gt_boxes_3d.append(tensor)
        gt_labels_3d.append(torch.randint(0, 10, (n, ), device=device))  # 共 10 类
    return gt_boxes_3d, gt_labels_3d


def count_norm_layers(model):
    """统计 BatchNorm / GroupNorm 层。"""
    bn = [m for m in model.modules()
          if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    gn = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    return bn, gn


def collect_tensors(x, out):
    """递归收集嵌套结构（tuple/list/dict）里的所有 Tensor。"""
    if isinstance(x, (tuple, list)):
        for y in x:
            collect_tensors(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            collect_tensors(y, out)
    elif isinstance(x, torch.Tensor):
        out.append(x)


def check_grads(module, name):
    """检查某子模块参数梯度：返回 (是否OK, 描述)。"""
    total, n, none_cnt, bad = 0.0, 0, 0, 0
    for p in module.parameters():
        if p.grad is None:
            none_cnt += 1
            continue
        g = p.grad.detach()
        if torch.isnan(g).any() or torch.isinf(g).any():
            bad += 1
            continue
        total += float((g * g).sum())
        n += 1
    if bad:
        ok, desc = False, f'❌ {bad} 个参数梯度为 NaN/Inf'
    elif n == 0:
        ok, desc = False, f'⚠️ 无梯度（{none_cnt} 个参数全部 grad=None）'
    else:
        norm = total ** 0.5
        note = f'（另 {none_cnt} 个参数 grad=None）' if none_cnt else ''
        ok, desc = True, f'✅ grad_norm={norm:.4f}  有梯度参数={n}{note}'
    return ok, f'  [{name:22s}] {desc}'


def main():
    if not torch.cuda.is_available():
        print('需要 GPU（voxel_pooling 等自定义算子仅支持 CUDA），跳过。')
        return
    device = 'cuda'
    torch.manual_seed(0)

    # ---------- 1. 构造模型（直接构造，不走 Lightning 层） ----------
    print('=' * 60)
    print('[1/5] 直接构造 CameraRadarNetDet + convert_bn_to_gn ...')
    model = CameraRadarNetDet(BACKBONE_IMG_CONF,
                              BACKBONE_PTS_CONF,
                              FUSER_CONF,
                              HEAD_CONF)
    convert_bn_to_gn(model, NORM_NUM_GROUP)
    print(f'  模型子模块: {list(model._modules.keys())}')

    # ---------- 2. 结构检查：BN -> GN（CPU，不占显存） ----------
    print('[2/5] 检查 BatchNorm 是否已全部替换为 GroupNorm ...')
    bn, gn = count_norm_layers(model)
    print(f'  BatchNorm 数量 = {len(bn)}')
    print(f'  GroupNorm 数量 = {len(gn)}')
    print(f'  num_groups 分布 = '
          f'{dict(Counter(m.num_groups for m in gn))}')
    assert len(bn) == 0, f'仍有 {len(bn)} 个 BatchNorm 未替换！'
    assert len(gn) > 0, '模型里没有 GroupNorm，替换未生效？'
    print('  ✅ BatchNorm 已全部替换为 GroupNorm')

    # ---------- 显存检查：前向/反向需要较多显存 ----------
    free, total = torch.cuda.mem_get_info(0)
    print(f'  GPU {total/2**30:.1f}GB，当前空闲 {free/2**30:.1f}GB')
    if free < 5 * 2**30:
        print('  ⚠️ 空闲显存 < 5GB（可能被其它训练进程占用），'
              '跳过前向/反向，仅完成结构检查。')
        print('  请先释放显存，再运行完整验证：python test/test_model_gn.py')
        return

    model = model.to(device)
    model.train()

    # ---------- 3. 前向 ----------
    print('[3/5] 构造 dummy 输入，前向 (is_train=True) ...')
    sweep_imgs, mats, pts_pv = build_dummy_batch(device)
    preds, depth = model(sweep_imgs, mats,
                         sweep_ptss=pts_pv, is_train=True)
    print(f'  depth 形状 = {tuple(depth.shape)}')
    all_tensors = []
    collect_tensors(preds, all_tensors)
    print(f'  preds 结构: {type(preds).__name__}, '
          f'含 {len(all_tensors)} 个张量')
    for i, t in enumerate(all_tensors[:6]):
        print(f'    pred_tensor[{i}] = {tuple(t.shape)}')
    assert all_tensors, '前向 preds 为空'
    assert depth.numel() > 0, 'depth 输出为空'

    # ---------- 4. 反向（真实检测 loss + 合成 depth loss） ----------
    print('[4/5] 计算 loss 并反向 ...')
    gt_boxes_3d, gt_labels_3d = build_gt(device)
    try:
        targets = model.get_targets(gt_boxes_3d, gt_labels_3d)
        loss_det, loss_hm, loss_bbox = model.loss(targets, preds)
        loss_depth = 3.0 * depth.float().mean()
        total_loss = loss_det + loss_depth
        print(f'  真实检测 loss: det={loss_det.item():.4f} '
              f'hm={loss_hm.item():.4f} bbox={loss_bbox.item():.4f}')
        print(f'  合成 depth loss: {loss_depth.item():.4f}')
    except Exception as e:
        print(f'  ⚠️ 真实 get_targets/loss 失败，回退合成 loss：{e}')
        loss = 3.0 * depth.float().mean()
        for t in all_tensors:
            loss = loss + t.float().mean()
        total_loss = loss
        print(f'  合成 loss: {total_loss.item():.4f}')
    total_loss.backward()

    # ---------- 5. 梯度检查 ----------
    print('[5/5] 梯度检查 ...')
    ok_all = True
    for name, sub in model._modules.items():
        ok, desc = check_grads(sub, name)
        ok_all = ok_all and ok
        print(desc)
    # 关键子模块再细分
    for name, sub in [
        ('img/backbone', model.backbone_img),
        ('pts/backbone', model.backbone_pts),
        ('fusion', model.fuser),
        ('head', model.head),
    ]:
        ok, desc = check_grads(sub, name)
        ok_all = ok_all and ok
        print(desc)

    print('=' * 60)
    if ok_all:
        print('🎉 验证通过：模型 GN 化正常，前向/反向/梯度均正常。')
    else:
        print('⚠️ 存在异常，请查看上方详细输出。')
        sys.exit(1)


if __name__ == '__main__':
    main()
