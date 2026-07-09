#!/usr/bin/env python3
"""
通用模型分析工具: 参数量分析 + FLOPs + ONNX 导出 + 验证
支持多种配置文件: CRN, BEVDepth, FastBEV 等

用法:
    python exps/det/export_analyze.py --config exps/det/CRN_r18_256x704_128x128_4key.py
    python exps/det/export_analyze.py --config exps/det/CRN_r50_256x704_128x128_4key.py
    python exps/det/export_analyze.py --config exps/det/BEVDepth_r50_256x704_128x128_4key.py
    python exps/det/export_analyze.py --config exps/det/FastBEV_r18_256x704_200x200.py
    # 指定输出名称 (可选)
    python exps/det/export_analyze.py --config exps/det/CRN_r18_256x704_128x128_4key.py --name CRN_r18
"""
import argparse, importlib, inspect, os, subprocess, sys, time, warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── 命令行参数 ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description='通用模型分析工具: 参数量+FLOPs+ONNX导出+验证')
parser.add_argument('--config', type=str, required=True,
                    help='配置文件路径, 如 exps/det/CRN_r18_256x704_128x128_4key.py')
parser.add_argument('--name', type=str, default=None,
                    help='输出名称, 默认自动从配置文件名推断')
parser.add_argument('--data_mode', type=str, default='sub',
                    choices=['sub', 'mini', 'full'],
                    help='数据模式 (默认 sub)')
parser.add_argument('--opset', type=int, default=13,
                    help='ONNX opset 版本 (默认 13)')
parser.add_argument('--no_profile', action='store_true',
                    help='跳过 CUDA Profiler 分析')
parser.add_argument('--no_flops', action='store_true',
                    help='跳过 FLOPs 分析')
parser.add_argument('--no_export', action='store_true',
                    help='跳过 ONNX 导出')
parser.add_argument('--skip_steps', type=str, nargs='*', default=[],
                    help='跳过的步骤编号, 如 1 2 3 4 5')
args = parser.parse_args()

# ─── 解析配置 ─────────────────────────────────────────────────
config_path = args.config
if not os.path.isabs(config_path):
    config_path = os.path.join(BASE, config_path)
if not config_path.endswith('.py'):
    config_path += '.py'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件不存在: {config_path}")

# 从文件名推断输出名称
config_basename = os.path.splitext(os.path.basename(config_path))[0]
export_name = args.name or config_basename.replace('_export_analyze', '')
print(f"  配置文件: {config_path}")
print(f"  输出名称: {export_name}")

# 动态导入配置模块
config_dir = os.path.dirname(config_path)
config_module_name = os.path.splitext(os.path.basename(config_path))[0]
sys.path.insert(0, config_dir)
config_module = importlib.import_module(config_module_name)

# 查找 LightningModel 类
lightning_cls = None
# 优先匹配: 在 config_module 中定义 (而非导入) 的 LightningModel 类
for name in dir(config_module):
    obj = getattr(config_module, name)
    if isinstance(obj, type) and 'LightningModel' in name:
        # 检查是否定义在当前模块中 (而非从 base_exp 导入)
        mod_name = getattr(obj, '__module__', '')
        if mod_name == config_module.__name__:
            lightning_cls = obj
            break
# fallback: 任何非 Base 前缀的 LightningModel 类
if lightning_cls is None:
    for name in sorted(dir(config_module)):
        obj = getattr(config_module, name)
        if isinstance(obj, type) and 'LightningModel' in name and 'Base' not in name:
            lightning_cls = obj
            break
# fallback2: 任何含 LightningModel 且有 model 属性的类
if lightning_cls is None:
    for name in dir(config_module):
        obj = getattr(config_module, name)
        if isinstance(obj, type) and hasattr(obj, 'model'):
            lightning_cls = obj
            break
if lightning_cls is None:
    raise ImportError(f"未在 {config_path} 中找到 LightningModel 类")

LightningModel = lightning_cls
lightning_model_name = lightning_cls.__name__
print(f"  LightningModel: {lightning_model_name}")

# ─── 检测模型架构类型 ─────────────────────────────────────────
def detect_arch(model):
    """检测模型架构类型: 'crn', 'bevdepth', 'fastbev'"""
    cls_name = type(model).__name__
    if cls_name == 'CameraRadarNetDet':
        return 'crn'
    elif cls_name == 'BaseBEVDepth':
        # 进一步区分: 有 backbone_pts + fuser 的是 CRN, 否则是纯 BEVDepth
        if hasattr(model, 'backbone_pts') and hasattr(model, 'fuser'):
            return 'crn'
        return 'bevdepth'
    elif cls_name == 'FastBEV':
        return 'fastbev'
    # fallback: 通过属性检测
    if hasattr(model, 'backbone_pts') and hasattr(model, 'fuser'):
        return 'crn'
    if hasattr(model, 'backbone_img'):
        return 'bevdepth'
    return 'unknown'

# ─── 工具函数 ─────────────────────────────────────────────────
def print_sep(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─── Conv+BN 融合工具函数 (模块级, 可被外部导入) ─────────────
def fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """融合 Conv2d + BatchNorm2d → 新的 Conv2d (推理等价)"""
    assert conv.out_channels == bn.num_features, \
        f"通道不匹配: Conv({conv.out_channels}) vs BN({bn.num_features})"
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True,
    )
    w = conv.weight
    if conv.groups > 1:
        g = conv.groups
        oc_per_g = conv.out_channels // g
        ic_per_g = conv.in_channels // g
        w = w.view(g, oc_per_g, ic_per_g, *conv.kernel_size)
        bn_w = bn.weight.view(g, oc_per_g, 1, 1, 1)
        bn_b = bn.bias.view(g, oc_per_g, 1, 1, 1)
        bn_rm = bn.running_mean.view(g, oc_per_g, 1, 1, 1)
        bn_rv = bn.running_var.view(g, oc_per_g, 1, 1, 1)
    else:
        bn_w = bn.weight.view(-1, 1, 1, 1)
        bn_b = bn.bias.view(-1, 1, 1, 1)
        bn_rm = bn.running_mean.view(-1, 1, 1, 1)
        bn_rv = bn.running_var.view(-1, 1, 1, 1)
    fused.weight = nn.Parameter(w * bn_w / torch.sqrt(bn_rv + bn.eps))
    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias = nn.Parameter(
        bn.bias + (conv_bias - bn.running_mean) * bn.weight / torch.sqrt(bn.running_var + bn.eps)
    )
    return fused


def fuse_all_conv_bn(module):
    """递归遍历模型, 融合所有 Conv2d + BatchNorm2d 连续对 (原地修改)"""
    replacements = {}
    children = list(module.named_children())
    for i, (name, child) in enumerate(children):
        cls = type(child)
        if cls in (nn.ConvTranspose2d,):
            fuse_all_conv_bn(child)
            continue
        if cls == nn.Conv2d and i + 1 < len(children):
            next_name, next_child = children[i + 1]
            if isinstance(next_child, nn.BatchNorm2d) and \
               child.out_channels == next_child.num_features:
                replacements[name] = fuse_conv_bn_pair(child, next_child)
                replacements[next_name] = nn.Identity()
        if cls != nn.Identity:
            fuse_all_conv_bn(child)
    for name, new_child in replacements.items():
        setattr(module, name, new_child)


skip_steps = [int(s) for s in args.skip_steps]

# =====================================================================
# Step 1: 参数量分析
# =====================================================================
if 1 not in skip_steps:
    print_sep(f"[1/5] 模型参数量分析 ({export_name})")
    model = LightningModel(data_mode=args.data_mode)
    model = model.to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)\n")
    top_modules = {}
    for name, param in model.model.named_parameters():
        top = name.split('.')[0]
        top_modules[top] = top_modules.get(top, 0) + param.numel()
    print(f"  {'模块':<38s} {'参数量':>10s} {'占比':>8s}")
    print("  " + "-" * 58)
    for mod, cnt in sorted(top_modules.items(), key=lambda x: -x[1]):
        print(f"  model.{mod:<34s} {cnt:>10,}  {cnt/total_params*100:>7.1f}%")
    print("  " + "-" * 58)

    arch = detect_arch(model.model)
    # 子模块 (CRN 特有)
    if arch == 'crn':
        for sub_name in ['backbone_img', 'head', 'backbone_pts']:
            sub = {}
            for n, p in getattr(model.model, sub_name).named_parameters():
                t = n.split('.')[0]; sub[t] = sub.get(t, 0) + p.numel()
            print(f"\n  --- {sub_name} ---")
            for m, c in sorted(sub.items(), key=lambda x: -x[1]):
                print(f"    {m:<28s} {c:>10,}")
    elif arch == 'bevdepth':
        for sub_name in ['backbone_img', 'head']:
            sub = {}
            for n, p in getattr(model.model, sub_name).named_parameters():
                t = n.split('.')[0]; sub[t] = sub.get(t, 0) + p.numel()
            print(f"\n  --- {sub_name} ---")
            for m, c in sorted(sub.items(), key=lambda x: -x[1]):
                print(f"    {m:<28s} {c:>10,}")

    del model; torch.cuda.empty_cache()
else:
    print("[1/5] 跳过")

# =====================================================================
# Step 2: FLOPs 分析
# =====================================================================
if 2 not in skip_steps and not args.no_flops:
    print_sep(f"[2/5] FLOPs 分析 (子进程, {export_name})")

    # 写入临时脚本避免 raw string 中 ''' 冲突
    flops_script = os.path.join(BASE, 'outputs', f'_flops_{export_name}.py')
    os.makedirs(os.path.dirname(flops_script), exist_ok=True)
    flops_data_mode = args.data_mode
    fh = open(flops_script, 'w')
    fh.write("import sys\n")
    fh.write(f"sys.path.insert(0, {BASE!r})\n")
    fh.write(f"sys.path.insert(0, {config_dir!r})\n")
    fh.write("import torch, torch.nn as nn, warnings; warnings.filterwarnings('ignore')\n")
    fh.write("from thop import profile, clever_format\n")
    fh.write(f"import {config_module_name} as cfg_mod\n")
    fh.write(f"LightningModel = cfg_mod.{lightning_model_name}\n")
    fh.write("device = 'cuda' if torch.cuda.is_available() else 'cpu'\n")
    fh.write(f"m = LightningModel(data_mode={flops_data_mode!r})\n")
    fh.write("model = m.model.to(device).eval().float()\n")
    fh.write("for buf in model.buffers():\n")
    fh.write("    if buf.is_floating_point() and buf.dtype != torch.float32:\n")
    fh.write("        buf.data = buf.data.float()\n")
    fh.write("""
arch = 'unknown'
cls_name = type(model).__name__
if cls_name == 'CameraRadarNetDet':
    arch = 'crn'
elif cls_name == 'BaseBEVDepth':
    arch = 'bevdepth'
elif cls_name == 'FastBEV':
    arch = 'fastbev'
elif hasattr(model, 'backbone_pts') and hasattr(model, 'fuser'):
    arch = 'crn'
elif hasattr(model, 'backbone_img'):
    arch = 'bevdepth'

B, S, C = 1, 1, 6
imgs = torch.randn(B, S, C, 3, 256, 704, device=device)
pts = torch.randn(B, S, C, 1000, 5, device=device)
intrin = torch.randn(B, S, C, 4, 4, device=device)
ida = torch.randn(B, S, C, 4, 4, device=device)
s2e = torch.randn(B, S, C, 4, 4, device=device)
bda = torch.randn(B, 4, 4, device=device)

class W(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, imgs, intrin, ida, s2e, bda, pts):
        mats = {'intrin_mats':intrin,'ida_mats':ida,'sensor2ego_mats':s2e,'bda_mat':bda}
        if arch == 'crn':
            return self.m(imgs, mats, sweep_ptss=pts, is_train=False)
        else:
            return self.m(imgs, mats, is_train=False)

flops, params = profile(W(model), inputs=(imgs,intrin,ida,s2e,bda,pts), verbose=False)
f, _ = clever_format([flops, params], "%.3f")
print(f"FLOPs:{f},Params:{sum(p.numel() for p in model.parameters())}")
""")
    fh.close()

    try:
        result = subprocess.run([sys.executable, flops_script],
                               capture_output=True, text=True, timeout=300, cwd=BASE)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.startswith('FLOPs:'):
                    parts = line.split(',')
                    print(f"  总 FLOPs:   {parts[0].split(':')[1]}")
                    print(f"  总参数量:   {parts[1].split(':')[1]}")
        else:
            print(f"  ⚠️  thop FLOPs 分析失败 (常见于自定义算子)")
            print(f"  错误: {result.stderr.strip()[-300:]}")
    finally:
        if os.path.exists(flops_script):
            os.remove(flops_script)
else:
    print("[2/5] 跳过")

# ONNX 文件路径 (Step 3/4 共用)
onnx_dir = os.path.join(BASE, 'outputs')
onnx_path = os.path.join(onnx_dir, f'{export_name}.onnx')

# =====================================================================
# Step 3: ONNX 导出
# =====================================================================
if 3 not in skip_steps and not args.no_export:
    print_sep(f"[3/5] ONNX 导出 ({export_name})")

    # ─── 3a. 定义纯 PyTorch BEV Pooling ─────────────────────
    def average_voxel_pooling_pytorch(geom_xyz, input_features, input_pos, voxel_num):
        """纯 PyTorch scatter_add 实现，替代自定义 CUDA average_voxel_pooling"""
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(geom_xyz.shape[0], -1, input_features.shape[-1])
        input_pos = input_pos.reshape(geom_xyz.shape[0], -1, input_pos.shape[-1])
        B, N, C = input_features.shape
        H, W = int(voxel_num[1]), int(voxel_num[0])

        valid = (input_pos[..., 0] > 0).float()
        in_bound = ((geom_xyz[..., 0] >= 0) & (geom_xyz[..., 0] < W) &
                    (geom_xyz[..., 1] >= 0) & (geom_xyz[..., 1] < H)).float()
        mask = valid * in_bound

        flat_idx = (geom_xyz[..., 1] * W + geom_xyz[..., 0]).long()
        flat_idx = flat_idx * mask.long()

        offsets = torch.arange(B, device=geom_xyz.device) * H * W
        flat_idx = flat_idx + offsets.view(-1, 1)

        flat_idx_1d = flat_idx.reshape(-1)
        feat_flat = (input_features * mask.unsqueeze(-1)).reshape(-1, C)
        cnt_flat = mask.reshape(-1, 1)

        out = torch.zeros(B * H * W, C, device=input_features.device,
                          dtype=input_features.dtype)
        cnt = torch.zeros(B * H * W, 1, device=input_features.device,
                          dtype=input_features.dtype)

        out = out.scatter_add(0, flat_idx_1d.unsqueeze(1).expand(-1, C), feat_flat)
        cnt = cnt.scatter_add(0, flat_idx_1d.unsqueeze(1), cnt_flat)

        out = out / cnt.clamp(min=1)
        return out.view(B, H, W, C).permute(0, 3, 1, 2), cnt.view(B, H, W, 1).permute(0, 3, 1, 2)

    def voxel_pooling_pytorch(geom_xyz, input_features, voxel_num):
        """纯 PyTorch scatter_add 实现，替代 BEVDepth 的 CUDA voxel_pooling (无 input_pos)"""
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(geom_xyz.shape[0], -1, input_features.shape[-1])
        B, N, C = input_features.shape
        H, W = int(voxel_num[1]), int(voxel_num[0])

        in_bound = ((geom_xyz[..., 0] >= 0) & (geom_xyz[..., 0] < W) &
                    (geom_xyz[..., 1] >= 0) & (geom_xyz[..., 1] < H)).float()

        flat_idx = (geom_xyz[..., 1] * W + geom_xyz[..., 0]).long()
        flat_idx = flat_idx * in_bound.long()

        offsets = torch.arange(B, device=geom_xyz.device) * H * W
        flat_idx = flat_idx + offsets.view(-1, 1)

        flat_idx_1d = flat_idx.reshape(-1)
        feat_flat = (input_features * in_bound.unsqueeze(-1)).reshape(-1, C)
        cnt_flat = in_bound.reshape(-1, 1)

        out = torch.zeros(B * H * W, C, device=input_features.device,
                          dtype=input_features.dtype)
        cnt = torch.zeros(B * H * W, 1, device=input_features.device,
                          dtype=input_features.dtype)

        out = out.scatter_add(0, flat_idx_1d.unsqueeze(1).expand(-1, C), feat_flat)
        cnt = cnt.scatter_add(0, flat_idx_1d.unsqueeze(1), cnt_flat)

        out = out / cnt.clamp(min=1)
        return out.view(B, H, W, C).permute(0, 3, 1, 2)

    # ─── 3c. DCN → Conv2d ──────────────────────────────────
    def _replace_dcn_with_conv(module):
        """递归替换 DCN (DeformableConv2d) 为标准 Conv2d"""
        replacements = {}
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if 'DeformableConv2d' in cls_name:
                new_conv = nn.Conv2d(
                    child.in_channels, child.out_channels,
                    child.kernel_size, stride=child.stride if hasattr(child, 'stride') else 1,
                    padding=child.padding if hasattr(child, 'padding') else 0,
                    bias=child.bias is not None,
                )
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)
                replacements[name] = new_conv
            else:
                _replace_dcn_with_conv(child)
        for name, new_child in replacements.items():
            setattr(module, name, new_child)

    # ─── 3c. 4x4 矩阵求逆 (ONNX 兼容) ─────────────────────
    def _mat4x4_inverse(mat):
        """纯代数 4x4 矩阵求逆 (ONNX opset 9+ 兼容)"""
        a11 = mat[..., 0, 0]; a12 = mat[..., 0, 1]; a13 = mat[..., 0, 2]; a14 = mat[..., 0, 3]
        a21 = mat[..., 1, 0]; a22 = mat[..., 1, 1]; a23 = mat[..., 1, 2]; a24 = mat[..., 1, 3]
        a31 = mat[..., 2, 0]; a32 = mat[..., 2, 1]; a33 = mat[..., 2, 2]; a34 = mat[..., 2, 3]
        a41 = mat[..., 3, 0]; a42 = mat[..., 3, 1]; a43 = mat[..., 3, 2]; a44 = mat[..., 3, 3]
        def d3(b11,b12,b13,b21,b22,b23,b31,b32,b33):
            return b11*(b22*b33-b23*b32) - b12*(b21*b33-b23*b31) + b13*(b21*b32-b22*b31)
        c11 =  d3(a22,a23,a24, a32,a33,a34, a42,a43,a44)
        c12 = -d3(a21,a23,a24, a31,a33,a34, a41,a43,a44)
        c13 =  d3(a21,a22,a24, a31,a32,a34, a41,a42,a44)
        c14 = -d3(a21,a22,a23, a31,a32,a33, a41,a42,a43)
        c21 = -d3(a12,a13,a14, a32,a33,a34, a42,a43,a44)
        c22 =  d3(a11,a13,a14, a31,a33,a34, a41,a43,a44)
        c23 = -d3(a11,a12,a14, a31,a32,a34, a41,a42,a44)
        c24 =  d3(a11,a12,a13, a31,a32,a33, a41,a42,a43)
        c31 =  d3(a12,a13,a14, a22,a23,a24, a42,a43,a44)
        c32 = -d3(a11,a13,a14, a21,a23,a24, a41,a43,a44)
        c33 =  d3(a11,a12,a14, a21,a22,a24, a41,a42,a44)
        c34 = -d3(a11,a12,a13, a21,a22,a23, a41,a42,a43)
        c41 = -d3(a12,a13,a14, a22,a23,a24, a32,a33,a34)
        c42 =  d3(a11,a13,a14, a21,a23,a24, a31,a33,a34)
        c43 = -d3(a11,a12,a14, a21,a22,a24, a31,a32,a34)
        c44 =  d3(a11,a12,a13, a21,a22,a23, a31,a32,a33)
        det = a11*c11 + a12*c12 + a13*c13 + a14*c14
        inv = torch.stack([
            torch.stack([c11, c21, c31, c41], -1),
            torch.stack([c12, c22, c32, c42], -1),
            torch.stack([c13, c23, c33, c43], -1),
            torch.stack([c14, c24, c34, c44], -1),
        ], -1)
        return inv / det.unsqueeze(-1).unsqueeze(-1)

    # ─── 3d. 架构专用: 创建 Export Wrapper ─────────────────
    model_for_export = LightningModel(data_mode=args.data_mode)
    m = model_for_export.model.to(device).eval().float()
    # 确保所有 buffer (如 frustum) 也是 float32, 避免 Float/Half 不匹配
    for buf in m.buffers():
        if buf.is_floating_point() and buf.dtype != torch.float32:
            buf.data = buf.data.float()
    arch = detect_arch(m)
    print(f"  检测到架构: {arch}  (class: {type(m).__name__})")
    # 验证关键子模块存在性
    print(f"    有 backbone_img: {hasattr(m, 'backbone_img')}, "
          f"有 backbone_pts: {hasattr(m, 'backbone_pts')}, "
          f"有 fuser: {hasattr(m, 'fuser')}")

    # 通用 patch: torch.inverse → 纯代数版本
    print("  [patch] 替换 torch.inverse → ONNX 兼容版本 ...")

    def _patch_get_geometry(backbone_img):
        """替换 get_geometry 相关方法，用代数求逆替代 torch.inverse"""
        def _get_geometry_collapsed_onnx(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat,
                                          z_min=-5., z_max=3.):
            batch_size, num_cams, _, _ = sensor2ego_mat.shape
            points = self.frustum
            ida_mat_v = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = _mat4x4_inverse(ida_mat_v).matmul(points.unsqueeze(-1)).float()
            points = torch.cat(
                (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                 points[:, :, :, :, :, 2:]), 5)
            combine = sensor2ego_mat.matmul(_mat4x4_inverse(intrin_mat)).float()
            points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points).float()
            if bda_mat is not None:
                bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                    batch_size, num_cams, 1, 1, 1, 4, 4)
                points = (bda_mat @ points).squeeze(-1)
            else:
                points = points.squeeze(-1)
            points_out = points[:, :, :, 0:1, :, :3]
            points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max)).float()
            return points_out, points_valid_z

        if hasattr(backbone_img, 'get_geometry_collapsed'):
            backbone_img.get_geometry_collapsed = \
                lambda *a, **kw: _get_geometry_collapsed_onnx(backbone_img, *a, **kw)

    if arch == 'crn':
        # ── CRN 专用: patch + wrapper ──────────────────────
        print("  [patch] 替换 average_voxel_pooling → scatter_add ...")
        import layers.backbones.rvt_lss_fpn as rvt_module
        rvt_module.average_voxel_pooling = average_voxel_pooling_pytorch
        import ops.average_voxel_pooling_v2 as avg_pool_mod
        avg_pool_mod.average_voxel_pooling = average_voxel_pooling_pytorch

        print("  [patch] 替换 DCN → Conv2d ...")
        _replace_dcn_with_conv(m)

        print("  [optimize] Conv+BN 融合 ...")
        fuse_all_conv_bn(m)

        print("  [patch] 替换 Fuser 注意力层 → 恒等版 ...")
        class PlaceholderAttn(nn.Module):
            def __init__(self, embed_dims=256, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
            def forward(self, queries, value_img=None, value_pts=None, **kwargs):
                return self.dropout(queries)

        if hasattr(m, 'fuser') and hasattr(m.fuser, 'attn_layers'):
            embed_dims = getattr(m.fuser, 'embed_dims', 256)
            for i in range(len(m.fuser.attn_layers)):
                m.fuser.attn_layers[i] = PlaceholderAttn(embed_dims=embed_dims).to(device).eval()

        # 禁用 timing
        for attr in ['idx', 'times']:
            if hasattr(m, attr): setattr(m, attr, None)
            if hasattr(m, 'backbone_img') and hasattr(m.backbone_img, attr):
                setattr(m.backbone_img, attr, None)
            if hasattr(m, 'fuser') and hasattr(m.fuser, attr):
                setattr(m.fuser, attr, None)
            if hasattr(m, 'head') and hasattr(m.head, attr):
                setattr(m.head, attr, None)

        _patch_get_geometry(m.backbone_img)

        class CRNExportWrapper(nn.Module):
            """CRN ONNX 导出专用 Wrapper (pts_context/pts_occupancy 在 Host 预处理)"""
            def __init__(self, original_model):
                super().__init__()
                self.backbone_img = original_model.backbone_img
                self.fuser = original_model.fuser
                self.head = original_model.head

            def forward(self, sweep_imgs, intrin_mats, ida_mats, sensor2ego_mats,
                        bda_mat, pts_context, pts_occupancy):
                mats_dict = {
                    'intrin_mats': intrin_mats, 'ida_mats': ida_mats,
                    'sensor2ego_mats': sensor2ego_mats, 'bda_mat': bda_mat,
                }
                feats, _ = self.backbone_img(sweep_imgs, mats_dict,
                                              ptss_context=pts_context,
                                              ptss_occupancy=pts_occupancy,
                                              times=None, return_depth=False)
                fused, _ = self.fuser(feats)
                preds, _ = self.head(fused)
                flat_preds = []
                for task_pred in preds:
                    if isinstance(task_pred, dict):
                        for v in task_pred.values():
                            flat_preds.append(v)
                    else:
                        for v in task_pred:
                            flat_preds.append(v)
                return tuple(flat_preds)

        export_wrapper = CRNExportWrapper(m).to(device).eval()

        # ── CRN 输入数据 ──────────────────────────────────
        print("  [data] 构建 4 帧模拟输入 (CRN) ...")
        B, S, C_exp = 1, 4, 6
        sweep_imgs = torch.randn(B, S, C_exp, 3, 256, 704, device=device)
        pts_context = torch.randn(B * C_exp, S, 80, 70, 44, device=device)
        pts_occupancy = torch.randn(B * C_exp, S, 1, 70, 44, device=device).sigmoid()
        intrin = torch.randn(B, S, C_exp, 4, 4, device=device)
        ida = torch.randn(B, S, C_exp, 4, 4, device=device)
        s2e = torch.randn(B, S, C_exp, 4, 4, device=device)
        bda = torch.randn(B, 4, 4, device=device)
        dummy_inputs = (sweep_imgs, intrin, ida, s2e, bda, pts_context, pts_occupancy)

        input_names = [
            'sweep_imgs', 'intrin_mats', 'ida_mats', 'sensor2ego_mats',
            'bda_mat', 'pts_context', 'pts_occupancy',
        ]
        dynamic_axes = {
            'sweep_imgs': {0: 'batch'},
            'intrin_mats': {0: 'batch'},
            'ida_mats': {0: 'batch'},
            'sensor2ego_mats': {0: 'batch'},
            'bda_mat': {0: 'batch'},
            'pts_context': {0: 'batch_times_cams'},
            'pts_occupancy': {0: 'batch_times_cams'},
        }

    elif arch == 'bevdepth':
        # ── BEVDepth 专用: patch + wrapper ────────────────
        # BEVDepth (BaseLSSFPN) 使用 voxel_pooling 而非 average_voxel_pooling
        print("  [patch] 替换 voxel_pooling → scatter_add ...")
        import ops.voxel_pooling_v2 as vox_pool_mod
        # average_voxel_pooling_pytorch 与 voxel_pooling 签名一致, 可复用
        vox_pool_mod.voxel_pooling = average_voxel_pooling_pytorch

        _patch_get_geometry(m.backbone_img)

        # 禁用 timing
        for attr in ['idx', 'times']:
            if hasattr(m, attr): setattr(m, attr, None)
            if hasattr(m, 'backbone_img') and hasattr(m.backbone_img, attr):
                setattr(m.backbone_img, attr, None)
            if hasattr(m, 'head') and hasattr(m.head, attr):
                setattr(m.head, attr, None)

        class BEVDepthExportWrapper(nn.Module):
            """BEVDepth ONNX 导出专用 Wrapper (仅图像输入, 无雷达)"""
            def __init__(self, original_model):
                super().__init__()
                self.backbone_img = original_model.backbone_img
                self.head = original_model.head

            def forward(self, sweep_imgs, intrin_mats, ida_mats, sensor2ego_mats, bda_mat):
                mats_dict = {
                    'intrin_mats': intrin_mats, 'ida_mats': ida_mats,
                    'sensor2ego_mats': sensor2ego_mats, 'bda_mat': bda_mat,
                }
                feats, _ = self.backbone_img(sweep_imgs, mats_dict,
                                              times=None, is_return_depth=False)
                preds, _ = self.head(feats)
                flat_preds = []
                for task_pred in preds:
                    if isinstance(task_pred, dict):
                        for v in task_pred.values():
                            flat_preds.append(v)
                    else:
                        for v in task_pred:
                            flat_preds.append(v)
                return tuple(flat_preds)

        export_wrapper = BEVDepthExportWrapper(m).to(device).eval()

        # ── BEVDepth 输入数据 ─────────────────────────────
        print("  [data] 构建 4 帧模拟输入 (BEVDepth) ...")
        B, S = 1, 4
        # 从模型配置读取 num_cams
        num_cams = 6
        sweep_imgs = torch.randn(B, S, num_cams, 3, 256, 704, device=device)
        intrin = torch.randn(B, S, num_cams, 4, 4, device=device)
        ida = torch.randn(B, S, num_cams, 4, 4, device=device)
        s2e = torch.randn(B, S, num_cams, 4, 4, device=device)
        bda = torch.randn(B, 4, 4, device=device)
        dummy_inputs = (sweep_imgs, intrin, ida, s2e, bda)

        input_names = [
            'sweep_imgs', 'intrin_mats', 'ida_mats', 'sensor2ego_mats', 'bda_mat',
        ]
        dynamic_axes = {
            'sweep_imgs': {0: 'batch'},
            'intrin_mats': {0: 'batch'},
            'ida_mats': {0: 'batch'},
            'sensor2ego_mats': {0: 'batch'},
            'bda_mat': {0: 'batch'},
        }

    elif arch == 'fastbev':
        # ── FastBEV 专用: patch + wrapper ─────────────────
        print("  [patch] FastBEV: 替换 torch.inverse → ONNX 兼容版本 ...")
        _patch_get_geometry(m.backbone_img)

        class FastBEVExportWrapper(nn.Module):
            """FastBEV ONNX 导出专用 Wrapper"""
            def __init__(self, original_model):
                super().__init__()
                self.backbone_img = original_model.backbone_img
                self.head = original_model.head

            def forward(self, sweep_imgs, intrin_mats, ida_mats, sensor2ego_mats, bda_mat):
                mats_dict = {
                    'intrin_mats': intrin_mats, 'ida_mats': ida_mats,
                    'sensor2ego_mats': sensor2ego_mats, 'bda_mat': bda_mat,
                }
                feats, _ = self.backbone_img(sweep_imgs, mats_dict,
                                              times=None, is_return_depth=False)
                preds, _ = self.head(feats)
                flat_preds = []
                for task_pred in preds:
                    if isinstance(task_pred, dict):
                        for v in task_pred.values():
                            flat_preds.append(v)
                    else:
                        for v in task_pred:
                            flat_preds.append(v)
                return tuple(flat_preds)

        export_wrapper = FastBEVExportWrapper(m).to(device).eval()

        print("  [data] 构建模拟输入 (FastBEV) ...")
        B, S = 1, 1
        num_cams = 6
        sweep_imgs = torch.randn(B, S, num_cams, 3, 256, 704, device=device)
        intrin = torch.randn(B, S, num_cams, 4, 4, device=device)
        ida = torch.randn(B, S, num_cams, 4, 4, device=device)
        s2e = torch.randn(B, S, num_cams, 4, 4, device=device)
        bda = torch.randn(B, 4, 4, device=device)
        dummy_inputs = (sweep_imgs, intrin, ida, s2e, bda)

        input_names = [
            'sweep_imgs', 'intrin_mats', 'ida_mats', 'sensor2ego_mats', 'bda_mat',
        ]
        dynamic_axes = {
            'sweep_imgs': {0: 'batch'},
            'intrin_mats': {0: 'batch'},
            'ida_mats': {0: 'batch'},
            'sensor2ego_mats': {0: 'batch'},
            'bda_mat': {0: 'batch'},
        }

    else:
        raise NotImplementedError(f"不支持的架构: {arch}")

    # ─── 3e. ONNX 导出 ─────────────────────────────────────
    print("\n  [export] 导出 ONNX ...")
    os.makedirs(onnx_dir, exist_ok=True)

    # warmup forward
    with torch.no_grad():
        wrapper_preds = export_wrapper(*dummy_inputs)

    output_names = [f'pred_{i}' for i in range(len(wrapper_preds))]

    torch.onnx.export(
        export_wrapper,
        dummy_inputs,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  ✅ 导出成功: {onnx_path}")
    print(f"  文件大小: {size_mb:.1f} MB")

    # ─── 3f. ONNX Runtime 验证 ─────────────────────────────
    print("\n  [验证] ONNX Runtime 推理 ...")
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

        ort_inputs = {}
        for name, tensor in zip(input_names, dummy_inputs):
            ort_inputs[name] = tensor.cpu().numpy().astype(np.float32)

        for _ in range(3):
            ort_session.run(None, ort_inputs)

        ort_outs = ort_session.run(None, ort_inputs)

        print(f"  ✅ ONNX Runtime 推理成功! 输出 {len(ort_outs)} 个")
        print(f"     第1个输出 shape: {ort_outs[0].shape}")
        print(f"     值范围 [{ort_outs[0].min():.3f}, {ort_outs[0].max():.3f}]")
        print("  ✅ ONNX 模型验证通过!")
    except ImportError:
        print("  ⚠️  onnxruntime 未安装, 跳过验证")
    except Exception as e:
        print(f"  ⚠️  ONNX Runtime 验证跳过: {type(e).__name__}: {e}")

    # 清理
    del model_for_export, m, export_wrapper
    torch.cuda.empty_cache()

else:
    print("[3/5] 跳过")

# =====================================================================
# Step 4: ONNX 结构验证 (独立于 Step 3)
# =====================================================================
if 4 not in skip_steps:
    print_sep(f"[4/5] ONNX 结构验证 ({export_name})")
    print(f"  检查文件: {onnx_path}")
    if os.path.exists(onnx_path):
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ✅ 结构验证通过")
            print(f"  输入: {[i.name for i in onnx_model.graph.input]}")
            print(f"  输出: {[o.name for o in onnx_model.graph.output]}")
        except ImportError:
            print(f"  ⚠️  onnx 库未安装，跳过结构验证")
            print(f"  ✅ 文件已生成: {onnx_path}")
    else:
        print(f"  ❌ 文件不存在 ({onnx_path})")
else:
    print("[4/5] 跳过")

# =====================================================================
# Step 5: 推理速度测试 (仅当有 CUDA 时)
# =====================================================================
if 5 not in skip_steps and not args.no_profile and torch.cuda.is_available():
    print_sep(f"[5/5] 推理延时分析 (CUDA Profiler, FP32, {export_name})")

    speed_model = LightningModel(data_mode=args.data_mode)
    speed_model = speed_model.to(device).eval()
    arch_speed = detect_arch(speed_model.model)

    B, S, C = 1, 1, 6
    imgs = torch.randn(B, S, C, 3, 256, 704, device=device)
    pts = torch.randn(B, S, C, 1000, 5, device=device)
    mats = {'intrin_mats': torch.randn(B, S, C, 4, 4, device=device),
            'ida_mats': torch.randn(B, S, C, 4, 4, device=device),
            'sensor2ego_mats': torch.randn(B, S, C, 4, 4, device=device),
            'bda_mat': torch.randn(B, 4, 4, device=device)}

    # 总推理速度
    with torch.no_grad():
        for _ in range(10):
            if arch_speed == 'crn':
                _ = speed_model.model(imgs, mats, sweep_ptss=pts, is_train=False)
            else:
                _ = speed_model.model(imgs, mats, is_train=False)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            if arch_speed == 'crn':
                _ = speed_model.model(imgs, mats, sweep_ptss=pts, is_train=False)
            else:
                _ = speed_model.model(imgs, mats, is_train=False)
        torch.cuda.synchronize()
        t_total = (time.time() - start) / 50
    print(f"\n  总推理 (wall-clock): {t_total*1000:.1f} ms  ({1/t_total:.1f} FPS)  [FP32]")

    # CUDA Profiler
    with torch.no_grad():
        for _ in range(5):
            if arch_speed == 'crn':
                speed_model.model(imgs, mats, sweep_ptss=pts, is_train=False)
            else:
                speed_model.model(imgs, mats, is_train=False)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(5):
            if arch_speed == 'crn':
                speed_model.model(imgs, mats, sweep_ptss=pts, is_train=False)
            else:
                speed_model.model(imgs, mats, is_train=False)
        torch.cuda.synchronize()

    del speed_model

    op_times = defaultdict(float)
    op_counts = defaultdict(int)
    for e in prof.events():
        if e.self_cuda_time_total > 0:
            name = e.name.split('(')[0].split('::')[-1].split('.')[-1][:48]
            ms = e.self_cuda_time_total / 1000
            op_times[name] += ms
            op_counts[name] += e.count

    total_p = sum(op_times.values())
    print(f"\n{'算子':<50s} {'耗时(ms)':>10s} {'调用':>6s} {'占比':>7s}")
    print("-" * 77)
    categories = {
        '卷积 (Conv/GEMM)': ['cutlass_tensorop', 'sm80_xmma_fprop', 'implicit_convolve',
                            'precomputed_convolve', 'winograd'],
        '归一化 (BN/Activation)': ['bn_fw_inf', 'relu', 'nchwToNhwc', 'nhwcToNchw'],
        '注意力 (Deformable Attn)': ['ms_deformable'],
        '体素化 (Voxelization)': ['point_to_voxelidx', 'determin_voxel_num'],
        '其他算子': [],
    }
    cat_times = {}
    for cat, keywords in categories.items():
        t_cat = 0
        for name, t in list(op_times.items()):
            if any(k in name.lower() for k in keywords):
                t_cat += t
        cat_times[cat] = t_cat

    for name, op_time in sorted(op_times.items(), key=lambda x: -x[1]):
        if op_time < 0.15:
            continue
        print(f"  {name:<48s} {op_time:>8.2f} {op_counts[name]:>4d} {op_time/total_p*100:>6.1f}%")

    other_sum = sum(t for n, t in op_times.items() if t < 0.15)
    other_cnt = sum(op_counts[n] for n, t in op_times.items() if t < 0.15)
    if other_sum > 0:
        print(f"  {'<其他小算子>':<48s} {other_sum:>8.2f} {other_cnt:>4d} {other_sum/total_p*100:>6.1f}%")
    print("-" * 77)
    print(f"  {'CUDA Kernel 总计 (含重叠)':<48s} {total_p:>8.2f} {'':>4s}")

    print(f"\n  类别汇总:")
    for cat, t in cat_times.items():
        print(f"    {cat:<20s} {t:>8.2f} ms ({t/total_p*100:.1f}%)")
    print(f"    {'Kernel 自累加 (含重叠)':<20s} {total_p:>8.2f} ms")
    print(f"  注: wall-clock 总推理 ({t_total*1000:.1f}ms) 低于 kernel 累加 ({total_p:.1f}ms)")
    print(f"      因为 CUDA kernel 异步执行且存在重叠, 累加时间 > 实际耗时")

    print(f"\n{'='*60}\n✅ 全部完成!")

elif 5 in skip_steps:
    print("[5/5] 跳过")
elif not torch.cuda.is_available():
    print("[5/5] 跳过 (无 CUDA)")
elif args.no_profile:
    print("[5/5] 跳过 (--no_profile)")
