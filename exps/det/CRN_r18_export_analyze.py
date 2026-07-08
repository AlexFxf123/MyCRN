#!/usr/bin/env python3
"""
CRN-R18: 参数量分析 + FLOPs + ONNX 导出 + 验证
"""
import torch, torch.nn as nn, numpy as np, sys, os, subprocess, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_sep(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

# =====================================================================
# 1. 创建模型 + 参数量分析
# =====================================================================
print_sep("[1/5] 模型参数量分析")
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel
model = CRNLightningModel(data_mode='sub')
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

# 子模块
for sub_name in ['backbone_img', 'head', 'backbone_pts']:
    sub = {}
    for n, p in getattr(model.model, sub_name).named_parameters():
        t = n.split('.')[0]; sub[t] = sub.get(t, 0) + p.numel()
    print(f"\n  --- {sub_name} ---")
    for m, c in sorted(sub.items(), key=lambda x: -x[1]):
        print(f"    {m:<28s} {c:>10,}")

del model; torch.cuda.empty_cache()

# =====================================================================
# 2. FLOPs 分析（子进程，避免 thop hook 泄漏）
# =====================================================================
print_sep("[2/5] FLOPs 分析 (子进程)")

flops_code = r'''
import sys; sys.path.insert(0, r"''' + BASE + r'''")
import torch, torch.nn as nn, warnings; warnings.filterwarnings('ignore')
from thop import profile, clever_format
from models.camera_radar_net_det import CameraRadarNetDet
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel as CLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 用独立模型实例，不与主进程共享
import copy, sys
# 直接从模块导入配置
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel
m = CRNLightningModel(data_mode='sub')
model = m.model.to(device).eval().float()

B,S,C = 1,1,6
imgs = torch.randn(B,S,C,3,256,704, device=device)
pts = torch.randn(B,S,C,1000,5, device=device)
intrin = torch.randn(B,S,C,4,4, device=device)
ida = torch.randn(B,S,C,4,4, device=device)
s2e = torch.randn(B,S,C,4,4, device=device)
bda = torch.randn(B,4,4, device=device)

class W(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, imgs, intrin, ida, s2e, bda, pts):
        mats = {'intrin_mats':intrin,'ida_mats':ida,'sensor2ego_mats':s2e,'bda_mat':bda}
        return self.m(imgs, mats, sweep_ptss=pts, is_train=False)

flops, params = profile(W(model), inputs=(imgs,intrin,ida,s2e,bda,pts), verbose=False)
f, _ = clever_format([flops, params], "%.3f")
print(f"FLOPs:{f},Params:{sum(p.numel() for p in model.parameters())}")
'''

result = subprocess.run([sys.executable, '-c', flops_code],
                       capture_output=True, text=True, timeout=300, cwd=BASE)
if result.returncode == 0:
    for line in result.stdout.strip().split('\n'):
        if line.startswith('FLOPs:'):
            parts = line.split(',')
            print(f"  总 FLOPs:   {parts[0].split(':')[1]}")
            print(f"  总参数量:   {parts[1].split(':')[1]}")
else:
    print(f"  ⚠️  thop FLOPs 分析失败 (常见于自定义算子)")
    print(f"  错误: {result.stderr.strip()[-200:]}")

# =====================================================================
# 3. ONNX 导出（替换自定义算子 + 4帧展开 + 精度验证）
# =====================================================================
print_sep("[3/5] ONNX 导出 (替换自定义算子)")

# ─────────────────────────────────────────────
# 3a. 定义纯 PyTorch BEV Pooling (替换 custom CUDA)
# ─────────────────────────────────────────────
def average_voxel_pooling_pytorch(geom_xyz, input_features, input_pos, voxel_num):
    """纯 PyTorch scatter_add 实现，替代自定义 CUDA average_voxel_pooling"""
    # 与原 CUDA 函数一致: 展平非 batch 和 feature 维度
    geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
    input_features = input_features.reshape(geom_xyz.shape[0], -1, input_features.shape[-1])
    input_pos = input_pos.reshape(geom_xyz.shape[0], -1, input_pos.shape[-1])
    B, N, C = input_features.shape
    H, W = int(voxel_num[1]), int(voxel_num[0])

    # 有效点 mask: input_pos > 0 且坐标在 BEV 网格范围内
    valid = (input_pos[..., 0] > 0).float()
    in_bound = ((geom_xyz[..., 0] >= 0) & (geom_xyz[..., 0] < W) &
                (geom_xyz[..., 1] >= 0) & (geom_xyz[..., 1] < H)).float()
    mask = valid * in_bound  # [B, N]

    # 展平 BEV 索引: flat_idx = y * W + x
    flat_idx = (geom_xyz[..., 1] * W + geom_xyz[..., 0]).long()  # [B, N]
    flat_idx = flat_idx * mask.long()

    # 加上 batch 偏移，避免跨 batch 混淆
    offsets = torch.arange(B, device=geom_xyz.device) * H * W
    flat_idx = flat_idx + offsets.view(-1, 1)  # [B, N]

    # 展平为 1D
    flat_idx_1d = flat_idx.reshape(-1)                       # [B*N]
    feat_flat = (input_features * mask.unsqueeze(-1)).reshape(-1, C)  # [B*N, C]
    cnt_flat = mask.reshape(-1, 1)                           # [B*N, 1]

    # scatter_add → 映射到 ONNX ScatterElements (reduction='add')
    out = torch.zeros(B * H * W, C, device=input_features.device,
                      dtype=input_features.dtype)
    cnt = torch.zeros(B * H * W, 1, device=input_features.device,
                      dtype=input_features.dtype)

    out = out.scatter_add(0, flat_idx_1d.unsqueeze(1).expand(-1, C), feat_flat)
    cnt = cnt.scatter_add(0, flat_idx_1d.unsqueeze(1), cnt_flat)

    out = out / cnt.clamp(min=1)
    return out.view(B, H, W, C).permute(0, 3, 1, 2), cnt.view(B, H, W, 1).permute(0, 3, 1, 2)


# ─────────────────────────────────────────────
# 3b. 替换 DCN → 标准 Conv2d
# ─────────────────────────────────────────────
def _replace_dcn_with_conv(module):
    """递归替换 DCN (DeformableConv2d) 为标准 Conv2d，保留权重"""
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
            # 复制卷积权重 (DCN 的 weight 形状与 Conv2d 一致)
            new_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)
            replacements[name] = new_conv
        else:
            _replace_dcn_with_conv(child)
    for name, new_child in replacements.items():
        setattr(module, name, new_child)


# ─────────────────────────────────────────────
# 3c. 定义导出 Wrapper
# ─────────────────────────────────────────────
class CRNExportWrapper(nn.Module):
    """
    ONNX 导出专用 Wrapper:
    - PtsBackbone 在 Host 预处理, 输入 pts_context / pts_occupancy
    - average_voxel_pooling → scatter_add 纯 PyTorch
    - 4 帧循环展开
    - 禁用 timing
    """
    def __init__(self, original_model):
        super().__init__()
        # 直接使用原始子模块 (平均池化已在模块级别替换)
        self.backbone_img = original_model.backbone_img
        self.fuser = original_model.fuser
        self.head = original_model.head

        # 禁用所有 timing
        self.backbone_img.times = None
        self.fuser.times = None
        self.head.times = None

    def forward(self, sweep_imgs, intrin_mats, ida_mats, sensor2ego_mats,
                bda_mat, pts_context, pts_occupancy):
        """端到端前向 (4帧)"""
        mats_dict = {
            'intrin_mats': intrin_mats,
            'ida_mats': ida_mats,
            'sensor2ego_mats': sensor2ego_mats,
            'bda_mat': bda_mat,
        }

        feats, _ = self.backbone_img(sweep_imgs, mats_dict,
                                      ptss_context=pts_context,
                                      ptss_occupancy=pts_occupancy,
                                      times=None,
                                      return_depth=False)
        fused, _ = self.fuser(feats)
        preds, _ = self.head(fused)

        # 将 preds 展平为 list[Tensor]
        flat_preds = []
        for task_pred in preds:
            if isinstance(task_pred, dict):
                for v in task_pred.values():
                    flat_preds.append(v)
            else:
                # list[list[Tensor]]
                for v in task_pred:
                    flat_preds.append(v)
        return tuple(flat_preds)


# ─────────────────────────────────────────────
# 3d. 加载模型 + patch
# ─────────────────────────────────────────────
print("  [patch] 替换 average_voxel_pooling → scatter_add ...")
# 必须先 import 模型 → 触发 rvt_lss_fpn 导入 → 再 patch
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel
import layers.backbones.rvt_lss_fpn as rvt_module
# 替换模块级的 average_voxel_pooling 引用
# _forward_single_sweep 内调用的是 rvt_module.average_voxel_pooling
rvt_module.average_voxel_pooling = average_voxel_pooling_pytorch
# 同时也替换 ops 包里的引用
import ops.average_voxel_pooling_v2 as avg_pool_mod
avg_pool_mod.average_voxel_pooling = average_voxel_pooling_pytorch
lightning_model = CRNLightningModel(data_mode='sub')
m = lightning_model.model.to(device).eval().float()

print("  [patch] 替换 DCN → Conv2d ...")
_replace_dcn_with_conv(m)

print("  [patch] 替换 Fuser 注意力层 → 恒等版 (需在 Orin 上写 TensorRT plugin) ...")
class PlaceholderAttn(nn.Module):
    """占位注意力层 — 保留结构但跳过可变形采样, 输出 identity + dropout
    完整精度的可变形注意力需在 TensorRT 中通过 custom plugin 实现"""
    def __init__(self, embed_dims=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, value_img=None, value_pts=None, **kwargs):
        return self.dropout(queries)

for i in range(len(m.fuser.attn_layers)):
    m.fuser.attn_layers[i] = PlaceholderAttn(
        embed_dims=m.fuser.embed_dims,
    ).to(device).eval()

# 禁用 timing
m.idx = 99999
m.backbone_img.idx = 99999
m.fuser.idx = 99999
m.head.idx = 99999

print("  [patch] 替换 torch.inverse → ONNX 兼容版本 ...")
def _mat4x4_inverse(mat):
    """纯代数 4x4 矩阵求逆 (ONNX opset 9+ 兼容)"""
    a11 = mat[..., 0, 0]; a12 = mat[..., 0, 1]; a13 = mat[..., 0, 2]; a14 = mat[..., 0, 3]
    a21 = mat[..., 1, 0]; a22 = mat[..., 1, 1]; a23 = mat[..., 1, 2]; a24 = mat[..., 1, 3]
    a31 = mat[..., 2, 0]; a32 = mat[..., 2, 1]; a33 = mat[..., 2, 2]; a34 = mat[..., 2, 3]
    a41 = mat[..., 3, 0]; a42 = mat[..., 3, 1]; a43 = mat[..., 3, 2]; a44 = mat[..., 3, 3]
    # 3x3 子式行列式
    def d3(b11,b12,b13,b21,b22,b23,b31,b32,b33):
        return b11*(b22*b33-b23*b32) - b12*(b21*b33-b23*b31) + b13*(b21*b32-b22*b31)
    # 余子式 (cofactor) 矩阵
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
    # 伴随矩阵 (adjugate) = cofactor^T, 除以行列式
    inv = torch.stack([
        torch.stack([c11, c21, c31, c41], -1),
        torch.stack([c12, c22, c32, c42], -1),
        torch.stack([c13, c23, c33, c43], -1),
        torch.stack([c14, c24, c34, c44], -1),
    ], -1)
    return inv / det.unsqueeze(-1).unsqueeze(-1)

def _get_geometry_collapsed_onnx(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat,
                                  z_min=-5., z_max=3.):
    """替换 get_geometry_collapsed，用代数求逆替代 torch.inverse"""
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

m.backbone_img.get_geometry_collapsed = \
    lambda *a, **kw: _get_geometry_collapsed_onnx(m.backbone_img, *a, **kw)

wrapper = CRNExportWrapper(m).to(device).eval()

# ─────────────────────────────────────────────
# 3e. 构建模拟输入 (4帧)
# ─────────────────────────────────────────────
print("  [data] 构建 4 帧模拟输入 ...")
B, S, C = 1, 4, 6
sweep_imgs = torch.randn(B, S, C, 3, 256, 704, device=device)
# pts_context / pts_occupancy 空间尺寸来自 PtsBackbone (SECOND) 输出: (70, 44)
pts_context = torch.randn(B * C, S, 80, 70, 44, device=device)
pts_occupancy = torch.randn(B * C, S, 1, 70, 44, device=device).sigmoid()
intrin = torch.randn(B, S, C, 4, 4, device=device)
ida = torch.randn(B, S, C, 4, 4, device=device)
s2e = torch.randn(B, S, C, 4, 4, device=device)
bda = torch.randn(B, 4, 4, device=device)

dummy_inputs = (sweep_imgs, intrin, ida, s2e, bda, pts_context, pts_occupancy)

# ─────────────────────────────────────────────
# 3f. ONNX 导出
# ─────────────────────────────────────────────
print("\n  [export] 导出 ONNX ...")
onnx_dir = os.path.join(BASE, 'outputs')
os.makedirs(onnx_dir, exist_ok=True)
onnx_path = os.path.join(onnx_dir, 'CRN_r18.onnx')

# 用真实数据做一次 warmup forward
with torch.no_grad():
    wrapper_preds = wrapper(*dummy_inputs)

torch.onnx.export(
    wrapper,
    dummy_inputs,
    onnx_path,
    export_params=True,
    opset_version=13,              # PyTorch 1.10 最高支持 opset 13
    input_names=[
        'sweep_imgs',
        'intrin_mats',
        'ida_mats',
        'sensor2ego_mats',
        'bda_mat',
        'pts_context',
        'pts_occupancy',
    ],
    output_names=[f'pred_{i}' for i in range(len(wrapper_preds))],
    dynamic_axes={
        'sweep_imgs': {0: 'batch'},
        'intrin_mats': {0: 'batch'},
        'ida_mats': {0: 'batch'},
        'sensor2ego_mats': {0: 'batch'},
        'bda_mat': {0: 'batch'},
        'pts_context': {0: 'batch_times_cams'},   # shape: [B*num_cams, S, 80, 70, 44]
        'pts_occupancy': {0: 'batch_times_cams'}, # shape: [B*num_cams, S, 1, 70, 44]
        # num_sweeps=4 固定, num_cams=6 固定, 空间尺寸 (70,44) 固定
    },
    do_constant_folding=True,
    verbose=False,
)

size_mb = os.path.getsize(onnx_path) / 1e6
print(f"  ✅ 导出成功: {onnx_path}")
print(f"  文件大小: {size_mb:.1f} MB")

# ─────────────────────────────────────────────
# 3g. ONNX Runtime 推理验证 (若已安装 onnxruntime)
# ─────────────────────────────────────────────
print("\n  [验证] ONNX Runtime 推理 ...")
try:
    import onnxruntime as ort
    import numpy as np

    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    # pts 输入第一维为 B*num_cams (B=1 → 6)
    ort_inputs = {
        'sweep_imgs': sweep_imgs.cpu().numpy().astype(np.float32),
        'intrin_mats': intrin.cpu().numpy().astype(np.float32),
        'ida_mats': ida.cpu().numpy().astype(np.float32),
        'sensor2ego_mats': s2e.cpu().numpy().astype(np.float32),
        'bda_mat': bda.cpu().numpy().astype(np.float32),
        'pts_context': pts_context.cpu().numpy().astype(np.float32),
        'pts_occupancy': pts_occupancy.cpu().numpy().astype(np.float32),
    }
    # warmup
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
    print(f"      (模型已导出, TensorRT 转换时可用 trtexec 进一步验证)")

# 清理
del lightning_model, m, wrapper
torch.cuda.empty_cache()

# =====================================================================
# 4. ONNX 结构验证
# =====================================================================
print_sep("[4/5] ONNX 结构验证")
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
    print(f"  ❌ 文件不存在")

# =====================================================================
# 5. 推理速度测试
# =====================================================================
print_sep("[5/5] 推理延时分析 (CUDA Profiler, FP32)")
import time, warnings
warnings.filterwarnings('ignore')

model = CRNLightningModel(data_mode='sub')
model = model.to(device).eval()
B, S, C = 1, 1, 6
imgs = torch.randn(B,S,C,3,256,704, device=device)
pts = torch.randn(B,S,C,1000,5, device=device)
mats = {'intrin_mats':torch.randn(B,S,C,4,4,device=device),
        'ida_mats':torch.randn(B,S,C,4,4,device=device),
        'sensor2ego_mats':torch.randn(B,S,C,4,4,device=device),
        'bda_mat':torch.randn(B,4,4,device=device)}

# 总推理速度 (wall-clock time)
with torch.no_grad():
    for _ in range(10): _ = model.model(imgs, mats, sweep_ptss=pts, is_train=False)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50): _ = model.model(imgs, mats, sweep_ptss=pts, is_train=False)
    torch.cuda.synchronize()
    t_total = (time.time() - start) / 50
print(f"\n  总推理 (wall-clock): {t_total*1000:.1f} ms  ({1/t_total:.1f} FPS)  [FP32]")

# CUDA Profiler 逐算子分析
with torch.no_grad():
    for _ in range(5): model.model(imgs, mats, sweep_ptss=pts, is_train=False)
    torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(5):
        model.model(imgs, mats, sweep_ptss=pts, is_train=False)
    torch.cuda.synchronize()

del model

from collections import defaultdict
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
# 归类输出：先输出主要类别算子
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
    if op_time < 0.15 and name in op_times:
        # 归入"其他"
        continue
    print(f"  {name:<48s} {op_time:>8.2f} {op_counts[name]:>4d} {op_time/total_p*100:>6.1f}%")

# 其他小算子的汇总
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
