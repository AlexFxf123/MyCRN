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
# 3. ONNX 导出（独立子进程）
# =====================================================================
print_sep("[3/5] ONNX 导出 (子进程)")

onnx_path = os.path.join(BASE, 'CRN_r18.onnx')
onnx_code = r'''
import sys; sys.path.insert(0, r"''' + BASE + r'''")
import torch, torch.nn as nn, os, warnings; warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from models.camera_radar_net_det import CameraRadarNetDet
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel

model = CRNLightningModel(data_mode='sub')
m = model.model.to(device).eval().float()

B,S,C = 1,1,6
imgs = torch.randn(B,S,C,3,256,704, device=device)
pts = torch.randn(B,S,C,1000,5, device=device)
intrin = torch.randn(B,S,C,4,4, device=device)
ida = torch.randn(B,S,C,4,4, device=device)
s2e = torch.randn(B,S,C,4,4, device=device)
bda = torch.randn(B,4,4, device=device)

class ONNXWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, imgs, intrin, ida, s2e, bda, pts):
        mats = {'intrin_mats':intrin,'ida_mats':ida,'sensor2ego_mats':s2e,'bda_mat':bda}
        return self.m(imgs, mats, sweep_ptss=pts, is_train=False)

wrapper = ONNXWrapper(m).to(device).eval()
dummy = (imgs, intrin, ida, s2e, bda, pts)

with torch.no_grad():
    ref = wrapper(*dummy)

path = r"''' + onnx_path + r'''"
torch.onnx.export(wrapper, dummy, path,
    export_params=True, opset_version=9,
    input_names=['imgs','intrin_mats','ida_mats','sensor2ego_mats','bda_mat','pts_pv'],
    output_names=[f'out_{i}' for i in range(len(ref))],
    dynamic_axes={'imgs':{0:'B',1:'S',2:'C'},'pts_pv':{0:'B',1:'S',2:'C',3:'N'},
                  'intrin_mats':{0:'B'},'ida_mats':{0:'B'},
                  'sensor2ego_mats':{0:'B'},'bda_mat':{0:'B'}},
    do_constant_folding=True, verbose=False)
size = os.path.getsize(path) / 1e6
print(f"OK:{size:.1f}")
'''

result = subprocess.run([sys.executable, '-c', onnx_code],
                       capture_output=True, text=True, timeout=600, cwd=BASE)
if result.returncode == 0:
    for line in result.stdout.strip().split('\n'):
        if line.startswith('OK:'):
            print(f"  ✅ 导出成功: {onnx_path}")
            print(f"  文件大小: {line.split(':')[1]} MB")
else:
    print(f"  ❌ 导出失败")
    print(f"  错误: {result.stderr.strip()[-500:]}")

# =====================================================================
# 4. ONNX 结构验证
# =====================================================================
print_sep("[4/5] ONNX 结构验证")
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

# 总推理速度
with torch.no_grad():
    for _ in range(10): _ = model.model(imgs, mats, sweep_ptss=pts, is_train=False)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50): _ = model.model(imgs, mats, sweep_ptss=pts, is_train=False)
    torch.cuda.synchronize()
    t = (time.time() - start) / 50
print(f"\n  总推理: {t*1000:.1f} ms  ({1/t:.1f} FPS)  [FP32]")

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

for name, t in sorted(op_times.items(), key=lambda x: -x[1]):
    if t < 0.15 and name in op_times:
        # 归入"其他"
        continue
    print(f"  {name:<48s} {t:>8.2f} {op_counts[name]:>4d} {t/total_p*100:>6.1f}%")

# 其他小算子的汇总
other_sum = sum(t for n, t in op_times.items() if t < 0.15)
other_cnt = sum(op_counts[n] for n, t in op_times.items() if t < 0.15)
if other_sum > 0:
    print(f"  {'<其他小算子>':<48s} {other_sum:>8.2f} {other_cnt:>4d} {other_sum/total_p*100:>6.1f}%")
print("-" * 77)
print(f"  {'CUDA Kernel 总计':<48s} {total_p:>8.2f} {'':>4s}")

print(f"\n  类别汇总:")
for cat, t in cat_times.items():
    print(f"    {cat:<20s} {t:>8.2f} ms ({t/total_p*100:.1f}%)")
print(f"    {'Kernel 总计':<20s} {total_p:>8.2f} ms")
print(f"  单次推理: {t*1000:.1f} ms")

print(f"\n{'='*60}\n✅ 全部完成!")
