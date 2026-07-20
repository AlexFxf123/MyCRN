# MyCRN —— 个人修改版

基于 [CRN: Camera Radar Net](https://github.com/youngskkim/CRN) (ICCV 2023)，适配自定义数据集管理、多模式训练评估。

## 环境配置

### 依赖版本（CRN conda 环境）

| 包 | 版本 |
|---|---|
| Python | 3.8 |
| PyTorch | **1.10.0+cu113** |
| CUDA | **11.3** |
| torchvision | 0.11.0+cu113 |
| pytorch-lightning | 1.6.0 |
| nuscenes-devkit | 1.1.9 |
| mmcv-full | 1.6.2 |
| mmdet | 2.25.2 |
| mmdet3d | 1.0.0rc4 |

### 安装步骤

```bash
# 创建环境
conda env create -f CRN.yaml
conda activate CRN

# PyTorch（CUDA 11.3）
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Lightning
pip install pytorch-lightning==1.6.0

# mm 系列
pip install mmcv-full==1.6.2
pip install mmsegmentation==0.28.0
pip install mmdet==2.25.2

# mmdetection3d
cd mmdetection3d
pip install -v -e .
cd ..

# 本项目
python setup.py develop
```

## 数据准备

### 目录结构

```
/home/fxf/data/nuScenes/     → nuScenes 原始数据
MyCRN/data/
├── info/                    ← gen_info.py 生成
│   ├── nuscenes_infos_train.pkl        (700 scenes, ~28130 samples)
│   ├── nuscenes_infos_val.pkl          (150 scenes, ~6019 samples)
│   ├── nuscenes_infos_sub_train.pkl    (233 scenes, ~9364 samples)
│   ├── nuscenes_infos_sub_val.pkl      (50 scenes,  ~2006 samples)
│   ├── nuscenes_infos_mini_train.pkl   (8 scenes,   ~323 samples)
│   └── nuscenes_infos_mini_val.pkl     (2 scenes,   ~81 samples)
├── depth_gt/                ← gen_depth_gt.py 生成
└── radar_pv_filter/         ← gen_radar_pv.py 生成
```

### 生成 pkl

```bash
conda run -n CRN python scripts/gen_info.py
```

## 训练

### 命令格式

```bash
conda run -n CRN python exps/det/CRN_r18_256x704_128x128_4key.py --train [参数]
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-b`, `--batch-size` | `1` | 每张卡的 batch size |
| `--max-epochs` | `24` | 训练轮数 |
| `--gpus` | `1` | GPU 数量 |
| `--data_mode` | `sub` | 数据集：`sub`（均衡子集）、`full`（全集）、`mini`（mini调试） |
| `--eval_interval` | `0` | 每 N 个 epoch 评估一次 mAP/NDS，0表示不做评估 |
| `--resume` | - | 从最新 checkpoint 恢复 |

### 示例

```bash
# mini 快速调试 (~323 train / ~81 val)
python exps/det/CRN_r18_256x704_128x128_4key.py --train --data_mode mini --max-epochs 5 --eval_interval 1

# 均衡子集训练 (~9364 train / ~2006 val，推荐)
python exps/det/CRN_r18_256x704_128x128_4key.py --train --data_mode sub --max-epochs 24 --eval_interval 3

# 全集训练 (~28130 train / ~6019 val)
python exps/det/CRN_r18_256x704_128x128_4key.py --train --data_mode full --max-epochs 24 --eval_interval 5

# 恢复训练
python exps/det/CRN_r18_256x704_128x128_4key.py --train --data_mode sub --resume
```

## 评估

### 快速评估（不重新推理）

```bash
python tools/eval_nusc.py --data_mode sub
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_mode` | `full` | `mini` / `sub` / `full` |
| `--result_path` | `./outputs/r18/results_nusc.json` | 预测结果文件 |
| `--output_dir` | `./outputs/r18` | 输出目录 |

### 完整评估（重新推理 + 评估）

```bash
python exps/det/CRN_r18_256x704_128x128_4key.py -e \
    --ckpt_path outputs/r18/lightning_logs/version_X/checkpoints/epoch=N-step=XXXX.ckpt \
    --data_mode sub -b 1 --gpus 1
```

### 训练中自动评估

```bash
# 每 1 个 epoch 评估一次
python exps/det/CRN_r18_256x704_128x128_4key.py --train --data_mode sub --eval_interval 1
```


### 查看训练log
tensorboard --logdir ./outputs/r18/lightning_logs/ --bind_all

## 输出目录

```
outputs/
├── r18/                ← CRN-R18
│   ├── lightning_logs/version_X/
│   │   ├── checkpoints/
│   │   └── ...
│   ├── results_nusc.json
│   └── metrics_summary.json
├── r50/                ← CRN-R50
└── bevdepth/           ← BEVDepth
```
