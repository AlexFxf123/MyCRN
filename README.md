# NMyCRN，根据个人使用修改
在nuScenes数据集上运行CRN模型，调整子集的train和val
read pkl是从生成的pkl文件中读取信息，并判断当前数据中是否存在相应sample的文件
read nuscenes是从nuscenes数据集中读取信息，判断当前数据是否存在相应的sample文件，这里用了子集01，共85个scene，因此只循环100次，最终得到子集包含的scene信息
splits是nuscenes自带的划分train和val的文件，这里根据子集的场景做新的划分

# CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception

https://github.com/youngskkim/CRN/assets/24770858/1bf85a3a-ad22-4875-ab0c-deeee347b03f

> [**CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception**](https://arxiv.org/abs/2304.00670)  
> [Youngseok Kim](https://youngskkim.github.io/),
> Juyeb Shin, Sanmin Kim, In-Jae Lee, 
> [Jun Won Choi](https://www.spa.hanyang.ac.kr/),
> [Dongsuk Kum](http://vdclab.kaist.ac.kr/)  
> [*ICCV 2023*](https://iccv2023.thecvf.com/)


## Abstract
In this paper, we propose Camera Radar Net (CRN), a novel camera-radar fusion framework that generates a semantically rich and spatially accurate bird's-eye-view (BEV) feature map for various tasks.
To overcome the lack of spatial information in an image, we transform perspective view image features to BEV with the help of sparse but accurate radar points.
We further aggregate image and radar feature maps in BEV using multi-modal deformable attention designed to tackle the spatial misalignment between inputs.
CRN with real-time setting operates at 20 FPS while achieving comparable performance to LiDAR detectors on nuScenes, and even outperforms at a far distance on 100m setting.
Moreover, CRN with offline setting yields 62.4% NDS, 57.5% mAP on nuScenes test set and ranks first among all camera and camera-radar 3D object detectors.


## Getting Started

### Installation
```shell
# clone repo
git clone https://github.com/youngskkim/CRN.git

cd CRN

# setup conda environment
conda env create --file CRN.yaml
conda activate CRN

# install dependencies
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
mim install mmcv==1.6.0
mim install mmsegmentation==0.28.0
mim install mmdet==2.25.2

cd mmdetection3d
pip install -v -e .
cd ..

python setup.py develop  # GPU required
```

### Data preparation
**Step 0.** Download [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).

**Step 1.** 已创建符号链接 `/home/fxf/data/nuScenes` → `/home/fxf/data/nuScenes-full`

**Step 2.** Create annotation file. 
This will generate `nuscenes_infos_{train,val}.pkl`.
```
python scripts/gen_info.py
```

**Step 3.** Generate ground truth depth.  
*Note: this process requires LiDAR keyframes.*
```
python scripts/gen_depth_gt.py
```

**Step 4.** Generate radar point cloud in perspective view. 
You can download pre-generated radar point cloud [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/youngseok_kim_kaist_ac_kr/EcEoswDVWu9GpGV5NSwGme4BvIjOm-sGusZdCQRyMdVUtw?e=OpZoQ4).  
*Note: this process requires radar blobs (in addition to keyframe) to utilize sweeps.*  
```
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_pv.py  # transform to camera coords
```

The folder structure will be as follows:
```
/home/fxf/data/nuScenes/  →  symlink to nuScenes-full (仅原始数据)
├── maps
├── samples
├── v1.0-trainval

MyCRN/data/  (生成文件)
├── info/                        ← 由 gen_info.py 生成
│   ├── nuscenes_infos_train.pkl
│   └── nuscenes_infos_val.pkl
├── depth_gt/                    ← 由 gen_depth_gt.py 生成
└── radar_pv_filter/             ← 由 gen_radar_pv.py 生成
```

### Training and Evaluation

输出目录结构：
```
outputs/
├── r18/          ← CRN_r18 训练输出
├── r50/          ← CRN_r50 训练输出
└── bevdepth/     ← BEVDepth_r50 训练输出
```

运行模式说明：
| 参数 | 模式 | 说明 |
|------|------|------|
| `--train` (或无参数) | **训练** | 默认模式，从零开始训练 |
| `--resume` | **恢复训练** | 从 `outputs/` 中最新的 checkpoint 恢复 |
| `-e` | **评估** | 需要 `--ckpt_path` 或 output dir 中有已训练的 checkpoint |
| `-p` | **预测** | 同评估，输出预测结果 |

**常用参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-b`, `--batch-size` | `1` | 每张卡的 batch size |
| `--max-epochs` | `24` | 训练轮数 |
| `--gpus` | `1` | GPU 数量 |
| `--resume` | - | 从最新 checkpoint 恢复训练 |
| `--train` | - | 训练模式（默认） |
| `-e` | - | 评估模式 |
| `--ckpt_path` | - | 指定 checkpoint 路径 |

**Training (from scratch)**
```
# CRN-R18 (batch_size=1, 24 epochs, 1 GPU)
python ./exps/det/CRN_r18_256x704_128x128_4key.py -b 1 --gpus 1

# CRN-R18 (自定义 batch size 和 epoch 数)
python ./exps/det/CRN_r18_256x704_128x128_4key.py -b 4 --gpus 4 --max-epochs 50

# CRN-R50
python ./exps/det/CRN_r50_256x704_128x128_4key.py -b 4 --gpus 4

# BEVDepth-R50
python ./exps/det/BEVDepth_r50_256x704_128x128_4key.py -b 4 --gpus 4
```

**Resume training (从最近的 checkpoint 恢复)**
```
python ./exps/det/CRN_r18_256x704_128x128_4key.py --resume -b 1 --gpus 1
```

**Evaluation**  
*Note: use `-b 1 --gpus 1` to measure inference time.*
```
# 指定 checkpoint 评估
python ./exps/det/CRN_r18_256x704_128x128_4key.py -e --ckpt_path ./outputs/r18/epoch=23-step=100000.ckpt -b 4 --gpus 4

# 自动使用 output dir 中最新的 checkpoint
python ./exps/det/CRN_r18_256x704_128x128_4key.py -e -b 4 --gpus 4
```

## Model Zoo
All models use 4 keyframes and are trained without CBGS.  
All latency numbers are measured with batch size 1, GPU warm-up, and FP16 precision.

|  Method  | Backbone | NDS  | mAP  | FPS  | Params | Config                                                  | Checkpoint                                                                                                  |
|:--------:|:--------:|:----:|:----:|:----:|:------:|:-------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
| BEVDepth |   R50    | 47.1 | 36.7 | 29.7 | 77.6 M | [config](exps/det/BEVDepth_r50_256x704_128x128_4key.py) | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/BEVDepth_r50_256x704_128x128_4key.pth) |
|   CRN    |   R18    | 54.2 | 44.9 | 29.4 | 37.2 M | [config](exps/det/CRN_r18_256x704_128x128_4key.py)      | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/CRN_r18_256x704_128x128_4key.pth)      |
|   CRN    |   R50    | 56.2 | 47.3 | 22.7 | 61.4 M | [config](exps/det/CRN_r50_256x704_128x128_4key.py)      | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/CRN_r50_256x704_128x128_4key.pth)      |


## Features
- [ ] BEV segmentation checkpoints 
- [ ] BEV segmentation code 
- [x] 3D detection checkpoints 
- [x] 3D detection code 
- [x] Code release 


## Acknowledgement
This project is based on excellent open source projects:
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)


## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{kim2023crn,
    title={Crn: Camera radar net for accurate, robust, efficient 3d perception},
    author={Kim, Youngseok and Shin, Juyeb and Kim, Sanmin and Lee, In-Jae and Choi, Jun Won and Kum, Dongsuk},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={17615--17626},
    year={2023}
}
```
