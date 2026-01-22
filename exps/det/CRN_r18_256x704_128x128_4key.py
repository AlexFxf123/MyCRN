"""
mAP: 0.4492
mATE: 0.5236
mASE: 0.2857
mAOE: 0.5640
mAVE: 0.2781
mAAE: 0.1792
NDS: 0.5415
Eval time: 185.7s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.702	0.312	0.172	0.146	0.306	0.197
truck	0.406	0.501	0.221	0.153	0.235	0.207
bus	0.506	0.542	0.210	0.130	0.404	0.178
trailer	0.227	0.880	0.252	0.600	0.205	0.100
construction_vehicle	0.133	0.819	0.518	1.251	0.111	0.352
pedestrian	0.450	0.558	0.291	0.683	0.368	0.174
motorcycle	0.478	0.413	0.257	0.820	0.425	0.213
bicycle	0.442	0.409	0.268	1.140	0.171	0.012
traffic_cone	0.544	0.414	0.378	nan	nan	nan
barrier	0.604	0.388	0.291	0.153	nan	nan

img: 10.84
  img_backbone: 3.62
  img_dep: 1.35
  img_transform: 5.01
  img_pool: 0.54
pts: 8.46
  pts_voxelize: 1.87
  pts_backbone: 5.27
  pts_head: 0.64
fusion: 6.77
  fusion_pre: 0.81
  fusion_layer: 5.31
  fusion_post: 0.07
head: 7.97
  head_backbone: 2.14
  head_head: 5.83
total: 34.04

FPS: 29.38

   | Name                                    | Type                      | Params
---------------------------------------------------------------------------------------
0  | model                                   | CameraRadarNetDet         | 37.2 M
1  | model.backbone_img                      | RVTLSSFPN                 | 17.0 M
2  | model.backbone_img.img_backbone         | ResNet                    | 11.2 M
3  | model.backbone_img.img_neck             | SECONDFPN                 | 246 K
4  | model.backbone_img.depth_net            | DepthNet                  | 4.8 M
5  | model.backbone_img.view_aggregation_net | ViewAggregation           | 807 K
6  | model.backbone_pts                      | PtsBackbone               | 3.1 M
7  | model.backbone_pts.pts_voxel_layer      | Voxelization              | 0
8  | model.backbone_pts.pts_voxel_encoder    | PillarFeatureNet          | 2.3 K
9  | model.backbone_pts.pts_middle_encoder   | PointPillarsScatter       | 0
10 | model.backbone_pts.pts_backbone         | SECOND                    | 2.7 M
11 | model.backbone_pts.pts_neck             | SECONDFPN                 | 90.5 K
12 | model.backbone_pts.pred_context         | Sequential                | 173 K
13 | model.backbone_pts.pred_occupancy       | Sequential                | 166 K
14 | model.fuser                             | MFAFuser                  | 1.2 M
15 | model.fuser.norm_img                    | LayerNorm                 | 160
16 | model.fuser.norm_pts                    | LayerNorm                 | 160
17 | model.fuser.input_proj                  | Linear                    | 20.6 K
18 | model.fuser.positional_encoding         | LearnedPositionalEncoding | 16.4 K
19 | model.fuser.ffn_layers                  | ModuleList                | 395 K
20 | model.fuser.norm_layers1                | ModuleList                | 1.5 K
21 | model.fuser.norm_layers2                | ModuleList                | 1.5 K
22 | model.fuser.attn_layers                 | ModuleList                | 198 K
23 | model.fuser.reduce_conv                 | Sequential                | 590 K
24 | model.head                              | BEVDepthHead              | 15.8 M
25 | model.head.loss_cls                     | GaussianFocalLoss         | 0
26 | model.head.loss_bbox                    | L1Loss                    | 0
27 | model.head.shared_conv                  | ConvModule                | 147 K
28 | model.head.task_heads                   | ModuleList                | 1.4 M
29 | model.head.trunk                        | ResNet                    | 11.9 M
30 | model.head.neck                         | SECONDFPN                 | 2.4 M
---------------------------------------------------------------------------------------
"""
import torch
from utils.torch_dist import synchronize

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel

from models.camera_radar_net_det import CameraRadarNetDet


class CRNLightningModel(BEVDepthLightningModel):    # 继承自BEVDepthLightningModel基类
    def __init__(self, *args, **kwargs) -> None:    # 初始化函数
        super().__init__(*args, **kwargs)           # 调用父类的初始化函数

        self.return_image = True                # 是否返回图像特征
        self.return_depth = True                # 是否返回深度信息
        self.return_radar_pv = True             # 是否返回雷达点云鸟瞰图特征
        ################################################
        self.optimizer_config = dict(           # 优化器配置  
            type='AdamW',                       # 制定优化器为AdamW优化器
            lr=2e-4,                            # 学习率设置为0.0002
            weight_decay=1e-4)                  # 权重衰减设置为0.0001
        ################################################
        self.ida_aug_conf = {                   # 图像数据增强配置
            'resize_lim': (0.386, 0.55),        # 随机缩放比例范围
            'final_dim': (256, 704),            # 输出图像的最终尺寸 (高, 宽)
            'rot_lim': (0., 0.),                # 随机旋转角度范围（这里不旋转）
            'H': 900,                           # 原始图像高度
            'W': 1600,                          # 原始图像宽度
            'rand_flip': True,                  # 是否随机水平翻转
            'bot_pct_lim': (0.0, 0.0),          # 裁剪底部百分比范围（这里不裁剪）
            'cams': [
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],                                  # 使用的相机名称
            'Ncams': 6,                         # 相机数量
        }
        self.bda_aug_conf = {                   # BEV数据增强配置
            'rot_ratio': 1.0,                   # 旋转增强的概率（1.0表示每次都旋转）
            'rot_lim': (-22.5, 22.5),           # 随机旋转角度范围（单位：度）
            'scale_lim': (0.9, 1.1),            # 随机缩放比例范围
            'flip_dx_ratio': 0.5,               # 沿x轴（前后）翻转的概率
            'flip_dy_ratio': 0.5                # 沿y轴（左右）翻转的概率
        }
        ################################################
        self.backbone_img_conf = {              # 图像骨干网络配置
            'x_bound': [-51.2, 51.2, 0.8],      # X轴范围和分辨率
            'y_bound': [-51.2, 51.2, 0.8],      # Y轴范围和分辨率
            'z_bound': [-5, 3, 8],              # Z轴范围和分辨率
            'd_bound': [2.0, 58.0, 0.8],        # 深度范围和分辨率
            'final_dim': (256, 704),            # BEV特征图的最终尺寸 (高, 宽)
            'downsample_factor': 16,            # 下采样因子
            'img_backbone_conf': dict(          # 图像骨干网络配置
                type='ResNet',                  # 使用ResNet作为骨干网络
                depth=18,                       # ResNet深度为18层
                frozen_stages=0,                # 冻结的阶段数
                out_indices=[0, 1, 2, 3],       # 输出的层索引
                norm_eval=False,                # 归一化层不进行评估模式
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),  # 预训练模型初始化
            ),
            'img_neck_conf': dict(              # 图像颈部网络配置
                type='SECONDFPN',               # 使用SECONDFPN作为颈部网络
                in_channels=[64, 128, 256, 512],    # 输入通道数
                upsample_strides=[0.25, 0.5, 1, 2], # 上采样步长
                out_channels=[64, 64, 64, 64],      # 输出通道数
            ),
            'depth_net_conf': dict(             # 深度网络配置  
                in_channels=256,                # 输入通道数
                mid_channels=256,               # 中间通道数
                ),
            'radar_view_transform': True,       # 是否进行雷达视图变换
            'camera_aware': False,              # 是否使用相机感知模块 
            'output_channels': 80,              # 输出通道数
        }
        ################################################
        self.backbone_pts_conf = {              # 点云骨干网络配置
            'pts_voxel_layer': dict(            # 点云体素化配置
                max_num_points=8,               # 每个体素的最大点数
                voxel_size=[8, 0.4, 2],         # 体素大小
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],    # 点云范围
                max_voxels=(768, 1024)          # 最大体素数量 (训练, 测试)
            ),
            'pts_voxel_encoder': dict(          # 点云体素编码器配置
                type='PillarFeatureNet',        # 使用PillarFeatureNet作为体素编码器
                in_channels=5,                  # 输入通道数（点的特征维度）
                feat_channels=[32, 64],         # 特征提取通道数
                with_distance=False,            # 是否使用点到雷达原点的距离作为特征
                with_cluster_center=False,      # 是否使用点到体素中心的偏移作为特征
                with_voxel_center=True,         # 是否使用点到体素几何中心的偏移作为特征
                voxel_size=[8, 0.4, 2],         # 体素大小
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],            # 点云范围
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),    # 归一化配置   
                legacy=True                                             # 兼容旧版本
            ),
            'pts_middle_encoder': dict(         # 点云中间编码器配置
                type='PointPillarsScatter',     # 使用PointPillarsScatter进行散射操作
                in_channels=64,                 # 输入通道数
                output_shape=(140, 88)          # 输出特征图的形状 (宽, 高)
            ),
            'pts_backbone': dict(               # 点云骨干网络配置
                type='SECOND',                  # 使用SECOND作为骨干网络
                in_channels=64,                 # 输入通道数
                out_channels=[64, 128, 256],    # 各层输出通道数
                layer_nums=[2, 3, 3],           # 每层卷积块数量
                layer_strides=[1, 2, 2],        # 每层下采样步长
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),              # 归一化配置
                conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect') # 卷积配置 
            ),
            'pts_neck': dict(                   # 点云颈部网络配置
                type='SECONDFPN',               # 使用SECONDFPN作为颈部网络
                in_channels=[64, 128, 256],     # 输入通道数
                out_channels=[64, 64, 64],      # 输出通道数
                upsample_strides=[0.5, 1, 2],   # 上采样步长
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),          # 归一化配置
                upsample_cfg=dict(type='deconv', bias=False),               # 上采样配置
                use_conv_for_no_stride=True                                 # 是否使用卷积进行无步长上采样
                ),
            'out_channels_pts': 80,             # 点云特征输出通道数
        }
        ################################################
        self.fuser_conf = {                     # 多模态特征融合配置    
            'img_dims': 80,                     # 图像特征维度
            'pts_dims': 80,                     # 点云特征维度
            'embed_dims': 128,                  # 融合后特征维度
            'num_layers': 6,                    # 融合层数
            'num_heads': 4,                     # 多头注意力机制的头数
            'bev_shape': (128, 128),            # BEV特征图形状 (宽, 高)
        }
        ################################################
        self.head_conf = {                      # 检测头配置
            'bev_backbone_conf': dict(          # BEV骨干网络配置
                type='ResNet',                  # 使用ResNet作为骨干网络
                in_channels=128,                # 输入通道数
                depth=18,                       # ResNet深度为18层
                num_stages=3,                   # 使用的阶段数
                strides=(1, 2, 2),              # 各阶段的步长
                dilations=(1, 1, 1),            # 各阶段的膨胀率
                out_indices=[0, 1, 2],          # 输出的层索引
                norm_eval=False,                # 归一化层不进行评估模式
                base_channels=128,              # 基础通道数
            ),
            'bev_neck_conf': dict(              # BEV颈部网络配置
                type='SECONDFPN',               # 使用SECONDFPN作为颈部网络
                in_channels=[128, 128, 256, 512],   # 输入通道数
                upsample_strides=[1, 2, 4, 8],      # 上采样步长
                out_channels=[64, 64, 64, 64]       # 输出通道数
            ),
            'tasks': [                                              # 任务配置（每个任务对应一类或多类目标检测）
                dict(num_class=1, class_names=['car']),                 
                dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                dict(num_class=2, class_names=['bus', 'trailer']),
                dict(num_class=1, class_names=['barrier']),
                dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
            ],
            'common_heads': dict(                                                   # 公共头配置
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),     # 各任务的回归头输出维度
            'bbox_coder': dict(                                                     # 边界框编码器配置
                type='CenterPointBBoxCoder',                                        # 使用CenterPointBBoxCoder进行边界框编码
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],          # 后处理中心范围
                max_num=500,                                                        # 最大检测目标数量  
                score_threshold=0.01,                                               # 分数阈值        
                out_size_factor=4,                                                  # 输出尺寸因子     
                voxel_size=[0.2, 0.2, 8],                                           # 体素大小
                pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],                         # 点云范围
                code_size=9,                                                        # 边界框编码大小
            ),
            'train_cfg': dict(
                point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                grid_size=[512, 512, 1],
                voxel_size=[0.2, 0.2, 8],
                out_size_factor=4,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            'test_cfg': dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.01,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=200,
                nms_thr=0.2,
            ),
            'in_channels': 256,  # Equal to bev_neck output_channels.
            'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),           # 分类损失配置
            'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),   # 回归损失配置
            'gaussian_overlap': 0.1,                                                # 高斯重叠阈值      
            'min_radius': 2,                                                        # 最小半径
        }
        ################################################
        self.key_idxes = [-2, -4, -6]
        self.model = CameraRadarNetDet(self.backbone_img_conf,
                                       self.backbone_pts_conf,
                                       self.fuser_conf,
                                       self.head_conf)

    def forward(self, sweep_imgs, mats, is_train=False, **inputs):
        return self.model(sweep_imgs, mats, sweep_ptss=inputs['pts_pv'], is_train=is_train)

    def training_step(self, batch):
        if self.global_rank == 0:
            for pg in self.trainer.optimizers[0].param_groups:
                self.log('learning_rate', pg["lr"])

        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            if self.return_radar_pv:
                pts_pv = pts_pv.cuda()
            gt_boxes_3d = [gt_box.cuda() for gt_box in gt_boxes_3d]
            gt_labels_3d = [gt_label.cuda() for gt_label in gt_labels_3d]
        preds, depth_preds = self(sweep_imgs, mats,
                                  pts_pv=pts_pv,
                                  is_train=True)
        targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
        loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...].contiguous()
        loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=3.)

        self.log('train/detection', loss_detection)
        self.log('train/heatmap', loss_heatmap)
        self.log('train/bbox', loss_bbox)
        self.log('train/depth', loss_depth)
        return loss_detection + loss_depth

    def validation_epoch_end(self, validation_step_outputs):
        detection_losses = list()
        heatmap_losses = list()
        bbox_losses = list()
        depth_losses = list()
        for validation_step_output in validation_step_outputs:
            detection_losses.append(validation_step_output[0])
            heatmap_losses.append(validation_step_output[1])
            bbox_losses.append(validation_step_output[2])
            depth_losses.append(validation_step_output[3])
        synchronize()

        self.log('val/detection', torch.mean(torch.stack(detection_losses)), on_epoch=True)
        self.log('val/heatmap', torch.mean(torch.stack(heatmap_losses)), on_epoch=True)
        self.log('val/bbox', torch.mean(torch.stack(bbox_losses)), on_epoch=True)
        self.log('val/depth', torch.mean(torch.stack(depth_losses)), on_epoch=True)

    def validation_step(self, batch, batch_idx):
        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            if self.return_radar_pv:
                pts_pv = pts_pv.cuda()
            gt_boxes_3d = [gt_box.cuda() for gt_box in gt_boxes_3d]
            gt_labels_3d = [gt_label.cuda() for gt_label in gt_labels_3d]
        with torch.no_grad():
            preds, depth_preds = self(sweep_imgs, mats,
                                      pts_pv=pts_pv,
                                      is_train=True)

            targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
            loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)

            if len(depth_labels.shape) == 5:
                # only key-frame will calculate depth loss
                depth_labels = depth_labels[:, 0, ...].contiguous()
            loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=3.)
        return loss_detection, loss_heatmap, loss_bbox, loss_depth


if __name__ == '__main__':
    train_set = True
    # 训练参数

    # 验证参数

    if train_set:
        run_cli(CRNLightningModel, 'det/CRN_r18_256x704_128x128_4key')
    else:
        run_cli(CRNLightningModel, 'det/CRN_r18_256x704_128x128_4key')
