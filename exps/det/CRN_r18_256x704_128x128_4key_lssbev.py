"""
mAP: 0.4492
mATE: 0.5236
mASE: 0.2857
mAOE: 0.5640
mAVE: 0.2781
mAAE: 0.1792
NDS: 0.5415
Eval time: 185.7s

LSSBEVBackbone + ConvFuser 配置
- 图像分支: 纯LSS (voxel_pooling_v2)，无RVT
- 点云分支: PtsBackbone，独立输出后上采样到BEV分辨率
- 融合: ConvFuser
"""
import torch
from utils.torch_dist import synchronize

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel

from models.camera_radar_net_det import CameraRadarNetDet


class CRNLightningModel(BEVDepthLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.return_image = True
        self.return_depth = True
        self.return_radar_pv = True
        ################################################
        self.optimizer_config = dict(
            type='AdamW',
            lr=2e-4,
            weight_decay=1e-4)
        ################################################
        self.ida_aug_conf = {
            'resize_lim': (0.386, 0.55),
            'final_dim': (256, 704),
            'rot_lim': (0., 0.),
            'H': 900,
            'W': 1600,
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.0),
            'cams': [
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            'Ncams': 6,
        }
        self.bda_aug_conf = {
            'rot_ratio': 1.0,
            'rot_lim': (-22.5, 22.5),
            'scale_lim': (0.9, 1.1),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
        }
        ################################################
        self.backbone_img_conf = {
            'type': 'LSSBEVBackbone',           # ← 使用纯LSS骨干（无RVT）
            'x_bound': [-51.2, 51.2, 0.8],
            'y_bound': [-51.2, 51.2, 0.8],
            'z_bound': [-5, 3, 8],
            'd_bound': [2.0, 58.0, 0.8],
            'final_dim': (256, 704),
            'downsample_factor': 16,
            'img_backbone_conf': dict(
                type='ResNet',
                depth=18,
                frozen_stages=0,
                out_indices=[0, 1, 2, 3],
                norm_eval=False,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
            ),
            'img_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[64, 128, 256, 512],
                upsample_strides=[0.25, 0.5, 1, 2],
                out_channels=[64, 64, 64, 64],
            ),
            'depth_net_conf': dict(
                in_channels=256,
                mid_channels=256,
            ),
            'camera_aware': False,              # ← 可配置，默认False
            'output_channels': 80,
        }
        ################################################
        self.backbone_pts_conf = {
            'pts_voxel_layer': dict(
                max_num_points=8,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
                max_voxels=(768, 1024)
            ),
            'pts_voxel_encoder': dict(
                type='PillarFeatureNet',
                in_channels=5,
                feat_channels=[32, 64],
                with_distance=False,
                with_cluster_center=False,
                with_voxel_center=True,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                legacy=True
            ),
            'pts_middle_encoder': dict(
                type='PointPillarsScatter',
                in_channels=64,
                output_shape=(140, 88)
            ),
            'pts_backbone': dict(
                type='SECOND',
                in_channels=64,
                out_channels=[64, 128, 256],
                layer_nums=[2, 3, 3],
                layer_strides=[1, 2, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
            ),
            'pts_neck': dict(
                type='SECONDFPN',
                in_channels=[64, 128, 256],
                out_channels=[64, 64, 64],
                upsample_strides=[0.5, 1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True
            ),
            'out_channels_pts': 80,
        }
        ################################################
        self.fuser_conf = {
            'type': 'ConvFuser',                # ← 使用卷积融合
            'img_dims': 80,
            'pts_dims': 80,
            'embed_dims': 128,
            'num_layers': 6,
        }
        ################################################
        self.head_conf = {
            'bev_backbone_conf': dict(
                type='ResNet',
                in_channels=128,
                depth=18,
                num_stages=3,
                strides=(1, 2, 2),
                dilations=(1, 1, 1),
                out_indices=[0, 1, 2],
                norm_eval=False,
                base_channels=128,
            ),
            'bev_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[128, 128, 256, 512],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64]
            ),
            'tasks': [
                dict(num_class=1, class_names=['car']),
                dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                dict(num_class=2, class_names=['bus', 'trailer']),
                dict(num_class=1, class_names=['barrier']),
                dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
            ],
            'common_heads': dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            'bbox_coder': dict(
                type='CenterPointBBoxCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.01,
                out_size_factor=4,
                voxel_size=[0.2, 0.2, 8],
                pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                code_size=9,
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
            'in_channels': 256,
            'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
            'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            'gaussian_overlap': 0.1,
            'min_radius': 2,
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
            depth_labels = depth_labels[:, 0, ...].contiguous()
        loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=3.)

        self.log('train/detection', loss_detection)
        self.log('train/heatmap', loss_heatmap)
        self.log('train/bbox', loss_bbox)
        self.log('train/depth', loss_depth)
        return loss_detection + loss_depth


if __name__ == '__main__':
    run_cli(CRNLightningModel, 'r18')
