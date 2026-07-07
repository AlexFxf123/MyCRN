"""Fast-BEV M0 (R18, 256x704, 200x200 BEV)
移植到 MyCRN 框架。

参考: https://github.com/Sense-GVT/Fast-BEV
"""
import torch
from utils.torch_dist import synchronize

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel as BaseLightningModel
from models.fast_bev import FastBEV


class FastBEVLightningModel(BaseLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.return_image = True
        self.return_depth = True
        self.return_radar_pv = False
        self.remove_z_axis = True

        self.optimizer_config = dict(type='AdamW', lr=2e-4, weight_decay=1e-4)

        self.ida_aug_conf = {
            'resize_lim': (0.386, 0.55), 'final_dim': (256, 704),
            'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600,
            'rand_flip': True, 'bot_pct_lim': (0.0, 0.0),
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': 6,
        }
        self.bda_aug_conf = {
            'rot_ratio': 1.0, 'rot_lim': (-22.5, 22.5),
            'scale_lim': (0.95, 1.05), 'flip_dx_ratio': 0.5, 'flip_dy_ratio': 0.5
        }
        ################################################
        # Fast-BEV M0 模型配置
        backbone = dict(type='ResNet', depth=18)
        neck = dict(in_channels=[64, 128, 256, 512], out_channels=64, num_outs=4)
        neck_fuse = dict(in_channels=[256], out_channels=[64])
        neck_3d = dict(
            in_channels=64*4 * 4,
            out_channels=192,
            num_layers=2,
            stride=2,
            is_transpose=False,
            fuse=dict(in_channels=64*4 * 4 * 4, out_channels=64 * 4),
        )
        bbox_head = dict(
            num_classes=10,
            in_channels=192,
            feat_channels=192,
            num_convs=0,
            is_transpose=True,
            pre_anchor_topk=25,
            bbox_thr=0.5,
            gamma=2.0,
            alpha=0.5,
            anchor_generator=dict(
                ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
                sizes=[[0.8660, 2.5981, 1.], [0.5774, 1.7321, 1.],
                       [1., 1., 1.], [0.4, 0.4, 1.]],
                custom_values=[0, 0],
                rotations=[0, 1.57],
                reshape_out=True),
            bbox_coder=dict(code_size=9),
        )
        n_voxels = [[200, 200, 4]]
        voxel_size = [[0.5, 0.5, 1.5]]
        multi_scale_id = [0]

        ################################################
        self.model = FastBEV(
            backbone=backbone,
            neck=neck,
            neck_fuse=neck_fuse,
            neck_3d=neck_3d,
            bbox_head=bbox_head,
            n_voxels=n_voxels,
            voxel_size=voxel_size,
            multi_scale_id=multi_scale_id,
        )

        self.key_idxes = [-2, -4, -6]
        self.dbound = [2.0, 58.0, 0.5]
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])

    def forward(self, sweep_imgs, mats, is_train=False, **inputs):
        return self.model(sweep_imgs, mats, is_train=is_train)

    def training_step(self, batch, batch_idx):
        sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, _ = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            gt_boxes_3d = [g.cuda() for g in gt_boxes_3d]
            gt_labels_3d = [g.cuda() for g in gt_labels_3d]

        preds, depth_preds = self(sweep_imgs, mats, is_train=True)
        targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
        loss_det, loss_hm, loss_box = self.model.loss(targets, preds)

        # 兼容 depth loss (Fast-BEV 实际不用)
        if len(depth_labels.shape) == 5:
            depth_labels = depth_labels[:, 0, ...].contiguous()
        loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=0.)

        self.log('train/detection', loss_det)
        self.log('train/bbox', loss_box)
        self.log('train/depth', loss_depth)
        return loss_det + loss_box + loss_depth

    def validation_step(self, batch, batch_idx):
        sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, _ = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            gt_boxes_3d = [g.cuda() for g in gt_boxes_3d]
            gt_labels_3d = [g.cuda() for g in gt_labels_3d]
        with torch.no_grad():
            preds, depth_preds = self(sweep_imgs, mats, is_train=True)
            targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
            loss_det, loss_hm, loss_box = self.model.loss(targets, preds)
            if len(depth_labels.shape) == 5:
                depth_labels = depth_labels[:, 0, ...].contiguous()
            loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=0.)
        return loss_det, loss_hm, loss_box, loss_depth


if __name__ == '__main__':
    run_cli(FastBEVLightningModel, 'fastbev_r18')
