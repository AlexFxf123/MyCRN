"""Fast-BEV M0 (R18, 256x704, 200x200 BEV)
移植到 MyCRN 框架，暂时用不到，先放到 backup 目录下。

参考: https://github.com/Sense-GVT/Fast-BEV (v1 style, single scale)
"""
import torch
import torch.nn as nn
from functools import partial
from pytorch_lightning.core import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from mmcv.runner import build_optimizer

from exps.base_cli import run_cli
from models.fast_bev import FastBEV
from datasets.nusc_det_dataset import NuscDatasetRadarDet, collate_fn
from evaluators.det_evaluators import DetNuscEvaluator
from utils.torch_dist import all_gather_object, synchronize


class FastBEVLightningModel(LightningModule):
    """Fast-BEV Lightning 模型 (独立实现，对齐原版 Fast-BEV 配置)。"""

    MYCRN_DATA = '/home/fxf/projects/BEV_Projects/MyCRN/data'

    def __init__(self,
                 gpus=1,
                 data_root='/home/fxf/data/nuScenes',
                 eval_interval=5,
                 batch_size_per_device=8,
                 default_root_dir='./outputs/',
                 data_mode='sub',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.default_root_dir = default_root_dir
        self.data_mode = data_mode

        # 评估器
        class_names = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]
        self.evaluator = DetNuscEvaluator(
            class_names=class_names,
            output_dir=self.default_root_dir,
            data_root=self.data_root)

        # 数据配置
        mode_map = {'sub': '_sub', 'mini': '_mini', 'full': ''}
        infix = mode_map.get(data_mode, '')
        self.train_info_paths = f'{self.MYCRN_DATA}/info/nuscenes_infos{infix}_train.pkl'
        self.val_info_paths = f'{self.MYCRN_DATA}/info/nuscenes_infos{infix}_val.pkl'

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
        self.img_conf = dict(
            img_mean=[123.675, 116.28, 103.53],
            img_std=[58.395, 57.12, 57.375],
            to_rgb=True)

        self.return_image = True
        self.return_depth = False
        self.return_radar_pv = False
        self.remove_z_axis = True

        ################################################
        # 模型配置 (匹配原版 fastbev_m0_r18_s256x704)
        backbone = dict(type='ResNet', depth=18)
        neck = dict(in_channels=[64, 128, 256, 512], out_channels=64, num_outs=4)
        neck_fuse = dict(in_channels=[256], out_channels=[64])
        neck_3d = dict(
            in_channels=64 * 4,
            out_channels=192,
            num_layers=2,
            stride=2,
            is_transpose=False,
            fuse=dict(in_channels=64 * 4, out_channels=64 * 4),
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
            diff_rad_by_sin=True,
            dir_offset=0.7854,
            anchor_generator=dict(
                ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
                sizes=[[0.8660, 2.5981, 1.], [0.5774, 1.7321, 1.],
                       [1., 1., 1.], [0.4, 0.4, 1.]],
                custom_values=[0, 0],
                rotations=[0, 1.57],
                reshape_out=True),
            bbox_coder=dict(code_size=9),
            train_cfg=dict(
                code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            test_cfg=dict(
                score_thr=0.05,
                nms_pre=1000,
                max_num=500,
                use_scale_nms=True,
                nms_type_list=[
                    'rotate', 'rotate', 'rotate', 'rotate', 'rotate',
                    'rotate', 'rotate', 'rotate', 'rotate', 'circle'],
                nms_thr_list=[
                    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2],
                nms_radius_thr_list=[
                    4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1],
            ),
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
            with_cp=True,
        )

        # 优化器: lr=4e-4, backbone lr_mult=0.1 (匹配原版)
        self.optimizer_config = dict(
            type='AdamW',
            lr=4e-4,
            weight_decay=0.01,
            paramwise_cfg=dict(
                custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

        # 评估状态
        self._val_results = []
        self._val_metas = []
        # eval_interval: 每 N 个 epoch 做一次评估, 0=不评估
        # 训练结束时也会自动评估一次

    def forward(self, sweep_imgs, mats, is_train=False, **inputs):
        return self.model(sweep_imgs, mats, is_train=is_train)

    # ==================== 训练 ====================



    def training_step(self, batch, batch_idx):
        if self.global_rank == 0:
            for pg in self.trainer.optimizers[0].param_groups:
                self.log('learning_rate', pg['lr'])

        sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, _, _ = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            gt_boxes_3d = [g.cuda() for g in gt_boxes_3d]
            gt_labels_3d = [g.cuda() for g in gt_labels_3d]

        preds, _ = self(sweep_imgs, mats, is_train=True)
        targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
        loss_pos, loss_hm, loss_neg = self.model.loss(targets, preds)
        loss_total = loss_pos + loss_neg

        self.log('train/positive_bag_loss', loss_pos)
        self.log('train/negative_bag_loss', loss_neg)
        return loss_total

    def configure_optimizers(self):
        """优化器 + warmup+poly 学习率调度 (对齐原版)。"""
        optimizer = build_optimizer(self.model, self.optimizer_config)

        # 估算总步数: 20 epoch, sub 约 6000 samples, bs=8 => ~15000 steps
        warmup_iters = 1000
        if self.trainer is not None and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            total_steps = 20000

        def lr_lambda(step):
            if step < warmup_iters:
                return step / max(1, warmup_iters)  # linear warmup
            else:
                progress = (step - warmup_iters) / max(1, total_steps - warmup_iters)
                return max(0.0, (1 - progress) ** 1.0)  # poly decay

        scheduler = LambdaLR(optimizer, lr_lambda)
        return [[optimizer], [scheduler]]

    # ==================== 验证 ====================

    def on_validation_epoch_start(self):
        self._do_eval = self.eval_interval > 0 and (
            self.current_epoch % self.eval_interval == 0)
        if self._do_eval:
            self._val_results = []
            self._val_metas = []

    def validation_step(self, batch, batch_idx):
        (sweep_imgs, mats, img_metas, gt_boxes_3d, gt_labels_3d,
         _, _, _) = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
            gt_boxes_3d = [g.cuda() for g in gt_boxes_3d]
            gt_labels_3d = [g.cuda() for g in gt_labels_3d]

        with torch.no_grad():
            preds, _ = self(sweep_imgs, mats, is_train=True)
            targets = self.model.get_targets(gt_boxes_3d, gt_labels_3d)
            loss_pos, loss_hm, loss_neg = self.model.loss(targets, preds)

            if self._do_eval:
                preds_eval = self(sweep_imgs, mats, is_train=False)
                results = self.model.get_bboxes(preds_eval, img_metas)
                for i in range(len(results)):
                    results[i] = list(results[i])
                    results[i][0] = results[i][0].tensor.detach().cpu().numpy()
                    results[i][1] = results[i][1].detach().cpu().numpy()
                    results[i][2] = results[i][2].detach().cpu().numpy()
                    results[i].append(img_metas[i])
                self._val_results.extend([r[:3] for r in results])
                self._val_metas.extend([r[3] for r in results])

        return loss_pos, loss_hm, loss_neg

    def validation_epoch_end(self, validation_step_outputs):
        pos_losses = [o[0] for o in validation_step_outputs]
        synchronize()
        self.log('val/positive_bag_loss',
                 torch.mean(torch.stack(pos_losses)), on_epoch=True)

        if self._do_eval and len(self._val_results) > 0:
            mode_name = {'full': '全集', 'sub': '均衡子集',
                         'mini': 'mini'}.get(self.data_mode, self.data_mode)
            print(f'\n[Eval] Epoch {self.current_epoch}: '
                  f'running {mode_name} evaluation...')
            synchronize()
            if self.global_rank == 0:
                self.evaluator.evaluate(self._val_results, self._val_metas)
            print(f'[Eval] Epoch {self.current_epoch}: evaluation done.')

    # ==================== 测试/评估 ====================

    def eval_step(self, batch, batch_idx, prefix: str = ''):
        (sweep_imgs, mats, img_metas, _, _, _, _, _) = batch
        if torch.cuda.is_available():
            if self.return_image:
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()

        preds = self(sweep_imgs, mats, is_train=False)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)

        for i in range(len(results)):
            results[i] = list(results[i])
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'predict')

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = []
        all_img_metas = []
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()

        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(
            map(list, zip(*all_gather_object(all_img_metas))),
            [])[:dataset_length]

        if self.global_rank == 0:
            if torch.cuda.is_available():
                self.model.cpu()
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    # ==================== 数据加载 ====================

    def _build_dataset(self, is_train=True):
        info_paths = self.train_info_paths if is_train else self.val_info_paths
        # img_backbone_conf 需要 d_bound 字段供 dataset 初始化
        dummy_backbone_conf = dict(d_bound=[2.0, 58.0, 0.5])
        return NuscDatasetRadarDet(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            rda_aug_conf=dict(N_sweeps=6, N_use=5, drop_ratio=0.1),
            img_backbone_conf=dummy_backbone_conf,
            classes=self.evaluator.class_names,
            data_root=self.data_root,
            info_paths=info_paths,
            is_train=is_train,
            img_conf=self.img_conf,
            load_interval=1,
            num_sweeps=1,
            sweep_idxes=[],
            key_idxes=[],
            return_image=self.return_image,
            return_depth=self.return_depth,
            return_radar_pv=self.return_radar_pv,
            remove_z_axis=self.remove_z_axis,
            depth_path=self.MYCRN_DATA + '/depth_gt',
            radar_pv_path=self.MYCRN_DATA + '/radar_pv_filter',
        )

    def train_dataloader(self):
        dataset = self._build_dataset(is_train=True)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_image=self.return_image,
                               is_return_depth=self.return_depth,
                               is_return_radar_pv=self.return_radar_pv),
            sampler=None,
        )

    def val_dataloader(self):
        dataset = self._build_dataset(is_train=False)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_image=self.return_image,
                               is_return_depth=self.return_depth,
                               is_return_radar_pv=self.return_radar_pv),
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    run_cli(FastBEVLightningModel, 'fastbev_r18')
