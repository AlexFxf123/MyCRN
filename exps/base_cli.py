# Copyright (c) Megvii Inc. All rights reserved.
import glob
import os
import re
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary

from callbacks.ema import EMACallback
from utils.torch_dist import all_gather_object, synchronize

from .base_exp import BEVDepthLightningModel


def find_latest_checkpoint(output_dir):
    """自动查找 output_dir 中最新的 checkpoint 文件"""
    ckpt_pattern = os.path.join(output_dir, '**', '*.ckpt')
    ckpt_files = glob.glob(ckpt_pattern, recursive=True)
    if not ckpt_files:
        return None
    # 按 epoch 号排序，取最大的
    def epoch_key(path):
        match = re.search(r'epoch=(\d+)', path)
        return int(match.group(1)) if match else 0
    ckpt_files.sort(key=epoch_key)
    return ckpt_files[-1]


def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            ckpt_path=None):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--train',
                               action='store_true',
                               help='train model (default mode if no -e/-p specified)')
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch-size', '--batch_size_per_device',
                               dest='batch_size_per_device', type=int,
                               help='batch size per device (default: 1)')
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str,
                               help='path to checkpoint for evaluation/prediction')
    parent_parser.add_argument('--max-epochs', type=int,
                               help='number of epochs to train (default: 24)')
    parent_parser.add_argument('--resume',
                               action='store_true',
                               help='resume training from the latest checkpoint in the output dir')
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    default_root_dir = os.path.join('./outputs/', exp_name)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=24,
                        # strategy='ddp',          # 多卡时取消注释
                        batch_size_per_device=1,
                        gpus=1,
                        # strategy='ddp_find_unused_parameters_false',
                        num_sanity_val_steps=0,
                        check_val_every_n_epoch=1,
                        gradient_clip_val=5,
                        accumulate_grad_batches=8,  # 梯度累积，等效于batchsize*n
                        limit_val_batches=1.0,
                        log_every_n_steps=1,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=default_root_dir)
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback, ModelSummary(max_depth=3)])
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ModelSummary(max_depth=3)])

    # 确定运行模式
    is_evaluate = args.evaluate
    is_predict = args.predict
    is_train = args.train or not (is_evaluate or is_predict)

    if is_train:
        print(f'[Mode] Training mode (use -e for evaluation, -p for prediction)')
    elif is_evaluate:
        print(f'[Mode] Evaluation mode')
    elif is_predict:
        print(f'[Mode] Prediction mode')

    if is_evaluate or is_predict:
        # 优先使用 --ckpt_path，否则尝试自动查找最新的 checkpoint
        ckpt = args.ckpt_path or find_latest_checkpoint(default_root_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f'No checkpoint found in {default_root_dir}. '
                'Please specify --ckpt_path explicitly.')

        if is_evaluate:
            trainer.test(model, ckpt_path=ckpt)
        else:
            predict_step_outputs = trainer.predict(model, ckpt_path=ckpt)
            all_pred_results = list()
            all_img_metas = list()
            for predict_step_output in predict_step_outputs:
                for i in range(len(predict_step_output)):
                    all_pred_results.append(predict_step_output[i][:3])
                    all_img_metas.append(predict_step_output[i][3])
            synchronize()
            len_dataset = len(model.test_dataloader().dataset)
            all_pred_results = sum(
                map(list, zip(*all_gather_object(all_pred_results))),
                [])[:len_dataset]
            all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                                [])[:len_dataset]
            model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                         os.path.dirname(ckpt))
    else:
        # 训练模式 (默认): 支持 --resume 或显式 ckpt_path
        resume_ckpt = None
        if args.resume:
            resume_ckpt = find_latest_checkpoint(default_root_dir)
            if resume_ckpt is None:
                print(f'[Warning] No checkpoint found in {default_root_dir}, starting training from scratch.')
            else:
                print(f'[Info] Resuming training from checkpoint: {resume_ckpt}')
        elif ckpt_path:
            resume_ckpt = ckpt_path
            print(f'[Info] Resuming training from checkpoint: {resume_ckpt}')

        trainer.fit(model, ckpt_path=resume_ckpt)

        # ====== 训练完成后自动评估 ======
        print(f'\n[AutoEval] 训练完成，自动在验证集上评估...')
        best_ckpt = find_latest_checkpoint(default_root_dir)
        if best_ckpt is not None:
            print(f'[AutoEval] 使用 checkpoint: {best_ckpt}')
            trainer.test(model, ckpt_path=best_ckpt)
            print(f'[AutoEval] 评估完成，结果已保存至 {default_root_dir}')
        else:
            print(f'[AutoEval] 未找到 checkpoint，跳过评估。')
