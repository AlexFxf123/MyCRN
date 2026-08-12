# Copyright (c) Megvii Inc. All rights reserved.
import glob
import os
import re
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_summary import ModelSummary

from callbacks.ema import EMACallback
from utils.torch_dist import all_gather_object, synchronize
from pytorch_lightning.callbacks import Callback

from .base_exp import BEVDepthLightningModel


class ExtractPTHCallback(Callback):
    """每个 epoch 保存 checkpoint 时，同时输出纯模型权重 .pth 文件。"""
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        epoch = trainer.current_epoch
        step = trainer.global_step
        pth_path = os.path.join(self.output_dir, f'epoch={epoch}-step={step}.pth')
        torch.save(checkpoint['state_dict'], pth_path)
        print(f'[ExtractPTH] Saved: {pth_path}')


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
                        max_epochs=24,             # 默认训练 24 个 epoch
                        # strategy='ddp',          # 多卡时取消注释
                        batch_size_per_device=4,
                        gpus=1,
                        # strategy='ddp_find_unused_parameters_false',
                        num_sanity_val_steps=0,
                        check_val_every_n_epoch=2,
                        gradient_clip_val=5,
                        limit_val_batches=0.25,     # 默认只验证 25% 的 val 数据集，节省时间和内存，不然会OOM
                        accumulate_grad_batches=4,  # 默认梯度累积 4 个 batch
                        log_every_n_steps=50,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=default_root_dir)
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    # 每个 epoch 保存 .pth 权重文件，用于复现
    extract_pth_callback = ExtractPTHCallback(default_root_dir)

    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback, ModelSummary(max_depth=3), extract_pth_callback])
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[ModelSummary(max_depth=3), extract_pth_callback])

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

        # === 先跑预测，拿到所有检测结果 ===
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

        # === 保存 results_nusc.json ===
        if model.save_results_json:
            out_dir = os.path.dirname(ckpt)
        else:
            import tempfile
            out_dir = tempfile.mkdtemp(prefix='nusc_pred_')
        result_json = model.evaluator._format_bbox(all_pred_results, all_img_metas, out_dir)

        # === -e 模式：子进程跑 NuScenesEval（防 OOM） ===
        if is_evaluate:
            import subprocess, sys
            script = os.path.join(os.path.dirname(__file__), '..', 'tools', 'eval_nusc.py')
            cmd = f'{sys.executable} {script} --result_path {result_json} --output_dir {out_dir} --data_mode {model.data_mode}'
            print(f'[Eval] 启动子进程评估...')
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode == 0:
                print(f'[Eval] 评估完成')
                if not model.save_results_json:
                    for fname in os.listdir(out_dir):
                        if fname == 'results_nusc.json' or fname.endswith('.pdf'):
                            os.remove(os.path.join(out_dir, fname))
            else:
                print(f'[Eval] 评估异常退出 (code={ret.returncode})')
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

        # ====== 训练完成后自动评估（子进程，防 OOM） ======
        print(f'\n[AutoEval] 训练完成，自动在验证集上评估...')
        best_ckpt = find_latest_checkpoint(default_root_dir)
        if best_ckpt is not None:
            print(f'[AutoEval] 使用 checkpoint: {best_ckpt}')
            import subprocess, sys
            eval_cmd = f'{sys.executable} {sys.argv[0]} -e --ckpt_path {best_ckpt}'
            for arg in ['--data_mode', '--batch-size', '--save-results']:
                if arg in sys.argv:
                    idx = sys.argv.index(arg)
                    eval_cmd += f' {arg}' if arg == '--save-results' else (
                        f' {arg} {sys.argv[idx+1]}' if idx + 1 < len(sys.argv) else '')
            print(f'[AutoEval] 启动子进程: {eval_cmd}')
            ret = subprocess.run(eval_cmd, shell=True)
            if ret.returncode == 0:
                print(f'[AutoEval] 评估完成，结果已保存至 {default_root_dir}')
            else:
                print(f'[AutoEval] 评估进程异常退出 (code={ret.returncode})')
        else:
            print(f'[AutoEval] 未找到 checkpoint，跳过评估。')
