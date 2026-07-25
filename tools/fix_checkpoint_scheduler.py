"""
修复 checkpoint 中的 scheduler milestones 并重置 optimizer LR，
使其与新的训练配置匹配。

用法:
    python tools/fix_checkpoint_scheduler.py                          # 自动查找最新 ckpt
    python tools/fix_checkpoint_scheduler.py --ckpt_path <path>       # 指定 ckpt
    python tools/fix_checkpoint_scheduler.py --milestones 40 44       # 自定义 milestones
    python tools/fix_checkpoint_scheduler.py --lr 2e-4                # 自定义初始 LR
    python tools/fix_checkpoint_scheduler.py --overwrite              # 直接覆盖原文件
"""
import argparse
import glob
import os
import re
import torch


def find_latest_checkpoint(output_dir):
    """自动查找 output_dir 中最新的 checkpoint 文件"""
    ckpt_pattern = os.path.join(output_dir, '**', '*.ckpt')
    ckpt_files = glob.glob(ckpt_pattern, recursive=True)
    if not ckpt_files:
        return None

    def epoch_key(path):
        match = re.search(r'epoch=(\d+)', path)
        return int(match.group(1)) if match else 0

    ckpt_files.sort(key=epoch_key)
    return ckpt_files[-1]


def fix_checkpoint_scheduler(ckpt_path, new_milestones, lr, overwrite=False):
    print(f'[Info] 加载 checkpoint: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # ====== 1. 修复 scheduler milestones ======
    if 'lr_schedulers' not in ckpt or len(ckpt['lr_schedulers']) == 0:
        print('[Error] checkpoint 中未找到 lr_schedulers 字段')
        return

    old_state = ckpt['lr_schedulers'][0]
    old_milestones = old_state.get('milestones', {})
    old_last_epoch = old_state.get('last_epoch', 'N/A')

    print(f'[Info] 旧 scheduler milestones: {old_milestones}')
    print(f'[Info] 旧 scheduler last_epoch: {old_last_epoch}')

    new_milestones_dict = {m: 0.1 for m in new_milestones}
    ckpt['lr_schedulers'][0]['milestones'] = new_milestones_dict
    ckpt['lr_schedulers'][0]['last_epoch'] = -1

    print(f'[Info] 新 scheduler milestones: {new_milestones_dict}')
    print(f'[Info] 新 scheduler last_epoch: -1 (重置)')

    # ====== 2. 重置 optimizer LR ======
    if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
        for i, pg in enumerate(ckpt['optimizer_states'][0]['param_groups']):
            old_lr = pg.get('lr', 'N/A')
            pg['lr'] = lr
            # 也重置 initial_lr / base_lr 等字段
            for key in ('initial_lr', 'base_lr'):
                if key in pg:
                    old_val = pg[key]
                    pg[key] = lr
                    print(f'[Info]   param_group[{i}] {key}: {old_val} -> {lr}')
            print(f'[Info]   param_group[{i}] lr: {old_lr} -> {lr}')
    else:
        print('[Warning] checkpoint 中未找到 optimizer_states，跳过 LR 重置')

    # ====== 3. 重置 callbacks 中的 EMA 状态（可选） ======
    # EMA callback 可能也保存了模型参数，但不需要动 LR

    if overwrite:
        save_path = ckpt_path
    else:
        base, ext = os.path.splitext(ckpt_path)
        save_path = f'{base}_fixed{ext}'

    torch.save(ckpt, save_path)
    print(f'[Done] 已保存修改后的 checkpoint 至: {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='修复 checkpoint 中的 scheduler milestones 并重置 optimizer LR')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='checkpoint 文件路径（默认自动查找最新 ckpt）')
    parser.add_argument('--output_dir', type=str, default='./outputs/',
                        help='搜索 checkpoint 的目录（自动查找时使用）')
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 44],
                        help='新的 milestones 列表，例如: 40 44')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='初始学习率（默认 2e-4）')
    parser.add_argument('--overwrite', action='store_true',
                        help='直接覆盖原文件（默认另存为 *_fixed.ckpt）')
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(args.output_dir)
        if ckpt_path is None:
            print(f'[Error] 在 {args.output_dir} 下未找到 .ckpt 文件，请通过 --ckpt_path 指定')
            return
        print(f'[Info] 自动找到最新 checkpoint: {ckpt_path}')

    fix_checkpoint_scheduler(ckpt_path, args.milestones, args.lr, args.overwrite)


if __name__ == '__main__':
    main()
