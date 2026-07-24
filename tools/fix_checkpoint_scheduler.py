"""
修复 checkpoint 中的 scheduler milestones，使其与新的训练配置匹配。

用法:
    python tools/fix_checkpoint_scheduler.py                          # 自动查找最新 ckpt
    python tools/fix_checkpoint_scheduler.py --ckpt_path <path>       # 指定 ckpt
    python tools/fix_checkpoint_scheduler.py --milestones 40 44       # 自定义 milestones
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


def fix_checkpoint_scheduler(ckpt_path, new_milestones, overwrite=False):
    print(f'[Info] 加载 checkpoint: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'lr_schedulers' not in ckpt or len(ckpt['lr_schedulers']) == 0:
        print('[Error] checkpoint 中未找到 lr_schedulers 字段')
        return

    old_state = ckpt['lr_schedulers'][0]
    old_milestones = old_state.get('milestones', {})
    old_last_epoch = old_state.get('last_epoch', 'N/A')

    print(f'[Info] 旧 milestones: {old_milestones}')
    print(f'[Info] 旧 last_epoch: {old_last_epoch}')

    # 构造新的 milestones dict
    new_milestones_dict = {m: 0.1 for m in new_milestones}
    ckpt['lr_schedulers'][0]['milestones'] = new_milestones_dict
    ckpt['lr_schedulers'][0]['last_epoch'] = -1

    print(f'[Info] 新 milestones: {new_milestones_dict}')
    print(f'[Info] 新 last_epoch: -1 (重置)')

    if overwrite:
        save_path = ckpt_path
    else:
        base, ext = os.path.splitext(ckpt_path)
        save_path = f'{base}_fixed{ext}'

    torch.save(ckpt, save_path)
    print(f'[Done] 已保存修改后的 checkpoint 至: {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='修复 checkpoint 中的 scheduler milestones')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='checkpoint 文件路径（默认自动查找最新 ckpt）')
    parser.add_argument('--output_dir', type=str, default='./outputs/',
                        help='搜索 checkpoint 的目录（自动查找时使用）')
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 44],
                        help='新的 milestones 列表，例如: 40 44')
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

    fix_checkpoint_scheduler(ckpt_path, args.milestones, args.overwrite)


if __name__ == '__main__':
    main()
