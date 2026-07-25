"""
从 PyTorch Lightning 的 .ckpt 文件中提取纯模型权重为 .pth 文件。

用法:
    python tools/extract_pth.py --ckpt_path <path>
    python tools/extract_pth.py --ckpt_path <path> --output model.pth
    python tools/extract_pth.py --ckpt_path <path> --strip_prefix model.
"""
import argparse
import os
import torch


def extract_pth(ckpt_path, output_path=None, strip_prefix=None):
    print(f'[Info] 加载 checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' not in ckpt:
        print('[Error] checkpoint 中未找到 state_dict')
        return

    state_dict = ckpt['state_dict']

    # 可选：去除前缀（如 "model."）
    if strip_prefix:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(strip_prefix):
                new_key = key[len(strip_prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        print(f'[Info] 已去除前缀 "{strip_prefix}"')

    if output_path is None:
        base, _ = os.path.splitext(ckpt_path)
        output_path = base + '.pth'

    print(f'[Info] 保存纯模型权重至: {output_path}')
    torch.save(state_dict, output_path)
    print(f'[Done] 完成，共 {len(state_dict)} 个参数组')


def main():
    parser = argparse.ArgumentParser(
        description='从 .ckpt 文件中提取纯模型权重为 .pth 文件')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='checkpoint 文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 .pth 文件路径（默认与 ckpt 同目录同名但后缀为 .pth）')
    parser.add_argument('--strip_prefix', type=str, default=None,
                        help='去除权重 key 的前缀，例如 "model."')
    args = parser.parse_args()
    extract_pth(args.ckpt_path, args.output, args.strip_prefix)


if __name__ == '__main__':
    main()