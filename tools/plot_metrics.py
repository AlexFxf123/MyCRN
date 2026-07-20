import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(exp_dir):
    """绘制评估指标条形图"""
    metrics_path = os.path.join(exp_dir, 'metrics_summary.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # 1. 绘制各类别 mAP
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = list(metrics['label_aps'].keys())
    aps = [metrics['mean_dist_aps'][cls] for cls in classes]

    axes[0].barh(classes, aps, color='steelblue')
    axes[0].set_xlabel('mAP')
    axes[0].set_title('Per-class Mean AP')
    for i, v in enumerate(aps):
        axes[0].text(v + 0.005, i, f'{v:.3f}', va='center')

    # 2. 绘制 TP errors
    err_names = {'trans_err': 'mATE', 'scale_err': 'mASE',
                 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'}
    labels = list(err_names.values())
    values = [metrics['tp_errors'][k] for k in err_names]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))

    axes[1].barh(labels, values, color=colors)
    axes[1].set_xlabel('Error')
    axes[1].set_title('TP Errors (lower is better)')
    for i, v in enumerate(values):
        axes[1].text(v + 0.005, i, f'{v:.3f}', va='center')

    plt.suptitle(f'NDS: {metrics["nd_score"]:.4f}  |  mAP: {metrics["mean_ap"]:.4f}')
    plt.tight_layout()
    # === 修复：确保输出目录存在 ===
    save_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'metrics_cvpr2019.png'), dpi=150)
    plt.show()
    print(f'NDS: {metrics["nd_score"]:.4f}, mAP: {metrics["mean_ap"]:.4f}')

# 使用示例
plot_metrics('outputs/r18/4key_baseline')