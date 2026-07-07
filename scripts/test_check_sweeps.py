"""
验证 gen_info.py 中 sweep 收集的 bug

用法：
  1. 先用原版 gen_info.py 生成 tiny 数据：
     python scripts/gen_info.py    # 这会生成全量数据
  2. 运行本脚本检查 bug
"""
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def check_existing_pkl(pkl_path):
    """检查已生成的 pkl 文件中 cam_sweeps 和 lidar_sweeps 的情况"""
    print(f"\n{'='*60}")
    print(f"检查文件: {pkl_path}")
    print('='*60)
    infos = mmcv.load(pkl_path)
    total = len(infos)
    
    cam_sweep_counts = []
    lidar_sweep_counts = []
    
    for i, info in enumerate(infos):
        cam_sweep_counts.append(len(info['cam_sweeps']))
        lidar_sweep_counts.append(len(info['lidar_sweeps']))
        
        # 只打印前 5 个样本的详细信息
        if i < 5:
            print(f"\n样本 {i}:")
            print(f"  场景: {info['scene_name']}")
            print(f"  sample_token: {info['sample_token']}")
            print(f"  cam_sweeps 数量: {len(info['cam_sweeps'])}")
            print(f"  lidar_sweeps 数量: {len(info['lidar_sweeps'])}")
            if info['cam_sweeps']:
                print(f"  cam_sweeps[0] 包含的相机: {list(info['cam_sweeps'][0].keys())}")
            if info['lidar_sweeps']:
                print(f"  lidar_sweeps[0] 包含的 LiDAR: {list(info['lidar_sweeps'][0].keys())}")
    
    cam_arr = np.array(cam_sweep_counts)
    lidar_arr = np.array(lidar_sweep_counts)
    
    print(f"\n{'='*60}")
    print(f"统计汇总 (共 {total} 个样本):")
    print(f"  cam_sweeps:")
    print(f"    最大值: {cam_arr.max()}")
    print(f"    最小值: {cam_arr.min()}")
    print(f"    平均值: {cam_arr.mean():.2f}")
    print(f"    中位数: {np.median(cam_arr):.1f}")
    print(f"    为 0 的数量: {(cam_arr == 0).sum()} / {total} ({(cam_arr == 0).mean()*100:.1f}%)")
    print(f"  lidar_sweeps:")
    print(f"    最大值: {lidar_arr.max()}")
    print(f"    最小值: {lidar_arr.min()}")
    print(f"    平均值: {lidar_arr.mean():.2f}")
    print(f"    中位数: {np.median(lidar_arr):.1f}")
    print(f"    为 0 的数量: {(lidar_arr == 0).sum()} / {total} ({(lidar_arr == 0).mean()*100:.1f}%)")
    
    return cam_arr, lidar_arr


def verify_bug_with_nuscenes():
    """
    直接从 nuScenes 数据中验证 prev 链的行为
    对比 sample_token 在不同 prev 步下的变化
    """
    print(f"\n{'='*60}")
    print("直接从 nuScenes API 验证 prev 链的行为")
    print('='*60)
    
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='/home/fxf/data/nuScenes/',
                    verbose=False)
    
    # 取第一个 scene 的第一个 sample
    scene = nusc.scene[0]
    first_sample = nusc.get('sample', scene['first_sample_token'])
    
    print(f"\n场景: {scene['name']}")
    print(f"第一个 sample token: {first_sample['token']}")
    
    # 检查 LIDAR_TOP 的 prev 链
    print(f"\n--- LIDAR_TOP prev 链 (前 12 步) ---")
    lidar_token = first_sample['data']['LIDAR_TOP']
    sd = nusc.get('sample_data', lidar_token)
    print(f"  当前 keyframe: sample_token={sd['sample_token']}, timestamp={sd['timestamp']}")
    
    for i in range(12):
        if sd['prev'] == '':
            print(f"  第 {i+1} 步: prev 为空，终止")
            break
        sd = nusc.get('sample_data', sd['prev'])
        same = "✓ 同 sample" if sd['sample_token'] == first_sample['token'] else "✗ 不同 sample"
        print(f"  第 {i+1} 步: sample_token={sd['sample_token']}  {same}, timestamp={sd['timestamp']}")
    
    # 检查 CAM_FRONT 的 prev 链
    print(f"\n--- CAM_FRONT prev 链 (前 8 步) ---")
    cam_token = first_sample['data']['CAM_FRONT']
    sd = nusc.get('sample_data', cam_token)
    print(f"  当前 keyframe: sample_token={sd['sample_token']}, timestamp={sd['timestamp']}")
    
    for i in range(8):
        if sd['prev'] == '':
            print(f"  第 {i+1} 步: prev 为空，终止")
            break
        sd = nusc.get('sample_data', sd['prev'])
        same = "✓ 同 sample" if sd['sample_token'] == first_sample['token'] else "✗ 不同 sample"
        print(f"  第 {i+1} 步: sample_token={sd['sample_token']}  {same}, timestamp={sd['timestamp']}")


if __name__ == '__main__':
    # 1. 验证原 bug 是否触发
    verify_bug_with_nuscenes()
    
    # 2. 检查已生成的 pkl 文件
    MYCRN_DATA = '/home/fxf/projects/BEV_Projects/MyCRN/data'
    
    try:
        # 先尝试检查小数据集
        check_existing_pkl(f'{MYCRN_DATA}/info/nuscenes_infos_train-tiny.pkl')
    except FileNotFoundError:
        print("\n[提示] 未找到 tiny 数据集，检查完整数据集...")
        check_existing_pkl(f'{MYCRN_DATA}/info/nuscenes_infos_train.pkl')