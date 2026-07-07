"""直接评估已有的 results_nusc.json（不重新推理）。
用法:
  python tools/eval_only.py                           # 补齐到 full val
  python tools/eval_only.py --data_mode sub           # 补齐到 balanced val
  python tools/eval_only.py --data_mode mini          # 补齐到 mini val
"""
import sys, json, os, tempfile, argparse, warnings
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_mode', type=str, default='full', choices=['full', 'sub', 'mini'])
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
args = parser.parse_args()

data_root = '/home/fxf/data/nuScenes'
result_path = args.result_path or f'./outputs/r18/results_nusc.json'
output_dir = args.output_dir or './outputs/r18'

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

# 目标 val 场景
val_split_map = {'full': splits.val, 'sub': splits.val_balanced, 'mini': splits.mini_val}
target_scenes = val_split_map[args.data_mode]
print(f'Data mode: {args.data_mode}, target val scenes: {len(target_scenes)}', flush=True)

nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)

with open(result_path) as f:
    pred_data = json.load(f)
pred_tokens = set(pred_data['results'].keys())

target_tokens = set()
for s in nusc.sample:
    scene = nusc.get('scene', s['scene_token'])
    if scene['name'] in target_scenes:
        target_tokens.add(s['token'])

missing = target_tokens - pred_tokens
extra = pred_tokens - target_tokens
if missing or extra:
    for t in missing:
        pred_data['results'][t] = []
    for t in extra:
        del pred_data['results'][t]
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.json', prefix='nusc_eval_')
    with os.fdopen(tmp_fd, 'w') as f:
        json.dump(pred_data, f)
    result_path = tmp_path
    print(f'Padded: +{len(missing)} empty, -{len(extra)} extra → {len(target_tokens)} total', flush=True)

cfg = config_factory('detection_cvpr_2019')

# 让 load_gt 只加载目标场景
import nuscenes.eval.common.loaders as _loaders
_orig_create = _loaders.create_splits_scenes
_loaders.create_splits_scenes = lambda verbose=False: {
    'train': [], 'val': list(target_scenes), 'test': [],
    'mini_train': [], 'mini_val': [],
    'train_detect': [], 'train_track': [],
}
try:
    nusc_eval = NuScenesEval(nusc, cfg, result_path, 'val',
                             output_dir=output_dir, verbose=False)
    nusc_eval.main(render_curves=False)
finally:
    _loaders.create_splits_scenes = _orig_create
print('Done! Results in', output_dir)
