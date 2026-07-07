# nuScenes库函数测试，判断子集内的场景
import os
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/fxf/data/nuScenes', verbose=True)
root_path = '/home/fxf/data/nuScenes/'
count = 0               # 总帧数
true_count = 0          # 有文件的帧数
true_scene = []
for my_scene in nusc.scene:     # 判断nuscenes数据集的场景
    # print('scene name:', my_scene['name'])
    scene_token = my_scene['token']
    sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', sample_token)

    sensor = 'CAM_FRONT'
    cam_front = nusc.get('sample_data', my_sample['data'][sensor])
    cam_front_path = root_path + cam_front['filename']

    cam_front_flag = os.path.exists(cam_front_path)
    print('cam_front_path:', cam_front_path, 'true of false:', cam_front_flag)
    if cam_front_flag:
        true_count = true_count + 1
        true_scene.append(my_scene['name'])
    
    count = count + 1 

    if count > 100:
        break

    test = 1
# 打印所有场景
print('total scenes!')
print()
print(true_scene)

# my_sample = nusc.get('sample', sample_token)
# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
# nusc.render_sample_data(cam_front_data['token'])