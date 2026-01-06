# 测试读取pkl文件，打印文件名，判断是否存在
import os
import pickle

# 方法1: 使用with语句自动处理文件关闭
with open('/home/fxf/data/nuScenes/nuscenes_infos_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('/home/fxf/data/nuScenes/nuscenes_infos_val.pkl', 'rb') as f:
    val_data = pickle.load(f)
root_path = '/home/fxf/data/nuScenes/'
count = 0               # 总帧数
true_count = 0          # 有文件的帧数
for my_scene in train_data:     # train_data或val_data

    print('scene name:', my_scene['scene_name'])
    sample_token = my_scene['sample_token']

    cam_infos = my_scene['cam_infos']
    cam_front = cam_infos['CAM_FRONT']
    cam_front_path = root_path + cam_front['filename']

    cam_front_flag = os.path.exists(cam_front_path)
    print('cam_front_path:', cam_front_path, 'true of false:', cam_front_flag)
    if cam_front_flag:
        true_count = true_count + 1
    
    count = count + 1 

    test = 1
print('total frames: ', count, ', true frames: ', true_count)