import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('/home/fxf/projects/MyCRN')

# 导入必要的模块
from models.camera_radar_net_det import CameraRadarNetDet
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel

class CRNModelWrapper(nn.Module):
    """
    CRN模型包装器，用于导出ONNX格式
    将复杂的输入参数封装为模型期望的格式
    """
    def __init__(self, crn_model):
        super(CRNModelWrapper, self).__init__()
        self.model = crn_model
        self.eval()  # 设置为评估模式
    
    def forward(self, sweep_imgs, mats, pts_pv):
        """
        前向传播函数，适配ONNX导出
        
        参数:
        - sweep_imgs: 图像数据 [batch_size, num_sweeps, num_cams, 3, H, W]
        - mats: 相机参数字典，但ONNX需要拆分成独立张量
        - pts_pv: 雷达点云数据 [batch_size, num_sweeps, num_cams, num_points, features]
        """
        # 确保所有输入都是float32类型
        sweep_imgs = sweep_imgs.float()
        pts_pv = pts_pv.float()
        
        # 确保mats中的所有张量都是float32类型
        for key in mats:
            if isinstance(mats[key], torch.Tensor):
                mats[key] = mats[key].float()
        
        # 将输入转换为模型期望的格式
        return self.model(sweep_imgs, mats, sweep_ptss=pts_pv, is_train=False)

def create_mats_dict(batch_size=1, num_sweeps=1, num_cams=6, device='cuda'):
    """
    创建相机参数字典
    
    参数:
    - batch_size: 批次大小
    - num_sweeps: 雷达扫描次数
    - num_cams: 相机数量
    - device: 设备类型 ('cuda' 或 'cpu')
    
    返回:
    - mats_dict: 包含所有相机参数的字典
    """
    # 根据CRN模型配置创建模拟的相机参数
    mats_dict = {
        'intrin_mats': torch.randn(batch_size, num_sweeps, num_cams, 4, 4, dtype=torch.float32, device=device),
        'ida_mats': torch.randn(batch_size, num_sweeps, num_cams, 4, 4, dtype=torch.float32, device=device),
        'sensor2ego_mats': torch.randn(batch_size, num_sweeps, num_cams, 4, 4, dtype=torch.float32, device=device),
        'bda_mat': torch.randn(batch_size, 4, 4, dtype=torch.float32, device=device),
    }
    return mats_dict

def create_dummy_inputs(batch_size=1, num_sweeps=1, num_cams=6, num_points=1000, device='cuda'):
    """
    创建模拟输入数据
    
    参数:
    - batch_size: 批次大小
    - num_sweeps: 雷达扫描次数
    - num_cams: 相机数量
    - num_points: 每个雷达的点数
    - device: 设备类型 ('cuda' 或 'cpu')
    
    返回:
    - sweep_imgs: 模拟图像数据
    - mats_dict: 模拟相机参数字典
    - pts_pv: 模拟雷达点云数据
    """
    # 模拟图像数据: [batch_size, num_sweeps, num_cams, 3, 256, 704]
    sweep_imgs = torch.randn(batch_size, num_sweeps, num_cams, 3, 256, 704, dtype=torch.float32, device=device)
    
    # 模拟相机参数字典
    mats_dict = create_mats_dict(batch_size, num_sweeps, num_cams, device)
    
    # 模拟雷达点云数据: [batch_size, num_sweeps, num_cams, num_points, features]
    # 假设每个点有5个特征: x, y, z, intensity, timestamp
    pts_pv = torch.randn(batch_size, num_sweeps, num_cams, num_points, 5, dtype=torch.float32, device=device)
    
    return sweep_imgs, mats_dict, pts_pv

def export_crn_model_to_onnx():
    """
    导出CRN模型为ONNX格式
    """
    print("=== 开始在GPU上导出CRN模型为ONNX格式 ===")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法在GPU上运行")
        print("请检查您的CUDA安装和PyTorch配置")
        return
    
    device = 'cuda'
    print(f"使用设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 创建CRN模型
    print("1. 创建CRN模型...")
    
    # 加载预训练的CRN模型
    # 注意：这里假设您有训练好的模型权重文件
    # 如果没有，可以使用随机初始化的权重进行测试
    try:
        # 尝试从配置创建模型
        model_config = {
            'backbone_img_conf': {
                'x_bound': [-51.2, 51.2, 0.8],
                'y_bound': [-51.2, 51.2, 0.8],
                'z_bound': [-5, 3, 8],
                'd_bound': [2.0, 58.0, 0.8],
                'final_dim': (256, 704),
                'downsample_factor': 16,
                'img_backbone_conf': {
                    'type': 'ResNet',
                    'depth': 18,
                    'frozen_stages': 0,
                    'out_indices': [0, 1, 2, 3],
                    'norm_eval': False,
                },
                'img_neck_conf': {
                    'type': 'SECONDFPN',
                    'in_channels': [64, 128, 256, 512],
                    'upsample_strides': [0.25, 0.5, 1, 2],
                    'out_channels': [64, 64, 64, 64],
                },
                'depth_net_conf': {
                    'in_channels': 256,
                    'mid_channels': 256,
                },
                'radar_view_transform': True,
                'camera_aware': False,
                'output_channels': 80,
            },
            'backbone_pts_conf': {
                'pts_voxel_layer': {
                    'max_num_points': 8,
                    'voxel_size': [8, 0.4, 2],
                    'point_cloud_range': [0, 2.0, 0, 704, 58.0, 2],
                    'max_voxels': (768, 1024)
                },
                'pts_voxel_encoder': {
                    'type': 'PillarFeatureNet',
                    'in_channels': 5,
                    'feat_channels': [32, 64],
                    'with_distance': False,
                    'with_cluster_center': False,
                    'with_voxel_center': True,
                    'voxel_size': [8, 0.4, 2],
                    'point_cloud_range': [0, 2.0, 0, 704, 58.0, 2],
                    'norm_cfg': {'type': 'BN1d', 'eps': 1e-3, 'momentum': 0.01},
                    'legacy': True
                },
                'pts_middle_encoder': {
                    'type': 'PointPillarsScatter',
                    'in_channels': 64,
                    'output_shape': (140, 88)
                },
                'pts_backbone': {
                    'type': 'SECOND',
                    'in_channels': 64,
                    'out_channels': [64, 128, 256],
                    'layer_nums': [2, 3, 3],
                    'layer_strides': [1, 2, 2],
                    'norm_cfg': {'type': 'BN', 'eps': 1e-3, 'momentum': 0.01},
                    'conv_cfg': {'type': 'Conv2d', 'bias': True, 'padding_mode': 'reflect'}
                },
                'pts_neck': {
                    'type': 'SECONDFPN',
                    'in_channels': [64, 128, 256],
                    'out_channels': [64, 64, 64],
                    'upsample_strides': [0.5, 1, 2],
                    'norm_cfg': {'type': 'BN', 'eps': 1e-3, 'momentum': 0.01},
                    'upsample_cfg': {'type': 'deconv', 'bias': False},
                    'use_conv_for_no_stride': True
                },
                'out_channels_pts': 80,
            },
            'fuser_conf': {
                'img_dims': 80,
                'pts_dims': 80,
                'embed_dims': 128,
                'num_layers': 6,
                'num_heads': 4,
                'bev_shape': (128, 128),
            },
            'head_conf': {
                'bev_backbone_conf': {
                    'type': 'ResNet',
                    'in_channels': 128,
                    'depth': 18,
                    'num_stages': 3,
                    'strides': (1, 2, 2),
                    'dilations': (1, 1, 1),
                    'out_indices': [0, 1, 2],
                    'norm_eval': False,
                    'base_channels': 128,
                },
                'bev_neck_conf': {
                    'type': 'SECONDFPN',
                    'in_channels': [128, 128, 256, 512],
                    'upsample_strides': [1, 2, 4, 8],
                    'out_channels': [64, 64, 64, 64]
                },
                'tasks': [
                    {'num_class': 1, 'class_names': ['car']},
                    {'num_class': 2, 'class_names': ['truck', 'construction_vehicle']},
                    {'num_class': 2, 'class_names': ['bus', 'trailer']},
                    {'num_class': 1, 'class_names': ['barrier']},
                    {'num_class': 2, 'class_names': ['motorcycle', 'bicycle']},
                    {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']},
                ],
                'common_heads': {
                    'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 
                    'rot': (2, 2), 'vel': (2, 2)
                },
                'bbox_coder': {
                    'type': 'CenterPointBBoxCoder',
                    'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    'max_num': 500,
                    'score_threshold': 0.01,
                    'out_size_factor': 4,
                    'voxel_size': [0.2, 0.2, 8],
                    'pc_range': [-51.2, -51.2, -5, 51.2, 51.2, 3],
                    'code_size': 9,
                },
                'train_cfg': {
                    'point_cloud_range': [-51.2, -51.2, -5, 51.2, 51.2, 3],
                    'grid_size': [512, 512, 1],
                    'voxel_size': [0.2, 0.2, 8],
                    'out_size_factor': 4,
                    'dense_reg': 1,
                    'gaussian_overlap': 0.1,
                    'max_objs': 500,
                    'min_radius': 2,
                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                'test_cfg': {
                    'post_center_limit_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    'max_per_img': 500,
                    'max_pool_nms': False,
                    'min_radius': [4, 12, 10, 1, 0.85, 0.175],
                    'score_threshold': 0.01,
                    'out_size_factor': 4,
                    'voxel_size': [0.2, 0.2, 8],
                    'nms_type': 'circle',
                    'pre_max_size': 1000,
                    'post_max_size': 200,
                    'nms_thr': 0.2,
                },
                'in_channels': 256,
                'loss_cls': {'type': 'GaussianFocalLoss', 'reduction': 'mean'},
                'loss_bbox': {'type': 'L1Loss', 'reduction': 'mean', 'loss_weight': 0.25},
                'gaussian_overlap': 0.1,
                'min_radius': 2,
            }
        }
        
        # 创建CRN模型
        crn_model = CameraRadarNetDet(
            model_config['backbone_img_conf'],
            model_config['backbone_pts_conf'],
            model_config['fuser_conf'],
            model_config['head_conf']
        )
        
        # 关键修复：确保模型所有参数和缓冲区都是float32
        # 1. 递归地将所有子模块转换为float32
        def recursive_float32(module):
            for child in module.children():
                recursive_float32(child)
            
            # 转换当前模块的参数
            for param in module.parameters(recurse=False):
                param.data = param.data.float()
            
            # 转换当前模块的缓冲区
            for buffer in module.buffers(recurse=False):
                buffer.data = buffer.data.float()
        
        # 应用递归转换
        crn_model.eval()
        recursive_float32(crn_model)
        
        # 2. 确保模型是float32类型
        crn_model = crn_model.float()
        
        # 3. 将模型移动到GPU
        crn_model = crn_model.to(device)
        
        # 4. 禁用混合精度训练
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        print(f"模型创建成功")
        print(f"模型参数总数: {sum(p.numel() for p in crn_model.parameters()):,}")
        print(f"模型数据类型: {next(crn_model.parameters()).dtype}")
        print(f"模型所在设备: {next(crn_model.parameters()).device}")
        
        # 检查模型所有参数的数据类型
        for name, param in crn_model.named_parameters():
            if param.dtype != torch.float32:
                print(f"警告: 参数 {name} 的数据类型是 {param.dtype}，正在转换为float32")
                param.data = param.data.float()
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        return
    
    # 创建模型包装器
    print("2. 创建模型包装器...")
    wrapped_model = CRNModelWrapper(crn_model)
    wrapped_model.eval()
    
    # 创建模拟输入
    print("3. 创建模拟输入...")
    batch_size = 1
    num_sweeps = 1
    num_cams = 6
    num_points = 1000
    
    sweep_imgs, mats_dict, pts_pv = create_dummy_inputs(
        batch_size=batch_size,
        num_sweeps=num_sweeps,
        num_cams=num_cams,
        num_points=num_points,
        device=device
    )
    
    print(f"输入数据形状:")
    print(f"  sweep_imgs: {sweep_imgs.shape}, dtype: {sweep_imgs.dtype}, device: {sweep_imgs.device}")
    print(f"  pts_pv: {pts_pv.shape}, dtype: {pts_pv.dtype}, device: {pts_pv.device}")
    
    # 测试模型前向传播
    print("4. 测试模型前向传播...")
    try:
        with torch.no_grad():
            # 注意：这里需要将mats_dict作为参数传递
            output = wrapped_model(sweep_imgs, mats_dict, pts_pv)
        
        print(f"模型前向传播测试成功")
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if torch.is_tensor(out):
                    print(f"  输出 {i}: {out.shape}, dtype: {out.dtype}, device: {out.device}")
                else:
                    print(f"  输出 {i}: 非张量")
        else:
            print(f"  输出: {output.shape}, dtype: {output.dtype}, device: {output.device}")
            
    except Exception as e:
        print(f"模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 导出ONNX模型
    print("5. 导出ONNX模型...")
    onnx_path = "crn_model_gpu.onnx"
    
    try:
        # 由于CRN模型有多个输出，我们需要创建一个包装器来处理
        class ONNXExportWrapper(nn.Module):
            def __init__(self, model):
                super(ONNXExportWrapper, self).__init__()
                self.model = model
            
            def forward(self, sweep_imgs, intrin_mats, ida_mats, sensor2ego_mats, bda_mat, pts_pv):
                # 组装mats字典
                mats = {
                    'intrin_mats': intrin_mats.float(),
                    'ida_mats': ida_mats.float(),
                    'sensor2ego_mats': sensor2ego_mats.float(),
                    'bda_mat': bda_mat.float()
                }
                # 调用模型
                output = self.model(sweep_imgs.float(), mats, pts_pv.float())
                
                # 如果输出是元组，将其展开
                if isinstance(output, tuple):
                    return output
                else:
                    return (output,)
        
        # 创建导出包装器
        export_wrapper = ONNXExportWrapper(wrapped_model)
        export_wrapper.eval()
        
        # 确保导出包装器也在GPU上
        export_wrapper = export_wrapper.to(device)
        
        # 准备ONNX输入
        dummy_inputs = (
            sweep_imgs,
            mats_dict['intrin_mats'],
            mats_dict['ida_mats'],
            mats_dict['sensor2ego_mats'],
            mats_dict['bda_mat'],
            pts_pv
        )
        
        # 导出ONNX
        torch.onnx.export(
            export_wrapper,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['sweep_imgs', 'intrin_mats', 'ida_mats', 
                        'sensor2ego_mats', 'bda_mat', 'pts_pv'],
            output_names=['output'],
            dynamic_axes={
                'sweep_imgs': {0: 'batch_size', 1: 'num_sweeps', 2: 'num_cams'},
                'intrin_mats': {0: 'batch_size', 1: 'num_sweeps'},
                'ida_mats': {0: 'batch_size', 1: 'num_sweeps'},
                'sensor2ego_mats': {0: 'batch_size', 1: 'num_sweeps'},
                'bda_mat': {0: 'batch_size'},
                'pts_pv': {0: 'batch_size', 1: 'num_sweeps', 2: 'num_cams', 3: 'num_points'},
                'output': {0: 'batch_size'}
            },
            verbose=False,
            do_constant_folding=True
        )
        
        print(f"ONNX导出成功: {onnx_path}")
        print(f"模型已保存到: {os.path.abspath(onnx_path)}")
        
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 禁用自动混合精度
    torch.backends.cudnn.enabled = False
    torch.set_grad_enabled(False)
    
    # 导出CRN模型
    export_crn_model_to_onnx()
