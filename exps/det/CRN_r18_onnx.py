#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project path
sys.path.append('/home/fxf/projects/MyCRN')

# Import necessary modules
from models.camera_radar_net_det import CameraRadarNetDet
from exps.det.CRN_r18_256x704_128x128_4key import CRNLightningModel

class CRNModelWrapper(nn.Module):
    """
    CRN model wrapper for ONNX export
    Convert complex input parameters to model expected format
    """
    def __init__(self, crn_model):
        super(CRNModelWrapper, self).__init__()
        self.model = crn_model
        self.eval()  # Set to evaluation mode
    
    def forward(self, sweep_imgs, mats, pts_pv):
        """
        Forward function for ONNX export
        
        Args:
        - sweep_imgs: image data [batch_size, num_sweeps, num_cams, 3, H, W]
        - mats: camera parameter dict, but ONNX needs separate tensors
        - pts_pv: radar point cloud data [batch_size, num_sweeps, num_cams, num_points, features]
        """
        # Ensure all inputs are float32
        sweep_imgs = sweep_imgs.float()
        pts_pv = pts_pv.float()
        
        # Ensure all tensors in mats are float32
        for key in mats:
            if isinstance(mats[key], torch.Tensor):
                mats[key] = mats[key].float()
        
        # Convert input to model expected format
        return self.model(sweep_imgs, mats, sweep_ptss=pts_pv, is_train=False)

def create_mats_dict(batch_size=1, num_sweeps=1, num_cams=6, device='cuda'):
    """
    Create camera parameter dict
    
    Args:
    - batch_size: batch size
    - num_sweeps: number of radar sweeps
    - num_cams: number of cameras
    - device: device type ('cuda' or 'cpu')
    
    Returns:
    - mats_dict: dict containing all camera parameters
    """
    # Create simulated camera parameters based on CRN model config
    mats_dict = {
        'intrin_mats': torch.randn(batch_size, num_sweeps, num_cams, 4, 4, dtype=torch.float32, device=device),
        'ida_mats': torch.randn(batch_size, num_sweeps, num_cams, 4, 4, dtype=torch.float32, device=device),
        'sensor2ego_mats': torch.randn(batch_size, num_sweeps, num_cams, 4, 4, dtype=torch.float32, device=device),
        'bda_mat': torch.randn(batch_size, 4, 4, dtype=torch.float32, device=device),
    }
    return mats_dict

def create_dummy_inputs(batch_size=1, num_sweeps=1, num_cams=6, num_points=1000, device='cuda'):
    """
    Create dummy input data
    
    Args:
    - batch_size: batch size
    - num_sweeps: number of radar sweeps
    - num_cams: number of cameras
    - num_points: number of points per radar
    - device: device type ('cuda' or 'cpu')
    
    Returns:
    - sweep_imgs: simulated image data
    - mats_dict: simulated camera parameter dict
    - pts_pv: simulated radar point cloud data
    """
    # Simulated image data: [batch_size, num_sweeps, num_cams, 3, 256, 704]
    sweep_imgs = torch.randn(batch_size, num_sweeps, num_cams, 3, 256, 704, dtype=torch.float32, device=device)
    
    # Simulated camera parameter dict
    mats_dict = create_mats_dict(batch_size, num_sweeps, num_cams, device)
    
    # Simulated radar point cloud data: [batch_size, num_sweeps, num_cams, num_points, features]
    # Assume each point has 5 features: x, y, z, intensity, timestamp
    pts_pv = torch.randn(batch_size, num_sweeps, num_cams, num_points, 5, dtype=torch.float32, device=device)
    
    return sweep_imgs, mats_dict, pts_pv

def export_crn_model_to_onnx():
    """
    Export CRN model to ONNX format
    """
    print("=== Start exporting CRN model to ONNX format ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available, cannot run on GPU")
        print("Please check your CUDA installation and PyTorch configuration")
        return
    
    device = 'cuda'
    print(f"Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Get current PyTorch version supported ONNX opset version
    try:
        # Convert producer_version to integer
        producer_version_str = torch.onnx.producer_version
        print(f"ONNX opset support: Current PyTorch supports opset 7-{producer_version_str}")
        
        # Try to convert to integer, if fails use default
        try:
            producer_version_int = int(producer_version_str)
        except (ValueError, TypeError):
            print(f"Warning: Cannot convert producer_version '{producer_version_str}' to integer, using default 11")
            producer_version_int = 11
    except AttributeError:
        print(f"Warning: torch.onnx.producer_version not found, using default 11")
        producer_version_int = 11
    
    # Create CRN model
    print("1. Creating CRN model...")
    
    # Load pre-trained CRN model
    # Note: Assuming you have trained model weight files
    # If not, you can use randomly initialized weights for testing
    try:
        # Try to create model from config
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
        
        # Create CRN model
        crn_model = CameraRadarNetDet(
            model_config['backbone_img_conf'],
            model_config['backbone_pts_conf'],
            model_config['fuser_conf'],
            model_config['head_conf']
        )
        
        # Key fix: Ensure all model parameters and buffers are float32
        # 1. Recursively convert all submodules to float32
        def recursive_float32(module):
            for child in module.children():
                recursive_float32(child)
            
            # Convert current module's parameters
            for param in module.parameters(recurse=False):
                param.data = param.data.float()
            
            # Convert current module's buffers
            for buffer in module.buffers(recurse=False):
                buffer.data = buffer.data.float()
        
        # Apply recursive conversion
        crn_model.eval()
        recursive_float32(crn_model)
        
        # 2. Ensure model is float32 type
        crn_model = crn_model.float()
        
        # 3. Move model to GPU
        crn_model = crn_model.to(device)
        
        # 4. Disable mixed precision training
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        print(f"Model created successfully")
        print(f"Total model parameters: {sum(p.numel() for p in crn_model.parameters()):,}")
        print(f"Model data type: {next(crn_model.parameters()).dtype}")
        print(f"Model device: {next(crn_model.parameters()).device}")
        
        # Check data type of all model parameters
        for name, param in crn_model.named_parameters():
            if param.dtype != torch.float32:
                print(f"Warning: Parameter {name} data type is {param.dtype}, converting to float32")
                param.data = param.data.float()
        
    except Exception as e:
        print(f"Model creation failed: {e}")
        return
    
    # Create model wrapper
    print("2. Creating model wrapper...")
    wrapped_model = CRNModelWrapper(crn_model)
    wrapped_model.eval()
    
    # Create dummy inputs
    print("3. Creating dummy inputs...")
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
    
    print(f"Input data shapes:")
    print(f"  sweep_imgs: {sweep_imgs.shape}, dtype: {sweep_imgs.dtype}, device: {sweep_imgs.device}")
    print(f"  pts_pv: {pts_pv.shape}, dtype: {pts_pv.dtype}, device: {pts_pv.device}")
    
    # Test model forward propagation
    print("4. Testing model forward propagation...")
    try:
        with torch.no_grad():
            # Note: mats_dict needs to be passed as parameter
            output = wrapped_model(sweep_imgs, mats_dict, pts_pv)
        
        print(f"Model forward propagation test successful")
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if torch.is_tensor(out):
                    print(f"  Output {i}: {out.shape}, dtype: {out.dtype}, device: {out.device}")
                else:
                    print(f"  Output {i}: non-tensor")
        else:
            print(f"  Output: {output.shape}, dtype: {output.dtype}, device: {output.device}")
            
    except Exception as e:
        print(f"Model forward propagation test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Export ONNX model
    print("5. Exporting ONNX model...")
    
    try:
        # Since CRN model has multiple outputs, we need to create a wrapper to handle it
        class ONNXExportWrapper(nn.Module):
            def __init__(self, model):
                super(ONNXExportWrapper, self).__init__()
                self.model = model
            
            def forward(self, sweep_imgs, intrin_mats, ida_mats, sensor2ego_mats, bda_mat, pts_pv):
                # Assemble mats dictionary
                mats = {
                    'intrin_mats': intrin_mats.float(),
                    'ida_mats': ida_mats.float(),
                    'sensor2ego_mats': sensor2ego_mats.float(),
                    'bda_mat': bda_mat.float()
                }
                # Call model
                output = self.model(sweep_imgs.float(), mats, pts_pv.float())
                
                # If output is a tuple, expand it
                if isinstance(output, tuple):
                    return output
                else:
                    return (output,)
        
        # Create export wrapper
        export_wrapper = ONNXExportWrapper(wrapped_model)
        export_wrapper.eval()
        
        # Ensure export wrapper is also on GPU
        export_wrapper = export_wrapper.to(device)
        
        # Prepare ONNX inputs
        dummy_inputs = (
            sweep_imgs,
            mats_dict['intrin_mats'],
            mats_dict['ida_mats'],
            mats_dict['sensor2ego_mats'],
            mats_dict['bda_mat'],
            pts_pv
        )
        
        # Try opset versions from 9 to 13 (inverse operation introduced in opset 9)
        min_opset = 9
        # We'll try up to 13, but we know 13 might fail
        max_opset = min(13, producer_version_int)  # Now both are integers
        
        print(f"PyTorch supported opset range: 7-{producer_version_int}")
        print(f"Trying opset versions from {min_opset} to {max_opset}")
        
        # First try default export type, then try ONNX_FALLTHROUGH if needed
        operator_export_types = [torch.onnx.OperatorExportTypes.ONNX]
        operator_export_types.append(torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
        
        export_success = False
        last_error = None
        successful_opset = None
        successful_export_type = None
        successful_path = None
        
        for operator_export_type in operator_export_types:
            if export_success:
                break
            for opset_version in range(min_opset, max_opset + 1):
                try:
                    onnx_path = f"crn_model_opset_{opset_version}.onnx"
                    if operator_export_type != torch.onnx.OperatorExportTypes.ONNX:
                        onnx_path = f"crn_model_opset_{opset_version}_fallthrough.onnx"
                    
                    print(f"Trying opset version {opset_version} with operator_export_type {operator_export_type}...")
                    
                    torch.onnx.export(
                        export_wrapper,
                        dummy_inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=opset_version,
                        operator_export_type=operator_export_type,
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
                    
                    print(f"ONNX export successful: {onnx_path}")
                    print(f"Model saved to: {os.path.abspath(onnx_path)}")
                    
                    export_success = True
                    successful_opset = opset_version
                    successful_export_type = operator_export_type
                    successful_path = onnx_path
                    break
                    
                except Exception as e:
                    last_error = e
                    print(f"Failed with opset {opset_version} and operator_export_type {operator_export_type}: {e}")
                    continue
        
        if export_success:
            print(f"\nExport successful!")
            print(f"Successful opset version: {successful_opset}")
            print(f"Successful export type: {successful_export_type}")
            print(f"Model saved as: {successful_path}")
        else:
            print(f"\nAll opset versions failed. Last error: {last_error}")
            print(f"Tried opset versions from {min_opset} to {max_opset}")
            
            # Try one more time with opset 11 (common version)
            print("\nTrying one more time with opset 11 (common supported version)...")
            try:
                torch.onnx.export(
                    export_wrapper,
                    dummy_inputs,
                    "crn_model_final_attempt.onnx",
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
                    verbose=True,  # Set to True to see more details
                    do_constant_folding=True
                )
                print(f"ONNX export successful with opset 11: crn_model_final_attempt.onnx")
            except Exception as e:
                print(f"Final attempt with opset 11 also failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"ONNX export setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Disable auto mixed precision
    torch.backends.cudnn.enabled = False
    torch.set_grad_enabled(False)
    
    # Export CRN model
    export_crn_model_to_onnx()