import mmcv

from models.base_bev_depth import BaseBEVDepth
from layers.backbones.rvt_lss_fpn import RVTLSSFPN
from layers.backbones.pts_backbone import PtsBackbone
from layers.fuser.multimodal_feature_aggregation import MFAFuser
from layers.fuser.conv_fuser import ConvFuser
from layers.heads.bev_depth_head_det import BEVDepthHead

logger = mmcv.utils.get_logger('mmdet')
logger.setLevel('WARNING')

__all__ = ['CameraRadarNetDet']


class CameraRadarNetDet(BaseBEVDepth):
    """Source code of `CRN`, `https://arxiv.org/abs/2304.00670`.

    Args:
        backbone_img_conf (dict): Config of image backbone.
        backbone_pts_conf (dict): Config of point backbone.
        fuser_conf (dict): Config of BEV feature fuser.
        head_conf (dict): Config of head.
    """

    def __init__(self, backbone_img_conf, backbone_pts_conf, fuser_conf, head_conf):
        super(BaseBEVDepth, self).__init__()
        self.backbone_img = RVTLSSFPN(**backbone_img_conf)              # 图像骨干网络
        self.backbone_pts = PtsBackbone(**backbone_pts_conf)            # 点云骨干网络
        fuser_type = fuser_conf.pop('type', 'MFAFuser')                # 融合模块类型
        if fuser_type == 'ConvFuser':
            self.fuser = ConvFuser(**fuser_conf)                       # 卷积融合模块
        else:
            self.fuser = MFAFuser(**fuser_conf)                        # 注意力融合模块
        self.head = BEVDepthHead(**head_conf)                           # 检测头

        self.radar_view_transform = backbone_img_conf['radar_view_transform']

        # inference time measurement
        self.idx = 0
        self.times_dict = {
            'img': [],
            'img_backbone': [],
            'img_dep': [],
            'img_transform': [],
            'img_pool': [],

            'pts': [],
            'pts_voxelize': [],
            'pts_backbone': [],
            'pts_head': [],

            'fusion': [],
            'fusion_pre': [],
            'fusion_layer': [],
            'fusion_post': [],

            'head': [],
            'head_backbone': [],
            'head_head': [],
        }

    def forward(self,
                sweep_imgs,
                mats_dict,
                sweep_ptss=None,
                is_train=False
                ):
        """Forward function for BEVDepth

        Args:
            sweep_imgs (Tensor): Input images.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sweep_ptss (Tensor): Input points.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if is_train:
            self.time = None

            ptss_context, ptss_occupancy, _ = self.backbone_pts(sweep_ptss)
            feats, depth, _ = self.backbone_img(sweep_imgs,
                                                mats_dict,
                                                ptss_context,
                                                ptss_occupancy,
                                                return_depth=True)
            fused, _ = self.fuser(feats)
            preds, _ = self.head(fused)
            return preds, depth
        else:
            if self.idx < 100:  # skip few iterations for warmup
                self.times = None
            elif self.idx == 100:
                self.times = self.times_dict

            ptss_context, ptss_occupancy, self.times = self.backbone_pts(sweep_ptss,
                                                                         times=self.times)
            feats, self.times = self.backbone_img(sweep_imgs,
                                                  mats_dict,
                                                  ptss_context,
                                                  ptss_occupancy,
                                                  times=self.times)
            fused, self.times = self.fuser(feats, times=self.times)
            preds, self.times = self.head(fused, times=self.times)

            if self.idx == 1000:
                time_mean = {}
                for k, v in self.times.items():
                    if len(v) > 0:
                        time_mean[k] = sum(v) / len(v)

                def _print_time(key, indent=''):
                    if key in time_mean:
                        print('%s%s: %.2f' % (indent, key, time_mean[key]))

                _print_time('img')
                _print_time('img_backbone', '  ')
                _print_time('img_dep', '  ')
                _print_time('img_transform', '  ')
                _print_time('img_pool', '  ')
                _print_time('pts')
                _print_time('pts_voxelize', '  ')
                _print_time('pts_backbone', '  ')
                _print_time('pts_head', '  ')
                _print_time('fusion')
                _print_time('fusion_pre', '  ')
                _print_time('fusion_layer', '  ')
                _print_time('fusion_post', '  ')
                _print_time('head')
                _print_time('head_backbone', '  ')
                _print_time('head_head', '  ')

                total_keys = ['pts', 'img', 'fusion', 'head']
                if all(k in time_mean for k in total_keys):
                    total = sum(time_mean[k] for k in total_keys)
                    print('total: %.2f' % total)
                    print(' ')
                    print('FPS: %.2f' % (1000/total))

            self.idx += 1
            return preds
