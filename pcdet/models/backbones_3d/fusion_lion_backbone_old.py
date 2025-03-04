from functools import partial

# import math
# import numpy as np
import torch
import torch.nn as nn
# import torch_scatter
# from mamba_ssm import Block as MambaBlock
# from torch.nn import functional as F

# from ..model_utils.retnet_attn import Block as RetNetBlock
# from ..model_utils.rwkv_cls import Block as RWKVBlock
# from ..model_utils.vision_lstm2 import xLSTM_Block
# from ..model_utils.ttt import TTTBlock
from ...utils.spconv_utils import replace_feature, spconv
# import torch.utils.checkpoint as cp

from .lion_backbone_one_stride import LIONBlock, PatchMerging3D, LIONLayer
from ..fusion_layers.fusion_layer import FusionLayer


#for waymo and nuscenes, kitti, once
class FusionLION3DBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        self.stages = nn.ModuleList(
            LIONBlock(self.layer_dim[i],
                      depths[i],
                      layer_down_scales[i],
                      self.window_shape[i],
                      self.group_size[i],
                      direction,
                      shift=shift,
                      operator=self.linear_operator,
                      layer_id=i * depths[i] * 2 * 2,
                      n_layer=self.n_layer
                      )
            for i in range(num_layers)

        )
        self.downsamples = nn.ModuleList(
            PatchMerging3D(
                self.layer_dim[i],
                self.layer_dim[i],
                down_scale=[1, 1, 2],
                norm_layer=nn.LayerNorm,
                diffusion=diffusion,
                diff_scale=diff_scale
                )
            for i in range(num_layers)
        )
        
        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 128,
            'x_conv2': 128,
            'x_conv3': 128,
            'x_conv4': 128
        }
        
        
        self.image_dim = model_cfg.IMAGE_DIM
        self.fusion = nn.ModuleList(
            FusionLayer(
                self.layer_dim[i],
                self.image_dim[i],
                grid_size,
                )
            for i in range(num_layers)
        )

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        image_features = batch_dict['image_features']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # x = self.linear_1(x)
        # x1, _ = self.dow1(x)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
        # x = self.linear_2(x1)
        # x2, _ = self.dow2(x)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
        # x = self.linear_3(x2)
        # x3, _ = self.dow3(x)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
        # x = self.linear_4(x3)
        # x4, _ = self.dow4(x)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
        fusion_feats = dict()
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if image_features[idx] is not None:
                x = self.fusion[idx](x, image_features[idx], batch_dict)
            fusion_feats[f'x_conv{idx+1}'] = x
            x, _ = self.downsamples[idx](x)
        x = self.linear_out(x)

        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 1
        })

        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x1,
        #         'x_conv2': x2,
        #         'x_conv3': x3,
        #         'x_conv4': x4,
        #     }
        # })
        if fusion_feats is not None:
            batch_dict.update({
                'multi_scale_fusion_features': fusion_feats
            })
            batch_dict.update({
                # 'multi_scale_3d_strides': {
                'multi_scale_fusion_strides': {
                    'x_conv1': torch.tensor([1,1,2], device=x.features.device).float(),
                    'x_conv2': torch.tensor([1,1,4], device=x.features.device).float(),
                    'x_conv3': torch.tensor([1,1,8], device=x.features.device).float(),
                    'x_conv4': torch.tensor([1,1,16], device=x.features.device).float(),
                }
            })
        return batch_dict
