from functools import partial
import copy

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
from ..model_utils.transform_utils import img_point_sampling
from ...utils.spconv_utils import replace_feature, spconv

# import torch.utils.checkpoint as cp

from .lion_backbone_hilbert import (
    LION3DBackboneHilbert,
    LIONBlockHilbert,
    PatchMerging3D,
)


# for waymo and nuscenes, kitti, once
class FusionLION3DBackbone(LION3DBackboneHilbert):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)

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
        self.image_dim = model_cfg.IMAGE_DIM
        self.use_img = model_cfg.USE_IMG

        if self.use_img:
            self.sample_img = img_point_sampling
            self.stem_img = nn.Sequential(
                spconv.SparseConv3d(
                    input_channels, dim, 3, padding=1, stride=1, bias=False
                ),
                nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )

            self.stages_img = nn.ModuleList(
                LIONBlockHilbert(
                    self.layer_dim[i],
                    depths[i],
                    layer_down_scales[i],
                    self.window_shape[i],
                    self.group_size[i],
                    direction,
                    shift=shift,
                    operator=self.linear_operator,
                    layer_id=sum(depths[:i]) * 2 if i > 0 else 0,
                    n_layer=self.n_layer,
                    scan_version=self.scan_version,
                    template=self.template,
                )
                for i in range(num_layers)
            )
            
            self.pts_conv = nn.ModuleList(
                nn.Sequential(
                    spconv.SparseConv3d(
                        self.layer_dim[i], self.layer_dim[i], 3, padding=1, stride=1, bias=False
                    ),
                    nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
                    nn.SiLU(),
                )
                for i in range(num_layers)
            )
            
            self.img_conv = nn.ModuleList(
                nn.Sequential(
                    spconv.SparseConv3d(
                        self.layer_dim[i], self.layer_dim[i], 3, padding=1, stride=1, bias=False
                    ),
                    nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
                    nn.SiLU(),
                )
                for i in range(num_layers)
            )

            self.downsp_imgs = copy.deepcopy(self.downsamples)
            # self.fusion = nn.ModuleList(
            #     FusionLayer(
            #         self.layer_dim[i],
            #         self.image_dim[i],
            #         grid_size,
            #         )
            #     for i in range(num_layers)
            # )


    def fusion(self, x, x_img, last_pts_feat, last_img_feat, idx):
        last_pts_feat = self.pts_conv[idx](last_pts_feat)
        last_img_feat = self.img_conv[idx](last_img_feat)
        x = replace_feature(x, x.features + last_img_feat)
        x_img = replace_feature(x_img, x_img.features + last_pts_feat)

        self.downsamples(x)
        self.downsp_imgs(x_img)
        return x, x_img

    def forward(self, batch_dict):
        voxel_features = batch_dict["voxel_features"]
        voxel_coords = batch_dict["voxel_coords"]
        batch_size = batch_dict["batch_size"]
        image_features = batch_dict["image_features"]

        self.use_img &= image_features is not None

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        if self.use_img:
            x_img = spconv.SparseConvTensor(
                features=self.sample_img(voxel_coords, image_features, batch_dict),
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size,
            )
            x_img = self.stem_img(x_img)
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
            pts_feat = x.features
            x = stage(x)
            if self.use_img:
                img_feat = x_img.features
                x_img = self.stages_img[idx](x_img)
                x, x_img = self.fusion(x, x_img, pts_feat, img_feat, idx)
                # fusion_feats[f'x_conv{idx+1}'] = x_fus
            else:
                x, _ = self.downsamples[idx](x)
        if self.use_img:
            x = replace_feature(x, x.features + x_img.features)
        x = self.linear_out(x)

        batch_dict.update(
            {"encoded_spconv_tensor": x, "encoded_spconv_tensor_stride": 1}
        )

        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x1,
        #         'x_conv2': x2,
        #         'x_conv3': x3,
        #         'x_conv4': x4,
        #     }
        # })
        # if fusion_feats is not None:
        #     batch_dict.update({
        #         'multi_scale_fusion_features': fusion_feats
        #     })
        #     batch_dict.update({
        #         # 'multi_scale_3d_strides': {
        #         'multi_scale_fusion_strides': {
        #             'x_conv1': torch.tensor([1,1,2], device=x.features.device).float(),
        #             'x_conv2': torch.tensor([1,1,4], device=x.features.device).float(),
        #             'x_conv3': torch.tensor([1,1,8], device=x.features.device).float(),
        #             'x_conv4': torch.tensor([1,1,16], device=x.features.device).float(),
        #         }
        #     })
        return batch_dict
