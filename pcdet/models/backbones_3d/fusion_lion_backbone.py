from functools import partial
import copy

import math
# import numpy as np
import torch
import torch.nn as nn

import torch_scatter
# from mamba_ssm import Block as MambaBlock
# from torch.nn import functional as F

# from ..model_utils.retnet_attn import Block as RetNetBlock
# from ..model_utils.rwkv_cls import Block as RWKVBlock
# from ..model_utils.vision_lstm2 import xLSTM_Block
# from ..model_utils.ttt import TTTBlock
from ..model_utils.transform_utils import img_point_sampling
from ...utils.spconv_utils import replace_feature, spconv

# import torch.utils.checkpoint as cp

from .lion_backbone_hilbert import LION3DBackboneHilbert, LIONBlockHilbert, PatchMerging3D

class Downsample(PatchMerging3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_conv_img = spconv.SparseSequential(
            spconv.SubMConv3d(self.dim, self.dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(self.dim),
            nn.GELU(),
        )


    def forward(self, x, x_img, img_feats, idx, batch_dict, coords_shift=1, diffusion_scale=4):
        x = self.sub_conv(x)
        x_img = self.sub_conv_img(x_img)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:
            x_feat_att = x.features.mean(-1)
            batch_size = x.indices[:, 0].max() + 1
            selected_diffusion_pts_feats_list = [x.features.clone()]
            selected_diffusion_img_feats_list = [x_img.features.clone()]
            selected_diffusion_coords_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K)

                selected_coords_copy = x.indices[mask][indices].clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                selected_pts_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0
                selected_img_feats_expand = x_img.features[mask][indices].repeat(diffusion_scale, 1) * 0.0


                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
                            selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
                            selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
                        selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
                        selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale==4:
#                         print('####diffusion_scale==4')
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
                        selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
                        selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_pts_feats_list.append(selected_pts_feats_expand)
                selected_diffusion_img_feats_list.append(selected_img_feats_expand)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_pts_feats = torch.cat(selected_diffusion_pts_feats_list)
            final_diffusion_img_feats = torch.cat(selected_diffusion_img_feats_list)

        else:
            coords = x.indices.clone()
            final_diffusion_pts_feats = x.features.clone()
            final_diffusion_img_feats = x_img.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (x.spatial_shape[0] // down_scale[2])


        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        pts_merge = torch_scatter.scatter_add(final_diffusion_pts_feats, unq_inv, dim=0)
        img_merge = torch_scatter.scatter_add(final_diffusion_img_feats, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        reverse_scale = torch.tensor([x ** (idx + 1) for x in down_scale[::-1]], device=voxel_coords.device, dtype=voxel_coords.dtype)
        reverse_coords = voxel_coords.clone()
        reverse_coords[:, 1:4] *= reverse_scale
        sample_feats = img_point_sampling(reverse_coords, img_feats, batch_dict)
        img_merge += sample_feats

        pts_merge = self.norm(pts_merge)
        img_merge = self.norm(img_merge)

        pts_merge = spconv.SparseConvTensor(
            features=pts_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        
        img_merge = spconv.SparseConvTensor(
            features=img_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return pts_merge, img_merge
        

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
        self.use_img = model_cfg.USE_IMG

        if self.use_img:
            self.stem_img = spconv.SparseSequential(
                spconv.SubMConv3d(
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
                spconv.SparseSequential(
                    spconv.SubMConv3d(self.layer_dim[i], self.layer_dim[i], 3, padding=1, stride=1, bias=False, indice_key='subm'),
                    nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
                    nn.SiLU(),
                )
                for i in range(num_layers)
            )
            
            self.img_conv = nn.ModuleList(
                spconv.SparseSequential(
                    spconv.SubMConv3d(self.layer_dim[i], self.layer_dim[i], 3, padding=1, stride=1, bias=False, indice_key='subm'),
                    nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
                    nn.SiLU(),
                )
                for i in range(num_layers)
            )

            self.downsamples = nn.ModuleList(
                Downsample(
                    self.layer_dim[i],
                    self.layer_dim[i],
                    down_scale=[1, 1, 2],
                    norm_layer=self.norm_fn,
                    diffusion=diffusion,
                    diff_scale=diff_scale
                    )
                for i in range(num_layers)
            )
            # self.fusion = nn.ModuleList(
            #     FusionLayer(
            #         self.layer_dim[i],
            #         self.image_dim[i],
            #         grid_size,
            #         )
            #     for i in range(num_layers)
            # )


    def fusion(self, x, x_img, last_pts_feat, last_img_feat):
        x = replace_feature(x, x.features + last_img_feat)
        x_img = replace_feature(x_img, x_img.features + last_pts_feat)
        return x, x_img

    def forward(self, batch_dict):
        voxel_features = batch_dict["voxel_features"]
        voxel_coords = batch_dict["voxel_coords"]
        batch_size = batch_dict["batch_size"]
        image_features = batch_dict["image_fpn"][0]

        self.use_img &= image_features is not None

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        if self.use_img:
            x_img = spconv.SparseConvTensor(
                features=img_point_sampling(voxel_coords, image_features, batch_dict),
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
        # fusion_feats = dict()
        for idx, stage in enumerate(self.stages):
            if self.use_img:
                pts_feat = self.pts_conv[idx](x).features
                img_feat = self.img_conv[idx](x_img).features
            x = stage(x)
            if self.use_img:
                x_img = self.stages_img[idx](x_img)
                x, x_img = self.fusion(x, x_img, pts_feat, img_feat)
                x, x_img = self.downsamples[idx](x, x_img, image_features, idx, batch_dict)
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
