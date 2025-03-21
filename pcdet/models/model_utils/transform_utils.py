from functools import partial
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import replace_feature, spconv

def img_point_sampling(voxel_coords, img_feats, batch_dict):  # from UVTR
    batch_size = batch_dict['batch_size']
    img_aug_matrix = batch_dict.get('img_aug_matrix', None)
    lidar_aug_matrix = batch_dict.get('lidar_aug_matrix', None)
    lidar2image = batch_dict.get('lidar2image', None)
    image_size = batch_dict['ori_shape']
    # Transfer to Point cloud range with X,Y,Z

    BN, C, H, W = img_feats.size()
    N = BN // batch_size
    img_feats = img_feats.view(batch_size, N, C, H, W).transpose(0, 1)
    # multi_cam_feats = []
    sampled_feats = img_feats.new_zeros(size=(C, voxel_coords.shape[0]))
    for b in range(batch_size):
        voxel_batch_idx = voxel_coords[:, 0] == b
        # cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
        cur_coords = torch.flip(voxel_coords[voxel_batch_idx, 1:], dims=[1]).clone()
        if img_aug_matrix is not None:
            cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
        if lidar_aug_matrix is not None:
            cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else lidar_aug_matrix[0][b]
        cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, list) else lidar2image[0][b]

        cur_coords = cur_coords.to(dtype=cur_lidar2image.dtype)
        # inverse aug for pseudo points
        if lidar_aug_matrix is not None:
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
        else:
            cur_coords = cur_coords.transpose(1, 0)

        # lidar2image
        cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)  # cur_coords: [3, N]
        cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

        # if self.random_noise is not None and self.training:
        #     seed = np.random.rand()
        #     if seed > 0.5:
        #         cur_coords += np.random.uniform(-self.random_noise, self.random_noise)

        # get 2d coords
        dist = cur_coords[:, 2, :].clone()
        this_mask = (dist > 1e-5)

        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        if img_aug_matrix is not None:
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)


        cur_coords[..., 0] /= image_size[0][1]
        cur_coords[..., 1] /= image_size[0][0]
        cur_coords = (cur_coords - 0.5) * 2  # to [-1, +1]

        # this_mask = (this_mask & (cur_coords[..., 0] > -1.0)
        #              & (cur_coords[..., 0] < 1.0)
        #              & (cur_coords[..., 1] > -1.0)
        #              & (cur_coords[..., 1] < 1.0)
        #              )

        # mask.append(this_mask)
        # reference_voxel_cam.append(cur_coords)

    # sample img features
        multi_cam_feats = img_feats.new_zeros(size=(C, cur_coords.shape[1]))
        for k in range(N):
            reference_points_cam = cur_coords[k].view(-1, 1, 2)
            sampled_feat = F.grid_sample(img_feats[k][b].unsqueeze(dim=0), reference_points_cam.unsqueeze(dim=0))  # feat: [24,256,32,88]; reference_points_cam_lvl: [24, num_query, 1, 2]
            sampled_feat = sampled_feat.view(C, -1)
            multi_cam_feats += sampled_feat
        sampled_feats[..., voxel_batch_idx] = multi_cam_feats.view(C, voxel_coords.shape[0])
    sampled_feats = sampled_feats.transpose(0, 1)

    # return reference_voxel_cam, mask, sampled_feats
    return sampled_feats