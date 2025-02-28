from functools import partial
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import replace_feature, spconv

class FusionLayer(nn.Module):
    def __init__(
        self,
        pts_dim = 128,
        img_dim = 96,
        grid_size=(448, 512),
        random_noise: Optional[float] = 0.01
        ):
        super().__init__()
        self.random_noise = random_noise
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.fuseconv = spconv.SparseSequential(
            spconv.SparseConv3d(pts_dim + img_dim, pts_dim, 3, 1, padding=1, bias=False, indice_key="subm0"),
            spconv.SparseBatchNorm(pts_dim),
            nn.ReLU()
        )
    
    def img_point_sampling(self, voxel_points, voxel_coords, img_feats, batch_dict):  # from UVTR
        # img_aug_matrix = kwargs.get('img_aug_matrix', None)
        # lidar_aug_matrix = kwargs.get('lidar_aug_matrix', None)
        # lidar2image = kwargs.get('lidar2img', None)
        # image_size = kwargs['img_metas'][0]['input_shape']
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
        sampled_feats = img_feats.new_zeros(size=(C, voxel_points.shape[0]))
        for b in range(batch_size):
            # cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
            cur_coords = voxel_points[voxel_coords[:, 0] == b, :, :3].reshape(-1, 3).clone()
            cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else lidar_aug_matrix[0][b]
            cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, list) else lidar2image[0][b]

            # inverse aug for pseudo points
            cur_coords = cur_coords.to(dtype=cur_lidar_aug_matrix.dtype)
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)  # cur_coords: [3, N]
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += np.random.uniform(-self.random_noise, self.random_noise)

            # get 2d coords
            dist = cur_coords[:, 2, :].clone()
            this_mask = (dist > 1e-5)

            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
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
            sampled_feats[..., voxel_coords[:, 0] == b] = multi_cam_feats.view(C, *voxel_points.shape[:2]).sum(-1)
            del cur_coords, multi_cam_feats
        sampled_feats = sampled_feats.transpose(0, 1)

        # return reference_voxel_cam, mask, sampled_feats
        return sampled_feats

    def img_point_sampling_v2(self, voxel_coords, img_feats, batch_dict):  # from UVTR
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

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += np.random.uniform(-self.random_noise, self.random_noise)

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
                multi_cam_feats += F.grid_sample(img_feats[k][b].unsqueeze(dim=0), reference_points_cam.unsqueeze(dim=0), align_corners=True).view(C, -1)  # feat: [24,256,32,88]; reference_points_cam_lvl: [24, num_query, 1, 2]
            sampled_feats[..., voxel_batch_idx] = multi_cam_feats.view(C, voxel_coords.shape[0])
        sampled_feats = sampled_feats.transpose(0, 1)

        # return reference_voxel_cam, mask, sampled_feats
        return sampled_feats

    def img_fv_to_bev(self, img_feats, voxel_coords, batch_dict):
        # bs = batch_dict['batch_size']
        voxel_points = batch_dict['voxel_points']
        # pillar_coors_list = []

        # for i in range(bs):
        #     this_idx = voxel_coords[:, 0]==i
        #     this_coors = voxel_coords[this_idx]
        #     pillar_coors_list.append(this_coors)

        sampled_img_feats = self.img_point_sampling(voxel_points, voxel_coords, img_feats, batch_dict)
        # sampled_img_feats = self.img_point_sampling_v2(voxel_coords, img_feats, batch_dict)

        # decorated_img_feat = torch.zeros([bs, self.embed_dims, self.bev_size, self.bev_size]).type_as(img_feats[0])
        # for b in range(bs):
        #     this_pillar_coors = pillar_coors_list[b]
        #     output = sampled_img_feats[0][b].reshape(self.embed_dims, -1, voxel_points.shape[1])
        #     decorated_img_feat[b, :, this_pillar_coors[:, 2].long(), this_pillar_coors[:, 3].long()] = output.sum(dim=2)

        return sampled_img_feats
    
    
    def forward(self, pts_feats, img_feats, batch_dict):
        sampled_img_feats = self.img_point_sampling_v2(pts_feats.indices, img_feats, batch_dict)
        fuse_feats = torch.cat([pts_feats.features, sampled_img_feats], dim=1)
        del sampled_img_feats
        fuse_feats = replace_feature(pts_feats, fuse_feats)
        fuse_feats = self.fuseconv(fuse_feats)
        return fuse_feats