import torch

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .dynamic_voxel_vfe import DynamicVoxelVFE


class DynamicVoxelVFE_V2(DynamicVoxelVFE):
    def __init__(self, *args, **kwargs):
        self.num_point_per_voxel = 3
        super().__init__(*args, **kwargs)

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        voxel_points = unq_coords.new_zeros(size=(unq_coords.shape[0], self.num_point_per_voxel, 3))
        for voxel_idx in range(unq_coords.shape[0]):
            points_in_voxel = points[unq_inv == voxel_idx, 1:4]
            if points_in_voxel.size(0) == 0:
                continue
            if points_in_voxel.size(0) >= self.num_point_per_voxel:
                voxel_point = points_in_voxel[:self.num_point_per_voxel]
            else:
                voxel_point = points.new_zeros(size=(self.num_point_per_voxel, 3))
                voxel_point[:points_in_voxel.size(0)] = points_in_voxel
            voxel_points[voxel_idx] = voxel_point


        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['voxel_points'] = voxel_points

        return batch_dict
