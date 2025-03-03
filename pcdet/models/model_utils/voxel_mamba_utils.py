import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import math


def get_z_axe_hilbert_index_3d_mamba_lite(template, coords, batch_size, z_dim, hilbert_spatial_size, scan_version=0, debug=True):
    '''
    coords: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D 
    _, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coords[:, 3] #+ shift[2]
    y = coords[:, 2] #+ shift[1]
    z = coords[:, 1] #+ shift[0]

    if scan_version == 0:
        flat_coords = (y * hil_size_x + x).long()
        hil_inds = (template[flat_coords] * z_dim + z).long()
        # operates = [lambda x: x, lambda x: x.flip(0)]
        operates = [lambda x: x]
        reversed_operates = operates
    elif scan_version == 1:
        s = hil_size_x >> 1
        flat_coords = y * s  + (x // 2) + (x % 2) * 2 * s * s
        hil_inds = (template[flat_coords] * z_dim + z).long()
        operates = [lambda x: x]
        reversed_operates = operates
        

    # inds_curt_to_next = torch.argsort(hil_inds)
    # x_coords = x[inds_curt_to_next]
    # y_coords = y[inds_curt_to_next]
    # z_coords = z[inds_curt_to_next]
    
    # inds_curt_to_next = {}
    # inds_next_to_curt = {}
    # indices = {}
    # for b in range(batch_size):
    #     batch_mask = coors[:, 0] == b
    #     inds_curt_to_next[b] = torch.argsort(hil_inds[batch_mask])
    #     inds_next_to_curt[b] = torch.argsort(inds_curt_to_next[b])
    #     # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])
    #     indice = torch.eq(batch_mask, True).nonzero().squeeze().tolist()
    #     back_indices = torch.zeros_like(indices)
    #     back_indices[batch_mask] = indices[batch_mask][inds_curt_to_next[b]].flip(0)[inds_next_to_curt[b]]
    #     indices[b] = {'forward': indice, 'backward': back_indices}
  
    # index_info = {}
    # index_info['flat2win'] = inds_curt_to_next
    # index_info['win2flat'] = inds_next_to_curt
    # index_info['indices'] = indices
    index_info = {}
    for b in range(batch_size):
        batch_mask = coords[:, 0] == b
        inds_curt_to_next = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt = torch.argsort(inds_curt_to_next)
        # indice = torch.nonzero(batch_mask, as_tuple=False).squeeze()
        # back_indices = torch.zeros_like(indice)
        # back_indices[inds_curt_to_next] = indice[inds_curt_to_next].flip(0)
        # indices = {'forward': indice, 'backward': back_indices}
        # index_info[b] = {'flat2win': inds_curt_to_next, 'win2flat': inds_next_to_curt, 'indices': indices}
        index_info[b] = {'flat2win': inds_curt_to_next, 'win2flat': inds_next_to_curt, 'operates': operates, 'reversed_operates': reversed_operates}
        
        
    
        
    return index_info

def get_hilbert_index_3d_mamba_lite(template, coors, batch_size, z_dim, hilbert_spatial_size, shift=(0, 0, 0), debug=True):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D
    hil_size_z, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coors[:, 3] + shift[2]
    y = coors[:, 2] + shift[1]
    z = coors[:, 1] + shift[0]

    flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
        # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info



def get_hilbert_index_2d_mamba_lite(template, coors, batch_size, hilbert_spatial_size, shift=(0, 0), debug=True):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D
    _, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coors[:, 3] + shift[1]
    y = coors[:, 2] + shift[0]
    # z = coors[:, 1] + shift[0]

    # flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    flat_coors = (y * hil_size_x + x).long()
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
        # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info