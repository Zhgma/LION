# -----------------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# -----------------------------------------------------------------------------------
# VMamba: Visual State Space Model
# Copyright (c) 2024 MzeroMiko
# -----------------------------------------------------------------------------------
# Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion
# Modified by Chaodong Xiao
# -----------------------------------------------------------------------------------

import math
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch_scatter
from mamba_ssm import Block as MambaBlock
from torch.nn import functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from .swin import SwinTransformer
from .vmamba import SS2D, Mlp
from ...utils.spconv_utils import replace_feature, spconv
from ..model_utils.swin_utils import swin_converter
from ..model_utils.swin_utils import PatchEmbed, PatchMerging
from ..model_utils.swin_utils import FFN, DropPath, to_2tuple, trunc_normal_, trunc_normal_init, constant_init

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.patches_resolution = patches_resolution
        # self.num_patches = patches_resolution[0] * patches_resolution[1]

        # self.in_chans = in_chans
        # self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = rearrange(x, 'b d h w -> b h w d')

        return x


class DownSampling(nn.Module):
    r""" DownSampling Layer.

    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, ratio=4.0):
        super().__init__()

        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.conv(rearrange(x, 'b h w d -> b d h w').contiguous())
        x = rearrange(x, 'b d h w -> b h w d')

        return x


class MambaBlock2D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # attn_drop_rate: float = 0,
        # d_state: int = 16,
        # dt_init: str = "random",
        # num_heads: int = 8,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        # =============================
        mlp_ratio = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim*mlp_ratio), act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
    
    def forward(self, x: torch.Tensor):
        x = x + self.cpe1(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class MambaBlockSequence2D(nn.Module):
    def __init__(
        self, 
        embed_dim,
        depth=2,
        drop_path=[0.1, 0.1], 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2",
        # ===========================
        # attn_drop=0.,
        # drop_path=0., 
        # downsample=None, 
        # d_state=16,
        # dt_init="random",
        # mlp_ratio=4.0,
        # norm_layer=nn.LayerNorm, 
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        with_cp=False, 
        **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.with_cp = with_cp

        self.blocks = nn.ModuleList([
            MambaBlock2D(
                hidden_dim=embed_dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                # attn_drop_rate=attn_drop,
                # d_state=d_state,
                # dt_init=dt_init,
                # mlp_ratio=mlp_ratio,
                # norm_layer=norm_layer,
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=embed_dim)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_down, x

        return x, x


class MambaBackbone2D(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        embed_dims = self.model_cfg.get('EMBED_DIMS', [128, 128, 128, 128])
        depths = self.model_cfg.get('DEPTH', [2, 2, 2, 2])
        # layer_down_scales = self.model_cfg.get('LAYER_DOWN_SCALES', [[2, 2], [2, 2]])
        # direction = self.model_cfg.get('DIRECTION', ['x', 'y'])
        # diffusion = self.model_cfg.get('DIFFUSION', False)
        # shift = self.model_cfg.get('SHIFT', False)
        # diff_scale = self.model_cfg.get('DIFF_SCALE', 1)
        drop_path=self.model_cfg.get('DROP_PATH', [0.1, 0.1, 0.1, 0.1])
        # =========================
        ssm_d_state=self.model_cfg.get('SSM_D_STATE', 16)
        ssm_ratio=self.model_cfg.get('SSM_RATIO', 2.0)
        ssm_rank_ratio=self.model_cfg.get('SSM_RANK_RATIO', 2.0)
        ssm_dt_rank=self.model_cfg.get('SSM_DT_RANK', "auto")
        ssm_act_layer=self.model_cfg.get('SSM_ACT_LAYER', nn.SiLU)
        ssm_conv=self.model_cfg.get('SSM_CONV', 3)
        ssm_conv_bias=self.model_cfg.get('SSM_CONV_BIAS', True)
        ssm_drop_rate=self.model_cfg.get('SSM_DROP_RATE', 0.0)
        ssm_simple_init=self.model_cfg.get('SSM_SIMPLE_INIT', False)
        forward_type=self.model_cfg.get('FORWARD_TYPE', "v2")
        # =========================
        mlp_ratio=self.model_cfg.get('MLP_RATIO', 4.0)
        mlp_act_layer=self.model_cfg.get('MLP_ACT_LAYER', nn.GELU)
        mlp_drop_rate=self.model_cfg.get('MLP_DROP_RATE', 0.0)
        # =========================
        with_cp = self.model_cfg.get('WITH_CP', False)
        self.embed_dims = embed_dims
        self.patch_embed = Stem(in_chans=3, embed_dim=embed_dims[0])
        
        
        pretrain_img_size = self.model_cfg.get('PRETRAIN_IMG_SIZE', 224)
        init_cfg = self.model_cfg.get('INIT_CFG', None)
        # depths = self.model_cfg.DEPTHS
        # strides = self.model_cfg.get('STRIDES', (4, 2, 2, 2))
        patch_size = self.model_cfg.get('PATCH_SIZE', 4)
        # embed_dims = self.model_cfg.EMBED_DIMS
        drop_rate = self.model_cfg.get('DROP_RATE', 0.0)
        out_indices = self.model_cfg.get('OUT_INDICES', [0, 1, 2, 3])
        # with_cp = self.model_cfg.get('WITH_CP', False)
        use_abs_pos_embed = self.model_cfg.get('USE_ABS_POS_EMBED', False)
        
        self.convert_weights = self.model_cfg.get('CONVERT_WEIGHTS', False)
        self.frozen_stages = self.model_cfg.get('FROZEN_STAGES', -1)

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = (pretrain_img_size, pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = (pretrain_img_size[0], pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        self.init_cfg = init_cfg

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        # assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        
        
        self.stages = nn.ModuleList([
            MambaBlockSequence2D(
                embed_dim=embed_dims[idx],
                depth=depths[idx],
                drop_path=drop_path[idx],
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                # layer_down_scales[idx],
                # direction,
                # diffusion,
                # shift,
                # diff_scale,
                downsample=DownSampling if idx < num_layers - 1 else None,
                with_cp=with_cp
            )
            for idx in range(num_layers)])
        # self.add_module('layer{}'.format(idx), stages)
        # self.downsp.append(
        #     DownSampling(dim[idx], ratio=layer_down_scales[idx]))
        
        # Add a norm layer for each output
        for i in out_indices:
            layer =  nn.LayerNorm(embed_dims[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MambaBackbone2D, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            print(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = torch.load(self.init_cfg.checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    print(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, batch_dict):
        x = batch_dict['camera_imgs']
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        img_feats = []
        for i, stage in enumerate(self.stages):
            x, img_feat = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                img_feat = norm_layer(img_feat)
                img_feat = img_feat.permute(0, 3, 1, 2).contiguous()  # B * N, H, W, C -> B * N, C, H, W
                img_feats.append(img_feat)
        batch_dict['image_features'] = img_feats
        return batch_dict

# class MambaBackbone2D(nn.Module):
#     def __init__(self, model_cfg):
#         super().__init__()
#         self.model_cfg = model_cfg
#         pretrain_img_size = self.model_cfg.get('PRETRAIN_IMG_SIZE', 224)
#         init_cfg = self.model_cfg.get('INIT_CFG', None)
#         depths = self.model_cfg.DEPTHS
#         in_channels = self.model_cfg.get('IN_CHANNELS', 3)
#         strides = self.model_cfg.get('STRIDES', (4, 2, 2, 2))
#         patch_size = self.model_cfg.get('PATCH_SIZE', 4)
#         embed_dims = self.model_cfg.EMBED_DIMS
#         num_heads = self.model_cfg.NUM_HEADS
#         window_size = self.model_cfg.WINDOW_SIZE
#         mlp_ratio = self.model_cfg.MLP_RATIO
#         qkv_bias = self.model_cfg.get('QKV_BIAS', True)
#         qk_scale = self.model_cfg.get('QK_SCALE', None)
#         drop_rate = self.model_cfg.DROP_RATE
#         attn_drop_rate = self.model_cfg.ATTN_DROP_RATE
#         drop_path_rate = self.model_cfg.DROP_PATH_RATE
#         patch_norm = self.model_cfg.get('PATCH_NORM', True)
#         out_indices = self.model_cfg.get('OUT_INDICES', [0, 1, 2, 3])
#         with_cp = self.model_cfg.get('WITH_CP', False)
#         use_abs_pos_embed = self.model_cfg.get('USE_ABS_POS_EMBED', False)
#         act_cfg=dict(type='GELU')
#         norm_cfg=dict(type='LN')
        
#         self.convert_weights = self.model_cfg.get('CONVERT_WEIGHTS', False)
#         self.frozen_stages = self.model_cfg.get('FROZEN_STAGES', -1)

#         if isinstance(pretrain_img_size, int):
#             pretrain_img_size = to_2tuple(pretrain_img_size)
#         elif isinstance(pretrain_img_size, tuple):
#             if len(pretrain_img_size) == 1:
#                 pretrain_img_size = to_2tuple(pretrain_img_size[0])
#             assert len(pretrain_img_size) == 2, \
#                 f'The size of image should have length 1 or 2, ' \
#                 f'but got {len(pretrain_img_size)}'
                
#         self.patch_embed = Stem(in_chans=3, embed_dim=self.dim[0])
        
#         if self.use_abs_pos_embed:
#             patch_row = pretrain_img_size[0] // patch_size
#             patch_col = pretrain_img_size[1] // patch_size
#             num_patches = patch_row * patch_col
#             self.absolute_pos_embed = nn.Parameter(
#                 torch.zeros((1, num_patches, embed_dims)))
        
#         self.layer = nn.ModuleList()
#         self.downsp = nn.ModuleList()
#         for idx in len(num_layers):
#             self.layer.append(
#                 MambaLayer2D(dim[idx], depths[idx], layer_down_scales[idx], direction, diffusion, shift, diff_scale))
#             self.add_module('layer{}'.format(idx), layer)
#             self.downsp.append(
#                 DownSampling(dim[idx], ratio=layer_down_scales[idx]))
            
#     def forward(self, batch_dict):
#         x = batch_dict['camera_imgs']
#         B, N, C, H, W = x.size()
#         x = x.view(B * N, C, H, W)
#         x, hw_shape = self.patch_embed(x)

#         if self.use_abs_pos_embed:
#             x = x + self.absolute_pos_embed
            
#         for idx in range(len(self.layer)):
#             x = self.layer[idx](x)
#             x = self.downsp[idx](x)
#         return x
        
        