import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

class ConvMamba(nn.Module):
    def __init__(
            self,
            in_channels,
            norm_layer,
            channel_first,
            ssm_act_layer,
            mlp_act_layer,
            **kwargs
    ):
        super(ConvMamba, self).__init__()
        self.conv2d1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.SILU_ = nn.SiLU()

        self.conv2d2 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.vssm = VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'])


    def forward(self,x):
        x1 = self.vssm(x)
        # print(f"x1: {x1.shape}\n")

        x2 = x.permute(0, 3, 1, 2) # Change from [B, H, W, C] to [B, C, H, W]
        x2 = self.SILU_(self.conv2d1(x2))
        # print(f"x2: {x2.shape}\n")

        x3 = self.conv2d2(x2)
        # print(f"x3: {x3.shape}\n")

        x3 = x3.permute(0, 2, 3, 1) # Change from [B, C, H, W] to [B, H, W, C]
        # print(f"x4: {x3.shape}\n")

        x_out = x1 + x3

        return x_out

encoder_dims = [96, 192, 384, 768]
kwargs = {'patch_size': 4, 'in_chans': 3, 'num_classes': 1000, 'depths': [2, 2, 15, 2], 'dims': 96, 'ssm_d_state': 1, 'ssm_ratio': 2.0, 'ssm_rank_ratio': 2.0, 'ssm_dt_rank': 'auto', 'ssm_conv': 3, 'ssm_conv_bias': False, 'ssm_drop_rate': 0.0, 'ssm_init': 'v0', 'forward_type': 'v3noz', 'mlp_ratio': 4.0, 'mlp_drop_rate': 0.0, 'drop_path_rate': 0.3, 'patch_norm': True, 'downsample_version': 'v3', 'patchembed_version': 'v2', 'gmlp': False, 'use_checkpoint': False}


class ConvMamba_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(ConvMamba_Encoder, self).__init__()
        self.convMamba = ConvMamba(
            in_channels= 128,
            **kwargs
        )

    def forward(self, x):
        x = x.Permtue(0, 2, 3, 1) # Change from [B, C, H, W] to [B, H, W, C]
        x = self.convMamba(x)
        return x