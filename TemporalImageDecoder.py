import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from MambaCD.changedetection.models.SemanticDecoder import ResBlock
from ChangeDetection.ConvMamba import ConvMamba, Conv1DMamba, Conv1DMamba_v2


SELECTED_MODEL = Conv1DMamba_v2

class TemporalImageDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(TemporalImageDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_4_temporal = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SELECTED_MODEL(
                in_channels=128,
                encoder_dims=encoder_dims,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                **kwargs
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_3_temporal = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SELECTED_MODEL(
                in_channels=128,
                encoder_dims=encoder_dims,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                **kwargs
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2_temporal = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SELECTED_MODEL(
                in_channels=128,
                encoder_dims=encoder_dims,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                **kwargs
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_1_temporal = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SELECTED_MODEL(
                in_channels=128,
                encoder_dims=encoder_dims,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                **kwargs
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.smooth_layer_3 = ResBlock(
            in_channels=128,
            out_channels=128,
            stride=1,
        )
        self.smooth_layer_2 = ResBlock(
            in_channels=128,
            out_channels=128,
            stride=1,
        )
        self.smooth_layer_1 = ResBlock(
            in_channels=128,
            out_channels=128,
            stride=1,
        )
        self.smooth_layer_0 = ResBlock(
            in_channels=128,
            out_channels=128,
            stride=1,
        )
        self.trans_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.trans_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        self.conv = nn.Conv2d(
            in_channels= 128,
            out_channels= 3,
            kernel_size= 1,
        )
        
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def forward(self, features):
        feat_1, feat_2, feat_3, feat_4 = features

        '''
            Stage I
        '''
        p4 = self.st_block_4_temporal(feat_4)
       
        '''
            Stage II
        '''
        p3 = self.trans_layer_3(feat_3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)
        p3 = self.st_block_3_temporal(p3)

        '''
            Stage III
        '''
        p2 = self.trans_layer_2(feat_2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
        p2 = self.st_block_2_temporal(p2)

        '''
            Stage IV
        '''
        p1 = self.trans_layer_1(feat_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)
        p1 = self.st_block_1_temporal(p1)
        p1 = self.smooth_layer_0(p1)


        '''
        convert to RGB
        '''
        p1 = self.conv(p1)

        return p1

