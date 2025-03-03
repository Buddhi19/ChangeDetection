import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleChangeGuidedAttention(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])
        
    def forward(self, features_list, change_map):
        """
        Args:
            features_list: List of feature maps at different scales
            change_map: Output from change detection branch [B, 128, H, W]
        Returns:
            List of conditioned feature maps
        """
        conditioned_features = []
        for i, features in enumerate(features_list):
            # Downsample change_map to match feature size
            _, _, H, W = features.shape
            change_resized = F.interpolate(change_map, size=(H, W), mode='bilinear')
            
            # Compute attention
            attention = torch.sigmoid(self.conv_layers[i](change_resized))
            
            # Condition features
            conditioned_features.append(features * (1 + attention))
        
        return conditioned_features


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class MultiScaleChangeGuidedAttentionV2(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])
        
        self.spatial_attn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, kernel_size=1)
            )
            for channels in channels_list
        ])
        
        self.channel_attn = nn.ModuleList([
            SEBlock(channels)
            for channels in channels_list
        ])
        
    def forward(self, features_list, change_map):
        conditioned_features = []
        for i, features in enumerate(features_list):
            B, C, H, W = features.shape
            change_resized = F.interpolate(change_map, size=(H, W), mode='bilinear')
            
            change_proj = self.conv_layers[i](change_resized)
            
            sp_attn = torch.sigmoid(self.spatial_attn[i](change_proj))
            
            ch_attn = self.channel_attn[i](change_proj)
            
            attn = sp_attn * ch_attn  # [B, C, H, W]
            
            conditioned = features * (1 + attn)
            conditioned_features.append(conditioned)
            
        return conditioned_features
