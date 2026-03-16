"""CondensateNet model architecture for condensate segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from .config import MODEL_PARAMS


class SparseAttention(nn.Module):
    def __init__(self, kernel_size=11):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        return x * self.attention(attention_input)

class NormProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(self.bn(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.swish = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(self.swish(self.bn(x)))

class StyleModulatedConv(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.conv_block = ConvBlock(channels, channels)
        self.style_projection = nn.Linear(style_dim, channels)

    def forward(self, x, style_vector):
        feat = self.conv_block(x)
        style_bias = self.style_projection(style_vector).unsqueeze(-1).unsqueeze(-1)
        return feat + style_bias

class DualResidualBlock(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.style_conv1 = StyleModulatedConv(channels, style_dim)
        self.style_conv2 = StyleModulatedConv(channels, style_dim)
        self.style_conv3 = StyleModulatedConv(channels, style_dim)
        self.projection = NormProjection(channels, channels)
        self.initial_conv = ConvBlock(channels, channels)

    def forward(self, x, lateral, style_vector):
        combined = self.initial_conv(x) + lateral
        x_intermediate = self.style_conv1(combined, style_vector) + self.projection(x)
        refined = self.style_conv2(x_intermediate, style_vector)
        output = self.style_conv3(refined, style_vector) + x_intermediate
        return output

class MultiScaleEncoder(nn.Module):
    def __init__(self, variant='rw_s', pyramid_channels=[24, 48, 64, 160]):
        super().__init__()
        encoder_name = f"efficientnetv2_{variant}"
        self.base_encoder = create_model(
            encoder_name, features_only=True, pretrained=True,
            in_chans=1, out_indices=[0, 1, 2, 3]
        )
        enc_channels = self.base_encoder.feature_info.channels()
        self.channel_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, target_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_ch)
            )
            for enc_ch, target_ch in zip(enc_channels[:4], pyramid_channels)
        ])

    def forward(self, x):
        features = self.base_encoder(x)[:4]
        return [adapter(feat) for feat, adapter in zip(features, self.channel_adapters)]

class CondensateSegmentationNet(nn.Module):
    def __init__(self, encoder_variant='rw_s', pyramid_channels=[24, 48, 64, 160],
                 use_spatial_attention=True, spatial_kernel_size=11, dropout_rate=0.15):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.pyramid_channels = pyramid_channels

        self.encoder = MultiScaleEncoder(encoder_variant, pyramid_channels)

        style_dim = pyramid_channels[-1]
        pyramid_dim = 32

        self.pyramid_block4 = DualResidualBlock(pyramid_dim, style_dim)
        self.pyramid_block3 = DualResidualBlock(pyramid_dim, style_dim)
        self.pyramid_block2 = DualResidualBlock(pyramid_dim, style_dim)
        self.pyramid_block1 = DualResidualBlock(pyramid_dim, style_dim)

        self.lateral_conv4 = nn.Conv2d(pyramid_channels[3], pyramid_dim, kernel_size=1, bias=False)
        self.lateral_conv3 = nn.Conv2d(pyramid_channels[2], pyramid_dim, kernel_size=1, bias=False)
        self.lateral_conv2 = nn.Conv2d(pyramid_channels[1], pyramid_dim, kernel_size=1, bias=False)
        self.lateral_conv1 = nn.Conv2d(pyramid_channels[0], pyramid_dim, kernel_size=1, bias=False)

        self.upsample_blocks = nn.ModuleList([
            DualResidualBlock(pyramid_dim, style_dim) for _ in range(3)
        ])

        self.output_projection = NormProjection(pyramid_dim, pyramid_dim)

        if use_spatial_attention:
            self.spatial_attention = SparseAttention(spatial_kernel_size)

        self.dropout = nn.Dropout2d(dropout_rate)
        self.mask_head = nn.Conv2d(pyramid_dim, 1, kernel_size=1)
        self.flow_head = nn.Conv2d(pyramid_dim, 2, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.Linear)):
                nn.init.constant_(m.weight, 1) if isinstance(m, nn.BatchNorm2d) else nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.mask_head.weight, gain=0.1)
        nn.init.constant_(self.mask_head.bias, -2.0) if self.mask_head.bias is not None else None
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias) if self.flow_head.bias is not None else None

    def upsample_and_refine(self, x, style_vector, block):
        x_up = F.interpolate(x, scale_factor=2, mode='nearest')
        return block(x_up, torch.zeros_like(x_up), style_vector)

    def forward(self, x):
        # Input validation
        if not torch.all(torch.isfinite(x)):
            raise ValueError("Input contains NaN or Inf")

        # Extract multi-scale features
        features = self.encoder(x)  # Using x directly, not x_normalized
        C1, C2, C3, C4 = features

        # Generate style vector from deepest features
        style_vector = F.normalize(F.adaptive_avg_pool2d(C4, 1).flatten(1), p=2, dim=1)

        # Build feature pyramid with style modulation
        P4 = self.pyramid_block4(self.lateral_conv4(C4), self.lateral_conv4(C4), style_vector)
        P3 = self.pyramid_block3(F.interpolate(P4, size=C3.shape[2:], mode='nearest'), self.lateral_conv3(C3), style_vector)
        P2 = self.pyramid_block2(F.interpolate(P3, size=C2.shape[2:], mode='nearest'), self.lateral_conv2(C2), style_vector)
        P1 = self.pyramid_block1(F.interpolate(P2, size=C1.shape[2:], mode='nearest'), self.lateral_conv1(C1), style_vector)

        # Progressive upsampling and refinement
        for i, P in enumerate([P4, P3, P2]):
            for j in range(3-i):
                P = self.upsample_and_refine(P, style_vector, self.upsample_blocks[0])
            if i == 0:
                P4_refined = P
            elif i == 1:
                P3_refined = P
            else:
                P2_refined = P

        # Combine multi-scale features
        combined = P1 + P2_refined + P3_refined + P4_refined

        # Apply spatial attention and dropout
        features_final = self.dropout(
            self.spatial_attention(self.output_projection(combined))
            if self.use_spatial_attention
            else self.output_projection(combined)
        )

        # Generate outputs at original input resolution
        target_size = x.shape[2:]
        return {
            'mask': F.interpolate(self.mask_head(features_final), size=target_size, mode='bilinear', align_corners=False),
            'flows': F.interpolate(self.flow_head(features_final), size=target_size, mode='bilinear', align_corners=False)
        }

def create_condensate_model():
    """Factory function to create model with default parameters"""
    return CondensateSegmentationNet(**MODEL_PARAMS)
