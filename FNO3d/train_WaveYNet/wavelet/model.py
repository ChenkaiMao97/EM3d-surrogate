import torch
import torch.nn as nn
import pywt
from dwt import DWTForward3d_Laplacian, DWTInverse3d_Laplacian


class WaveletBlock(nn.Module):
    def __init__(self, channels, wavelet_type, max_depth, block_depth, num_high_freq_terms, num_groups=1):
        super(WaveletBlock, self).__init__()
        self.wavelet_type = wavelet_type
        self.max_depth = max_depth
        self.num_high_freq_terms = num_high_freq_terms
        self.num_groups = num_groups
        self.wavelet = pywt.Wavelet(self.wavelet_type)
        self.dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=self.max_depth, wave=self.wavelet, mode="zero").to("cuda")
        self.dwt_forward_3d_lap = DWTForward3d_Laplacian(J=self.max_depth, wave=self.wavelet, mode="zero").to("cuda")

        self.conv_low = nn.ModuleList()
        self.conv_high = nn.ModuleList([nn.ModuleList() for _ in range(num_high_freq_terms)])
        self.conv_spatial = nn.ModuleList()
        self.channel_mix = nn.ModuleList()

        for _ in range(block_depth):
            self.conv_low.append(nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=num_groups))
            for i in range(max_depth - num_high_freq_terms):
                self.conv_high[i].append(nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=num_groups))
            self.conv_spatial.append(nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=num_groups))
            self.channel_mix.append(nn.Conv3d(channels, channels, kernel_size=1))

        self.conv_combined = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm3d(channels)
        self.activation = nn.LeakyReLU(0.05)

    def forward(self, x):
        # Wavelet transform
        wavelet_coeffs = self.dwt_forward_3d_lap(x)
        low_freq = wavelet_coeffs[0]
        high_freq = wavelet_coeffs[1][self.num_high_freq_terms :]

        # Process low-frequency component
        for i in range(len(self.conv_low)):
            low_freq = self.conv_low[i](low_freq)
            low_freq = self.channel_mix[i](low_freq)

        # Process high-frequency components
        high_freq_conv = []
        for i in range(self.num_high_freq_terms):
            hf_conv = high_freq[i]
            for j in range(len(self.conv_high[i])):
                hf_conv = self.conv_high[i][j](hf_conv)
                hf_conv = self.channel_mix[j](hf_conv)
            high_freq_conv.append(hf_conv)

        # Inverse wavelet transform
        wavelet_coeffs = [None, None]
        wavelet_coeffs[0] = low_freq
        wavelet_coeffs[1] = [None] * self.num_high_freq_terms + high_freq_conv
        wavelet_output = self.dwt_inverse_3d_lap(wavelet_coeffs)

        # Spatial convolution
        spatial_conv = x
        for i in range(len(self.conv_spatial)):
            spatial_conv = self.conv_spatial[i](spatial_conv)
            spatial_conv = self.channel_mix[i](spatial_conv)
            spatial_conv = self.activation(spatial_conv)

        # Combine the results and perform convolution
        combined = spatial_conv + wavelet_output
        combined = self.conv_combined(combined)
        combined = self.batch_norm(combined)
        combined = self.activation(combined)

        return combined + x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class Model(nn.Module):
    def __init__(self, block_depths, block_channels, block_groups=None, num_high_freq_terms=2):
        super(Model, self).__init__()
        self.wavelet_type = "sym5"
        self.max_depth = 4
        if block_groups is None:
            block_groups = [1] * len(block_depths)

        self.blocks = nn.ModuleList()
        in_channels = 3
        for depth, channels, group in zip(block_depths, block_channels, block_groups):
            transition = TransitionBlock(in_channels, channels)
            self.blocks.append(transition)
            block = WaveletBlock(channels, self.wavelet_type, self.max_depth, depth, num_high_freq_terms, group)
            self.blocks.append(block)
            in_channels = channels

        self.channel_wise = nn.Conv3d(block_channels[-1], 6, kernel_size=1)

    def forward(self, yeex, yeey, yeez):
        x = torch.stack((yeex, yeey, yeez), dim=1)  # Concatenate input tensors
        for block in self.blocks:
            x = block(x)
        x = self.channel_wise(x)
        return x.permute(0, 2, 3, 4, 1) * 6.0
