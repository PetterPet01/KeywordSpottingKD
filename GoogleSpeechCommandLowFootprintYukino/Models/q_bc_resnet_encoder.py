# File: Models/q_bc_resnet_encoder.py
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio  # For deltas

# Assuming these are in the Models directory or accessible in the Python path
from .subspectral_norm import SubSpectralNorm
from .quaternion_layers import QuaternionConv2dASM

DROPOUT_ENC = 0.1  # Default dropout, can be overridden


class NormalBlockEncoder(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT_ENC, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 5) if use_subspectral else nn.BatchNorm2d(n_chan)
        self.f2 = nn.Sequential(
            QuaternionConv2dASM(in_channels=n_chan, out_channels=n_chan, kernel_size=(3, 1), padding="same", stride=1,
                                groups=1),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            QuaternionConv2dASM(in_channels=n_chan, out_channels=n_chan, kernel_size=(1, 3), padding="same", stride=1,
                                groups=1, dilatation=(1, dilation) if dilation != 1 else 1),  # PyTorch uses dilation
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            QuaternionConv2dASM(in_channels=n_chan, out_channels=n_chan, kernel_size=(1, 1), padding=0, stride=1,
                                groups=1),
            nn.Dropout2d(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)
        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)
        return self.activation(x + x1 + x2)


class TransitionBlockEncoder(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT_ENC,
                 use_subspectral: bool = True):
        super().__init__()
        if stride == 1:
            conv = QuaternionConv2dASM(in_channels=out_chan, out_channels=out_chan, kernel_size=(3, 1), padding="same",
                                       groups=1, stride=(1, 1))
        else:
            # Original padding might be specific to input size 40. For stride=2, (H-K+2P)/S + 1. If K=3, (H-3+2P)/2+1.
            # If padding=(1,0) for kernel (3,1), then P_h=1. (H-3+2)/2+1 = (H-1)/2+1.
            # If H=40, (39)/2+1 = 19.5+1=20.5 (problematic). If H=20, (19)/2+1=9.5+1=10.5.
            # padding="same" might be safer if the conv layer supports it, or calculate carefully.
            # The original code used padding=(1,0) for nn.Conv2d with stride=(stride,1) and kernel (3,1).
            # For stride=(2,1), this means padding (1,0) for height, (0,0) for width dimension.
            # For QuaternionConv2dASM, let's assume padding values are (pad_h, pad_w)
            effective_padding = (1, 0)  # (padding_H, padding_W) for kernel (3,1)
            conv = QuaternionConv2dASM(in_channels=out_chan, out_channels=out_chan, kernel_size=(3, 1),
                                       stride=(stride, 1), padding=effective_padding, groups=1)

        norm_layer = SubSpectralNorm(out_chan, 5) if use_subspectral else nn.BatchNorm2d(out_chan)
        self.f2 = nn.Sequential(
            QuaternionConv2dASM(in_channels=in_chan, out_channels=out_chan, stride=(1, 1), kernel_size=(1, 1),
                                padding=0, groups=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )
        self.f1 = nn.Sequential(
            QuaternionConv2dASM(in_channels=out_chan, out_channels=out_chan, stride=(1, 1), kernel_size=(1, 3),
                                padding="same", dilatation=(1, dilation) if dilation != 1 else 1, groups=1),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            QuaternionConv2dASM(in_channels=out_chan, out_channels=out_chan, stride=(1, 1), kernel_size=(1, 1),
                                padding=0, groups=1),
            nn.Dropout2d(dropout)
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)
        return self.activation(x + x1)


class QBcResNetEncoderASM(nn.Module):
    def __init__(self, *, scale: int = 1, dropout: float = DROPOUT_ENC, use_subspectral: bool = True):
        super().__init__()
        # Input to this model will be (batch, 1, n_mels, n_frames)
        # We'll create the 4-channel quaternion input internally
        self.input_conv = QuaternionConv2dASM(in_channels=4, out_channels=16 * scale, stride=(2, 1), kernel_size=(5, 5),
                                              padding=2, groups=1)

        self.t1 = TransitionBlockEncoder(16 * scale, 8 * scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlockEncoder(8 * scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlockEncoder(8 * scale, 12 * scale, dilation=2, stride=2, dropout=dropout,
                                         use_subspectral=use_subspectral)
        self.n21 = NormalBlockEncoder(12 * scale, dilation=2, dropout=dropout, use_subspectral=use_subspectral)

        self.t3 = TransitionBlockEncoder(12 * scale, 16 * scale, dilation=4, stride=2, dropout=dropout,
                                         use_subspectral=use_subspectral)
        self.n31 = NormalBlockEncoder(16 * scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlockEncoder(16 * scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlockEncoder(16 * scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlockEncoder(16 * scale, 20 * scale, dilation=8, dropout=dropout,
                                         use_subspectral=use_subspectral)
        self.n41 = NormalBlockEncoder(20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlockEncoder(20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlockEncoder(20 * scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        # Original dw_conv might reduce dims significantly if padding is 0.
        # kernel_size=(5,5), padding=0. If H_in=5, H_out=1. If H_in=10, H_out=6.
        # Let's check shapes. Input n_mels=40.
        # input_conv (stride 2): 40 -> 20
        # t1: 20 -> 20
        # t2 (stride 2): 20 -> 10
        # t3 (stride 2): 10 -> 5
        # t4: 5 -> 5
        # So, before dw_conv, height is 5.
        # dw_conv (kernel 5, padding 0): 5 -> 1. Width also reduces.
        self.dw_conv = QuaternionConv2dASM(in_channels=20 * scale, out_channels=20 * scale, stride=(1, 1),
                                           kernel_size=(5, 5), padding=0, groups=1)  # Removed dilatation, assume 1
        self.onexone_conv = QuaternionConv2dASM(in_channels=20 * scale, out_channels=32 * scale, stride=(1, 1),
                                                kernel_size=(1, 1), padding=0, groups=1)  # Removed dilatation

        # Output features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.feature_dim = 32 * scale

    def _prepare_quaternion_input(self, data_mel):
        # data_mel shape: (batch, 1, n_mels, n_frames)
        # Detach from graph for delta computation if data_mel requires grad and deltas should not propagate there.
        # However, torchaudio.functional.compute_deltas typically handles this correctly.

        # Squeeze the channel dim for compute_deltas if it expects (batch, n_mels, n_frames)
        if data_mel.dim() == 4 and data_mel.size(1) == 1:
            data_squeezed = data_mel.squeeze(1)
        else:  # If it's already (batch, n_mels, n_frames)
            data_squeezed = data_mel

        data_first = torchaudio.functional.compute_deltas(data_squeezed)
        data_second = torchaudio.functional.compute_deltas(data_first)
        data_third = torchaudio.functional.compute_deltas(data_second)

        # UnSqueeze back to add channel dim if it was removed
        if data_mel.dim() == 4 and data_mel.size(1) == 1:
            data_first = data_first.unsqueeze(1)
            data_second = data_second.unsqueeze(1)
            data_third = data_third.unsqueeze(1)

        # Concatenate along the channel dimension (dim=1)
        quaternion_input = torch.cat([data_mel, data_first, data_second, data_third], dim=1)
        # Expected shape: (batch, 4, n_mels, n_frames)
        return quaternion_input

    def forward(self, x: torch.Tensor):
        # x shape: (batch, 1, n_mels, n_frames), e.g. (B, 1, 40, 431)
        quaternion_x = self._prepare_quaternion_input(x)
        # quaternion_x shape: (batch, 4, n_mels, n_frames)

        x = self.input_conv(quaternion_x)
        x = self.t1(x)
        x = self.n11(x)

        x = self.t2(x)
        x = self.n21(x)

        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)
        # x shape: (batch, 32*scale, H_final, W_final), e.g., (B, 32*s, 1, W_reduced)

        x = self.adaptive_pool(x)  # (batch, 32*scale, 1, 1)
        x = self.flatten(x)  # (batch, 32*scale)
        return x