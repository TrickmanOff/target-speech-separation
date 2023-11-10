"""
https://arxiv.org/abs/2005.04686
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from tss_lib.model.base_model import BaseModel


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels_dim,), 1.))
        self.beta = nn.Parameter(torch.full((channels_dim,), 0.))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        input:  (batch_dim, channels_dim, features_dim)
        output: (batch_dim, channels_dim, features_dim)
        """
        mean = torch.mean(x, (-2, -1), keepdim=True)  # (batch_dim, channels_dim, features_dim)
        var = torch.var(x, (-2, -1), unbiased=False, keepdim=True)  # (batch_dim, channels_dim, features_dim)
        normed_x = (x - mean) / torch.sqrt(var + self.eps)
        return normed_x * self.weight[None, :, None] + self.beta[None, :, None]


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels_dim: int, *args, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(channels_dim, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        input:  (batch_dim, channels_dim, features_dim)
        output: (batch_dim, channels_dim, features_dim)
        """
        return self.ln(x.transpose(-1, -2)).transpose(-1, -2)


class TemporalEncoder(nn.Module):
    def __init__(self, min_L: int, L: int, stride: int, N: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=N, kernel_size=L, stride=stride)
        self.relu = nn.ReLU()

        self.min_L = min_L
        self.stride = stride
        self.L = L

    def forward(self, x: Tensor) -> Tensor:
        """
        input:  (batch_dim, 1, time_dim)
        output: (batch_dim, N, K)
        """
        T = x.shape[-1]
        T = self.min_L * ((T + self.min_L - 1) // self.min_L)  # for correct inverse convolution
        K = (T - self.min_L) // self.stride + 1
        rp = max(0, self.stride * (K - 1) - T + self.L)
        x = F.pad(x, (0, rp), mode='constant', value=0)
        return self.relu(self.conv(x))


class SpeechEncoder(nn.Module):
    def __init__(self, L1: int = 40, L2: int = 160, L3: int = 320, N: int = 256):
        super().__init__()
        assert L1 <= L2 <= L3
        assert L1 % 2 == 0
        stride = L1 // 2
        self.short_encoder = TemporalEncoder(L1, L1, stride, N)
        self.mid_encoder = TemporalEncoder(L1, L2, stride, N)
        self.long_encoder = TemporalEncoder(L1, L3, stride, N)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        input:
            x:  (batch_dim, 1, T)
        output:
            ei: (batch_dim, N, K)
        """
        e1 = self.short_encoder(x)  # (batch_dim, N, K)
        e2 = self.mid_encoder(x)    # (batch_dim, N, K)
        e3 = self.long_encoder(x)   # (batch_dim, N, K)
        e = torch.concat((e1, e2, e3), dim=-2)
        return {
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'e': e
        }


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int,
                 de_cnn_dilation: int,
                 de_cnn_kernel_size: int = 3,
                 hidden_channels: int = 512,
                 ref_embed_dim: int = 0):
        """
        notations from the original paper:
            de_cnn_kernel_size - Q
            in_channels - O
            hidden_channels - P
            ref_embed_dim - D
        """
        super().__init__()
        conv1 = nn.Conv1d(in_channels+ref_embed_dim, hidden_channels, kernel_size=1)
        act1 = nn.PReLU()
        gln1 = GlobalLayerNorm(hidden_channels)
        assert de_cnn_dilation * (de_cnn_kernel_size - 1) % 2 == 0
        depthwise_conv_padding = de_cnn_dilation * (de_cnn_kernel_size - 1) // 2
        depthwise_conv = nn.Conv1d(hidden_channels,
                                   hidden_channels,
                                   kernel_size=de_cnn_kernel_size,
                                   dilation=de_cnn_dilation,
                                   padding=depthwise_conv_padding)
        act2 = nn.PReLU()
        gln2 = GlobalLayerNorm(hidden_channels)
        conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.block = nn.Sequential(
            conv1, act1, gln1,
            depthwise_conv,
            act2, gln2,
            conv2
        )

    def forward(self, x: Tensor, speaker_embed: Optional[Tensor] = None) -> Tensor:
        """
        input:            (batch_dim, in_channels, time_dim)
        [speaker_embed]:  (batch_dim, ref_embed_dim)
        output:           (batch_dim, in_channels, time_dim)
        """
        conv_input = x
        if speaker_embed is not None:
            speaker_embed = torch.tile(speaker_embed.unsqueeze(-1), (1, 1, x.shape[-1]))
            conv_input = torch.concat((x, speaker_embed), dim=1)
        return x + self.block(conv_input)


class TCNStack(nn.Module):
    def __init__(self, in_channels: int, ref_embed_dim: int, num_blocks: int = 8, **tcn_kwargs):
        super().__init__()
        self.block_with_embed = TCNBlock(in_channels, de_cnn_dilation=1, ref_embed_dim=ref_embed_dim, **tcn_kwargs)
        other_blocks = nn.ModuleList([
            TCNBlock(in_channels, de_cnn_dilation=2**block_idx, **tcn_kwargs)
            for block_idx in range(1, num_blocks)
        ])
        self.other_blocks = nn.Sequential(*other_blocks)

    def forward(self, x: Tensor, speaker_embed: Tensor) -> Tensor:
        """
        input:            (batch_dim, in_channels, time_dim)
        [speaker_embed]:  (batch_dim, ref_embed_dim)
        output:           (batch_dim, in_channels, time_dim)
        """
        x = self.block_with_embed(x, speaker_embed)
        return self.other_blocks(x)


class SpeakerExtractor(nn.Module):
    def __init__(self, in_channels: int, ref_embed_dim: int, to_channels: int = 256, num_tcn_stacks: int = 4,
                 n_parts: int = 3, **tcn_kwargs):
        super().__init__()
        self.n_parts = n_parts
        self.ln = ChannelLayerNorm(n_parts * in_channels)
        self.conv = nn.Conv1d(n_parts * in_channels, to_channels, kernel_size=1)
        self.tcn_stacks = nn.ModuleList([
            TCNStack(to_channels, ref_embed_dim, **tcn_kwargs)
            for _ in range(num_tcn_stacks)
        ])

        self.masks_convs = nn.ModuleList([
            nn.Conv1d(to_channels, in_channels, kernel_size=1)
            for _ in range(3)
        ])

    def forward(self, e: Tensor, ref_embed: Tensor, **e_parts: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        input:
            e:  (batch_dim, 3*in_channels, K)
            ei: (batch_dim, in_channels, K) for i = 1, 2, ..., n_parts

        output:
            {
                si: (batch_dim, in_channels, K) for i = 1, 2, ..., n_parts
            }
        """
        normed_e = self.ln(e)
        e_tilde = self.conv(normed_e)
        tcn_out = e_tilde
        for tcn_stack in self.tcn_stacks:
            tcn_out = tcn_stack(tcn_out, ref_embed)

        masks = {
            f'm{i+1}': F.relu(conv(tcn_out))
            for i, conv in enumerate(self.masks_convs)
        }

        result = {
            f's{i+1}': masks[f'm{i+1}'] * e_parts[f'e{i+1}']
            for i in range(self.n_parts)
        }

        return result


class SpeechDecoder(nn.Module):
    def __init__(self, L1: int = 40, L2: int = 160, L3: int = 320, in_channels: int = 256):
        super().__init__()
        assert L1 <= L2 <= L3
        assert L1 % 2 == 0
        stride = L1 // 2
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose1d(in_channels, 1, kernel_size=L, stride=stride)
            for L in [L1, L2, L3]
        ])

    def forward(self, **s_parts: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        input:
            si: (batch_dim, N, K)

        output:
            wi: (batch_dim, 1, >=T)
        """
        return {
            f'w{i+1}': deconv(s_parts[f's{i+1}'])
            for i, deconv in enumerate(self.deconvs)
        }


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, maxpooling_kernel: int = 3):
        # TODO: try to vary the number of channels
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.PReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels)
        )
        self.block2 = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(maxpooling_kernel)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x:      (batch_dim, in_channels, features_dim)

        output: (batch_dim, out_channels, ...)
        """
        out1 = x + self.block1(x)  # (batch_dim, in_channels, features_dim)
        return self.block2(out1)


class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels: int,
                 ref_embed_dim: int, num_classes: int,
                 num_resnet_blocks: int = 3,
                 hidden_channels_1: int = 256):
        super().__init__()
        blocks = [
            ChannelLayerNorm(in_channels),
            nn.Conv1d(in_channels, hidden_channels_1, kernel_size=1)
        ]
        blocks += [
            ResNetBlock(hidden_channels_1)
            for _ in range(num_resnet_blocks)
        ]
        blocks += [
            nn.Conv1d(hidden_channels_1, ref_embed_dim, kernel_size=1)
        ]
        self.block = nn.Sequential(
            *blocks
        )

        self.linear = nn.Linear(ref_embed_dim, num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        x:      (batch_dim, in_channels, features_dim)

        output:
            embed:  (batch_dim, ref_embed_dim)
            logits: (batch_dim, num_classes)
        """
        out = self.block(x)       # (batch_dim, ref_embed_dim, *)
        embed = out.mean(dim=-1)  # (batch_dim, ref_embed_dim)
        logits = self.linear(embed)
        return {
            'embed': embed,
            'logits': logits
        }


class SpexPlus(BaseModel):
    def __init__(self, num_classes: int,
                 N: int = 256, ref_embed_dim: int = 256,
                 O: int = 256, P: int = 512,
                 num_tcn_stacks: int = 4,
                 L1: int = 40, L2: int = 160, L3: int = 320,
                 **tcn_kwargs):
        super().__init__()
        self.speech_encoder = SpeechEncoder(L1, L2, L3, N)
        tcn_kwargs['hidden_channels'] = P
        self.speaker_extractor = SpeakerExtractor(in_channels=N, ref_embed_dim=ref_embed_dim,
                                                  to_channels=O, num_tcn_stacks=num_tcn_stacks,
                                                  n_parts=3, **tcn_kwargs)
        self.speaker_encoder = SpeakerEncoder(in_channels=3*N, ref_embed_dim=ref_embed_dim,
                                              num_classes=num_classes, hidden_channels_1=O)
        self.speech_decoder = SpeechDecoder(L1, L2, L3, in_channels=N)

    def forward(self, mix: Tensor, ref: Tensor) -> Dict[str, Tensor]:
        """
        input:
            mix:  (batch_dim, 1, T)
            ref:  (batch_dim, 1, T_ref)

        output:
            si:   (batch_dim, 1, T) - predicted target audio from mix
                for i = 1, 2, 3
            speakers_logits: (batch_dim, num_classes) - predicted logits of target speaker
        """

        encoded_mix = self.speech_encoder(mix)
        encoded_ref = self.speech_encoder(ref)['e']  # (batch_dim, 3*N, K)
        speaker_encoder_res = self.speaker_encoder(encoded_ref)
        ref_embed, speakers_logits = speaker_encoder_res['embed'], speaker_encoder_res['logits']
        encoded_pred_audios = self.speaker_extractor(ref_embed=ref_embed, **encoded_mix)
        pred_audios = self.speech_decoder(**encoded_pred_audios)
        pred_audios = {key: audio[:, :, :mix.shape[-1]] for key, audio in pred_audios.items()}
        result = pred_audios | {'speakers_logits': speakers_logits}
        return result
