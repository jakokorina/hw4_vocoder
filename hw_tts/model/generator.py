import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from .mrf import MRFBlock


class Generator(nn.Module):
    def __init__(self,
                 upsample_rates: tuple[int, ...],
                 upsample_kernel_sizes: tuple[int, ...],
                 upsample_init_channel: int,
                 resblock_kernel_sizes: tuple[int, ...],
                 resblock_dilation_sizes: tuple[tuple[int, ...], ...],
                 leaky_coef: float = 0.1
                 ):
        super().__init__()
        # Pre-conv
        self.pre_conv = weight_norm(nn.Conv1d(80, upsample_init_channel, 7, 1, padding=3))
        self.leaky = leaky_coef

        # Conv1dT Blocks
        self.convT = nn.ModuleList([])
        for i, (kernel, up) in enumerate(zip(upsample_kernel_sizes, upsample_rates)):
            self.convT.append(weight_norm(nn.ConvTranspose1d(
                upsample_init_channel // (2 ** i),
                upsample_init_channel // (2 ** (i + 1)),
                kernel,
                up,
                padding=(kernel - up) // 2)
            ))
        self._reset_parameters(self.convT)

        # MRF Blocks
        self.mrf_blocks = nn.ModuleList([])
        for i in range(len(self.convT)):
            channels = upsample_init_channel // (2 ** (i + 1))
            self.mrf_blocks.append(MRFBlock(channels, resblock_kernel_sizes, resblock_dilation_sizes, leaky_coef))

        self.post_conv = weight_norm(nn.Conv1d(upsample_init_channel // (2 ** (len(self.convT))), 1, 7, 1, padding=3))
        self._reset_parameters(self.post_conv)

    @staticmethod
    def _reset_parameters(module: nn.Module, mean=0.0, std=1.0):
        for layer in module.modules():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.normal_(mean, std)

    def forward(self, input: torch.Tensor):
        output = self.pre_conv(input)

        for convT, mrf in zip(self.convT, self.mrf_blocks):
            output = convT(F.leaky_relu(output))
            output = mrf(output)

        output = self.post_conv(F.leaky_relu(output, self.leaky))
        output = F.tanh(output)
        return output
