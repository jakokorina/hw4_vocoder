import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation, leaky_coef: float = 0.1):
        super().__init__()

        self.conv1 = nn.ModuleList([])
        self.conv2 = nn.ModuleList([])

        for d_i in dilation:
            self.conv1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d_i, padding="SAME")))
            self.conv2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding="SAME")))

        self.leaky = leaky_coef
        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0.0, std: float = 1.0):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                layer.weight.data.normal_(mean, std)

    def forward(self, input: torch.Tensor):
        output = input
        for conv1, conv2 in zip(self.conv1, self.conv2):
            # Conv1d
            x = conv1(F.leaky_relu(output, self.leaky))
            x = conv2(F.leaky_relu(x, self.leaky))

            # Residual connection
            output = x + output

        return output


class MRFBlock(nn.Module):
    def __init__(self,
                 channel: int,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 leaky_coef: float = 0.1):
        super().__init__()

        # Residual blocks
        self.res_blocks = nn.ModuleList([])
        for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            self.res_blocks.append(ResBlock(channel, k, d, leaky_coef))

    def forward(self, input: torch.Tensor):
        output = None
        for block in self.res_blocks:
            if output is None:
                output = block(input)
            else:
                output += block(input)

        output = output / len(self.res_blocks)
        return output
