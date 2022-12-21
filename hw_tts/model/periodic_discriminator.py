import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm


def compute_pad(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class PDiscriminator(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, leaky_coef: float = 0.1):
        super().__init__()

        self.period = period
        self.leaky = leaky_coef

        padding = compute_pad(5)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(padding, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.post_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, input: torch.Tensor):
        feature_map = []

        # Reshape from 1d to 2d
        batch, channels, t = input.shape
        if t % self.period != 0:  # pad first
            input = F.pad(input, (0, self.period - (t % self.period)), "reflect")

        output = input.view(batch, channels, -1, self.period)

        for conv in self.convs:
            output = F.leaky_relu(conv(output), self.leaky)
            feature_map.append(output)

        output = self.conv_post(output)
        feature_map.append(output)
        output = torch.flatten(output, 1, -1)

        return output, feature_map


class MultiPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PDiscriminator(2),
            PDiscriminator(3),
            PDiscriminator(5),
            PDiscriminator(7),
            PDiscriminator(11)
        ])

    def forward(self, input: torch.Tensor):
        outputs = []
        feature_maps = []
        for discriminator in self.discriminators:
            out, fmap = discriminator(input)
            outputs.append(out)
            feature_maps.append(fmap)

        return outputs, feature_maps
