import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, spectral_norm


class SDiscriminator(nn.Module):
    def __init__(self, use_spectral=False, leaky_coef: float = 0.1):
        super().__init__()
        norm = spectral_norm if use_spectral else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

        self.leaky = leaky_coef

    def forward(self, input: torch.Tensor):
        feature_map = []
        output = input

        for conv in self.convs:
            output = F.leaky_relu(conv(output), self.leaky)
            feature_map.append(output.clone())

        output = self.conv_post(output)
        feature_map.append(output)
        output = torch.flatten(output, 1, -1)

        return output, feature_map


class MultiSDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SDiscriminator(use_spectral=True),
            SDiscriminator(),
            SDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, input: torch.Tensor):
        outputs = []
        feature_maps = []
        output = input
        for i, d in enumerate(self.discriminators):
            if i > 0:
                output = self.meanpools[i - 1](input)
            out, fmap = d(output)
            outputs.append(out)
            feature_maps.append(fmap)

        return outputs, feature_maps
