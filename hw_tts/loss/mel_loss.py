import torch.nn as nn
import torch.nn.functional as F


class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.L1Loss(reduction='mean')

    def forward(self, mel_real, mel_generated):
        if mel_real.shape[-1] != mel_generated.shape[-1]:
            mel_real = F.pad(mel_real, (0, mel_generated.shape[-1] - mel_real.shape[-1]))
        return self.loss_function(mel_real, mel_generated)
