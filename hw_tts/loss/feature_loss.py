import torch.nn as nn


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.L1Loss(reduction='mean')

    def forward(self, fmaps_real, fmaps_generated):
        loss = 0
        for fmap_r, fmap_g in zip(fmaps_real, fmaps_generated):
            for r, g in zip(fmap_r, fmap_g):
                loss += self.loss_function(r, g)

        return loss
