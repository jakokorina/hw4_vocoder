import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = F.mse_loss

    def forward(self, outputs):
        cum_loss = 0
        for dg in outputs:
            loss = self.loss_function(dg, torch.tensor([1.0], device=outputs.device))
            cum_loss += loss

        return cum_loss
