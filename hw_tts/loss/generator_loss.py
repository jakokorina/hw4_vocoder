import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = F.mse_loss

    def forward(self, outputs):
        cum_loss = 0
        gen_losses = []
        for dg in outputs:
            loss = self.loss_function(dg, torch.tensor([1.0]))
            gen_losses.append(loss)
            cum_loss += loss

        return cum_loss, gen_losses
