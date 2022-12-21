import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = F.mse_loss

    def forward(self, real_outputs, generated_outputs):
        loss = 0
        for dr, dg in zip(real_outputs, generated_outputs):
            r_loss = self.loss_function(dr, torch.tensor([1.0]))
            g_loss = self.loss_function(dg, torch.tensor([0.0]))
            loss += (r_loss + g_loss)

        return loss
