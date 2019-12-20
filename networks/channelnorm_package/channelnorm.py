import torch
from torch.nn.modules.module import Module

class ChannelNorm(Module):

    def __init__(self, norm_deg: int = 2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        """
        Applies Lp normalization across channels

        Args:

        input1: tensor of size (B, C, H, W) - data to be normalized
        Return:

        output: tensor of size (B, 1, H, W) - data normalized across channels

        """
        return torch.sum(
            input1 ** self.norm_deg, dim=1, keepdim=True) ** (1.0 / self.norm_deg)
        
