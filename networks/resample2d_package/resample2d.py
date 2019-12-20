import torch
from torch.nn import functional as F
from torch.nn.modules.module import Module

class Resample2d(Module):
    """
    Applies optical flow to the data and performs resampling
    The current PyTorch implementation is ~3 times slower than the custom one.
    It is provided for compatibility with TorchHub
    N.B. Timings are going to change once LazyTensor is out
    """

    def __init__(
            self,
            mode: str = "bilinear",
            padding_mode: str = "border"):
        super(Resample2d, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input1: tensor of size (B, D, H, W) - data to be resampled
            input2: tensor of size (B, 2, H, W) - optical flow data
        Return:
            tensor of size (B, D, H, W) - resampled data
        """
        B = input2.shape[0]
        H, W = input2.shape[2:]
        # creiate a grid of shape (B, H, W, 2)
        # TODO: good candidate for LazyTensor, once it's implemented
        gridx = 2 * torch.arange(W, device=input2.device, dtype=torch.float) / (W - 1) - 1
        gridy = 2 * torch.arange(H, device=input2.device, dtype=torch.float) / (H - 1) - 1
        gridx_expanded = gridx[None, None, :].expand(B, H, W)
        gridy_expanded = gridy[None, :, None].expand(B, H, W)
        gridx_optflow = gridx_expanded + input2[:, 0, :, :] / (W - 1) * 2
        gridy_optflow = gridy_expanded + input2[:, 1, :, :] / (H - 1) * 2
        grid = torch.stack((gridx_optflow, gridy_optflow), dim=3)
        # resample input1 based on the grid
        input1_resampled = F.grid_sample(
            input1, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)
        return input1_resampled
