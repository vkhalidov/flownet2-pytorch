import math
import torch
from torch.nn import functional as F
from torch.nn.modules.module import Module


class Correlation(Module):
    """
    PyTorch implementation of the custom Correlation operator
    ~1.7x slower than the custom implementation 
    """

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        # input sizes
        B, C, H, W = input1.shape
        # padded input sizes
        piH = H + 2 * self.pad_size
        piW = W + 2 * self.pad_size
        # output sizes
        dr = self.max_displacement // self.stride2
        ds = 2 * dr + 1
        oC = ds ** 2
        kr = (self.kernel_size - 1) // 2
        br = kr + self.max_displacement
        oH = math.ceil((piH - 2 * br) / self.stride1)
        oW = math.ceil((piW - 2 * br) / self.stride1)
        output = torch.zeros((B, oC, oH, oW), device=input1.device)
        tinput2 = torch.zeros_like(input2)
        for dj in range(-dr, dr + 1):
            for di in range(-dr, dr + 1):
                tinput2.zero_()
                tj1 = max(min(0 - dj * self.stride2, H), 0)
                tj2 = max(min(H - dj * self.stride2, H), 0)
                ti1 = max(min(0 - di * self.stride2, W), 0)
                ti2 = max(min(W - di * self.stride2, W), 0)
                if (tj1 >= tj2) or (ti1 >= ti2):
                    continue
                j1 = max(min(0 + dj * self.stride2, H), 0)
                j2 = max(min(H + dj * self.stride2, H), 0)
                i1 = max(min(0 + di * self.stride2, W), 0)
                i2 = max(min(W + di * self.stride2, W), 0)
                tinput2[:, :, tj1:tj2, ti1:ti2] = input2[:, :,j1:j2, i1:i2]
                output[:, (dj + dr) * ds + (di + dr), :, :] = torch.sum(
                    F.avg_pool2d(
                        input1[:, :, ::self.stride1, ::self.stride1]
                        * tinput2[:, :, ::self.stride1, ::self.stride1],
                        self.kernel_size,
                        padding=min(self.pad_size, self.kernel_size // 2)),
                    dim=1) / C
        return output

