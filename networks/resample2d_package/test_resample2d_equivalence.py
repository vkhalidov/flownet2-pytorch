import os
import sys
import time
import unittest
import torch

RESAMPLE2D_CUDA_PATH = os.path.join(os.path.dirname(__file__), "build", "lib.linux-x86_64-3.7")
sys.path.append(RESAMPLE2D_CUDA_PATH)

from resample2d import Resample2d as Resample2dPt
from resample2d_vanilla import Resample2d as Resample2dCu

class Resample2dTest(unittest.TestCase):

    TOLERANCE = 1e-5

    def _create_data_01(self, B, C, H, W):
        gpu0 = torch.device("cuda:0")
        data = []
        flow = []
        for b in range(B):
            data_b = torch.zeros((C, H, W), device=gpu0)
            data_b[:, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1
            data.append(data_b)
            flow.append(torch.ones((2, H, W), device=gpu0))
        data_res = torch.stack(data, 0).contiguous()
        flow_res = torch.stack(flow, 0).contiguous()
        return data_res, flow_res

    def _create_data_02(self, B, C, H, W):
        gpu0 = torch.device("cuda:0")
        data = torch.rand((B, C, H, W), device=gpu0)
        flow = torch.rand((B, 2, H, W), device=gpu0)
        return data, flow

    def _create_data_03(self, B, C, H, W, dx, dy):
        gpu0 = torch.device("cuda:0")
        data = []
        flow = []
        for b in range(B):
            data_b = torch.zeros((C, H, W), device=gpu0)
            data_b[:, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1
            data.append(data_b)
            flow.append(
                torch.ones((2, H, W), device=gpu0)
                * torch.Tensor([dx, dy])[:, None, None].expand(2, H, W).to(device=gpu0))
        data_res = torch.stack(data, 0).contiguous()
        flow_res = torch.stack(flow, 0).contiguous()
        return data_res, flow_res

    def _create_data_04(self, B, C, H, W, dx, dy):
        gpu0 = torch.device("cuda:0")
        data = []
        flow = []
        for b in range(B):
            data_b = torch.zeros((C, H, W), device=gpu0)
            data_b[:, 0,  :] = 1
            data_b[:, :,  0] = 1
            data_b[:, H - 1,  :] = 1
            data_b[:, :,  W - 1] = 1
            data.append(data_b)
            flow.append(
                torch.ones((2, H, W), device=gpu0)
                * torch.Tensor([dx, dy])[:, None, None].expand(2, H, W).to(device=gpu0))
        data_res = torch.stack(data, 0).contiguous()
        flow_res = torch.stack(flow, 0).contiguous()
        return data_res, flow_res

    def _create_data_05(self, B, C, H, W):
        y, x = torch.meshgrid(torch.linspace(-2, 2, H), torch.linspace(-2, 2, W))
        data = []
        flow = []
        for b in range(B):
            xc = float(b) / B
            yc = 1 - float(b) / (2 * B)
            d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2)
            data.append(d.unsqueeze(2).repeat(C, 1, 1).float())
            flow.append(torch.ones((2, H, W)))
        gpu0 = torch.device("cuda:0")
        data_res = torch.stack(data, 0).to(device=gpu0)
        flow_res = torch.stack(flow, 0).to(device=gpu0)
        return data_res, flow_res

    def test_forward_01_1_1_10_10_bilinear_ks1(self):
        data, flow = self._create_data_01(1, 1, 10, 10)
        resample2d_pt = Resample2dPt()
        out_pt = resample2d_pt(data, flow).cpu()
        resample2d_cu = Resample2dCu()
        out_cu = resample2d_cu(data, flow).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_01_5_3_10_10_bilinear_ks1(self):
        data, flow = self._create_data_01(5, 3, 10, 10)
        resample2d_pt = Resample2dPt()
        out_pt = resample2d_pt(data, flow).cpu()
        resample2d_cu = Resample2dCu()
        out_cu = resample2d_cu(data, flow).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_02_1_1_10_10_bilinear_ks1(self):
        torch.manual_seed(0)
        data, flow = self._create_data_02(1, 1, 10, 10)
        resample2d_pt = Resample2dPt()
        out_pt = resample2d_pt(data, flow).cpu()
        resample2d_cu = Resample2dCu()
        out_cu = resample2d_cu(data, flow).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_02_5_3_10_10_bilinear_ks1(self):
        torch.manual_seed(0)
        data, flow = self._create_data_02(5, 3, 10, 10)
        resample2d_pt = Resample2dPt()
        out_pt = resample2d_pt(data, flow).cpu()
        resample2d_cu = Resample2dCu()
        out_cu = resample2d_cu(data, flow).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_03_1_1_10_10_bilinear_ks1(self):
        data, flow = self._create_data_03(1, 1, 10, 10, 0.5, 0.3)
        resample2d_pt = Resample2dPt()
        out_pt = resample2d_pt(data, flow).cpu()
        resample2d_cu = Resample2dCu()
        out_cu = resample2d_cu(data, flow).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_04_1_1_10_10_bilinear_ks1(self):
        data, flow = self._create_data_04(1, 1, 10, 10, 0.5, 0.3)
        resample2d_pt = Resample2dPt()
        out_pt = resample2d_pt(data, flow).cpu()
        resample2d_cu = Resample2dCu()
        out_cu = resample2d_cu(data, flow).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

def perf_test_data(B, C, H, W):
    gpu0 = torch.device("cuda:0")
    data = torch.rand((B, C, H, W), device=gpu0)
    flow = torch.rand((B, 2, H, W), device=gpu0)
    return data, flow

def perf_test(resample2d_obj, desc_str):
    N_WARMUP = 10
    N_ITERS = 100
    torch.manual_seed(0)
    data, flow = perf_test_data(5, 3, 100, 100)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    timings = []
    for i in range(N_ITERS):
        start = time.perf_counter()
        out = resample2d_obj(data, flow).cpu()
        end = time.perf_counter()
        timings.append(end - start)
    print(
        "Average " + desc_str + " timings: ",
        sum(timings[N_WARMUP:]) / (N_ITERS - N_WARMUP))

def perf_test_pt():
    resample2d_obj = Resample2dPt()
    perf_test(resample2d_obj, "Resample2D PyTorch")

def perf_test_cu():
    resample2d_obj = Resample2dCu()
    perf_test(resample2d_obj, "Resample2D Custom")


if __name__ == "__main__":
    perf_test_pt()
    perf_test_cu()
    unittest.main()
