import os
import sys
import time
import unittest
import torch

CORRELATION_CUDA_PATH = os.path.join(os.path.dirname(__file__), "build", "lib.linux-x86_64-3.7")
sys.path.append(CORRELATION_CUDA_PATH)

from correlation import Correlation as CorrelationPt
from correlation_vanilla import Correlation as CorrelationCu

class CorrelationTest(unittest.TestCase):

    TOLERANCE = 1e-5

    def _create_data_01(self, B, C, H, W):
        gpu0 = torch.device("cuda:0")
        data = torch.ones((B, C, H, W), device=gpu0)
        kernel = torch.ones((B, C, H, W), device=gpu0) * 0.5
        return data, kernel

    def _create_data_02(self, B, C, H, W):
        gpu0 = torch.device("cuda:0")
        data = torch.rand((B, C, H, W), device=gpu0)
        kernel = torch.rand((B, C, H, W), device=gpu0)
        return data, kernel

    def _generic_test(
            self, create_data, pad_size, kernel_size, max_displacement,
            stride1, stride2, corr_multiply):
        data, kernel = create_data()
        corr_pt = CorrelationPt(
            pad_size=pad_size, kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride1=stride1, stride2=stride2,
            corr_multiply=corr_multiply)
        out_pt = corr_pt(data, kernel).cpu()
        corr_cu = CorrelationCu(
            pad_size=pad_size, kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride1=stride1, stride2=stride2,
            corr_multiply=corr_multiply)
        out_cu = corr_cu(data, kernel).cpu()
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_01_1_1_3_3(self):
        create_data = lambda: self._create_data_01(1, 1, 3, 3)
        self._generic_test(create_data, 3, 1, 3, 1, 2, 1)

    def test_forward_01_1_3_3_3(self):
        create_data = lambda: self._create_data_01(1, 3, 3, 3)
        self._generic_test(create_data, 3, 1, 3, 1, 2, 1)

    def test_forward_02_5_7_100_100(self):
        torch.manual_seed(0)
        create_data = lambda: self._create_data_02(5, 3, 100, 100)
        self._generic_test(create_data, 20, 1, 20, 1, 2, 1)

def perf_test_data(B, C, H, W):
    gpu0 = torch.device("cuda:0")
    data = torch.rand((B, C, H, W), device=gpu0)
    kernel = torch.rand((B, C, H, W), device=gpu0)
    return data, kernel

def perf_test(cn_obj, desc_str):
    N_WARMUP = 10
    N_ITERS = 100
    torch.manual_seed(0)
    data, kernel = perf_test_data(5, 3, 100, 100)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    timings = []
    for i in range(N_ITERS):
        start = time.perf_counter()
        out = cn_obj(data, kernel).cpu()
        end = time.perf_counter()
        timings.append(end - start)
    print(
        "Average " + desc_str + " timings: ",
        sum(timings[N_WARMUP:]) / (N_ITERS - N_WARMUP))

def perf_test_pt():
    cn_obj = CorrelationPt(20, 1, 20, 1, 2, 1)
    perf_test(cn_obj, "Correlation PyTorch")

def perf_test_cu():
    cn_obj = CorrelationCu(20, 1, 20, 1, 2, 1)
    perf_test(cn_obj, "Correlation Custom")

if __name__ == "__main__":
    perf_test_pt()
    perf_test_cu()
    unittest.main()
