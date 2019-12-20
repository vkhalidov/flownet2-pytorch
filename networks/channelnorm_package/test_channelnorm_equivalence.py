import os
import sys
import time
import unittest
import torch

CHANNELNORM_CUDA_PATH = os.path.join(os.path.dirname(__file__), "build", "lib.linux-x86_64-3.7")
sys.path.append(CHANNELNORM_CUDA_PATH)

from channelnorm import ChannelNorm as ChannelNormPt
from channelnorm_vanilla import ChannelNorm as ChannelNormCu

class ChannelNormTest(unittest.TestCase) :

    TOLERANCE = 1e-5

    def _create_data_01(self, B, C, H, W, seed=0):
       torch.manual_seed(seed)
       gpu0 = torch.device("cuda:0")
       data = torch.rand((B, C, H, W), device=gpu0)
       return data

    def _generic_test(self, create_data):
        data = create_data()
        cn_pt = ChannelNormPt()
        out_pt = cn_pt(data)
        cn_cu = ChannelNormCu()
        out_cu = cn_cu(data)
        self.assertTrue((out_pt - out_cu).abs().max() < self.TOLERANCE)

    def test_forward_01_2_3_4_5_0(self):
        create_data = lambda: self._create_data_01(2, 3, 4, 5)
        self._generic_test(create_data)

    def test_forward_01_5_30_100_100_42(self):
        create_data = lambda: self._create_data_01(5, 30, 100, 100, 42)
        self._generic_test(create_data)

def perf_test_data(B, C, H, W):
    gpu0 = torch.device("cuda:0")
    data = torch.rand((B, C, H, W), device=gpu0)
    return data

def perf_test(cn_obj, desc_str):
    N_WARMUP = 10
    N_ITERS = 100
    torch.manual_seed(0)
    data = perf_test_data(5, 30, 100, 100)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    timings = []
    for i in range(N_ITERS):
        start = time.perf_counter()
        out = cn_obj(data).cpu()
        end = time.perf_counter()
        timings.append(end - start)
    print(
        "Average " + desc_str + " timings: ",
        sum(timings[N_WARMUP:]) / (N_ITERS - N_WARMUP))

def perf_test_pt():
    cn_obj = ChannelNormPt()
    perf_test(cn_obj, "ChannelNorm PyTorch")

def perf_test_cu():
    cn_obj = ChannelNormCu()
    perf_test(cn_obj, "ChannelNorm Custom")

if __name__ == "__main__":
    perf_test_pt()
    perf_test_cu()
    unittest.main()
