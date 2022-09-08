import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import random
import numpy as np
import time
import torch.distributed as dist


def compute_padding(kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return pad_beg, pad_end
    

class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        # compute padding here
        pad_beg, pad_end = compute_padding(kernel_size, rate=dilation)
        
        # layers
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, (pad_beg, pad_beg), dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        #x = F.pad(x, self.padding)
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

def main():
    
    torch.distributed.init_process_group(backend="nccl")
    
    torch.backends.cudnn.benchmark = True
    enable_amp = True
    num_warmup_steps=10
    num_steps=100
    B = 1
    C = 728
    H = 48
    W = 72
    
    x = torch.randn(B, C, H, W).cuda().contiguous(memory_format=torch.channels_last)
    conv = SeparableConv2d_same(inplanes=C, planes=C, kernel_size=3, stride=1, dilation=1, bias=False).cuda().to(memory_format=torch.channels_last)
    conv = torch.jit.script(conv)
    gscaler = amp.GradScaler(enabled = enable_amp) 

    # graph capture
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(20):
            with amp.autocast(enabled = enable_amp):
                static_output = conv(x)
                loss = torch.sum(static_output)
            gscaler.scale(loss).backward()
                                                            
        # wait and clean up
        capture_stream.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        # create graph
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        with amp.autocast(enabled = enable_amp):
            static_output = conv(x)
            static_loss = torch.sum(static_output)
        gscaler.scale(static_loss).backward()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(capture_stream)   

    # run
    torch.cuda.synchronize()
    t_time = time.perf_counter_ns()
    for _ in range(num_steps):
        graph.replay()
    torch.cuda.synchronize()
    t_time = time.perf_counter_ns() - t_time

    print(f"Time for separable conv time ({num_steps} steps): {t_time*1e-6} ms")


if __name__ == "__main__":
    
    main()