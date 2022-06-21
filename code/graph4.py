import torch
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

batch_size = 128
num_epochs = 10
N, D_in, H, D_out = 640, 4096, 2048, 1024

def evaluate(model, device):

    # Placeholders used for capture
    test_inputs = torch.randn(N, D_in, device=local_rank)
    test_labels = torch.randn(N, D_out, device=local_rank)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in zip(test_inputs, test_labels):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    dist = torch.distributed.init_process_group(backend="nccl", world_size=world_size, 
            rank=rank)

    # torch.cuda.set_device(local_rank)

    device = torch.device("cuda:{}".format(local_rank))

    model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                torch.nn.Dropout(p=0.2),
                                torch.nn.Linear(H, D_out),
                                torch.nn.Dropout(p=0.1)).cuda()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
            ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                output_device=local_rank)
    torch.cuda.current_stream().wait_stream(s)

    criterion = torch.nn.CrossEntropyLoss()

    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Placeholders used for capture
    static_input = torch.randn(N, D_in, device=local_rank)
    static_target = torch.randn(N, D_out, device=local_rank)
    

    # warmup
    # Uses static_input and static_target here for convenience,
    # but in a real setting, because the warmup includes optimizer.step()
    # you must use a few batches of real data.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(11):
            optimizer.zero_grad(set_to_none=True)
            y_pred = ddp_model(static_input)
            loss = criterion(y_pred, static_target)
            loss = loss
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    # Sets grads to None before capture, so backward() will create
    # .grad attributes with allocations from the graph's private pool
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_y_pred = ddp_model(static_input)
        static_loss = criterion(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

    real_inputs = [torch.rand_like(static_input) for _ in range(N)]
    real_labels = [torch.rand_like(static_target) for _ in range(N)]

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        for data, target in zip(real_inputs, real_labels):
                print('..')
                print(data.shape)
                print(target.shape)
                # Fills the graph's input memory with new data to compute on
                static_input.copy_(data)
                static_target.copy_(target)
                # replay() includes forward, backward, and step.
                # You don't even need to call optimizer.zero_grad() between iterations
                # because the captured backward refills static .grad tensors in place.
                g.replay()
                # Params have been updated. static_y_pred, static_loss, and .grad
                # attributes hold values from computing on this iteration's data.
