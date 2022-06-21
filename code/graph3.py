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

batch_size = 32
num_epochs = 10

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    print('Local rank: ', local_rank)
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    dist = torch.distributed.init_process_group(backend="nccl", world_size=world_size, 
            rank=rank)

    # torch.cuda.set_device(local_rank)

    device = torch.device("cuda:{}".format(local_rank))

    model = torchvision.models.resnet18(pretrained=False).cuda()

    #print(model)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
            ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                output_device=local_rank)
    torch.cuda.current_stream().wait_stream(s)

    #model = model.to(device, memory_format=torch.channels_last) 

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=True, download=True, transform=transform) 
    test_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=False, download=True, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()

    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


    N, C, H, W, O = batch_size, 3, 32, 32, 1000

    static_inputs = torch.randn(N, C, H, W).cuda().contiguous(memory_format=torch.channels_last)
    static_labels = torch.randn(N, O).cuda()

    # Placeholders used for capture
    #static_input = torch.randn(N, C, H, W, device=device)
    #static_labels = torch.randn(N, O, device=device)

    # warmup
    # Uses static_input and static_target here for convenience,
    # but in a real setting, because the warmup includes optimizer.step()
    # you must use a few batches of real data.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(11):
            optimizer.zero_grad(set_to_none=True)
            y_pred = ddp_model(static_inputs)
            loss = criterion(y_pred, static_labels)
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
        ddp_model.train()
        static_y_pred = ddp_model(static_inputs)
        static_loss = criterion(static_y_pred, static_labels)
        static_loss.backward()
        optimizer.step()

    #real_inputs = [torch.rand_like(static_input) for _ in range(batch_size)]
    #real_labels = [torch.rand_like(static_labels) for _ in range(batch_size)]

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        for data in train_loader:
            inputs = data[0].to(device, memory_format=torch.channels_last)

            #labels = data[1].to(device)

            #labels = static_labels
            #print(labels)

            # print(static_labels.shape)
            # Fills the graph's input memory with new data to compute on
            static_inputs.copy_(inputs)
            #static_labels.copy_(labels)
            # replay() includes forward, backward, and step.
            # You don't even need to call optimizer.zero_grad() between iterations
            # because the captured backward refills static .grad tensors in place.
            g.replay()
            # Params have been updated. static_y_pred, static_loss, and .grad
            # attributes hold values from computing on this iteration's data.