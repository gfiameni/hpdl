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

batch_size = 1000
num_epochs = 10
channels_last = False

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
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    dist = torch.distributed.init_process_group(backend="nccl", world_size=world_size, 
            rank=rank)

    # torch.cuda.set_device(local_rank)

    device = torch.device("cuda:{}".format(local_rank))

    model = torchvision.models.resnet18(pretrained=False).cuda()

    print(model)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=True, download=False, transform=transform) 
    test_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=False, download=False, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()

    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        t0 = time.time()
        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()
        
        for data in train_loader:
            if torch.backends.cudnn.version() >= 7603 and channels_last:
                inputs, labels = data[0].to(device, memory_format=torch.channels_last), data[1].to(device, memory_format=torch.channels_last)
            else:
                inputs, labels = data[0].to(device), data[1].to(device)

            print('Inputs ', inputs.shape)
            print('Labels ', labels.shape)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
        print("Time {} seconds".format(round(time.time() - t0, 2)))