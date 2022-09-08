import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def get_resources():

    if os.environ.get('OMPI_COMMAND'):
        # from mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else:
        # from slurm
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])

    return rank, local_rank, world_size


rank, local_rank, world_size = get_resources()

num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

batch_size = 32
learning_rate = 0.1

num_epochs = 10

dist.init_process_group("nccl", rank=rank, world_size=world_size)

if rank == 0:
    print("world_size", dist.get_world_size())

device = torch.device("cuda:{}".format(local_rank))

#print(f"rank {rank} local_rank {local_rank} device {device}")

# Prepare dataset and dataloader
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=True, download=False, transform=transform) 
test_set = torchvision.datasets.CIFAR10(root="./data/cifar-10", train=False, download=False, transform=transform)

torch.cuda.set_device(local_rank)

train_sampler = DistributedSampler(dataset=train_set)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=num_workers)

model = torchvision.models.resnet18(pretrained=False)
model = model.to(device)

ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

criterion = torch.nn.CrossEntropyLoss()

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

epochs_time = time.time()

for epoch in range(num_epochs):
    dt = time.time()

    ddp_model.train()


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Waits for everything to finish running
    torch.cuda.synchronize()
    start.record()

    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = ddp_model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    end.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"Elapsed time: {start.elapsed_time(end)*1e-6}")
        accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
        print(f"epoch {epoch} rank {rank} world_size {world_size} accuracy {accuracy}")

    dt = time.time() - dt

    if rank == 0:
        print(f"epoch {epoch} rank {rank} world_size {world_size} time {dt:2f}")

epochs_time = time.time() - epochs_time

if rank == 0:
    print(f"walltime: rank {rank} world_size {world_size} time {epochs_time:2f}")

dist.destroy_process_group()

