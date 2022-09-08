from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.batchnorm = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, rank):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if(rank == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, rank):
    model.eval()
    test_loss = 0
    correct = 0
    data_len = 0
    test_loss_tensor = 0
    correct_tensor = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(rank), target.to(rank)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss_tensor += F.nll_loss(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_tensor += pred.eq(target.view_as(pred)).sum()
            data_len += len(data)

    test_loss /= data_len

    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

    if(rank == 0):
        print("Test average loss: {}, correct predictons: {}, total: {}, accuracy: {}% \n".format(test_loss_tensor.item() / len(test_loader.dataset), correct_tensor.item(), len(test_loader.dataset),
             100.0 * correct_tensor.item() / len(test_loader.dataset)))

    #print('\nTest  set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #        test_loss, correct, data_len,
    #        100. * correct / data_len))


def demo_basic(rank, world_size, args, use_cuda):

    #--------------- Setup -------------#
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    #-----------------------------------#

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # ---------------- Data ------------#
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
                                               download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)    

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
                                               download=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,            
        num_workers=0,
        pin_memory=True,
        sampler=test_sampler)

    #------------- MODEL LOAD----------------------#

    model = Net().to(rank)

    #model.load_state_dict(torch.load('/home/centos/tf2/srijith/mnist_ddp.pt', map_location=torch.device(rank)), strict=False)

    model = DDP(model, device_ids=[rank])

    model.eval()

    test(model, device, test_loader, rank)

    #-----------------------------------------------#

    cleanup()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpus', type=int, default=1, metavar='N',
                        help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    world_size = args.gpus

    if torch.cuda.device_count() > 1:
      print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")

    #########################################################
    mp.spawn(demo_basic, args=(world_size, args, use_cuda), nprocs=world_size, join=True)    
    #########################################################


if __name__ == '__main__':
    print(torch.cuda.device_count())
    main()