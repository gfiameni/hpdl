# Welcome to the lab of the "High performance computing and large-scale models" course


- [School in AI: Deep Learning, Vision and Language for Industry](https://aischools.it/).


# Launching and configuring distributed data parallel applications

This notebook demonstrates how to structure a distributed model training application so it can be launched conveniently on multiple nodes, each with multiple GPUs using PyTorch's distributed
[launcher script](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py).

# Prerequisites
- [PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), the primitives it provides for [writing distributed applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html) as well as training [distributed models](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

The example program in this tutorial uses the [`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) class for training models in a _data parallel_ fashion: multiple workers train the same global model by processing different portions of a large dataset, computing local gradients (aka _sub_-gradients) independently and then collectively synchronizing gradients using the AllReduce primitive. In
HPC terminology, this model of execution is called _Single Program Multiple Data_ or SPMD since the same application runs on all application but each one operates on different portions of the training dataset.

# Application process topologies
A Distributed Data Parallel (DDP) application can be executed on multiple nodes where each node can consist of multiple GPU devices. Each node in turn can run multiple copies of the DDP application, each of which processes its models on multiple GPUs.

Let _N_ be the number of nodes on which the application is running and _G_ be the number of GPUs per node. The total number of application
processes running across all the nodes at one time is called the **World Size**, _W_ and the number of processes running on each node
is referred to as the **Local World Size**, _L_.

Each application process is assigned two IDs: a _local_ rank in \[0,_L_-1\] and a _global_ rank in \[0, _W_-1\].

To illustrate the terminology defined above, consider the case where a DDP application is launched on two nodes, each of which has four GPUs. We would then like each process to span two GPUs each. The mapping of processes to nodes is shown in the figure below:

![ProcessMapping](https://user-images.githubusercontent.com/875518/77676984-4c81e400-6f4c-11ea-87d8-f2ff505a99da.png)

While there are quite a few ways to map processes to nodes, a good rule of thumb is to have one process span a single GPU. This enables the DDP application to have as many parallel reader streams as there are GPUs and in practice provides a good balance between I/O and computational costs. In the rest of this tutorial, we assume that the application follows this heuristic.