import torch
import os

if __name__ == "__main__":

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    
    dist = torch.distributed.init_process_group(backend="nccl", world_size=world_size, 
            rank=rank)

    torch.cuda.set_device(local_rank)

    N, D_in, H, D_out = 640, 4096, 2048, 1024
    model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                torch.nn.Dropout(p=0.2),
                                torch.nn.Linear(H, D_out),
                                torch.nn.Dropout(p=0.1)).cuda()


    # Cuda Graphs requires that DDP is captured on a side stream
    # It is important to synchronize the streams after the DDP initialization
    # so anything after sees properly initialized model weights across GPUs
    s = torch.cuda.Stream()
    #s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
            ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], 
                output_device=local_rank)
    torch.cuda.current_stream().wait_stream(s)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    static_input = torch.randn(N, D_in, device=local_rank)
    static_target = torch.randn(N, D_out, device=local_rank)

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        output = torch.tensor([rank]).cuda(rank)
        for i in range(11):
            optimizer.zero_grad(set_to_none=True)
            y_pred = ddp_model(static_input)
            loss = loss_fn(y_pred, static_target)
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
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

        """
        if local_rank == 0:
            reqs = []
            req1 = dist.isend(tensor=static_input, dst=1) # enqueue the operation instead of issuing it
            reqs.append(req1)
            reqs.wait_all() # blocking until all operators are finished 
        if local_rank == 1:
            # Rank 1: 
            reqs = []
            req1 = dist.irecv(tensor=static_input, src=0)
            reqs.append(req1)
            reqs.wait_all() 
        """

    real_inputs = [torch.rand_like(static_input) for _ in range(1000)]
    real_targets = [torch.rand_like(static_target) for _ in range(1000)]

    for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        # replay() includes forward, backward, and step.
        g.replay()