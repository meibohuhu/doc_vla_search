import torch, torch.distributed as dist, time
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r); dev = f"cuda:{r}"
for mb in [64, 256, 1024]:
    n = mb*1024*1024//4
    t = torch.ones(n, device=dev)
    for _ in range(3): dist.all_reduce(t)  # warmup
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters): dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/iters
    # allreduce moves ~2*(N-1)/N * size; approximate busbw = 2*size/dt
    size = n*4
    busbw = 2*size/dt/1e9
    if r == 0:
        print(f"{mb:5d} MB: {dt*1000:7.1f} ms/allreduce  busbw ~{busbw:6.2f} GB/s", flush=True)
dist.destroy_process_group()
