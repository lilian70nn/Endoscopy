import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from lejepa import sigreg

WORLD_SIZE = 4


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    assert dist.get_rank() == rank
    assert dist.get_world_size() == 4

    shards = list(range(20))
    local_shards = shards[rank::world_size]

    print(f"[Rank {rank}] shards = {local_shards}", flush=True)

    model = torch.nn.Linear(768, 768)
    model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3):
        x = torch.randn(8, 768)

        embeddings = model(x)

        prediction_loss = embeddings.square().mean()
        sigreg_loss = sigreg(embeddings, step, num_slices=16).mean()

        loss = 0.9 * prediction_loss + 0.1 * sigreg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tensor = loss.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size

        if rank == 0:
            print(f"step={step} distributed_loss={loss_tensor.item():.6f}", flush=True)

    parameters = torch.cat([p.detach().flatten() for p in model.module.parameters()])
    gathered = [torch.zeros_like(parameters) for _ in range(world_size)]
    dist.all_gather(gathered, parameters)

    if rank == 0:
        for i in range(1, world_size):
            assert torch.allclose(gathered[0], gathered[i], atol=1e-6)

        print("[PASS] all 4 ranks have synchronized model parameters")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

    print("\nALL 4-RANK DDP TESTS PASSED")