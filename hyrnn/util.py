import torch.jit


@torch.jit.script
def extract_last_states(data, batch_sizes):
    last = torch.zeros(data.size(1), dtype=torch.int64)
    for i in range(int(batch_sizes.size(0))):
        last.index_add_(
            0,
            torch.arange(batch_sizes[i], dtype=torch.int64),
            torch.ones(batch_sizes[i], dtype=torch.int64),
        )
    last = last - 1
    indices = torch.where(last > 0, last, torch.zeros(1, dtype=torch.int64))
    return data[indices, torch.arange(data.size(1), dtype=torch.int64)]
