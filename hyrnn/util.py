import torch


def last_states_indices(batch_sizes):
    n_times = batch_sizes.size(0)
    n_sequences = batch_sizes[0]
    last = torch.zeros(n_sequences, dtype=torch.int64)
    for i in range(int(n_times)):
        last.index_add_(
                dim=0,
                index=torch.arange(batch_sizes[i], dtype=torch.int64),
                source=torch.ones(batch_sizes[i], dtype=torch.int64))
    indices = last.cumsum(0) - 1
    # indices = (last)_+
    return indices.flip(0)

