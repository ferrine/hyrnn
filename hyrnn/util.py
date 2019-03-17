import torch


def extract_last_states(data, batch_sizes):
    n_times, n_sequences = data.size(0), data.size(1)
    assert n_times == batch_sizes.size(0), "Inconsistent `data` and `batch_sizes` shapes"
    last = torch.zeros(n_sequences, dtype=torch.int64)
    for i in range(int(n_times)):
        last.index_add_(
                dim=0,
                index=torch.arange(batch_sizes[i], dtype=torch.int64),
                source=torch.ones(batch_sizes[i], dtype=torch.int64))
    last = last - 1
    # indices = (last)_+
    indices = torch.where(last > 0, last, torch.zeros(1, dtype=torch.int64))
    return data[indices, torch.arange(n_sequences, dtype=torch.int64)]
