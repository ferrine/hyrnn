import sys
sys.path.append('..')
import hyrnn
import collections
import torch
import torch.nn as nn
from experiment_run import model

PrefixBatch = collections.namedtuple("PrefixBatch", "sequences,alignment,label")


def genten(n):
    return torch.zeros(3, dtype=torch.long)


def test_models():
    cells = ["hyp_gru", "eucl_rnn", "eucl_gru"]
    for cell in cells:
        test_model(cell)


def test_model(cell_type, num_classes=2, batch_size=2):
    model_ = model.RNNBase(
        101,
        101,
        101,
        101,
        cell_type,
        device="cpu",
        use_distance_as_feature=True,
        num_classes=2,
    )
    first = (genten(3), genten(3))
    second = (genten(3), genten(3))
    align = torch.tensor((1, 0), dtype=torch.long)
    labels = torch.tensor((0, 1), dtype=torch.long)

    input = PrefixBatch(
        (
            torch.nn.utils.rnn.pack_sequence(first),
            torch.nn.utils.rnn.pack_sequence(second),
        ),
        torch.tensor(align),
        torch.tensor(labels),
    )
    model_input = (input[0][0], input[0][1], input[1])

    output = model_(model_input)
    assert output.shape[0] == batch_size
    assert output.shape[1] == num_classes
