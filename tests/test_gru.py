import hyrnn
import torch


def test_gru_just_works():
    gru = hyrnn.nets.MobiusGRU(4, 3, hyperbolic_input=False)
    sequence = torch.randn(10, 5, 4)
    h, ht = gru(sequence)
    assert h.shape[0] == 10
    assert ht.shape[0] == 5
