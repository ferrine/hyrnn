import hyrnn
import torch


def test_gru_just_works():
    gru = hyrnn.nets.MobiusGRU(4, 3, hyperbolic_input=False)
    sequence = torch.randn(10, 5, 4)
    h, ht = gru(sequence)
    assert h.shape[0] == 10
    assert ht.shape[1] == 5


def test_extract_last_states():
    tens = torch.tensor([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15],
                        [16, 17, 18, 19]])
    bs = torch.tensor([4, 4, 3, 2, 1])
    res = hyrnn.util.extract_last_states(tens, bs)
    assert (res == torch.tensor([16, 13, 10,  7])).all()
    res1 = hyrnn.util.extract_last_states(tens, bs-1)
    assert (res1 == torch.tensor([12,  9,  6,  3])).all()
