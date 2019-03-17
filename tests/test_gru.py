import hyrnn
import torch.nn


def test_MobiusGRU_no_packed_just_works():
    input_size = 4
    hidden_size = 3
    gru = hyrnn.nets.MobiusGRU(
            input_size,
            hidden_size,
            hyperbolic_input=False)
    timestops = 10
    sequence = torch.randn(timestops, 5, input_size)
    h, ht = gru(sequence)
    assert h.shape[0] == timestops
    assert ht.shape[1] == 5


def test_extract_last_states():
    seqs = torch.nn.utils.rnn.pack_sequence([
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([7])
    ])
    data, bs = seqs
    res = hyrnn.util.extract_last_states(data, bs)
    assert (res == torch.tensor([3, 5, 7])).all()


def test_mobius_gru_loop_just_works():
    input_size = 4
    hidden_size = 3
    seqs = torch.nn.utils.rnn.pack_sequence([
        torch.zeros(10, input_size),
        torch.zeros(5, input_size),
        torch.zeros(1, input_size)
        ])
    loop_params = dict()
    loop_params['h0'] = torch.zeros(input_size, requires_grad=False)
    loop_params['input'] = seqs.data
    loop_params['weight_ih'] = torch.nn.Parameter(
            torch.randn(3*hidden_size, input_size))
    loop_params['weight_hh'] = torch.nn.Parameter(
            torch.randn(3*hidden_size, hidden_size))
    loop_params['bias'] = torch.randn(3, hidden_size)
    loop_params['c'] = 1.
    loop_params['nonlin'] = None
    loop_params['hyperbolic_input'] = True
    loop_params['hyperbolic_hidden_state0'] = True
    loop_params['batch_sizes'] = seqs.batch_sizes
    outs = hyrnn.nets.mobius_gru_loop(**loop_params)

    print(outs.shape)


def test_MobiusGRU_with_packed_just_works():
    input_size = 4
    hidden_size = 3
    gru = hyrnn.nets.MobiusGRU(
        input_size,
        hidden_size,
        hyperbolic_input=False)
    seqs = torch.nn.utils.rnn.pack_sequence([
        torch.zeros(10, input_size),
        torch.zeros(5, input_size),
        torch.zeros(1, input_size)
        ])
    h, ht = gru(seqs)
    assert h.data.size(0) == 16  # sum of times
    assert ht.size(1) == 3
