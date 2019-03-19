import hyrnn
import torch.nn


def test_MobiusGRU_no_packed_just_works():
    input_size = 4
    hidden_size = 3
    batch_size = 5
    gru = hyrnn.nets.MobiusGRU(input_size, hidden_size, hyperbolic_input=False)
    timestops = 10
    sequence = torch.randn(timestops, batch_size, input_size)
    out, ht = gru(sequence)
    # out: (seq_len, batch, num_directions * hidden_size)
    # ht: (num_layers * num_directions, batch, hidden_size)
    assert out.shape[0] == timestops
    assert out.shape[1] == batch_size
    assert out.shape[2] == hidden_size
    assert ht.shape[0] == 1
    assert ht.shape[1] == batch_size
    assert ht.shape[2] == hidden_size


def test_MobiusGRU_2_layers_no_packed_just_works():
    input_size = 4
    hidden_size = 3
    batch_size = 5
    num_layers = 2
    gru = hyrnn.nets.MobiusGRU(
        input_size, hidden_size, num_layers=num_layers, hyperbolic_input=False
    )
    timestops = 10
    sequence = torch.randn(timestops, batch_size, input_size)
    out, ht = gru(sequence)
    # out: (seq_len, batch, num_directions * hidden_size)
    # ht: (num_layers * num_directions, batch, hidden_size)
    assert out.shape[0] == timestops
    assert out.shape[1] == batch_size
    assert out.shape[2] == hidden_size
    assert ht.shape[0] == num_layers
    assert ht.shape[1] == batch_size
    assert ht.shape[2] == hidden_size


def test_mobius_gru_loop_just_works():
    input_size = 4
    hidden_size = 3
    num_sequences = 3
    seqs = torch.nn.utils.rnn.pack_sequence(
        [
            torch.zeros(10, input_size),
            torch.zeros(5, input_size),
            torch.zeros(1, input_size),
        ]
    )
    loop_params = dict()
    loop_params["h0"] = torch.zeros(num_sequences, hidden_size, requires_grad=False)
    loop_params["input"] = seqs.data
    loop_params["weight_ih"] = torch.nn.Parameter(
        torch.randn(3 * hidden_size, input_size)
    )
    loop_params["weight_hh"] = torch.nn.Parameter(
        torch.randn(3 * hidden_size, hidden_size)
    )
    loop_params["bias"] = torch.randn(3, hidden_size)
    loop_params["c"] = 1.0
    loop_params["nonlin"] = None
    loop_params["hyperbolic_input"] = True
    loop_params["hyperbolic_hidden_state0"] = True
    loop_params["batch_sizes"] = seqs.batch_sizes
    hyrnn.nets.mobius_gru_loop(**loop_params)


def test_MobiusGRU_with_packed_just_works():
    input_size = 4
    hidden_size = 3
    gru = hyrnn.nets.MobiusGRU(input_size, hidden_size, hyperbolic_input=False)
    seqs = torch.nn.utils.rnn.pack_sequence(
        [
            torch.zeros(10, input_size),
            torch.zeros(5, input_size),
            torch.zeros(1, input_size),
        ]
    )
    h, ht = gru(seqs)
    assert h.data.size(0) == 16  # sum of times
    assert h.data.size(1) == hidden_size
    # ht: (num_layers * num_directions, batch, hidden_size)
    assert ht.size(2) == hidden_size
    assert ht.size(1) == 3  # batch size
    assert ht.size(0) == 1  # num layers


def test_MobiusGRU_2_layers_with_packed_just_works():
    input_size = 4
    hidden_size = 3
    gru = hyrnn.nets.MobiusGRU(
        input_size,
        hidden_size,
        num_layers=2,
        hyperbolic_input=False)
    seqs = torch.nn.utils.rnn.pack_sequence([
        torch.zeros(10, input_size),
        torch.zeros(5, input_size),
        torch.zeros(1, input_size)
        ])
    h, ht = gru(seqs)
    assert h.data.size(0) == 16  # sum of times
    assert h.data.size(1) == hidden_size
    # ht: (num_layers * num_directions, batch, hidden_size)
    assert ht.size(2) == hidden_size
    assert ht.size(1) == 3  # batch size
    assert ht.size(0) == 2  # num layers

