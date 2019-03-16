import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNBase(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim_size,
        hidden_size,
        cell_type="eucl_rnn",
        device=None,
        num_layers=1,
    ):
        super(RNNBase, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.linear = nn.Linear(hidden_size * 2 + 1, 2)
        self.softmax = nn.Softmax()
        self.device = device  # declaring device here due to fact we are using catalyst
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if cell_type.lower() == "eucl_rnn":
            self.cell = nn.RNN
        elif cell_type.lower() == "eucl_gru":
            self.cell = nn.GRU
        else:
            raise NotImplementedError("Unsuported cell type: {0}".format(cell_type))

        self.cell_source = self.cell(dim_size, self.hidden_size, self.num_layers)
        self.cell_target = self.cell(dim_size, self.hidden_size, self.num_layers)

    def forward(self, input):
        source_input = input[0]
        target_input = input[1]

        batch_size = source_input.shape[1]

        source_input = self.embedding(source_input)
        target_input = self.embedding(target_input)

        zero_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device
        )

        source_output, source_hidden = self.cell_source(source_input, zero_hidden)
        target_output, target_hidden = self.cell_target(target_input, zero_hidden)

        target_hidden = target_hidden[-1]
        source_hidden = source_hidden[-1]

        dist = torch.unsqueeze(torch.norm(source_hidden - target_hidden, dim=1), 1)

        hidden = torch.cat((target_hidden, source_hidden, dist), dim=1)

        hidden = self.softmax(self.linear(hidden))

        return hidden
