import torch
import torch.nn as nn
import torch.nn.functional as F
import prefix_dataset

from catalyst.dl.experiments import SupervisedRunner

class baseEucl(nn.Module):
    def __init__(self, vocab_size, dim_size, hidden_size):
        super(baseEucl, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.linear = nn.Linear(hidden_size*2+1, 2)
        self.softmax = nn.Softmax()

    def forward(self, input):
        source_input = input[0]
        target_input = input[1]

        source_embed = self.embedding(source_input)
        target_embed = self.embedding(target_input)

        source_output = source_embed
        target_output = target_embed

        source_output, source_hidden = self.cell_1(source_output, torch.zeros(2, source_input.shape[1],hidden_size))
        target_output, target_hidden = self.cell_2(target_output, torch.zeros(2, target_input.shape[1], hidden_size))

        target_hidden = target_hidden[-1]
        source_hidden = source_hidden[-1]

        dist = torch.unsqueeze(torch.norm(source_hidden - target_hidden, dim=1),1)

        hidden = torch.cat((target_hidden, source_hidden, dist), dim=1)

        hidden = self.softmax(self.linear(hidden))

        return hidden


class RNNEucl(baseEucl):
    def __init__(self, vocab_size, dim_size, hidden_size):
        super(RNNEucl, self).__init__(vocab_size, dim_size, hidden_size)
        self.cell_1 = nn.RNN(dim_size, hidden_size, 2, batch_first=False)
        self.cell_2 = nn.RNN(dim_size, hidden_size, 2, batch_first=False)


class GRUEucl(baseEucl):
    def __init__(self, vocab_size, dim_size, hidden_size):
        super(GRUEucl, self).__init__(vocab_size, dim_size, hidden_size)
        self.cell_1 = nn.GRU(dim_size, hidden_size, 2, batch_first=False)
        self.cell_2 = nn.GRU(dim_size, hidden_size, 2, batch_first=False)


logdir = './logdir'
n_epochs = 10
num = 10

dataset_train = prefix_dataset.PrefixDataset('./data', num=num, split='train', download=True)

dataset_test = prefix_dataset.PrefixDataset('./data', num=num, split='test', download=True)

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, collate_fn=prefix_dataset.packing_collate_fn)

loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, collate_fn=prefix_dataset.packing_collate_fn)


vocab_size = 400
dim_size = 5
hidden_size = 5

model = GRUEucl(vocab_size,dim_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

runner = SupervisedRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders={'train':loader_train, 'valid': loader_test},
    logdir=logdir,
    n_epochs=n_epochs,
    verbose=True
)
