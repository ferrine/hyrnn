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
        self.sofmtax = nn.Softmax()

    def forward(self, source_input, target_input):
        source_embed = self.embedding(source_input).view(1,1,-1)
        target_embed = self.embedding(target_input).view(1,1,-1)

        source_output = source_embed
        target_output = target_embed

        import pdb
        pdb.set_trace()

        source_output, source_hidden = self.cell_1(source_output, torch.zeros(2, source_input.shape[0],hidden_size))
        target_output, target_hidden = self.cell_2(target_output, torch.zeros(2, source_input.shape[0], hidden_size))

        dist = torch.norm(source_hidden - target_hidden)

        hidden = torch.cat((target_hidden, source_hidden, dist))

        hidden = self.softmax(self.linear(hidden))

        return hidden


class RNNEucl(baseEucl):
    def __init__(self, vocab_size, dim_size, hidden_size):
        super(RNNEucl, self).__init__(vocab_size, dim_size, hidden_size)
        self.cell_1 = nn.RNN(dim_size, hidden_size, 2)
        self.cell_2 = nn.RNN(dim_size, hidden_size, 2)


class GRUEucl(baseEucl):
    def __init__(self, vocab_size, dim_size, hidden_size):
        super(GRUEucl, self).__init__(vocab_size, dim_size, hidden_size)
        self.cell_1 = nn.GRU(dim_size, hidden_size, 2)
        self.cell_2 = nn.GRU(dim_size, hidden_size, 2)


logdir = './logdir'
n_epochs = 10
num = 10

dataset_train = prefix_dataset.PrefixDataset('./data', num=num, split='train', download=True)

dataset_test = prefix_dataset.PrefixDataset('./data', num=num, split='test', download=True)

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, collate_fn=prefix_dataset.packing_collate_fn)

loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, collate_fn=prefix_dataset.packing_collate_fn)


vocab_size = 100
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
