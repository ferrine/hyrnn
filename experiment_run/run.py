import sys
sys.path.append('..')

import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import prefix_dataset
import model, runner

from catalyst.dl.callbacks import PrecisionCallback


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, help="", default="./data")
parser.add_argument("--data_class", type=int, help="", default=10)
parser.add_argument("--num_epochs", type=int, help="", default=1)
parser.add_argument("--log_dir", type=str, help="", default="./logdir")
parser.add_argument("--batch_size", type=int, help="", default=64)
parser.add_argument("--vocab_size", type=int, help="", default=0)
parser.add_argument("--hidden_size", type=int, help="", default=5)
parser.add_argument("--dim_size", type=int, help="", default=5)
parser.add_argument("--cell_type", type=str, help="", default="eucl_rnn")
parser.add_argument("--num_layers", type=int, help="", default=1)
parser.add_argument("--verbose", type=bool, help="", default=True)


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = args.data_dir
logdir = args.log_dir

n_epochs = args.num_epochs
num = args.data_class
batch_size = args.batch_size

dataset_train = prefix_dataset.PrefixDataset(
    data_dir, num=num, split="train", download=True
)

dataset_test = prefix_dataset.PrefixDataset(
    data_dir, num=num, split="test", download=True
)

loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn
)

loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn
)

vocab_size = args.vocab_size
dim_size = args.dim_size
hidden_size = args.hidden_size
cell = args.cell_type
num_layers = args.num_layers

model = model.RNNBase(
    vocab_size,
    dim_size,
    hidden_size,
    cell_type=cell,
    device=device,
    num_layers=num_layers,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

runner = runner.CustomRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders={"train": loader_train, "valid": loader_test},
    callbacks=[PrecisionCallback(precision_args=[1])],
    logdir=logdir,
    n_epochs=n_epochs,
    verbose=args.verbose,
)
