import sys

sys.path.append("..")

import argparse
import os
import torch.utils.data
import torch.nn as nn
import geoopt
import prefix_dataset
import model, runner

from catalyst.dl.callbacks import PrecisionCallback


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, help="", default="./data")
parser.add_argument("--data_class", type=int, help="", default=10)
parser.add_argument("--num_epochs", type=int, help="", default=100)
parser.add_argument("--log_dir", type=str, help="", default="logdir")
parser.add_argument("--batch_size", type=int, help="", default=64)

parser.add_argument("--embedding_dim", type=int, help="", default=5)
parser.add_argument("--hidden_dim", type=int, help="", default=5)
parser.add_argument("--project_dim", type=int, help="", default=5)
parser.add_argument("--use_distance_as_feature", action="store_true", default="True")


parser.add_argument("--num_layers", type=int, help="", default=1)
parser.add_argument("--verbose", type=bool, help="", default=True)
parser.add_argument(
    "--cell_type", choices=("hyp_gru", "eucl_rnn", "eucl_gru"), default="eucl_gru"
)
parser.add_argument("--decision_type", choices=("hyp", "eucl"), default="eucl")
parser.add_argument("--embedding_type", choices=("hyp", "eucl"), default="eucl")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--sgd", action='store_true')
parser.add_argument("--adam_betas", type=str, default="0.9,0.999")
parser.add_argument("--wd", type=float, default=0.)
parser.add_argument("--c", type=float, default=1.)
parser.add_argument("--j", type=int, default=1)


args = parser.parse_args()
os.mkdir("./logs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = args.data_dir
logdir = os.path.join("./logs", args.log_dir)

n_epochs = args.num_epochs
num = args.data_class
batch_size = args.batch_size
adam_betas = args.adam_betas.split(",")

dataset_train = prefix_dataset.PrefixDataset(
    data_dir, num=num, split="train", download=True
)

dataset_test = prefix_dataset.PrefixDataset(
    data_dir, num=num, split="test", download=True
)

loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn,
    shuffle=True, num_workers=args.j,
)

loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn
)


model = model.RNNBase(
    dataset_train.vocab_size,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    project_dim=args.project_dim,
    cell_type=args.cell_type,
    device=device,
    num_layers=args.num_layers,
    use_distance_as_feature=args.use_distance_as_feature,
    num_classes=2,
    c=args.c
).double()

criterion = nn.CrossEntropyLoss()
if not args.sgd:
    optimizer = geoopt.optim.RiemannianAdam(
        model.parameters(),
        lr=args.lr,
        betas=(float(adam_betas[0]), float(adam_betas[1])),
        stabilize=10,
        weight_decay=args.wd
    )
else:
    optimizer = geoopt.optim.RiemannianSGD(
        model.parameters(), args.lr, stabilize=10,
        weight_decay=args.wd)

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
