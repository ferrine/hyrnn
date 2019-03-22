Hyperbolic RNN in Pytorch
=========================

## Prerequisites
You need `python3` to run the code. The list of the dependencies are in `requrements.txt`. Run:
```
python -m pip install -r requirements.txt
```

## Training

To reproduce the results from the Table 1 on the final report, run the following command:

```
python experiment_run/run.py --data_dir=./data --num_epochs=30 --log_dir=./logs --batch_size=1024 --num_layers=2 --cell_type=hyp_gru
```

where you can change the argument `cell_type=hyp_gru` to `cell_type=eucl_gru` if you want to run Euclidean version of GRU.

Note that training can take up to 12-15 hours, having the batchsize of 1024. You can increase the batchsize to get the results in more reasonable time, but wait for some slight accuracy drop.
