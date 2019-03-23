Hyperbolic RNN in Pytorch
=========================

## Prerequisites
You need `Python3.6` to run the code. The list of the dependencies are in `requrements.txt`. Run:
```
python -m pip install -r requirements.txt
```

## Training

To reproduce the results from the Table 1 on the final report, run the following command:

```
cd ./experiment_run
python run.py --data_dir=./data --num_epochs=30 --log_dir=./logs --batch_size=1024 --num_layers=2 --cell_type=hyp_gru
```

where you can change the argument `cell_type=hyp_gru` to `cell_type=eucl_gru` if you want to run Euclidean version of GRU.

Note that training can take up to 12-15 hours, having the batch size of 1024. You can increase the batch size to get the results more quicker, but wait for some slight accuracy drop.

## Results

|      Model      |            Value            |
| ---------------- | --------------------------- |
| Fully Euclidean GRU / B=64      | 93.25                 |
| Fully Hyperbolic GRU / B=1024  | 96.8                         |

## TODO

- [x] Abstract interface for Riemannian manifolds, embed-
ded in ambient real coordinate space.
- [x] Compatible generic `RSGD`.
- [x] Compatible generic `RAdam`.
- [x] Compatible implementation of Poincare ball and Mo-
bius arithmetics.
- [x] Test coverage for optimization routines.
- [x] GRU based on Mobius arithmetics, API-compatible
with `torch.nn.GRU`.
- [ ] Layers parameterized by pivots of Log and Exp, as
opposed to fixed pivot of 0 in Mobius arithmetics-based
layers.
- [ ] Test coverage for “Mobius” layers and RNN loops.
- [x] Numerical stability with `float64`.
- [ ] Numerical stability with `float32`.
- [ ] Investigation of possibility of using `cudnn` loop.
- [ ] `C++` implementation of core operations and loops.
