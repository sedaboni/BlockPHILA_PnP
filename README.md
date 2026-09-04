# Block-coordinate Plug-And-Play Methods with Armijo-like line-search for Image Restoration

### [Paper (arXiv)](https://arxiv.org/abs/2603.01734) 

## Setup Environment
1. Create the conda environment
```
conda env create --file "environment.yml"
```

2. Activate the environment
```
conda activate blockPHILA
```
## Run experiments
1. Replicate the experiment contained in the paper.

```
bash run_experiments.sh
```
2. Check the metrics.

```
tensorboard --logdir=results
```

## The Block-PHILA variants

Each variant is identified by three independent design choices, recorded in its name as

```
Block-PHILA-<splitting>-<steplength>[-I][-VM]
```

| tag | meaning |
| --- | --- |
| `FB` | forward-backward splitting: the proximal step acts on the data fidelity term |
| `GD` | all-smooth splitting: a gradient step is taken on the whole objective function |
| `BB` | adaptive Barzilai-Borwein steplength |
| `C` | constant steplength |
| `I` | FISTA-like inertial term (absent tag = no inertia) |
| `VM` | variable metric (absent tag = identity scaling matrix) |

The eight configurations obtained by combining the splitting, the steplength rule and the
inertial term are the following.

| variant | splitting | steplength $\alpha_k$ | inertia $\beta_k$ | $N=1$ reduces to | legacy name |
| --- | --- | --- | --- | --- | --- |
| `FB-BB-I` | FB | Barzilai-Borwein | FISTA | PHILA | `phila1` |
| `FB-BB` | FB | Barzilai-Borwein | $0$ | VMILA | `phila2` |
| `FB-C-I` | FB | constant | FISTA | PHILA | `phila3` |
| `FB-C` | FB | constant | $0$ | VMILA | `phila4` |
| `GD-BB-I` | GD | Barzilai-Borwein | FISTA | full-gradient PHILA | `phila5` |
| `GD-BB` | GD | Barzilai-Borwein | $0$ | full-gradient VMILA | `phila6` |
| `GD-C-I` | GD | constant | FISTA | full-gradient PHILA | `phila7` |
| `GD-C` | GD | constant | $0$ | full-gradient VMILA | `phila8` |

# Citation
Please consider citing Block PHILA PnP if you find it helpful.

```BibTex
@article{porta2026block,
  title={Block-coordinate Plug-And-Play Methods with Armijo-like line-search for Image Restoration},
  author={Porta, Federica and Rebegoldi, Simone and Sebastiani, Andrea},
  journal={arXiv preprint arXiv:2603.01734},
  year={2026}
}
 ```


