# DissectBatchNorm

Study of BatchNorm components in DL models.

- Python deep environment with PyTorch+CUDA
- [DeepOBS](https://github.com/fsschneider/DeepOBS)

## Table of Contents
1. [Installation](#installation)
2. [Getting started](#getting-started)
3. [Extending the code](#extending-the-code)

## Installation

Tested on CUDA-enabled Ubuntu 22.04 and conda 23.1.0

```
conda create -n dlnorm python=3.9
conda activate dlnorm
#
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge omegaconf==2.3.0
conda install -c conda-forge matplotlib==3.6.2
conda install -c anaconda h5py==3.7.0
#
pip install 'git+https://github.com/fsschneider/DeepOBS.git@develop#egg=deepobs'
pip install coloredlogs==15.0.1
```

Full environment description can be found [here](conda_env.yml)


## Getting started

### Run models

- the training loop is defined in [run_trainingLoop.py](batchnorm/run_trainingLoop.py). It can be started via the terminal with the following arguments:
    - `TUNING_CONF`: path to the config file containing the hyperparameters for training
    - `PROBLEM`: name of the testproblem to train on
    - `OPTIMIZER`: name of the optimizer to use
    - `RANDOM_SEED`: random seed for reproducibility
    - `ABLATION`: type of BatchNorm modification to apply
    - `POSITION`: position of the BatchNorm layers to modify (comma-separated list of integers)

``` python
python run_trainingLoop.py TUNING_CONF=configs/basic_config.yaml PROBLEM="cifar10_3c3d_bn" OPTIMIZER=SGD RANDOM_SEED=4842821 ABLATION=trained_bn POSITION=0,0,0,0,0
```

- To run on the cluster, we have the `.sbatch` files, which specify the resources and paramters to explore

- Results are stored in the [`output` folder](batchnorm/output/). Each run creates several files:
    - the layer biases (Vanilla) / learnable BatchNorm parameters (gamma/beta) in separate hdf5 files
    - a log file containing training, validation and test metrics
    - final model is stored in a pth file

- The [ressources folder](res/) contains the results and plots from previous experiments. Those can be used for further analysis without having to re-run the models.




### Analysis
- [bn_plot_functions.py](batchnorm/analysis/bn_plot_functions.py): contains various functions to aid the comparison of training performances and behavior of learnable parameters of BatchNorm. The resulting plots are stored in [`output/plots`](batchnorm/output/plots/)

- Terminal command:  `python -m batchnorm.analysis.bn_plot_functions.py`


- Change [bn_plot_functions.py](batchnorm/analysis/bn_plot_functions.py) to run the desired plot function:

    1. Example to compare training loss and accuracy of a model across all modifications of BatchNorm layers
    2. Example to compare the trajectory of the learnable parameters (gamma/beta) at all BatchNorm layers


```python
model_paths = [f"batchnorm/output/4842821__cifar100_3c3d_bn__{x}/Lr_0.16579130972807002/Ep_350/{y}" for x, y 
                in zip(["0,0,0,0,0", *["0,0,1,1,1"] * 4],
                 ["trained_bn", "trained_bn", "fixed_bn", "fixed_gamma", "fixed_beta"])]

performance_across_ablations(model_paths, smooth_curve=False)

```

```python
paths_AffineParams_gamma = [
    "batchnorm/output/4842821__cifar100_allcnnc_bn__1,0,0,0,0,0,0,0,0/Lr_0.16579130972807002/Ep_350/trained_bn",
    "batchnorm/output/4842821__cifar100_allcnnc_bn__1,0,0,0,0,0,0,0,0/Lr_0.16579130972807002/Ep_350/fixed_gamma"
]

for i, path in enumerate(paths_AffineParams_gamma):
    input_path = path
    bn_params_3Dsurface(True, "gamma", input_path)

```

## Extending the code

- Implement a new network with BatchNorm layers in [bn_models.py](batchnorm/testproblems/bn_models.py)
-  Add a new testproblem class in [batchnorm/testproblems](batchnorm/testproblems/): name convention: `bn_tp_{dataset}_{network}.py`
- Implement new testproblem in [problems.py](batchnorm/testproblems/problems.py)
- Add implemented model to [config file](basic_config.yml) with according hyperparameters for training