# Multilevel Generative Samplers for Investigating Critical Phenomena
This is the code for the paper [Multilevel Generative Samplers for Investigating Critical Phenomena](https://openreview.net/forum?id=YcUV5apdlq).

## Installation
You can follow the steps below to build the code from source. 
Clone the repository 
```bash
$ git clone https://github.com/mlneuralsampler/multilevel
$ cd multilevel
```
create a new python environment (optional you can do the same operation using `conda`)
```bash
$ python3 -m venv .venv
```
start your virtual environment
```bash
$ source .venv/bin/activate
```
install the package via `pip`
```bash
$ pip install -e .
```

## Quick Start
The file `main.py` can be used to train new  models and run the analysis once the models are trained. 
A detailed list of non-defualt hyperparameters used for the experiments is reported in the [paper](todo=-link).
For training the model it is sufficient to run
```bash
$ python3 rigcs.py --train 
```
Once the model has been trained, the analysis can be run right away with the commands below. 
Note that the path to the model is automatically built from the parser based on the same arguments used for training.
Therefore, it is important to not change the arguments default `args` or those supplied by command line for training 
the model. To compute the metrics associated with **importance sampling** one can run
```bash
$ python3 rigcs.py --measures_md 
```
to perform unbiased sampling via *neural-mcmc* as discussed in [Nicoli et al., Phys. Rev. E (2019)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.023304)
```bash
$ python3 rigcs.py --measures_imh
```

## Baselines
In the paper, our approach is compared in general to three different baselines:
- **VAN**: method from [Wu et al., Phys. Rev. Lett. (2019)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.080602).
- **HAN**: method from [Bialas et al., Comp. Phys. Comms. (2022)](https://www.sciencedirect.com/science/article/pii/S0010465522002211).
- **Cluster-AIS**: state-of-the art method for collective sampling of spin systems by  [Wolff Ulli, Phys. Rev. Lett. (1989)](https://link.aps.org/doi/10.1103/PhysRevLett.62.361) combined with annealed importance sampling (AIS) [Radford M. Neal, Statistics and computing 11 (2001)](https://link.springer.com/article/10.1023/a:1008923215028).
- **MLMC-HB**: multilevel Monte Carlo with Heathbath as introduced in [Jansen et al., Phys. Rev. D 102 (2020)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.114512).

For the baselines we used the following resources:
- **VAN**: the VAN baselines were run using the code provided in [https://github.com/wdphy16/stat-mech-van](https://github.com/wdphy16/stat-mech-van) with only minor modifications and slight hyperparameter tuning. 
- **HAN**: for HAN, we requested the code to the authors of the paper who kindly shared the source code for reproducing their experiments with us. 
- **Cluster**: for the cluster method we based our implementation on a highly optimized code for GPU based on [Komura et al., Comp. Phys. Sim. (2012)](https://www.sciencedirect.com/science/article/pii/S001046551200032X?via%3Dihub) and [Komura et al., Comp. Phys. Sim. (2014)](https://www.sciencedirect.com/science/article/pii/S0010465513003743?via%3Dihub).
- **MLMC-HB**: custom implementation of the algorithm following [Jansen et al., Phys. Rev. D 102 (2020)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.114512).

While the **VAN** experiments can be run from this repository, we do not provide the implementations of the **HAN** as the code we used is intellectual property of the authors. 
As this is not publicly available, we refer to them in case the reader is interested in re-run the **HAN** experiments as well.
As far as the **cluster** methods we refer to the source code linked to the papers cited above. 

## Examples
This repository contains a folder `examples` with two `ipython` notebooks which can be used to gain familiarity with the code and the model's structure.

- `multilevel.ipynb`: this notebook is intended to help the user understanding the protocol of our training. Namely, using this notebook the user can understand how the models are built and how it is stored fur future evaluation. All the functionalities shown therein are used in the `main.py` for training new models. 
**N.B.** Note to the user: in case you want to access the configurations in `data/config` those files need to be unzipped first. A util script unzip.py can be found in `/examples/.`. Configurations contained in `data/config` are for demonstration purposes and because they are quite heavy we only included them up to lattices of size `16x16`.

In order install your python environment for running the examples notebook you can run the following command (first make sure your virtual environment is active, otherwise first run `$ source .venv/bin/activate`)
```bash 
$ python3 -m ipykernel install --user --name=multilevel
```
You'll then find the kernel `multivel` among the possible kernels when opening the `.ipynb` notebook.

## References       

If you find this useful, please consider citing:
``` 
@inproceedings{
singha2025multilevel,
title={Multilevel Generative Samplers for Investigating Critical Phenomena},
author={Ankur Singha and Elia Cellini and Kim Andrea Nicoli and Karl Jansen and Stefan K{\"u}hn and Shinichi Nakajima},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=YcUV5apdlq}
}
```       
