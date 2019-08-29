# U-Net with Online Normalization

This repository contains simple PyTorch implementations of 
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) 
created by Ronneberger et al. 
The purpose of this implementation is to show Online Normalizations performance 
in this setting. 

We train U-Net on a 
[Synthetic Dataset](https://github.com/usuyama/pytorch-unet). 
Dataset generation as well as a skeleton for the U-Net model are adopted from 
Naoto Usuyama's [Simple PyTorch implementations of U-Net/FullyConvNet (FCN) for 
image segmentation](https://github.com/usuyama/pytorch-unet). 
PyTorch Training functions are adapted from 
[PyTorch Examples](https://github.com/pytorch/examples) 
[ImageNet Training Code](https://github.com/pytorch/examples/tree/master/imagenet).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Datasets

The synthetic dataset is automatically generated. No preparation is needed.

## Training

To train a model, run `main.py`:

```bash
python main.py
```

### Training with Online Normalization

To train a model using Online Normalization:

```bash
python main.py --norm_mode online
```

## Reproducing Experiments

The following bash commands will reproduce the experimental results:
```bash
python main.py --model_dir /path/to/cache/model --norm_mode online --lr 0.04 --afwd 0.984375 --abkw 0.5
python main.py --model_dir /path/to/cache/model --norm_mode batch --lr 0.04
python main.py --model_dir /path/to/cache/model --norm_mode none --lr 0.6
```
For our experimentation, each setting is run 50 times. 
The median of 50 runs is reported.

## Usage

```
usage: main.py [-h] [--model_dir MODEL_DIR] [--epochs N] [--start-epoch N]
               [-b N] [--lr LR] [--lr_milestone LR_MILESTONE]
               [--lr_multiplier M] [--momentum M] [--wd W] [-p N]
               [--resume PATH] [-e] [--seed SEED] [--norm_mode NORM]
               [--afwd AFWD] [--abkw ABKW] [--rm_layer_scaling] [--classes N]
               [--t_size N] [--v_size N] [--im_size N]

PyTorch UNet Model Training

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        dir to which model is saved
  --epochs N            number of total epochs to run (default: 40)
  --start-epoch N       manual epoch number (useful on restarts, default: 0)
  -b N, --batch-size N  mini-batch size (default: 25)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.04)
  --lr_milestone LR_MILESTONE
                        epoch at which we take a learning-rate step (default:
                        25)
  --lr_multiplier M     lr multiplier at lr_milestones (default: 0.1)
  --momentum M          optimizer momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 2e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training
  --norm_mode NORM      select normalization type. options: "batch" | "online"
                        | "none". (default: batch)
  --afwd AFWD, --decay_factor_forward AFWD
                        forward decay factor which sets momentum process
                        hyperparameter when using online normalization
                        (default: 63 / 64)
  --abkw ABKW, --hdecay_factor_backward ABKW
                        backward decay factor which sets control process
                        hyperparameter when using online normalization
                        (default: 1 / 2)
  --rm_layer_scaling    remove layer scaling in online normalization (default:
                        False)
  --classes N           classes (default: 6)
  --t_size N            train set size (default: 2000)
  --v_size N            val set size (default: 200)
  --im_size N           image height and width (default: 192)
```
