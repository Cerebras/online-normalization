# FMNIST training in PyTorch

This implements training of a 3 layer MLP on the 
[Fashion-MNIST](https://arxiv.org/abs/1708.07747) dataset. 
PyTorch Training functions are adapted from 
[PyTorch Examples](https://github.com/pytorch/examples) 
[ImageNet Training Code](https://github.com/pytorch/examples/tree/master/imagenet).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Datasets

Running `python fmnist_main.py /path/to/data` will automatically download the 
data to `/path/to/data` if it is not already in `/path/to/data`.

## Training

To train a model, run `fmnist_main.py`:

```bash
python fmnist_main.py [fmnist folder]
```

## Reproducing Experiments

The following bash commands will reproduce the experimental results:
```bash
python fmnist_main.py --model_dir /path/to/cache/model --norm_mode online --afwd 0.999 --abkw 0.99
python fmnist_main.py --model_dir /path/to/cache/model --norm_mode batch
python fmnist_main.py --model_dir /path/to/cache/model --norm_mode layer
python fmnist_main.py --model_dir /path/to/cache/model --norm_mode none
```
For our experimentation, each setting is run 400 times. 
The mean of 400 runs is reported.


## Usage

```
usage: fmnist_main.py [-h] [--model_dir MODEL_DIR] [-j N] [--epochs N]
                      [--start-epoch N] [-b N] [--lr LR] [--momentum M]
                      [--wd W] [-p N] [--resume PATH] [-e] [--seed SEED]
                      [--norm_mode NORM] [--afwd AFWD] [--abkw ABKW]
                      [--rm_layer_scaling]
                      DIR

PyTorch FashionMNIST Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        dir to which model is saved (default: ./model_dir)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 10)
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 32)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.04)
  --momentum M          optimizer momentum (default: 0)
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training
  --norm_mode NORM      select normalization type. options: "batch" | "layer"
                        | "online" | "none". (default: batch)
  --afwd AFWD, --decay_factor_forward AFWD
                        forward decay factor which sets momentum process
                        hyperparameter when using online normalization
                        (default: 0.999)
  --abkw ABKW, --decay_factor_backward ABKW
                        backward decay factor which sets control process
                        hyperparameter when using online normalization
                        (default: 0.99)
  --rm_layer_scaling    remove layer scaling in online normalization
```
