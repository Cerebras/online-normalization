This is a modified version of 
[pytorch/examples' imagenet](https://github.com/pytorch/examples/tree/master/imagenet). 
Training procedure adapted from 
[ResNet in TensorFlow r1.9.0](https://github.com/tensorflow/models/tree/r1.9.0/official/resnet)

# ResNet Training on the ImageNet and Cifar Datasets using PyTorch

This implements training of ResNet on the ImageNet and Cifar Datasets.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Datasets

#### CIFAR
Running `python cifar10_main.py /path/to/data` or 
`python cifar100_main.py /path/to/data`  will automatically download the data 
to `/path/to/data` if it is not already in `/path/to/data`.

#### ImageNet
Download the ImageNet dataset and move validation images to labeled subfolders
  - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training

To train a model, run `<DATASET>_main.py`:

```bash
python <DATASET>_main.py [folder with train and val folders]
```

## Reproducing Experiments

The following bash commands will reproduce the experimental results

### CIFAR

```bash
python cifar10_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode online --afwd 0.9990234375 --abkw 0.9921875
python cifar10_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode batch
python cifar10_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode layer
python cifar10_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode instance
python cifar10_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode group

python cifar100_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode online --afwd 0.998046875 --abkw 0.9375
python cifar100_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode batch
python cifar100_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode layer
python cifar100_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode instance
python cifar100_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode group
```
For our experimentation, each setting is run 5 times. The median of 5 runs is reported.

### ImageNet

```bash
python imagenet_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode online --afwd 0.999 --abkw 0.99
python imagenet_main.py /path/to/data --model-dir /path/to/cache/model --norm-mode batch
```
For our experimentation, each setting is run 1 time.

## Usage
### ImageNet Usage

```
usage: imagenet_main.py [-h] [--model-dir MODEL_DIR] [-a ARCH] [-j N]
                        [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                        [--lr-milestones LR_MILESTONES [LR_MILESTONES ...]]
                        [--lr-multiplier M] [--momentum M] [--wd W] [-p N]
                        [--resume PATH] [-e] [--pretrained] [--seed SEED]
                        [--norm-mode NORM] [--afwd AFWD] [--abkw ABKW]
                        [--ecm ECM] [--gn-num-groups GN_NUM_GROUPS]
                        DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        dir to which model is saved (default: ./model_dir)
  -a ARCH, --arch ARCH  model architecture: conv1x1 | conv3x3 | resnet101 |
                        resnet152 | resnet18 | resnet34 | resnet50 |
                        resnext101_32x8d | resnext50_32x4d (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 90)
  --start-epoch N       manual epoch number (useful on restarts, default: 0)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --lr-milestones LR_MILESTONES [LR_MILESTONES ...]
                        epochs at which we take a learning-rate step (default:
                        [30, 60, 80, 90])
  --lr-multiplier M     lr multiplier at lr_milestones (default: 0.1)
  --momentum M          optimizer momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training
  --norm-mode NORM      norm choices: batch | group | layer | instance |
                        online | none (default: batch)
  --afwd AFWD, --decay-factor-forward AFWD
                        forward decay factor which sets momentum process
                        hyperparameter when using online normalization
                        (default: 0.999)
  --abkw ABKW, --decay-factor-backward ABKW
                        backward decay factor which sets control process
                        hyperparameter when using online normalization
                        (default: 0.99)
  --ecm ECM             Online Norm ErrorCheckingMechanism choices: ls | ac |
                        none (default: ls)
  --gn-num-groups GN_NUM_GROUPS
                        number of groups in group norm if using group norm as
                        normalization method (default: 32)
```

### Cifar10 Usage

```
usage: cifar10_main.py [-h] [--model-dir MODEL_DIR] [-d D] [-j N] [--epochs N]
                       [--start-epoch N] [-b N] [--lr LR]
                       [--lr-milestones LR_MILESTONES [LR_MILESTONES ...]]
                       [--lr-multiplier M] [--momentum M] [--wd W] [-p N]
                       [--resume PATH] [-e] [--pretrained] [--seed SEED]
                       [--norm-mode NORM] [--afwd AFWD] [--abkw ABKW]
                       [--ecm ECM] [--gn-num-groups GN_NUM_GROUPS]
                       DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        dir to which model is saved (default: ./model_dir)
  -d D, --depth D       depth of ResNet (default: 20)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 250)
  --start-epoch N       manual epoch number (useful on restarts, default: 0)
  -b N, --batch-size N  mini-batch size (default: 128)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --lr-milestones LR_MILESTONES [LR_MILESTONES ...]
                        epochs at which we take a learning-rate step (default:
                        [100, 150, 200])
  --lr-multiplier M     lr multiplier at lr_milestones (default: 0.1)
  --momentum M          optimizer momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 2e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training
  --norm-mode NORM      norm choices: batch | group | layer | instance |
                        online | none (default: batch)
  --afwd AFWD, --decay-factor-forward AFWD
                        forward decay factor which sets momentum process
                        hyperparameter when using online normalization
                        (default: 1023 / 1024)
  --abkw ABKW, --decay-factor-backward ABKW
                        backward decay factor which sets control process
                        hyperparameter when using online normalization
                        (default: 127 / 128)
  --ecm ECM             Online Norm ErrorCheckingMechanism choices: ls | ac |
                        none (default: ls)
  --gn-num-groups GN_NUM_GROUPS
                        number of groups in group norm if using group norm as
                        normalization method (default: 8)
```

### Cifar100 Usage

```
usage: cifar100_main.py [-h] [--model-dir MODEL_DIR] [-d D] [-j N]
                        [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                        [--lr-milestones LR_MILESTONES [LR_MILESTONES ...]]
                        [--lr-multiplier M] [--momentum M] [--wd W] [-p N]
                        [--resume PATH] [-e] [--pretrained] [--seed SEED]
                        [--norm-mode NORM] [--afwd AFWD] [--abkw ABKW]
                        [--ecm ECM] [--gn-num-groups GN_NUM_GROUPS]
                        DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        dir to which model is saved (default: ./model_dir)
  -d D, --depth D       depth of ResNet (default: 20)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 250)
  --start-epoch N       manual epoch number (useful on restarts, default: 0)
  -b N, --batch-size N  mini-batch size (default: 128)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --lr-milestones LR_MILESTONES [LR_MILESTONES ...]
                        epochs at which we take a learning-rate step (default:
                        [100, 150, 200])
  --lr-multiplier M     lr multiplier at lr_milestones (default: 0.1)
  --momentum M          optimizer momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 2e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training
  --norm-mode NORM      norm choices: batch | group | layer | instance |
                        online | none (default: batch)
  --afwd AFWD, --decay-factor-forward AFWD
                        forward decay factor which sets momentum process
                        hyperparameter when using online normalization
                        (default: 511 / 512)
  --abkw ABKW, --decay-factor-backward ABKW
                        backward decay factor which sets control process
                        hyperparameter when using online normalization
                        (default: 15 / 16)
  --ecm ECM             Online Norm ErrorCheckingMechanism choices: ls | ac |
                        none (default: ls)
  --gn-num-groups GN_NUM_GROUPS
                        number of groups in group norm if using group norm as
                        normalization method (default: 8)
```
