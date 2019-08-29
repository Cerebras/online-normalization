This is a minimally modified version of 
[PyTorch Example's ImageNet Training](https://github.com/pytorch/examples/tree/27a6244452c5fcc2269dc59e26a50a4599771081).
This showcases the usage of Online Normalization in ResNet.
ResNet model is minimally modified from 
[PyTorch Torchvision's ResNet implementation](https://github.com/pytorch/vision/blob/9a481d0bec2700763a799ff148fe2e083b575441/torchvision/models/resnet.py).
We train using flags: `-b32 --lr 0.001635 --momentum 0.98691628`

# ImageNet training in PyTorch

This implements training ResNet on the ImageNet dataset.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: resnet101 | resnet152 | resnet18 | 
                        resnet34 | resnet50 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
```
