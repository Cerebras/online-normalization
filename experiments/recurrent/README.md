# Word-level language modeling RNN

This repository implements experimentation of recurrent Word-level language 
modeling on [Penn-Treebank](https://doi.org/10.3115/1075812.1075835) for the 
Online Normalization paper.
Data-Processing and recurrent model adapted from 
[PyTorch](https://pytorch.org/)'s 
[Word-level language modeling](https://github.com/pytorch/examples/tree/master/word_language_model). 
PyTorch Training functions are adapted from [PyTorch](https://pytorch.org/)'s 
[ImageNet Training Code](https://github.com/pytorch/examples/tree/master/imagenet).

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Dataset

Place Penn-Treebank's `train.txt`, `valid.txt`, and `test.txt` into `/path/to/data`.
This repository can be used to train similar datasets which have the same format.

## Training

This example trains a multi-layer recurrent model on a language modeling task.

```bash
python ptb_main.py /path/to/data --not-tied                   # Train a LSTM network
python ptb_main.py /path/to/data --ru-type LSTM               # Train a tied LSTM network (PyTorch Implementation of LSTM)
python ptb_main.py /path/to/data                              # Train a tied LSTM network using Online Normalization
python ptb_main.py /path/to/data --ru-type CustomRNN_tanh --norm-mode online  # Train a tied tanh RNN network using Online Normalization
python ptb_main.py /path/to/data --ru-type CustomLSTM --norm-mode layer      # Train a tied LSTM network using Layer Normalization
python ptb_main.py /path/to/data --ru-type CustomRNN_tanh --norm-mode layer  # Train a tied tanh RNN network using Layer Normalization
```

## Reproducing Experiments

The following bash commands will reproduce the experimental results

### RNN

```bash
python ptb_main.py /path/to/data --model_dir /path/to/cache/model --ru-type CustomRNN_tanh --norm-mode online --lr 1.7 --afwd 0.9999389648 --abkw 0.9921875
python ptb_main.py /path/to/data --model_dir /path/to/cache/model --ru-type CustomRNN_tanh --norm-mode layer --lr 0.95
python ptb_main.py /path/to/data --model_dir /path/to/cache/model --ru-type RNN_TANH --lr 0.5
```
For our experimentation, each setting is run 25 times. 
The median of 25 runs is reported.

### LSTM

```bash
python ptb_main.py /path/to/data --model_dir /path/to/cache/model --epochs 25 --ru-type CustomLSTM --norm-mode online --lr 6.5 --afwd 0.9998779297 --abkw 0.96875
python ptb_main.py /path/to/data --model_dir /path/to/cache/model --epochs 25 --ru-type CustomLSTM --norm-mode layer --lr 3.25
python ptb_main.py /path/to/data --model_dir /path/to/cache/model --epochs 25 --ru-type LSTM --lr 3.5
```
For our experimentation, each setting is run 25 times. 
The median of 25 runs is reported.


## Usage

```
usage: ptb_main.py [-h] [--model-dir MODEL_DIR] [--ru-type RUnit]
                   [--norm-mode NORM] [--emsize EMSIZE] [--nhid NHID]
                   [--nlayers NLAYERS] [--epochs N] [--start-epoch N] [-b N]
                   [--bptt BPTT] [--dropout DROPOUT] [--not-tied] [--lr LR]
                   [--lr-decay LR_DECAY] [--wd W] [-p N] [--resume PATH] [-e]
                   [--seed SEED] [--test-at-end] [--afwd AFWD] [--abkw ABKW]
                   [--ecm ECM] [--cell-norm]
                   DIR

PyTorch RNN/LSTM Language Model

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        dir to which model is saved (default: ./model_dir)
  --ru-type RUnit       recurrent unit choices: RNN_TANH | RNN_RELU | LSTM |
                        GRU | CustomLSTM | CustomRNN_tanh | CustomRNN_relu
                        (default: CustomLSTM)
  --norm-mode NORM      norm choices: layer | online | none (default: online)
  --emsize EMSIZE       size of word embeddings (default: 200)
  --nhid NHID           number of hidden units per layer (default: 200)
  --nlayers NLAYERS     number of layers (default: 1)
  --epochs N            number of total epochs to run (default: 40)
  --start-epoch N       manual epoch number (useful on restarts, default: 0)
  -b N, --batch-size N  mini-batch size (default: 20)
  --bptt BPTT           sequence length (default: 128)
  --dropout DROPOUT     dropout applied to layers (default: 0 = no dropout)
  --not-tied            do not tie the word embedding and softmax weights
  --lr LR, --learning-rate LR
                        initial learning rate (default: 6.5)
  --lr-decay LR_DECAY   per epoch exponential learning rate schedule decay
                        rate of lr (default: 1, value of 1 means no decay)
  --wd W, --weight-decay W
                        weight decay (default: 1e-6)
  -p N, --print-freq N  print frequency (default: 25)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        Perform a one time evaluation on the validation set.
                        Will not perform any training.
  --seed SEED           seed for initializing training
  --test-at-end         test at the end of training
  --afwd AFWD, --decay-factor-forward AFWD
                        forward decay factor which sets momentum process
                        hyperparameter when using online normalization
                        (default: 8191 / 8192)
  --abkw ABKW, --decay-factor-backward ABKW
                        backward decay factor which sets control process
                        hyperparameter when using online normalization
                        (default: 31 / 32)
  --ecm ECM             Online Norm ErrorCheckingMechanism choices: ls | ac |
                        none (default: ls)
  --cell-norm           normalize cell gate in LSTM
```

### Alternative model training
With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python ptb_main.py /path/to/data --emsize 650 --nhid 650 --dropout 0.5 --not-tied
python ptb_main.py /path/to/data --emsize 650 --nhid 650 --dropout 0.5
python ptb_main.py /path/to/data --emsize 1500 --nhid 1500 --dropout 0.65 --not-tied
python ptb_main.py /path/to/data --emsize 1500 --nhid 1500 --dropout 0.65
```
