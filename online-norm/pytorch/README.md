# PyTorch implementation of Online Normalization

This folder holds the 1d and 2d [Online Normalization Algorithm](https://arxiv.org/abs/1905.05894) implemented in [PyTorch](https://pytorch.org/) and integrates as a [Module](https://pytorch.org/docs/stable/nn.html?highlight=module) with the [PyTorch Deep Learning Framework](https://pytorch.org/docs/stable/nn).

## Installation

```bash
pip install .
```

## Usage

### For use with spatially 2D tensors
Given your input tensor is of shape: `(N, C, H, W)`
```
from online_norm_pytorch import OnlineNorm2d


# With Learnable Parameters
norm = OnlineNorm2d(100)
# Without Learnable Parameters
norm = OnlineNorm2d(100, weight=False, bias=False)
input = torch.randn(20, 100, 35, 45)
output = norm(input)
```

### For use with Linear Layer
Given your input tensor is of shape: `(N, C)`
```
from online_norm_pytorch import OnlineNorm1d

# With Learnable Parameters
norm = OnlineNorm1d(100)
# Without Learnable Parameters
norm = OnlineNorm1d(100, weight=False, bias=False)
input = torch.randn(20, 100)
output = norm(input)
```

## Testing installation

The test uses [Nose](https://nose.readthedocs.io/en/latest/) testing framework.
To test after installation:
```bash
python setup.py test
```
or
```bash
nosetests
```