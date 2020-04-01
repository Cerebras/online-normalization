# PyTorch implementation of Online Normalization

This folder holds the 1d and 2d [Online Normalization Algorithm](https://arxiv.org/abs/1905.05894) implemented in [PyTorch](https://pytorch.org/) and integrates as a [Module](https://pytorch.org/docs/stable/nn.html?highlight=module) with the [PyTorch Deep Learning Framework](https://pytorch.org/docs/stable/nn).

## Installation

```bash
python -m pip install -e .
```

### Environment
My enviornment uses is set up with:
```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
conda install -c psi4 gcc-5
```
Note: PyTorch is built with gcc5 so I needed to update my gcc to build this such that it interfaces with PyTorch well

## Usage

### For use with spatially 2D tensors
Given your input tensor is of shape: `(N, C, H, W)`
```
from online_norm_pytorch import OnlineNorm2d


# With Learnable Parameters
norm = OnlineNorm2d(100)
# Without Learnable Parameters
norm = OnlineNorm2d(100, affine=False)
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
norm = OnlineNorm1d(100, affine=False)
input = torch.randn(20, 100)
output = norm(input)
```

## Testing installation

To test after installation:
```bash
python -m unittest discover -v
```

## change log

### 2020-03-25

#### Added

- Added [activation clamping](LinkToActClampPaper) as an error checking mechanism (ecm) and make it the default ecm.

### 2020-04-01

#### Added

- Added CUDA kernel for online norm (removed python loop version of norm, deprecate BON)
