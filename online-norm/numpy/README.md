# Numpy implementation of Online Normalization

This folder holds the 1d and 2d Online Normalization Algorithm implemented
in [NumPy](https://www.numpy.org/). This is a reference implementation which
does NOT integrate with any deep learning framework.

## Usage

### For use with spatially 2D tensors

Given your input tensor is of shape: `(N, C, H, W)`

``` python
import numpy
from online_norm_2d import OnlineNorm2d

N, C, H, W = 64, 128, 32, 32
norm = OnlineNorm2d(C, .999, .99)
inputs = numpy.random.randn(N, C, H, W)  # generate fake input
output = norm(inputs)
grad_out = numpy.random.randn(N, C, H, W)  # generate fake gradient
grad_in = norm.backward(grad_out)
```

### For use with spatially 1D tensors

Given your input tensor is of shape: `(N, C)`

``` python
import numpy
from online_norm_1d import OnlineNorm1d

N, C = 64, 128
norm = OnlineNorm1d(C, .999, .99)
inputs = numpy.random.randn(N, C)  # generate fake input
output = norm(inputs)
grad_out = numpy.random.randn(N, C)  # generate fake gradient
grad_in = norm.backward(grad_out)
```

## change log

### 2020-03-25

#### Added

- Added [activation clamping](LinkToActClampPaper) as an error checking mechanism (ecm) and make it the defualt ecm.
