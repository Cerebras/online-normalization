# TensorFlow implementation of Online Normalization

This folder holds the [Online Normalization Algorithm](https://arxiv.org/abs/1905.05894) implemented in [TensorFlow](https://www.tensorflow.org//).

## Installation

```bash
pip install .
```

## Usage

### For use with spatially 2D tensors in channel_first format (GPU only)
NOTE: tf's CPU conv implementations do not support channel_first
Given your input tensor is of shape: `(N, C, H, W)`
```
import numpy
import tensorflow as tf
from online_norm_tf import online_norm

N, C, H, W = 8, 256, 32, 32
inputs = numpy.random.randn(N, C, H, W)

input_placeholder = tf.placeholder(tf.float32, shape=(N, C, H, W))
norm = online_norm(input_placeholder, training=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# get the output of the tf version of the layer
out = sess.run([norm], feed_dict={input_placeholder: inputs})
```

### For use with Dense Layers
Given your input tensor is of shape: `(N, C)`
```
import numpy
import tensorflow as tf
from online_norm_tf import online_norm

N, C = 8, 256
inputs = numpy.random.randn(N, C)

input_placeholder = tf.placeholder(tf.float32, shape=(N, C))
norm = online_norm(input_placeholder, training=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# get the output of the tf version of the layer
out = sess.run([norm], feed_dict={input_placeholder: inputs})
```

## Testing installation

```bash
python setup.py test
```

## change log

### 2020-03-25

#### Added

- Added [activation clamping](https://www.cerebras.net/error-compensation-mechanism-in-online-normalization/) as an error compensation mechanism (ecm) and make it the default ecm.

#### Changed

- Combine BatchedOnlineNorm and OnlineNorm into one class since its one algorithm.

### 2020-04-05

#### Added

- Loop version of OnlineNorm implemented
- Tests updated

### 2020-04-01

#### Added

- Added CUDA kernel for online norm (removed python loop version of norm and NormBatched)
