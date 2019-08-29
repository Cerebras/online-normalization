# Experiment Reproducibility

Here we analyze the reproducibility / run to run variability of experimentation. 
If experiments were run multiple times, the metrics are analyzed on the epoch 
during which performance is best. 
The epoch during which performance is best is defined on the median curve. 
Median is used instead of the mean because some of the networks without 
normalization (or using Layer Normalization) had some out lier training runs. 
Showing the mean would unfairly misrepresent those networks as having poor 
average performance. 
If a network setting is run only once, `Min`, `Max`, and `Standard Deviation` 
are set to `N/A`.

## Metric Statistic

Dataset       | Network      | Metric             | Norm | Best Epoch | Total Epochs | Median |  Min  | Max      | Mean    | Standard Deviation | Number of Runs 
--------------|--------------|--------------------|------|------------|--------------|--------|-------|----------|---------|--------------------|----------------
CIFAR10       | ResNet20     | Accuracy           |  ON  |       184  |      250     |  92.3  | 91.8  |     92.3 |   92.1  |     0.20           |       5         
CIFAR10       | ResNet20     | Accuracy           |  BN  |       202  |      250     |  92.2  | 91.9  |     92.5 |   92.2  |     0.22           |       5         
CIFAR10       | ResNet20     | Accuracy           |  LN  |       243  |      250     |  87.4  |  N/A  |      N/A |   87.4  |      N/A           |       1         
CIFAR10       | ResNet20     | Accuracy           |  IN  |       181  |      250     |  90.4  |  N/A  |      N/A |   90.4  |      N/A           |       1         
CIFAR10       | ResNet20     | Accuracy           |  GN  |       164  |      250     |  90.3  |  N/A  |      N/A |   90.3  |      N/A           |       1         
CIFAR100      | ResNet20     | Accuracy           |  ON  |       188  |      250     |  68.6  | 67.6  |     69.4 |   68.5  |     0.62           |       5         
CIFAR100      | ResNet20     | Accuracy           |  BN  |       159  |      250     |  68.6  | 68.1  |     69.2 |   68.6  |     0.38           |       5         
CIFAR100      | ResNet20     | Accuracy           |  LN  |       179  |      250     |  59.2  |  N/A  |      N/A |   59.2  |      N/A           |       1         
CIFAR100      | ResNet20     | Accuracy           |  IN  |       196  |      250     |  63.1  |  N/A  |      N/A |   63.1  |      N/A           |       1         
CIFAR100      | ResNet20     | Accuracy           |  GN  |       153  |      250     |  63.3  |  N/A  |      N/A |   63.3  |      N/A           |       1         
ImageNet      | ResNet50     | Accuracy           |  ON  |        91  |      100     |  76.3  |  N/A  |      N/A |   76.3  |      N/A           |       1         
ImageNet      | ResNet50     | Accuracy           |  BN  |        86  |      100     |  76.4  |  N/A  |      N/A |   76.4  |      N/A           |       1         
Synthetic     | U-Net        | Jaccard Similarity |  ON  |        35  |       40     |  0.977 | 0.972 |    0.980 |   0.976 |     0.0022         |      50         
Synthetic     | U-Net        | Jaccard Similarity |  BN  |        35  |       40     |  0.976 | 0.970 |    0.982 |   0.976 |     0.0023         |      50         
Synthetic     | U-Net        | Jaccard Similarity |  --  |        39  |       40     |  0.961 | 0.0   |    0.973 |   0.941 |     0.1373         |      50         
Fashion MNIST | FC           | Accuracy           |  ON  |         9  |       10     |  88.9  | 86.2  |     89.9 |   88.8  |     0.53           |     400         
Fashion MNIST | FC           | Accuracy           |  BN  |         9  |       10     |  88.8  | 86.0  |     89.7 |   88.6  |     0.58           |     400         
Fashion MNIST | FC           | Accuracy           |  LN  |         9  |       10     |  88.3  | 85.5  |     89.4 |   88.2  |     0.60           |     400         
Fashion MNIST | FC           | Accuracy           |  --  |         9  |       10     |  88.2  | 85.5  |     89.0 |   88.1  |     0.52           |     400         
Penn Treebank | RNN Network  | Perplexity         |  ON  |        33  |       40     | 129.2  | 127.0 |    181.3 |   134.4 |    12.78           |      25         
Penn Treebank | RNN Network  | Perplexity         |  LN  |        38  |       40     | 141.9  | 136.3 |    246.6 |   147.4 |    20.66           |      25         
Penn Treebank | RNN Network  | Perplexity         |  --  |        30  |       40     | 172.0  | 157.5 |   7106.9 |   633.6 |  1446.01           |      25         
Penn Treebank | LSTM Network | Perplexity         |  ON  |        20  |       25     | 113.4  | 110.8 |    123.0 |   114.8 |     3.63           |      25         
Penn Treebank | LSTM Network | Perplexity         |  LN  |        22  |       25     | 121.6  | 118.2 |  21217.6 |   950.0 |  4053.81           |      25         
Penn Treebank | LSTM Network | Perplexity         |  --  |        25  |       25     | 124.4  | 121.0 | 133293.5 |  5459.0 | 26094.11           |      25         


## Loss Statistic

Dataset       | Network      | Norm | Best Epoch | Total Epochs | Median | Min   | Max    | Mean  | Standard Deviation | Number of Runs 
--------------|--------------|------|------------|--------------|--------|-------|--------|-------|--------------------|----------------
CIFAR10       | ResNet20     |  ON  |       102  |      250     |  0.26  | 0.26  |  0.27  | 0.26  |  0.0017            |       5         
CIFAR10       | ResNet20     |  BN  |       103  |      250     |  0.26  | 0.26  |  0.27  | 0.26  |  0.0041            |       5         
CIFAR10       | ResNet20     |  LN  |       193  |      250     |  0.39  |  N/A  |   N/A  | 0.39  |     N/A            |       1         
CIFAR10       | ResNet20     |  IN  |       130  |      250     |  0.31  |  N/A  |   N/A  | 0.31  |     N/A            |       1         
CIFAR10       | ResNet20     |  GN  |       109  |      250     |  0.32  |  N/A  |   N/A  | 0.32  |     N/A            |       1         
CIFAR100      | ResNet20     |  ON  |       107  |      250     |  1.12  | 1.12  |  1.14  | 1.13  |  0.0079            |       5         
CIFAR100      | ResNet20     |  BN  |       103  |      250     |  1.14  | 1.14  |  1.15  | 1.14  |  0.0034            |       5         
CIFAR100      | ResNet20     |  LN  |       247  |      250     |  1.47  |  N/A  |   N/A  | 1.47  |     N/A            |       1         
CIFAR100      | ResNet20     |  IN  |       233  |      250     |  1.32  |  N/A  |   N/A  | 1.32  |     N/A            |       1         
CIFAR100      | ResNet20     |  GN  |       111  |      250     |  1.35  |  N/A  |   N/A  | 1.35  |     N/A            |       1         
ImageNet      | ResNet50     |  ON  |        97  |      100     |  0.94  |  N/A  |   N/A  | 0.94  |     N/A            |       1         
ImageNet      | ResNet50     |  BN  |        85  |      100     |  0.97  |  N/A  |   N/A  | 0.97  |     N/A            |       1         
Synthetic     | U-Net        |  ON  |        33  |       40     |  0.007 | 0.006 |  0.010 | 0.007 |  0.00098           |      50         
Synthetic     | U-Net        |  BN  |        35  |       40     |  0.007 | 0.005 |  0.009 | 0.007 |  0.00085           |      50         
Synthetic     | U-Net        |  --  |        39  |       40     |  0.010 | 0.008 |  0.511 | 0.021 |  0.071             |      50         
Fashion MNIST | FC           |  ON  |         7  |       10     |  0.323 | 0.301 |  0.440 | 0.326 |  0.017             |     400         
Fashion MNIST | FC           |  BN  |         6  |       10     |  0.328 | 0.299 |  0.391 | 0.332 |  0.017             |     400         
Fashion MNIST | FC           |  LN  |         9  |       10     |  0.326 | 0.301 |  0.421 | 0.329 |  0.017             |     400         
Fashion MNIST | FC           |  --  |         9  |       10     |  0.332 | 0.313 |  0.408 | 0.333 |  0.013             |     400         
Penn Treebank | RNN Network  |  ON  |        33  |       40     |  4.86  | 4.84  |  5.20  | 4.901 |  2.55              |      25         
Penn Treebank | RNN Network  |  LN  |        38  |       40     |  4.96  | 4.92  |  5.51  | 4.993 |  3.03              |      25         
Penn Treebank | RNN Network  |  --  |        30  |       40     |  5.14  | 5.06  |  8.87  | 6.451 |  7.28              |      25         
Penn Treebank | LSTM Network |  ON  |        20  |       25     |  4.73  | 4.70  |  4.81  | 4.74  |  1.29              |      25         
Penn Treebank | LSTM Network |  LN  |        22  |       25     |  4.80  | 4.77  |  9.96  | 6.85  |  8.30              |      25         
Penn Treebank | LSTM Network |  --  |        25  |       25     |  4.82  | 4.79  | 11.80  | 8.60  | 10.16              |      25         
