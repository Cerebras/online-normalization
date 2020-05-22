### To build:

1. Setup a virtual environment with TF1.14

```
virtualenv -p python3 venv_on
source venv_on/bin/activate
pip install tensorflow==1.14
```
2. Link cuda to tensorflow

```
cd venv_on/lib64/python3.6/site-packages/tensorflow/include/third_party
mkdir gpus
cd gpus
mkdir cuda
cd cuda
ln -s /usr/local/cuda/include ./include
```

3. Build the shared libraries

```
bash build.sh
```

4. Run the test file:

```
python kernel_example_test.py
```