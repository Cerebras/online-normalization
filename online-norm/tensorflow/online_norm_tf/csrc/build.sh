

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


g++ -std=c++11 -shared -o online_norm_cpu.so online_norm.cc ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]} -DCPU_ONLY=1


export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin


nvcc -std=c++11 -c -o online_norm_gpu.cu.o online_norm.cu.cc \
  ${TF_CFLAGS[@]} -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG


g++ -std=c++11 -shared -o online_norm_gpu.so online_norm.cc \
  online_norm_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64

# check if kernels are correctly linked by running:
# nm -C online_norm.so | grep Functor
