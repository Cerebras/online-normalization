// online_norm.cu.cc
#ifdef GOOGLE_CUDA
#include "online_norm.h"
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#define Idx3(n, c, d, N, C, D) (((n)*(C)*(D)) + ((c)*(D)) + (d))
#define Idx2(n, c, N, C) (((n)*(C)) + (c))


// kernel 1
template <typename T>
__global__ void norm_fwd_kernel(
  const float* __restrict__ mu,
  const float* __restrict__ var,
  const float* __restrict__ in_s_mu,
  const float* __restrict__ in_s_var,
  float* __restrict__ out_s_mu,
  float* __restrict__ out_s_var,
  T* __restrict__ mean,
  T* __restrict__ scale,
  const int C,
  const int N,
  const float afwd,
  const float eps
) {
  const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    unsigned int idx;
    float delta;
    float s_var_local = in_s_var[c];
    float s_mu_local = in_s_mu[c];

    for(int n = 0; n < N; ++n) {
      idx = Idx2(n, c, N, C);
      scale[idx] = (T)(sqrt(s_var_local + eps));
      mean[idx] = (T)(s_mu_local);
      delta = mu[idx] - s_mu_local;
      s_var_local = afwd * s_var_local + (1. - afwd) * var[idx] + \
          afwd * (1. - afwd) * delta * delta;
      s_mu_local = s_mu_local + (1. - afwd) * delta;
    };

    out_s_var[c] = s_var_local;
    out_s_mu[c] = s_mu_local;
  };
}


// // kernel 2
template <typename T>
__global__ void norm_uctrl_kernel(
  const float* __restrict__ mu_delta,
  const float* __restrict__ s_u,
  float* __restrict__ out_s_u,
  T* __restrict__ d_u,
  const unsigned int C,
  const unsigned int N,
  const float abkw
) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    int idx;
    float delta;
    float s_u_local = s_u[c];

    for(int n = 0; n < N; ++n){
      idx = Idx2(n, c, N, C);
      d_u[idx] = (T)(s_u_local);
      delta = mu_delta[idx] - s_u_local;
      s_u_local = s_u_local + (1. - abkw) * delta;
    };

    out_s_u[c] = s_u_local;
  };
}


// kernel 3
__device__ void warp_reduce(volatile float *s_mem, const unsigned int t_id, const unsigned int d) {
  for (unsigned int ridx = 32; ridx > 0; ridx /= 2) {
    if (d > ridx) {
      if (t_id < ridx) {
        if ((t_id + ridx) < d) {
          s_mem[t_id] += s_mem[t_id + ridx];
        }
      }
    }
  }
}


// //kerenl 4
template <typename T>
__global__ void norm_vctrl_kernel(
  const T* __restrict__ grad_out,
  const float* s_v,
  const T* __restrict__ out,
  float* out_s_v,
  T* __restrict__ grad_tmp,
  const int C,
  const int N,
  const int D,
  const float abkw
) {
  const unsigned int t_id = threadIdx.x;
  const unsigned int c = blockIdx.x;
  const unsigned int d = blockDim.x;
  unsigned int idx3, idx;

  extern __shared__ float s_mem[];
  out_s_v[c] =s_v[c];
  for(int n = 0; n < N; ++n){
    s_mem[t_id] = 0;                                // reset shared mem
    for (idx = t_id; idx < D; idx += d) {
      idx3 = Idx3(n, c, idx, N, C, D);              // idx in global mem
      // vctrl logic
      grad_tmp[idx3] = grad_out[idx3] - (T)(1. - abkw) * (T)(out_s_v[c]) * out[idx3];
      s_mem[t_id] += (float)(grad_tmp[idx3]) * (float)(out[idx3]);  // start reduction
    };
    __syncthreads();

    // reduce within thread block % warp reduction
    for (idx = 512; idx > 32; idx /= 2) {
      if (d > idx) {
        if (t_id < idx) {
          if ((t_id + idx) < d) {
            s_mem[t_id] += s_mem[t_id + idx];
          }
        }
      __syncthreads();
      }
    }
    if (t_id < 32) {
      warp_reduce(s_mem, t_id, d);
    }     // reduce within warp

    // update vctrl / mv reduction to global mem
    if (t_id == 0) {
      out_s_v[c] += (s_mem[0] / D);
    };
    __syncthreads();
  };
  __syncthreads();
}


template <typename T>
struct OnlineNormFwdFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    const float* mu,
    const float* var,
    const float* in_s_mu,
    const float* in_s_var,
    float* out_s_mu,
    float* out_s_var,
    T* mean,
    T* scale,
    const int C,
    const int N,
    const float afwd,
    const float eps
  ){
    int thread_per_block = min(int(C), 1024);
    int block_count = ceil(float(C) / thread_per_block);
    norm_fwd_kernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(
      mu,
      var,
      in_s_mu,
      in_s_var,
      out_s_mu,
      out_s_var,
      mean,
      scale,
      C,
      N,
      afwd,
      eps
    );
  }
};


template <typename T>
struct OnlineNormUCtrlFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    const float* mu_delta,
    const float* in_u,
    float* out_u,
    T* d_u,
    const int C,
    const int N,
    const float abkw
  ) {
    int thread_per_block = min(int(C), 1024);
    int block_count = ceil(float(C) / thread_per_block);
    norm_uctrl_kernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(
      mu_delta,
      in_u,
      out_u,
      d_u,
      C,
      N,
      abkw
    );
  }
};


template <typename T>
struct OnlineNormVCtrlFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    const T* grad_outputs,
    const T* outputs,
    const float* in_v,
    float* out_v,
    T* grad_tmp,
    const int C,
    const int N,
    const int D,
    const float abkw
  ) {
    int thread_per_block = min(int(D), 1024);
    int block_count = C;
    norm_vctrl_kernel<T><<<block_count, thread_per_block, thread_per_block * sizeof(float), d.stream()>>>(
      grad_outputs,
      in_v,
      outputs,
      out_v,
      grad_tmp,
      C,
      N,
      D,
      abkw
    );
  }
};


// Explicitly instantiate functors for the types of OpKernels registered.
template struct OnlineNormFwdFunctor<GPUDevice, float>;
template struct OnlineNormFwdFunctor<GPUDevice, Eigen::half>;

template struct OnlineNormUCtrlFunctor<GPUDevice, float>;
template struct OnlineNormUCtrlFunctor<GPUDevice, Eigen::half>;

template struct OnlineNormVCtrlFunctor<GPUDevice, float>;
template struct OnlineNormVCtrlFunctor<GPUDevice, Eigen::half>;

#endif  // GOOGLE_CUDA

