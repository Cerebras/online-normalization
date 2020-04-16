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


// forward kernel
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


// device reduction helper func
__device__ void warp_reduce(
  volatile float *s_mem,
  const unsigned int t_id,
  const unsigned int d
) {
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


/* 
 * OnlineNorm backward kernel implementation
 * The ON bwd algorithm is:
 *
 *    grad_tmp = grad_out - (1 - abkw) v_ctrl * out
 *    v_ctrl = v_ctrl + mean(grad_tmp * out)
 *    grad_tmp = grad_tmp / scale
 *    grad_in = grad_tmp - (1 - abkw) u_ctrl
 *    u_ctrl = u_ctrl + mean(grad_in)
 *
 * There out is the output of ON, scale is the std. dev. used to scale the data
 * in the fwd pass, grad_out is the gradient of the output, grad_in is the
 * gradient of the input, v_ctrl is the v control variable, u_ctrl is the u
 * control variable, abkw is the backward decay factor, and mean(.) is the mean
 * operator.
 *
 * The ON algorithm loops over N samples. Each sample has an associated
 * grad_out, out, and scale. The v and u control variables are applied to the
 * the gradient to produce the gradient of the input. s_mem_v and s_mem_u are
 * shared memory used in the reduction (reduction over D) needed to update
 * v_ctrl and u_ctrl. Each thread block operates on an one of C features
 * (ie. channel when operating on spatial data). Each channel has a v and u
 * control variable which are updated per sample by the reductions per thread
 * block.
 *
 * The kernel assumes contiguous inputs inputs of shape (N, C, *) where D is
 * the product of *.
 */
template <typename T>
__global__ void norm_bwd_kernel(
    const T* __restrict__ grad_out,
    const float* v_ctrl,
    const float* u_ctrl,
    const T* __restrict__ out,
    const T* __restrict__ scale,
    float* out_v_ctrl,
    float* out_u_ctrl,
    T* __restrict__ grad_in,
    const unsigned int C, const unsigned int N, const unsigned int D,
    const float abkw) {

  const unsigned int t_id = threadIdx.x;
  const unsigned int c = blockIdx.x;
  const unsigned int d = blockDim.x;
  unsigned int idx3, idx;

  extern __shared__ float s_mem_v[];
  float *s_mem_u = &s_mem_v[d];

  T grad_tmp;

  if (t_id == 0) {
    out_v_ctrl[c] = v_ctrl[c];
    out_u_ctrl[c] = u_ctrl[c];
  }
  __syncthreads();

  for(int n = 0; n < N; ++n){
    s_mem_v[t_id] = 0;                              // reset v_ctrl shared mem
    s_mem_u[t_id] = 0;                              // reset u_ctrl shared mem
    for (idx = t_id; idx < D; idx += d) {
      idx3 = Idx3(n, c, idx, N, C, D);              // idx in global mem

      // v_ctrl logic
      grad_tmp = grad_out[idx3] - ((T)(1. - abkw)) * (T)(out_v_ctrl[c]) * out[idx3];
      // start reduction for v_ctrl updt
      s_mem_v[t_id] += (float)(grad_tmp) * (float)(out[idx3]);
      
      // scale grad
      grad_tmp = grad_tmp / scale[Idx2(n, c, N, C)];

      // u_ctrl logic
      grad_tmp = grad_tmp - ((T)(1. - abkw)) * (T)(out_u_ctrl[c]);
      grad_in[idx3] = grad_tmp;
      // start reduction for u_ctrl updt
      s_mem_u[t_id] += (float)(grad_tmp);
    }
    __syncthreads();

    // reduce within thread block % warp reduction
    for (idx = 512; idx > 32; idx /= 2) {
      if (d > idx) {
        if ((t_id < idx) && ((t_id + idx) < d)) {
          s_mem_v[t_id] += s_mem_v[t_id + idx];
          s_mem_u[t_id] += s_mem_u[t_id + idx];
        }
        __syncthreads();
      }
    }
    __syncthreads();

    // reduce smem within warp
    if (t_id < 32) {
      warp_reduce(s_mem_v, t_id, d);
      warp_reduce(s_mem_u, t_id, d);
    }

    // move reduction to global mem to updt ctrl variables
    if (t_id == 0) {
      out_v_ctrl[c] += (s_mem_v[0] / D);    // update v_ctrl
      out_u_ctrl[c] += (s_mem_u[0] / D);    // update u_ctrl
    }
    __syncthreads();
  }
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
struct OnlineNormBwdFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    const T* grad_outputs,
    const float* in_v,
    const float* in_u,
    const T* outputs,
    const T* scale,
    float* out_v,
    float* out_u,
    T* grad_in,
    const int C,
    const int N,
    const int D,
    const float abkw
  ) {
    int thread_per_block = min(int(D), 512);
    int block_count = C;
    norm_bwd_kernel<T><<<block_count, thread_per_block, 2 * thread_per_block * sizeof(float), d.stream()>>>(
      grad_outputs,
      in_v,
      in_u,
      outputs,
      scale,
      out_v,
      out_u,
      grad_in,
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

template struct OnlineNormBwdFunctor<GPUDevice, float>;
template struct OnlineNormBwdFunctor<GPUDevice, Eigen::half>;

#endif  // GOOGLE_CUDA
