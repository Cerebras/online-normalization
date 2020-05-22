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


// device shared mem reduction within warp helper func
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
 * OnlineNorm forward kernel implementation
 * The ON fwd algorithm is:
 *
 *    scale = sqrt(s_var + eps)
 *    out = (input - s_mu) / scale
 *    mu, var = moments(input)
 *    diff = mu - s_mu
 *    s_var = afwd * s_var + (1 - afwd) * var + afwd * (1 - afwd) * diff * diff
 *    s_mu = s_mu + (1 - afwd) * diff
 *
 * where out is the output of ON, scale is the std. dev. used to scale the data
 * in the fwd pass and is cached for the bwd pass, eps is used for numerical
 * stability, s_mu and s_var are the streaming mean and variance,
 * mu and var are the sample mean and variance of the input, diff is an
 * intermediate stored variable, and afwd is the forward decay factor.
 *
 * The ON algorithm loops over N samples. s_mem_mu and s_mem_var are
 * shared memory used in the reduction (reduction over D) needed to update
 * s_mu and s_var. Each thread block operates on an one of C features
 * (ie. channel when operating on spatial data). Each channel has a s_mu and
 * s_var streaming statistics which are updated per sample by the reductions
 * per thread block.
 *
 * The kernel assumes contiguous inputs inputs of shape (N, C, *) where D is
 * the product of *.
 */
template <typename T>
__global__ void norm_fwd_kernel(
    const T* __restrict__ input,
    const float* in_s_mu,
    const float* in_s_var,
    float* s_mu,
    float* s_var,
    T* __restrict__ out,
    T* __restrict__ scale,
    const unsigned int C, const unsigned int N, const unsigned int D,
    const float afwd, const float eps) {

  const unsigned int t_id = threadIdx.x;
  const unsigned int c = blockIdx.x;
  const unsigned int d = blockDim.x;
  unsigned int idx3, idx;

  extern __shared__ float s_mem_mu[];
  float *s_mem_var = &s_mem_mu[d];

  float in_elem_f, sample_mu, sample_var, diff;
  T in_elem, m, s;

  if (t_id == 0) {
    s_mu[c] = in_s_mu[c];
    s_var[c] = in_s_var[c];
  }
  __syncthreads();

  for(int n = 0; n < N; ++n){
    s_mem_mu[t_id] = 0;                             // reset sample mu shared mem
    s_mem_var[t_id] = 0;                            // reset sample var shared mem
    // propagate fwd activations and start reduction to compute input mu and var
    m = (T)(s_mu[c]);
    s = (T)(sqrt(s_var[c] + eps));

    if (t_id == 0) { scale[Idx2(n, c, N, C)] = s; } // store scale used

    for (idx = t_id; idx < D; idx += d) {
      idx3 = Idx3(n, c, idx, N, C, D);              // idx in global mem
      in_elem = input[idx3];                        // get input element
      out[idx3] = (in_elem - m) / s;                // compute output
    
      // start 1st and 2nd moment reductions
      in_elem_f = (float)(in_elem);
      s_mem_mu[t_id] += in_elem_f;                  // 1st moment reduction
      s_mem_var[t_id] += in_elem_f * in_elem_f;     // 2nd moment reduction
    }
    __syncthreads();

    // reduce within thread block % warp reduction
    for (idx = 512; idx > 32; idx /= 2) {
      if (d > idx) {
        if ((t_id < idx) && ((t_id + idx) < d)) {
          s_mem_mu[t_id] += s_mem_mu[t_id + idx];   // 1st moment reduction
          s_mem_var[t_id] += s_mem_var[t_id + idx]; // 2nd moment reduction
        }
        __syncthreads();
      }
    }

    // reduce smem within warp
    if (t_id < 32) {
      warp_reduce(s_mem_mu, t_id, d);               // 1st moment reduction
      warp_reduce(s_mem_var, t_id, d);              // 2nd moment reduction
    }

    if (t_id == 0) {
      // compute sample mu and var to update streaming stats
      sample_mu = s_mem_mu[0] / D;
      sample_var = (s_mem_var[0] / D) - (sample_mu * sample_mu);

      // update streaming stats
      diff = sample_mu - s_mu[c];
      s_var[c] = afwd * s_var[c] + (1. - afwd) * sample_var + afwd * (1. - afwd) * diff * diff;
      s_mu[c] = s_mu[c] + (1. - afwd) * diff;
    }
    __syncthreads();
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
    const T* input,
    const float* in_s_mu,
    const float* in_s_var,
    float* out_s_mu,
    float* out_s_var,
    T* out,
    T* scale,
    const int C,
    const int N,
    const int D,
    const float afwd,
    const float eps
  ){
    int thread_per_block = min(int(D), 512);
    int block_count = C;
    norm_fwd_kernel<T><<<block_count, thread_per_block, 2 * thread_per_block * sizeof(float), d.stream()>>>(
      input,
      in_s_mu,
      in_s_var,
      out_s_mu,
      out_s_var,
      out,
      scale,
      C,
      N,
      D,
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
