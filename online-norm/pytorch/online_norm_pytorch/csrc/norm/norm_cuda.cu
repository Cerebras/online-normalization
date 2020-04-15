/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2019 Cerebras Systems Inc.
 * All rights reserved.
 *
 * Define norm fwd / bwd cpp functions and cuda kernels
 * TODO:
 *    1. This implemetation absorbs only the loop into the kernel. A more
 *       advanced kernel would absorb the entire operation into the CUDA kernel
 *       but for now this is fast enough
 *
 * Author:  Vitaliy Chiley
 * Contact: {vitaliy, info}@cerebras.net
 */

#include "norm.h"

#include <torch/extension.h>

#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <stdio.h>


#define Idx3(n, c, d, N, C, D) (((n)*(C)*(D)) + ((c)*(D)) + (d))
#define Idx2(n, c, N, C) (((n)*(C)) + (c))

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename scalar_t>
__global__ void norm_fwd_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ var,
    float* __restrict__ s_mu,
    float* __restrict__ s_var,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ scale,
    const unsigned int C, const unsigned int N,
    const float afwd, const float eps) {

  const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < C) {
    unsigned int idx;
    float delta;
    float s_var_local = s_var[c];
    float s_mu_local = s_mu[c];

    for(int n = 0; n < N; ++n){
      idx = Idx2(n, c, N, C);
      scale[idx] = (scalar_t)(sqrt(s_var_local + eps));
      mean[idx] = (scalar_t)(s_mu_local);

      delta = mu[idx] - s_mu_local;

      s_var_local = afwd * s_var_local + (1. - afwd) * var[idx] + afwd * (1. - afwd) * delta * delta;
      s_mu_local = s_mu_local + (1. - afwd) * delta;
    };

    s_var[c] = s_var_local;
    s_mu[c] = s_mu_local;
  };
}

std::vector<at::Tensor> norm_fwd_cuda(
    const at::Tensor input,
    at::Tensor s_mu,
    at::Tensor s_var,
    const float afwd,
    const float eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(s_mu);
  CHECK_INPUT(s_var);

  const auto input_shape = input.sizes();
  const auto inputs = input.reshape({input_shape[0], input_shape[1], -1}).toType(at::ScalarType::Float);

  const auto mu = inputs.mean({2});
  const auto var = (inputs - mu.unsqueeze(2)).pow(2).mean({2});

  auto scale = at::zeros_like(mu).toType(input.scalar_type());
  auto mean = at::zeros_like(mu).toType(input.scalar_type());

  const unsigned int N = input_shape[0];
  const unsigned int C = input_shape[1];
  
  const unsigned int threads = min(int(C), 1024);
  const dim3 blocks(ceil(float(C) / threads));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "norm_fwd", ([&] {
    norm_fwd_kernel<scalar_t><<<blocks, threads>>>(
        mu.data<float>(),
        var.data<float>(),
        s_mu.data<float>(),
        s_var.data<float>(),
        mean.data<scalar_t>(),
        scale.data<scalar_t>(),
        C, N, afwd, eps);
  }));
  THCudaCheck(cudaGetLastError());

  const auto out = ((inputs.toType(input.scalar_type()) - mean.unsqueeze(2)) / scale.unsqueeze(2)).reshape(input_shape);

  return {out, scale, s_mu, s_var};
}

__device__ void warp_reduce(volatile float *s_mem, const unsigned int t_id, const unsigned int d) {
  for (unsigned int ridx = 32; ridx > 0; ridx /= 2) {
    if (d > ridx) { if (t_id < ridx) { if ((t_id + ridx) < d) { s_mem[t_id] += s_mem[t_id + ridx]; } } __syncwarp(); }
  }
}

template <typename scalar_t>
__global__ void norm_bwd_ctrl_kernel(
    const scalar_t* __restrict__ grad_out,
    float* s_v,
    float* s_u,
    const scalar_t* __restrict__ out,
    scalar_t* __restrict__ scale,
    scalar_t* __restrict__ grad_in,
    const unsigned int C, const unsigned int N, const unsigned int D,
    const float abkw) {

  const unsigned int t_id = threadIdx.x;
  const unsigned int c = blockIdx.x;
  const unsigned int d = blockDim.x;
  unsigned int idx3, idx;

  extern __shared__ float s_mem_v[];
  float *s_mem_u = &s_mem_v[d];

  scalar_t grad_tmp;

  for(int n = 0; n < N; ++n){
    s_mem_v[t_id] = 0;                              // reset shared mem
    s_mem_u[t_id] = 0;                              // reset shared mem
    for (idx = t_id; idx < D; idx += d) {
      idx3 = Idx3(n, c, idx, N, C, D);              // idx in global mem

      // vctrl logic
      grad_tmp = grad_out[idx3] - (1. - (scalar_t)(abkw)) * (scalar_t)(s_v[c]) * out[idx3];
      s_mem_v[t_id] += (float)(grad_tmp) * (float)(out[idx3]);    // start reduction
      
      // scale grad
      grad_tmp = grad_tmp / scale[Idx2(n, c, N, C)];

      // uctrl logic
      grad_in[idx3] = grad_tmp - (1. - (scalar_t)(abkw)) * (scalar_t)(s_u[c]);
      s_mem_u[t_id] += (float)(grad_in[idx3]);      // start reduction
    };  
    __syncthreads();

    // update vctrl
    // reduce within thread block % warp reduction
    for (idx = 512; idx > 32; idx /= 2) {
      if (d > idx) { if (t_id < idx) { if ((t_id + idx) < d) { s_mem_v[t_id] += s_mem_v[t_id + idx]; } } __syncthreads(); }
    }
    if (t_id < 32) { warp_reduce(s_mem_v, t_id, d); }     // reduce within warp

    // update vctrl / mv reduction to global mem
    if (t_id == 0) { s_v[c] += (s_mem_v[0] / D); }

    // update uctrl
    // reduce within thread block % warp reduction
    for (idx = 512; idx > 32; idx /= 2) {
      if (d > idx) { if (t_id < idx) { if ((t_id + idx) < d) { s_mem_u[t_id] += s_mem_u[t_id + idx]; } } __syncthreads(); }
    }
    if (t_id < 32) { warp_reduce(s_mem_u, t_id, d); }     // reduce within warp

    // update uctrl / mv reduction to global mem
    if (t_id == 0) { s_u[c] += (s_mem_u[0] / D); }
    __syncthreads();
  }
  __syncthreads();
}

std::vector<at::Tensor> norm_bwd_cuda(
    const at::Tensor grad_out,
    at::Tensor u,
    at::Tensor v,
    const at::Tensor out,
    const at::Tensor scale,
    const float abkw) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(u);
  CHECK_INPUT(v);
  CHECK_INPUT(out);
  CHECK_INPUT(scale);

  // Assumes channel_first contiguous data

  const unsigned int N = grad_out.size(0);
  const unsigned int C = grad_out.size(1);
  const unsigned int D = grad_out[0][0].numel();

  auto grad_in = at::empty_like(grad_out);

  const unsigned int threads = min(int(D), 512);
  const dim3 blocks(C);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "norm_bwd_ctrl", ([&] {
    norm_bwd_ctrl_kernel<scalar_t><<<blocks, threads, 2 * threads * sizeof(float)>>>(
        grad_out.data<scalar_t>(),
        v.data<float>(),
        u.data<float>(),
        out.data<scalar_t>(),
        scale.data<scalar_t>(),
        grad_in.data<scalar_t>(),
        C, N, D, abkw);
  }));
  THCudaCheck(cudaGetLastError());

  return {grad_in, u, v};
}
