/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2019 Cerebras Systems Inc.
 * All rights reserved.
 *
 * Define norm func signatures
 *
 * Author:  Vitaliy Chiley
 * Contact: {vitaliy, info}@cerebras.net
 */

#pragma once
#include <torch/extension.h>
#include <torch/types.h>


std::vector<at::Tensor> norm_fwd_cpu(
    const at::Tensor input,
    at::Tensor s_mu,
    at::Tensor s_var,
    const float afwd,
    const float eps);

std::vector<at::Tensor> norm_bwd_cpu(
    const at::Tensor grad_out,
    at::Tensor u,
    at::Tensor v,
    const at::Tensor out,
    const at::Tensor scale,
    const float abkw);

#ifdef WITH_CUDA
std::vector<at::Tensor> norm_fwd_cuda(
    const at::Tensor input,
    at::Tensor s_mu,
    at::Tensor s_var,
    const float afwd,
    const float eps);

std::vector<at::Tensor> norm_bwd_cuda(
    const at::Tensor grad_out,
    at::Tensor u,
    at::Tensor v,
    const at::Tensor out,
    const at::Tensor scale,
    const float abkw);
#endif

// Interface for Python
inline std::vector<at::Tensor> norm_fwd(
    const at::Tensor& input,
    at::Tensor& s_mu,
    at::Tensor& s_var,
    const float afwd,
    const float eps) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return norm_fwd_cuda(
        input,
        s_mu,
        s_var,
        afwd,
        eps);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return norm_fwd_cpu(
      input,
      s_mu,
      s_var,
      afwd,
      eps);
}

inline std::vector<at::Tensor> norm_bwd(
    const at::Tensor& grad_out,
    at::Tensor& u,
    at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& scale,
    const float abkw) {
  if (grad_out.type().is_cuda()) {
#ifdef WITH_CUDA
    return norm_bwd_cuda(
        grad_out,
        u,
        v,
        out,
        scale,
        abkw);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return norm_bwd_cpu(
      grad_out,
      u,
      v,
      out,
      scale,
      abkw);
}
