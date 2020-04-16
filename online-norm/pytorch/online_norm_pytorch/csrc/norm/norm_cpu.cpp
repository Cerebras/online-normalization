/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2019 Cerebras Systems Inc.
 * All rights reserved.
 *
 * Define norm fwd / bwd for CPU
 *
 * Author:  Vitaliy Chiley
 * Contact: {vitaliy, info}@cerebras.net
 */

#include "norm.h"


#define CHECK_CUDA(x) AT_ASSERTM(!x.type().is_cuda(), #x " must not be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x);


std::vector<at::Tensor> norm_fwd_cpu(
    const at::Tensor input,
    at::Tensor s_mu,
    at::Tensor s_var,
    const float afwd,
    const float eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(s_mu);
  CHECK_INPUT(s_var);

  const auto input_shape = input.sizes();
  const auto inputs = input.view({input_shape[0], input_shape[1], -1}).toType(at::ScalarType::Float);

  const auto mu = inputs.mean({2});
  const auto var = (inputs - mu.unsqueeze(2)).pow(2).mean({2});

  auto mu_b = at::empty_like(mu).toType(input.scalar_type());
  auto var_b = at::empty_like(mu).toType(input.scalar_type());

  auto delta = at::empty_like(mu[0]);

  for (int i = 0; i < input.size(0); i++) {
    mu_b[i] = s_mu.toType(input.scalar_type());
    var_b[i] = s_var.toType(input.scalar_type());

    delta = mu[i] - s_mu;

    s_var = afwd * s_var + (1. - afwd) * var[i] + afwd * (1. - afwd) * delta.pow(2);
    s_mu += (1. - afwd) * delta;
  }

  auto scale = (var_b + eps).sqrt().toType(input.scalar_type());

  return {
    ((inputs.toType(input.scalar_type()) - mu_b.unsqueeze(2)) / scale.unsqueeze(2)).view(input_shape),
    scale,
    s_mu,
    s_var
  };
}

std::vector<at::Tensor> norm_bwd_cpu(
    const at::Tensor grad_out,
    at::Tensor u,
    at::Tensor v,
    const at::Tensor out,
    const at::Tensor scale,
    const float abwd) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(u);
  CHECK_INPUT(v);
  CHECK_INPUT(out);
  CHECK_INPUT(scale);
  const auto input_shape = grad_out.sizes();
  const auto grad_outputs = grad_out.view({input_shape[0], input_shape[1], -1});
  const auto outputs = out.view({input_shape[0], input_shape[1], -1});

  auto grad_in = at::empty_like(grad_outputs);
  auto grad_tmp = at::empty_like(grad_outputs[0]);

  for (int i = 0; i < grad_outputs.size(0); i++) {
    grad_tmp = grad_outputs[i] - (1 - abwd) * v.unsqueeze(1).toType(grad_out.scalar_type()) * outputs[i];
    v += (grad_tmp.toType(at::ScalarType::Float) * outputs[i].toType(at::ScalarType::Float)).mean({1});
    grad_tmp = grad_tmp / scale[i].unsqueeze(1);

    grad_in[i] = grad_tmp - (1 - abwd) * u.unsqueeze(1).toType(grad_out.scalar_type());
    u += grad_in[i].toType(at::ScalarType::Float).mean({1});
  }

  return {grad_in.view(input_shape), u, v};
}
