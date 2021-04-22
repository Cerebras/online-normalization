#ifndef ONLINE_NORM_H_
#define ONLINE_NORM_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// 2 kernels:
// 1. norm_fwd_kernel
// 2. norm_bwd_kernel

template <typename Device, typename T>
struct OnlineNormFwdFunctor {
  void operator()(
    const Device& d,
    const T* input,         // [N, C, D]
    const float* in_s_mu,   // [C]
    const float* in_s_var,  // [C]
    float* out_s_mu,        // [C]
    float* out_s_var,       // [C]
    T* out,                 // [N, C, D]
    T* scale,               // [N, C]
    const int C,
    const int N,
    const int D,
    const float afwd,
    const float eps
  );
};


template <typename Device, typename T>
struct OnlineNormBwdFunctor {
  void operator()(
    const Device& d,
    const T* grad_outputs,  // [N, C, D]
    const float* in_v,      // [C]
    const float* in_u,      // [C]
    const T* outputs,       // [N, C, D]
    const T* scale,         // [N, C]
    float* out_v,           // [C]
    float* out_u,           // [C]
    T* grad_in,             // [N, C, D]
    const int C,
    const int N,
    const int D,
    const float abkw
  );
};


#endif // ONLINE_NORM_H_
