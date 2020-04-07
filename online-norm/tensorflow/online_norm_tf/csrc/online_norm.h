#ifndef ONLINE_NORM_H_
#define ONLINE_NORM_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// 3 kernels:
// 1. norm_fwd_kernel
// 2. norm_uctrl_kernel
// 3. norm_vctrl_kernel

template <typename Device, typename T>
struct OnlineNormFwdFunctor {
  void operator()(
    const Device& d,
    const float* mu, // [N, C]
    const float* var, // [N, C]
    const float* in_s_mu, // [C]
    const float* in_s_var, // [C]
    float* out_s_mu, // [C]
    float* out_s_var, // [C]
    T* mean, // [N, C]
    T* scale, // [N, C]
    const int C,
    const int N,
    const float afwd,
    const float eps
  );
};

template <typename Device, typename T>
struct OnlineNormUCtrlFunctor {
  void operator()(
    const Device& d,
    const float* mu_delta, // [N, C]
    const float* in_u, // [C]
    float* out_u, // [C]
    T* d_u,  // [N, C]
    const int C,
    const int N,
    const float abkw
  );
};

template <typename Device, typename T>
struct OnlineNormVCtrlFunctor {
  void operator()(
    const Device& d,
    const T* grad_outputs, // [N, C, D]
    const T* outputs, // [N, C, D]
    const float* in_v, // [C]
    float* out_v, // [C]
    T* grad_tmp, // [N, C, D]
    const int C,
    const int N,
    const int D,
    const float abkw
  );
};


#endif // ONLINE_NORM_H_