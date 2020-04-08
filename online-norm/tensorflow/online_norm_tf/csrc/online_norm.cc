// online_norm.cc
#include "online_norm.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#ifndef CPU_ONLY
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#endif //CPU_ONLY


using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define Idx3(n, c, d, N, C, D) (((n)*(C)*(D)) + ((c)*(D)) + (d))
#define Idx2(n, c, N, C) (((n)*(C)) + (c))



// 1. norm_fwd_kernel
// 2. norm_uctrl_kernel
// 3. norm_vctrl_kernel

// Register TF operation
REGISTER_OP("OnlineNormFwd")
  .Attr("T: {float, half}")
  .Attr("afwd: float")
  .Attr("eps: float")
  .Input("mu: float")
  .Input("var: float")
  .Input("in_s_mu: float")
  .Input("in_s_var: float")
  .Output("mean: T")
  .Output("scale: T")
  .Output("out_s_mu: float")
  .Output("out_s_var: float");

REGISTER_OP("OnlineNormUCtrl")
  .Attr("T: {float, half}")
  .Attr("abkw: float")
  .Input("mu_delta: float")
  .Input("in_u: float")
  .Output("out_u: float")
  .Output("d_u: T");

REGISTER_OP("OnlineNormVCtrl")
  .Attr("T: {float, half}")
  .Attr("abkw: float")
  .Input("grad_out: T")
  .Input("out: T")
  .Input("in_v: float")
  .Output("out_v: float")
  .Output("grad_tmp: T");

#ifndef CPU_ONLY
extern template struct OnlineNormFwdFunctor<GPUDevice, float>;
extern template struct OnlineNormFwdFunctor<GPUDevice, Eigen::half>;

extern template struct OnlineNormUCtrlFunctor<GPUDevice, float>;
extern template struct OnlineNormUCtrlFunctor<GPUDevice, Eigen::half>;

extern template struct OnlineNormVCtrlFunctor<GPUDevice, float>;
extern template struct OnlineNormVCtrlFunctor<GPUDevice, Eigen::half>;
#endif //CPU_ONLY

// CPU specialization of actual computation.
template <typename T>
struct OnlineNormFwdFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
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
  ) {
    for(int c = 0; c < C; ++c) {
      unsigned int idx;
      float delta;
      float s_var_local = in_s_var[c];
      float s_mu_local = in_s_mu[c];
      for(int n = 0; n < N; ++n) {
        idx = Idx2(n, c, N, C);
        scale[idx] = (T)(sqrt(s_var_local + eps));
        mean[idx] = (T)(s_mu_local);

        delta = mu[idx] - s_mu_local;

        s_var_local = afwd * s_var_local + (1. - afwd) * var[idx] + afwd * (1. - afwd) * delta * delta;
        s_mu_local = s_mu_local + (1. - afwd) * delta;
      };

      out_s_var[c] = s_var_local;
      out_s_mu[c] = s_mu_local;
    };
  }
};


template <typename T>
struct OnlineNormUCtrlFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
    const float* mu_delta,
    const float* in_u,
    float* out_u,
    T* d_u,
    const int C,
    const int N,
    const float abkw
  ) {
    for(int c = 0; c < C; ++c) {
      int idx;
      float delta;
      float s_u_local = in_u[c];

      for(int n = 0; n < N; ++n) {
        idx = Idx2(n, c, N, C);
        d_u[idx] = (T)(s_u_local);

        delta = mu_delta[idx] - s_u_local;

        s_u_local = s_u_local + (1. - abkw) * delta;
      };

      out_u[c] = s_u_local;
    };
  }
};


template <typename T>
struct OnlineNormVCtrlFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
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
    float temp_mean;
    for(int j = 0; j < C; j++) {
      out_v[j] = in_v[j];
      for(int i = 0; i < N; i++) {
        temp_mean = 0;
        for(int k = 0; k< D; k++) {
          int idx = Idx3(i, j, k, N, C, D);
          grad_tmp[idx] = grad_outputs[idx] - ((T)(1.0 - abkw)) * ((T)out_v[j]) * outputs[idx];
          temp_mean += (float)grad_tmp[idx] * (float)outputs[idx];
        }
        temp_mean = temp_mean/D;
        out_v[j]+=temp_mean;
      }
    }
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class OnlineNormFwdOp : public OpKernel {
public:
  explicit OnlineNormFwdOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("afwd", &afwd));
    OP_REQUIRES_OK(context, context->GetAttr("eps", &eps));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& mu = context->input(0);
    const Tensor& var = context->input(1);
    const Tensor& in_s_mu = context->input(2);
    const Tensor& in_s_var = context->input(3);

    int N = mu.shape().dim_size(0);
    int C = mu.shape().dim_size(1);

    // Create an output tensor
    Tensor* out_s_mu = NULL;
    Tensor* out_s_var = NULL;
    Tensor* mean = NULL;
    Tensor* scale = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, mu.shape(), &mean));
    OP_REQUIRES_OK(context, context->allocate_output(1, mu.shape(), &scale));
    OP_REQUIRES_OK(context, context->allocate_output(2, in_s_mu.shape(), &out_s_mu));
    OP_REQUIRES_OK(context, context->allocate_output(3, in_s_var.shape(), &out_s_var));

    // Do the computation.
    OnlineNormFwdFunctor<Device, T>()(
      context->eigen_device<Device>(),
      mu.flat<float>().data(),
      var.flat<float>().data(),
      in_s_mu.flat<float>().data(),
      in_s_var.flat<float>().data(),
      out_s_mu->flat<float>().data(),
      out_s_var->flat<float>().data(),
      mean->flat<T>().data(),
      scale->flat<T>().data(),
      C,
      N,
      afwd,
      eps
    );
  }

private:
  float afwd;
  float eps;
};


template <typename Device, typename T>
class OnlineNormUCtrlOp : public OpKernel {
public:
  explicit OnlineNormUCtrlOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("abkw", &abkw));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& mu_delta = context->input(0);
    const Tensor& in_u = context->input(1);

    int N = mu_delta.shape().dim_size(0);
    int C = mu_delta.shape().dim_size(1);

    // Create an output tensor
    Tensor* out_u = NULL;
    Tensor* d_u = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, in_u.shape(), &out_u));
    OP_REQUIRES_OK(context, context->allocate_output(1, mu_delta.shape(), &d_u));

    // Do the computation.
    OnlineNormUCtrlFunctor<Device, T>()(
      context->eigen_device<Device>(),
      mu_delta.flat<float>().data(),
      in_u.flat<float>().data(),
      out_u->flat<float>().data(),
      d_u->flat<T>().data(),
      C,
      N,
      abkw
    );
  }

private:
  float abkw;
};


// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class OnlineNormVCtrlOp : public OpKernel {
public:
  explicit OnlineNormVCtrlOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("abkw", &abkw));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& grad_out = context->input(0); // shape should be [N, C, D]
    const Tensor& out = context->input(1); // shape should be [N, C, D]
    const Tensor& in_v = context->input(2);

    // Create an output tensor
    Tensor* out_v = NULL;
    Tensor* grad_tmp = NULL;

    TensorShape input_shape = grad_out.shape();
    const int N = input_shape.dim_size(0);
    const int C = input_shape.dim_size(1);
    const int D = input_shape.dim_size(2);

    OP_REQUIRES_OK(context, context->allocate_output(0, in_v.shape(), &out_v));
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_out.shape(), &grad_tmp));

    // Do the computation
    OnlineNormVCtrlFunctor<Device, T>()(
      context->eigen_device<Device>(),
      grad_out.flat<T>().data(),
      out.flat<T>().data(),
      in_v.flat<float>().data(),
      out_v->flat<float>().data(),
      grad_tmp->flat<T>().data(),
      C,
      N,
      D,
      abkw
    );
  }

private:
  float abkw;
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormFwd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OnlineNormFwdOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormUCtrl").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OnlineNormUCtrlOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormVCtrl").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OnlineNormVCtrlOp<CPUDevice, T>); \

REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);

// Register the GPU kernels.
#ifndef CPU_ONLY
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormFwd").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OnlineNormFwdOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormUCtrl").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OnlineNormUCtrlOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormVCtrl").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OnlineNormVCtrlOp<GPUDevice, T>); \

REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#endif //CPU_ONLY

