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
// 2. norm_bwd_kernel

// Register TF operation
REGISTER_OP("OnlineNormFwd")
  .Attr("T: {float, half}")
  .Attr("afwd: float")
  .Attr("eps: float")
  .Input("input: T")
  .Input("in_s_mu: float")
  .Input("in_s_var: float")
  .Output("out_s_mu: float")
  .Output("out_s_var: float")
  .Output("out: T")
  .Output("scale: T");

REGISTER_OP("OnlineNormBwd")
  .Attr("T: {float, half}")
  .Attr("abkw: float")
  .Input("grad_out: T")
  .Input("in_v: float")
  .Input("in_u: float")
  .Input("out: T")
  .Input("scale: T")
  .Output("out_v: float")
  .Output("out_u: float")
  .Output("grad_in: T");

#ifndef CPU_ONLY
extern template struct OnlineNormFwdFunctor<GPUDevice, float>;
extern template struct OnlineNormFwdFunctor<GPUDevice, Eigen::half>;

extern template struct OnlineNormBwdFunctor<GPUDevice, float>;
extern template struct OnlineNormBwdFunctor<GPUDevice, Eigen::half>;
#endif //CPU_ONLY


// CPU specialization of actual computation.
template <typename T>
struct OnlineNormFwdFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
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
  ) {
    int idx;
    T in_elem, m, s;
    float sample_mu, sample_var, in_elem_f, diff;

    for(int c = 0; c < C; ++c) {
      out_s_mu[c] = in_s_mu[c];
      out_s_var[c] = in_s_var[c];
      for(int n = 0; n < N; ++n) {
        sample_mu = 0;
        sample_var = 0;

        m = (T)(out_s_mu[c]);
        s = (T)(sqrt(out_s_var[c] + eps));

        scale[Idx2(n, c, N, C)] = s;                    // store scale used

        for(int d = 0; d < D; ++d) {
          idx = Idx3(n, c, d, N, C, D);                 // idx of elem
          in_elem = input[idx];                         // get input element
          out[idx] = (in_elem - m) / s;                 // compute output

          // 1st and 2nd moment reductions
          in_elem_f = (float)(in_elem);
          sample_mu += in_elem_f;                       // 1st moment reduction
          sample_var += in_elem_f * in_elem_f;          // 2nd moment reduction
        }
        // compute sample mu and var to update streaming stats
        sample_mu = sample_mu / D;
        sample_var = (sample_var / D) - (sample_mu * sample_mu);

        // update streaming stats
        diff = sample_mu - out_s_mu[c];
        out_s_var[c] = afwd * out_s_var[c] + (1. - afwd) * sample_var + afwd * (1. - afwd) * diff * diff;
        out_s_mu[c] = out_s_mu[c] + (1. - afwd) * diff;
      }
    }
  }
};


template <typename T>
struct OnlineNormBwdFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
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
    float temp_mean_v;
    float temp_mean_u;
    T grad_tmp;
    for(int c = 0; c < C; ++c) {
      out_v[c] = in_v[c];
      out_u[c] = in_u[c];
      for(int n = 0; n < N; ++n) {
        temp_mean_v = 0;
        temp_mean_u = 0;
        for(int d = 0; d < D; ++d) {
          int idx = Idx3(n, c, d, N, C, D);
          grad_tmp = grad_outputs[idx] - ((T)(1.0 - abkw)) * ((T)out_v[c]) * outputs[idx];
          temp_mean_v += (float)grad_tmp * (float)outputs[idx];

          grad_tmp = grad_tmp / scale[Idx2(n, c, N, C)];

          grad_in[idx] = grad_tmp - ((T)(1.0 - abkw)) * ((T)out_u[c]);
          temp_mean_u += (float)grad_in[idx];
        }
        temp_mean_v = temp_mean_v / D;
        out_v[c] += temp_mean_v;
        temp_mean_u = temp_mean_u / D;
        out_u[c] += temp_mean_u;
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
    const Tensor& input = context->input(0);
    const Tensor& in_s_mu = context->input(1);
    const Tensor& in_s_var = context->input(2);

    const int N = input.shape().dim_size(0);
    const int C = input.shape().dim_size(1);
    const int D = input.shape().dim_size(2);

    // Create an output tensor
    Tensor* out_s_mu = NULL;
    Tensor* out_s_var = NULL;
    Tensor* out = NULL; 
    Tensor* scale = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, in_s_mu.shape(), &out_s_mu));
    OP_REQUIRES_OK(context, context->allocate_output(1, in_s_var.shape(), &out_s_var));
    OP_REQUIRES_OK(context, context->allocate_output(2, input.shape(), &out));
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({N, C}), &scale));

    // Do the computation.
    OnlineNormFwdFunctor<Device, T>()(
      context->eigen_device<Device>(),
      input.flat<T>().data(),
      in_s_mu.flat<float>().data(),
      in_s_var.flat<float>().data(),
      out_s_mu->flat<float>().data(),
      out_s_var->flat<float>().data(),
      out->flat<T>().data(),
      scale->flat<T>().data(),
      C,
      N,
      D,
      afwd,
      eps
    );
  }

private:
  float afwd;
  float eps;
};


// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class OnlineNormBwdOp : public OpKernel {
public:
  explicit OnlineNormBwdOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("abkw", &abkw));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& grad_out = context->input(0); // shape should be [N, C, D]
    const Tensor& in_v = context->input(1);     // shape should be [C]
    const Tensor& in_u = context->input(2);     // shape should be [C]
    const Tensor& out = context->input(3);      // shape should be [N, C, D]
    const Tensor& scale = context->input(4);    // shape should be [N, C]

    // Create an output tensor
    Tensor* out_v = NULL;
    Tensor* out_u = NULL;
    Tensor* grad_in = NULL;

    TensorShape input_shape = grad_out.shape();
    const int N = input_shape.dim_size(0);
    const int C = input_shape.dim_size(1);
    const int D = input_shape.dim_size(2);

    OP_REQUIRES_OK(context, context->allocate_output(0, in_v.shape(), &out_v));
    OP_REQUIRES_OK(context, context->allocate_output(1, in_u.shape(), &out_u));
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_out.shape(), &grad_in));

    // Do the computation
    OnlineNormBwdFunctor<Device, T>()(
      context->eigen_device<Device>(),
      grad_out.flat<T>().data(),
      in_v.flat<float>().data(),
      in_u.flat<float>().data(),
      out.flat<T>().data(),
      scale.flat<T>().data(),
      out_v->flat<float>().data(),
      out_u->flat<float>().data(),
      grad_in->flat<T>().data(),
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
      Name("OnlineNormBwd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      OnlineNormBwdOp<CPUDevice, T>); \

REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);

// Register the GPU kernels.
#ifndef CPU_ONLY
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormFwd").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OnlineNormFwdOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("OnlineNormBwd").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      OnlineNormBwdOp<GPUDevice, T>); \

REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);
#endif //CPU_ONLY

