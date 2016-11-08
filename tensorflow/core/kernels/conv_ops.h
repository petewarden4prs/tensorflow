/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_CONV_OPS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
class LaunchConv2DOp {
 public:
  void launch(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
              const Tensor& input, const Tensor& filter, int row_stride,
              int col_stride, const Eigen::PaddingType& padding, Tensor* output,
              TensorFormat data_format);
};

// Used to keep track of persistent memory buffers used within the op.
template <class T, size_t size>
struct Im2ColBufferResource : public ResourceBase {
  Im2ColBufferResource<T, size>() {
    data = static_cast<T*>(malloc(size * sizeof(T)));
  }
  ~Im2ColBufferResource<T, size>() { free(data); }
  // This mutex ensures that only a single operation at a time is able to use
  // the buffer memory held by this resource.
  mutex mu;
  T* data;
  string DebugString() { return "Im2ColBufferResource"; }
};

#ifdef GOOGLE_CUDA
template <typename T>
class LaunchConv2DOp<Eigen::GpuDevice, T> {
 public:
  void launch(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
              const Tensor& input, const Tensor& filter, int row_stride,
              int col_stride, const Eigen::PaddingType& padding, Tensor* output,
              TensorFormat data_format);
};
#endif  // GOOGLE_CUDA

// Simple utility function used by ops to multithread basic workloads. To use
// it, pass begin and end values for the full workload and a std::function that
// receives a subset of that through the begin and end values for each worker's
// task. The division of the full workload into worker tasks is handled by the
// multithreading logic. Here's an example of how to use it:
// std::vector<float> my_vector(100);
// ...
// RunInParallel(context, 0, 100,
//   [&my_vector](int64 task_begin, int64 task_end) {
//     for (int64 current = task_begin; current != task_end; ++current) {
//       my_vector[current] *= 10.0f;
//     }
// });
void ParallelFor(OpKernelContext* context, int64 begin, int64 end,
                 std::function<void(int64, int64)> task_function);

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_H
