#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void spin8_scan_kernel(const scalar_t* action, const scalar_t* scale,
    const scalar_t* drive, const scalar_t* initial, scalar_t* output,
    int length, int channels, int reps) {
  const int lane = threadIdx.x;
  const int linear = blockIdx.x;
  const int rep = linear % reps;
  const int channel = (linear / reps) % channels;
  const int batch = linear / (reps * channels);
  __shared__ float state[8];
  const int initial_offset = ((batch * channels + channel) * reps + rep) * 8;
  state[lane] = static_cast<float>(initial[initial_offset + lane]);
  __syncthreads();
  for (int position = 0; position < length; ++position) {
    const int action_base = ((batch * length + position) * reps + rep) * 64;
    float next = 0.0f;
#pragma unroll
    for (int source = 0; source < 8; ++source) {
      next += static_cast<float>(action[action_base + lane * 8 + source]) * state[source];
    }
    const int scalar_offset = (batch * length + position) * channels + channel;
    const int value_offset = (((batch * length + position) * channels + channel) * reps + rep) * 8 + lane;
    next = static_cast<float>(scale[scalar_offset]) * next + static_cast<float>(drive[value_offset]);
    output[value_offset] = static_cast<scalar_t>(next);
    __syncthreads();
    state[lane] = next;
    __syncthreads();
  }
}

torch::Tensor spin8_scan_cuda(torch::Tensor action, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial) {
  TORCH_CHECK(action.is_cuda() && scale.is_cuda() && drive.is_cuda() && initial.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(action.is_contiguous() && scale.is_contiguous() && drive.is_contiguous() && initial.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(action.dim() == 5 && action.size(-1) == 8 && action.size(-2) == 8, "action must be (B,L,R,8,8)");
  TORCH_CHECK(scale.dim() == 3, "scale must be (B,L,C)");
  TORCH_CHECK(drive.dim() == 5 && drive.size(-1) == 8, "drive must be (B,L,C,R,8)");
  TORCH_CHECK(initial.dim() == 4 && initial.size(-1) == 8, "initial must be (B,C,R,8)");
  TORCH_CHECK(action.scalar_type() == scale.scalar_type() && action.scalar_type() == drive.scalar_type() && action.scalar_type() == initial.scalar_type(), "dtypes must match");
  TORCH_CHECK(action.device() == scale.device() && action.device() == drive.device() && action.device() == initial.device(), "devices must match");
  const int batch = scale.size(0), length = scale.size(1), channels = scale.size(2), reps = drive.size(3);
  TORCH_CHECK(action.size(0) == batch && action.size(1) == length && action.size(2) == reps, "action shape mismatch");
  TORCH_CHECK(drive.size(0) == batch && drive.size(1) == length && drive.size(2) == channels, "drive shape mismatch");
  TORCH_CHECK(initial.size(0) == batch && initial.size(1) == channels && initial.size(2) == reps, "initial shape mismatch");
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(action.device());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(action.scalar_type(), "spin8_scan_cuda", [&] {
    spin8_scan_kernel<scalar_t><<<batch * channels * reps, 8, 0, at::cuda::getCurrentCUDAStream()>>>(
        action.data_ptr<scalar_t>(), scale.data_ptr<scalar_t>(), drive.data_ptr<scalar_t>(), initial.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), length, channels, reps);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
