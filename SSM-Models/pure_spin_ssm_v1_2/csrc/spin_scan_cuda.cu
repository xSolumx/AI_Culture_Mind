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

namespace {

constexpr unsigned FULL_WARP = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(FULL_WARP, value, offset);
  }
  return value;
}

__device__ __forceinline__ float controller_angle(
    const float* features, const float* weight, const float* bias,
    int feature_base, int controller_row, int input_size, float gate_value) {
  const int lane = threadIdx.x;
  float partial = 0.0f;
  for (int feature = lane; feature < input_size; feature += 32) {
    partial = fmaf(weight[controller_row * input_size + feature],
        features[feature_base + feature], partial);
  }
  partial = warp_sum(partial);
  if (lane == 0) {
    partial = gate_value * (partial + bias[controller_row]);
  }
  return __shfl_sync(FULL_WARP, partial, 0);
}

__device__ __forceinline__ float generator_product(
    const float* generators, int representation, int coordinate, int row,
    float value, int factors) {
  float result = 0.0f;
  const int base = ((representation * factors + coordinate) * 8 + row) * 8;
#pragma unroll
  for (int source = 0; source < 8; ++source) {
    result = fmaf(__ldg(generators + base + source),
        __shfl_sync(FULL_WARP, value, representation * 8 + source), result);
  }
  return result;
}

__device__ __forceinline__ float apply_factor(
    const float* generators, int representation, int coordinate, int row,
    float state, float angle, int factors) {
  const float first = generator_product(
      generators, representation, coordinate, row, state, factors);
  const float second = generator_product(
      generators, representation, coordinate, row, first, factors);
  if (representation == 0) {
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    return state + sine * first + (1.0f - cosine) * second;
  }
  float sine, cosine;
  sincosf(0.5f * angle, &sine, &cosine);
  return cosine * state + (2.0f * sine) * first;
}

__device__ __forceinline__ float factor_derivative(
    const float* generators, int representation, int coordinate, int row,
    float state_before, float angle, int factors) {
  const float first = generator_product(
      generators, representation, coordinate, row, state_before, factors);
  const float second = generator_product(
      generators, representation, coordinate, row, first, factors);
  if (representation == 0) {
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    return cosine * first + sine * second;
  }
  float sine, cosine;
  sincosf(0.5f * angle, &sine, &cosine);
  return -0.5f * sine * state_before + cosine * first;
}

// In the isotypic schedule one warp owns only one of 8v, 8+, or 8-. The
// representation's eight state coordinates therefore live in lanes 0..7
// instead of occupying a representation-dependent octet of one packed warp.
__device__ __forceinline__ float isotypic_generator_product(
    const float* generators, int representation, int coordinate, int row,
    float value, int factors) {
  float result = 0.0f;
  const int base = ((representation * factors + coordinate) * 8 + row) * 8;
#pragma unroll
  for (int source = 0; source < 8; ++source) {
    result = fmaf(__ldg(generators + base + source),
        __shfl_sync(FULL_WARP, value, source), result);
  }
  return result;
}

__device__ __forceinline__ float isotypic_apply_factor(
    const float* generators, int representation, int coordinate, int row,
    float state, float angle, int factors) {
  const float first = isotypic_generator_product(
      generators, representation, coordinate, row, state, factors);
  const float second = isotypic_generator_product(
      generators, representation, coordinate, row, first, factors);
  float sine, cosine;
  if (representation == 0) {
    sincosf(angle, &sine, &cosine);
    return state + sine * first + (1.0f - cosine) * second;
  }
  sincosf(0.5f * angle, &sine, &cosine);
  return cosine * state + (2.0f * sine) * first;
}

__device__ __forceinline__ float isotypic_factor_derivative(
    const float* generators, int representation, int coordinate, int row,
    float state_before, float angle, int factors) {
  const float first = isotypic_generator_product(
      generators, representation, coordinate, row, state_before, factors);
  const float second = isotypic_generator_product(
      generators, representation, coordinate, row, first, factors);
  float sine, cosine;
  if (representation == 0) {
    sincosf(angle, &sine, &cosine);
    return cosine * first + sine * second;
  }
  sincosf(0.5f * angle, &sine, &cosine);
  return -0.5f * sine * state_before + cosine * first;
}

__global__ void controller_factorized_forward_kernel(
    const float* features, const float* weight, const float* bias,
    const float* generators, const float* scale, const float* drive,
    const float* initial, const float* gate, float* output, int length,
    int channels, int input_size) {
  const int lane = threadIdx.x;
  const int channel = blockIdx.x % channels;
  const int batch = blockIdx.x / channels;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = ((batch * channels + channel) * 3) * 8;
  float state = active ? initial[initial_base + lane] : 0.0f;

  for (int position = 0; position < length; ++position) {
    const int step = batch * length + position;
    const int feature_base = step * input_size;
    const float gate_value = gate[step];
#pragma unroll
    for (int coordinate = 0; coordinate < 28; ++coordinate) {
      const int controller_row = channel * 28 + coordinate;
      const float angle = controller_angle(
          features, weight, bias, feature_base, controller_row, input_size,
          gate_value);
      const float next_state = apply_factor(
          generators, representation, coordinate, row, state, angle, 28);
      if (active) state = next_state;
      __syncwarp();
    }
    if (active) {
      const int scalar_offset = step * channels + channel;
      const int output_base = (scalar_offset * 3) * 8;
      state = scale[scalar_offset] * state + drive[output_base + lane];
      output[output_base + lane] = state;
    }
    __syncwarp();
  }
}

__global__ void controller_factorized_backward_kernel(
    const float* features, const float* weight, const float* bias,
    const float* generators, const float* scale, const float* drive,
    const float* initial,
    const float* gate, const float* output, const float* output_gradient,
    float* feature_gradient, float* weight_gradient, float* bias_gradient,
    float* scale_gradient, float* drive_gradient, float* initial_gradient,
    int length, int channels, int input_size) {
  const int lane = threadIdx.x;
  const int channel = blockIdx.x % channels;
  const int batch = blockIdx.x / channels;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = ((batch * channels + channel) * 3) * 8;
  float carry = 0.0f;

  for (int reverse_position = 0; reverse_position < length; ++reverse_position) {
    const int position = length - 1 - reverse_position;
    const int step = batch * length + position;
    const int scalar_offset = step * channels + channel;
    const int output_base = (scalar_offset * 3) * 8;
    const int feature_base = step * input_size;
    const float gate_value = gate[step];
    float direct = active ? output_gradient[output_base + lane] + carry : 0.0f;
    // Division removes a complete replay of the forward factor chain. Retain
    // the replay path for zero/tiny float32 scales, where reconstruction would
    // be undefined or ill-conditioned.
    const float scale_value = scale[scalar_offset];
    float rotated = 0.0f;
    if (fabsf(scale_value) > 1.0e-7f) {
      rotated = active
          ? (output[output_base + lane] - drive[output_base + lane]) /
                scale_value
          : 0.0f;
    } else {
      rotated = active
          ? (position == 0
                ? initial[initial_base + lane]
                : output[output_base - channels * 3 * 8 + lane])
          : 0.0f;
#pragma unroll
      for (int coordinate = 0; coordinate < 28; ++coordinate) {
        const int controller_row = channel * 28 + coordinate;
        const float angle = controller_angle(
            features, weight, bias, feature_base, controller_row, input_size,
            gate_value);
        const float next_rotated = apply_factor(
            generators, representation, coordinate, row, rotated, angle, 28);
        if (active) rotated = next_rotated;
        __syncwarp();
      }
    }

    float scale_term = active ? direct * rotated : 0.0f;
    scale_term = warp_sum(scale_term);
    if (lane == 0) {
      scale_gradient[scalar_offset] = scale_term;
    }
    if (active) {
      drive_gradient[output_base + lane] = direct;
    }
    float adjoint = active ? scale_value * direct : 0.0f;
    float state_after = rotated;

#pragma unroll
    for (int reverse_coordinate = 0; reverse_coordinate < 28; ++reverse_coordinate) {
      const int coordinate = 27 - reverse_coordinate;
      const int controller_row = channel * 28 + coordinate;
      const float angle = controller_angle(
          features, weight, bias, feature_base, controller_row, input_size,
          gate_value);
      const float inverse_state = apply_factor(
          generators, representation, coordinate, row, state_after, -angle, 28);
      float state_before = active ? inverse_state : 0.0f;
      __syncwarp();
      const float raw_derivative = factor_derivative(
          generators, representation, coordinate, row, state_before, angle, 28);
      const float derivative = active ? raw_derivative : 0.0f;
      float angle_gradient = warp_sum(active ? adjoint * derivative : 0.0f);
      angle_gradient = __shfl_sync(FULL_WARP, angle_gradient, 0) * gate_value;
      for (int feature = lane; feature < input_size; feature += 32) {
        const float feature_value = features[feature_base + feature];
        const float weight_value = weight[controller_row * input_size + feature];
        atomicAdd(weight_gradient + controller_row * input_size + feature,
            angle_gradient * feature_value);
        atomicAdd(feature_gradient + feature_base + feature,
            angle_gradient * weight_value);
      }
      if (lane == 0) {
        atomicAdd(bias_gradient + controller_row, angle_gradient);
      }
      const float inverse_adjoint = apply_factor(
          generators, representation, coordinate, row, adjoint, -angle, 28);
      if (active) {
        adjoint = inverse_adjoint;
        state_after = state_before;
      }
      __syncwarp();
    }
    carry = adjoint;
  }
  if (active) {
    initial_gradient[initial_base + lane] = carry;
  }
}

__global__ void coordinate_factorized_forward_kernel(
    const float* coordinates, const float* generators, const float* scale,
    const float* drive, const float* initial, float* output, int length,
    int channels, int factors) {
  const int lane = threadIdx.x;
  const int channel = blockIdx.x % channels;
  const int batch = blockIdx.x / channels;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = ((batch * channels + channel) * 3) * 8;
  float state = active ? initial[initial_base + lane] : 0.0f;

  for (int position = 0; position < length; ++position) {
    const int scalar_offset = (batch * length + position) * channels + channel;
    const int coordinate_base = scalar_offset * factors;
#pragma unroll 4
    for (int coordinate = 0; coordinate < factors; ++coordinate) {
      const float angle = coordinates[coordinate_base + coordinate];
      const float next_state = apply_factor(
          generators, representation, coordinate, row, state, angle, factors);
      if (active) state = next_state;
      __syncwarp();
    }
    if (active) {
      const int output_base = (scalar_offset * 3) * 8;
      state = scale[scalar_offset] * state + drive[output_base + lane];
      output[output_base + lane] = state;
    }
    __syncwarp();
  }
}

__global__ void coordinate_factorized_backward_kernel(
    const float* coordinates, const float* generators, const float* scale,
    const float* drive, const float* initial, const float* output,
    const float* output_gradient,
    float* coordinate_gradient, float* scale_gradient, float* drive_gradient,
    float* initial_gradient, int length, int channels, int factors) {
  const int lane = threadIdx.x;
  const int channel = blockIdx.x % channels;
  const int batch = blockIdx.x / channels;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = ((batch * channels + channel) * 3) * 8;
  float carry = 0.0f;

  for (int reverse_position = 0; reverse_position < length; ++reverse_position) {
    const int position = length - 1 - reverse_position;
    const int scalar_offset = (batch * length + position) * channels + channel;
    const int coordinate_base = scalar_offset * factors;
    const int output_base = (scalar_offset * 3) * 8;
    float direct = active ? output_gradient[output_base + lane] + carry : 0.0f;
    const float scale_value = scale[scalar_offset];
    float rotated = 0.0f;
    if (fabsf(scale_value) > 1.0e-7f) {
      rotated = active
          ? (output[output_base + lane] - drive[output_base + lane]) /
                scale_value
          : 0.0f;
    } else {
      rotated = active
          ? (position == 0
                ? initial[initial_base + lane]
                : output[output_base - channels * 3 * 8 + lane])
          : 0.0f;
#pragma unroll 4
      for (int coordinate = 0; coordinate < factors; ++coordinate) {
        const float angle = coordinates[coordinate_base + coordinate];
        const float next_rotated = apply_factor(
            generators, representation, coordinate, row, rotated, angle,
            factors);
        if (active) rotated = next_rotated;
        __syncwarp();
      }
    }

    float scale_term = warp_sum(active ? direct * rotated : 0.0f);
    if (lane == 0) scale_gradient[scalar_offset] = scale_term;
    if (active) drive_gradient[output_base + lane] = direct;
    float adjoint = active ? scale_value * direct : 0.0f;
    float state_after = rotated;

#pragma unroll 4
    for (int reverse_coordinate = 0; reverse_coordinate < factors; ++reverse_coordinate) {
      const int coordinate = factors - 1 - reverse_coordinate;
      const float angle = coordinates[coordinate_base + coordinate];
      const float state_before_raw = apply_factor(
          generators, representation, coordinate, row, state_after, -angle,
          factors);
      const float state_before = active ? state_before_raw : 0.0f;
      __syncwarp();
      const float derivative_raw = factor_derivative(
          generators, representation, coordinate, row, state_before, angle,
          factors);
      const float angle_gradient = warp_sum(
          active ? adjoint * derivative_raw : 0.0f);
      if (lane == 0) {
        coordinate_gradient[coordinate_base + coordinate] = angle_gradient;
      }
      const float inverse_adjoint = apply_factor(
          generators, representation, coordinate, row, adjoint, -angle,
          factors);
      if (active) {
        adjoint = inverse_adjoint;
        state_after = state_before;
      }
      __syncwarp();
    }
    carry = adjoint;
  }
  if (active) initial_gradient[initial_base + lane] = carry;
}

__global__ void isotypic_coordinate_forward_kernel(
    const float* coordinates, const float* generators, const float* scale,
    const float* drive, const float* initial, float* output, int length,
    int channels, int factors) {
  const int lane = threadIdx.x;
  const int representation = blockIdx.x % 3;
  const int sequence = blockIdx.x / 3;
  const int channel = sequence % channels;
  const int batch = sequence / channels;
  const bool active = lane < 8;
  const int row = active ? lane : 0;
  const int initial_base =
      (((batch * channels + channel) * 3 + representation) * 8);
  float state = active ? initial[initial_base + lane] : 0.0f;

  for (int position = 0; position < length; ++position) {
    const int scalar_offset = (batch * length + position) * channels + channel;
    const int coordinate_base = scalar_offset * factors;
#pragma unroll 4
    for (int coordinate = 0; coordinate < factors; ++coordinate) {
      const float angle = coordinates[coordinate_base + coordinate];
      const float next_state = isotypic_apply_factor(
          generators, representation, coordinate, row, state, angle, factors);
      if (active) state = next_state;
      __syncwarp();
    }
    if (active) {
      const int output_base =
          ((scalar_offset * 3 + representation) * 8);
      state = scale[scalar_offset] * state + drive[output_base + lane];
      output[output_base + lane] = state;
    }
    __syncwarp();
  }
}

__global__ void isotypic_coordinate_backward_kernel(
    const float* coordinates, const float* generators, const float* scale,
    const float* drive, const float* initial, const float* output,
    const float* output_gradient, float* coordinate_gradient_by_representation,
    float* scale_gradient_by_representation, float* drive_gradient,
    float* initial_gradient, int length, int channels, int factors,
    int coordinate_elements, int scale_elements) {
  const int lane = threadIdx.x;
  const int representation = blockIdx.x % 3;
  const int sequence = blockIdx.x / 3;
  const int channel = sequence % channels;
  const int batch = sequence / channels;
  const bool active = lane < 8;
  const int row = active ? lane : 0;
  const int initial_base =
      (((batch * channels + channel) * 3 + representation) * 8);
  float carry = 0.0f;

  for (int reverse_position = 0; reverse_position < length; ++reverse_position) {
    const int position = length - 1 - reverse_position;
    const int scalar_offset = (batch * length + position) * channels + channel;
    const int coordinate_base = scalar_offset * factors;
    const int output_base = ((scalar_offset * 3 + representation) * 8);
    float direct = active ? output_gradient[output_base + lane] + carry : 0.0f;
    const float scale_value = scale[scalar_offset];
    float rotated = 0.0f;
    if (fabsf(scale_value) > 1.0e-7f) {
      rotated = active
          ? (output[output_base + lane] - drive[output_base + lane]) /
                scale_value
          : 0.0f;
    } else {
      rotated = active
          ? (position == 0
                ? initial[initial_base + lane]
                : output[output_base - channels * 3 * 8 + lane])
          : 0.0f;
#pragma unroll 4
      for (int coordinate = 0; coordinate < factors; ++coordinate) {
        const float angle = coordinates[coordinate_base + coordinate];
        const float next_rotated = isotypic_apply_factor(
            generators, representation, coordinate, row, rotated, angle,
            factors);
        if (active) rotated = next_rotated;
        __syncwarp();
      }
    }

    const float scale_term = warp_sum(active ? direct * rotated : 0.0f);
    if (lane == 0) {
      scale_gradient_by_representation[
          representation * scale_elements + scalar_offset] = scale_term;
    }
    if (active) drive_gradient[output_base + lane] = direct;
    float adjoint = active ? scale_value * direct : 0.0f;
    float state_after = rotated;

#pragma unroll 4
    for (int reverse_coordinate = 0; reverse_coordinate < factors;
         ++reverse_coordinate) {
      const int coordinate = factors - 1 - reverse_coordinate;
      const float angle = coordinates[coordinate_base + coordinate];
      const float state_before_raw = isotypic_apply_factor(
          generators, representation, coordinate, row, state_after, -angle,
          factors);
      const float state_before = active ? state_before_raw : 0.0f;
      __syncwarp();
      const float derivative_raw = isotypic_factor_derivative(
          generators, representation, coordinate, row, state_before, angle,
          factors);
      const float angle_gradient = warp_sum(
          active ? adjoint * derivative_raw : 0.0f);
      if (lane == 0) {
        coordinate_gradient_by_representation[
            representation * coordinate_elements + coordinate_base + coordinate]
            = angle_gradient;
      }
      const float inverse_adjoint = isotypic_apply_factor(
          generators, representation, coordinate, row, adjoint, -angle,
          factors);
      if (active) {
        adjoint = inverse_adjoint;
        state_after = state_before;
      }
      __syncwarp();
    }
    carry = adjoint;
  }
  if (active) initial_gradient[initial_base + lane] = carry;
}

// One warp owns the complete 2 x (8v + 8+ + 8-) state.  The Spin action is
// shared across the multiplicity copies, so it is applied once per copy with
// the same coordinates; the learned 2x2 left action then mixes only the
// multiplicity index.  This is the real-Schur-legal coupled recurrence
//
//     H_t = L_t H_{t-1} R_t^T + D_t.
__global__ void coupled_coordinate_forward_kernel(
    const float* coordinates, const float* generators, const float* left,
    const float* drive, const float* initial, float* output, int length,
    int factors) {
  const int lane = threadIdx.x;
  const int batch = blockIdx.x;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = batch * 48;
  float state0 = active ? initial[initial_base + lane] : 0.0f;
  float state1 = active ? initial[initial_base + 24 + lane] : 0.0f;

  for (int position = 0; position < length; ++position) {
    const int step = batch * length + position;
    const int coordinate_base = step * factors;
#pragma unroll 4
    for (int coordinate = 0; coordinate < factors; ++coordinate) {
      const float angle = coordinates[coordinate_base + coordinate];
      const float next0 = apply_factor(generators, representation, coordinate,
          row, state0, angle, factors);
      const float next1 = apply_factor(generators, representation, coordinate,
          row, state1, angle, factors);
      if (active) {
        state0 = next0;
        state1 = next1;
      }
      __syncwarp();
    }
    if (active) {
      const int left_base = step * 4;
      const int output_base = step * 48;
      const float rotated0 = state0;
      const float rotated1 = state1;
      state0 = left[left_base] * rotated0 + left[left_base + 1] * rotated1 +
          drive[output_base + lane];
      state1 = left[left_base + 2] * rotated0 + left[left_base + 3] * rotated1 +
          drive[output_base + 24 + lane];
      output[output_base + lane] = state0;
      output[output_base + 24 + lane] = state1;
    }
    __syncwarp();
  }
}

__global__ void coupled_coordinate_backward_kernel(
    const float* coordinates, const float* generators, const float* left,
    const float* drive, const float* initial, const float* output,
    const float* output_gradient, float* coordinate_gradient,
    float* left_gradient, float* drive_gradient, float* initial_gradient,
    int length, int factors) {
  const int lane = threadIdx.x;
  const int batch = blockIdx.x;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = batch * 48;
  float carry0 = 0.0f;
  float carry1 = 0.0f;

  for (int reverse_position = 0; reverse_position < length; ++reverse_position) {
    const int position = length - 1 - reverse_position;
    const int step = batch * length + position;
    const int coordinate_base = step * factors;
    const int left_base = step * 4;
    const int output_base = step * 48;
    const float direct0 = active
        ? output_gradient[output_base + lane] + carry0
        : 0.0f;
    const float direct1 = active
        ? output_gradient[output_base + 24 + lane] + carry1
        : 0.0f;

    // Replay the pre-affine state.  This avoids an inverse of L_t and remains
    // exact at zero retention or any singular learned left action.
    float rotated0 = active
        ? (position == 0 ? initial[initial_base + lane]
                         : output[output_base - 48 + lane])
        : 0.0f;
    float rotated1 = active
        ? (position == 0 ? initial[initial_base + 24 + lane]
                         : output[output_base - 24 + lane])
        : 0.0f;
#pragma unroll 4
    for (int coordinate = 0; coordinate < factors; ++coordinate) {
      const float angle = coordinates[coordinate_base + coordinate];
      const float next0 = apply_factor(generators, representation, coordinate,
          row, rotated0, angle, factors);
      const float next1 = apply_factor(generators, representation, coordinate,
          row, rotated1, angle, factors);
      if (active) {
        rotated0 = next0;
        rotated1 = next1;
      }
      __syncwarp();
    }

    const float left00 = warp_sum(active ? direct0 * rotated0 : 0.0f);
    const float left01 = warp_sum(active ? direct0 * rotated1 : 0.0f);
    const float left10 = warp_sum(active ? direct1 * rotated0 : 0.0f);
    const float left11 = warp_sum(active ? direct1 * rotated1 : 0.0f);
    if (lane == 0) {
      left_gradient[left_base] = left00;
      left_gradient[left_base + 1] = left01;
      left_gradient[left_base + 2] = left10;
      left_gradient[left_base + 3] = left11;
    }
    if (active) {
      drive_gradient[output_base + lane] = direct0;
      drive_gradient[output_base + 24 + lane] = direct1;
    }

    const float l00 = left[left_base];
    const float l01 = left[left_base + 1];
    const float l10 = left[left_base + 2];
    const float l11 = left[left_base + 3];
    float adjoint0 = active ? l00 * direct0 + l10 * direct1 : 0.0f;
    float adjoint1 = active ? l01 * direct0 + l11 * direct1 : 0.0f;
    float state_after0 = rotated0;
    float state_after1 = rotated1;

#pragma unroll 4
    for (int reverse_coordinate = 0; reverse_coordinate < factors;
         ++reverse_coordinate) {
      const int coordinate = factors - 1 - reverse_coordinate;
      const float angle = coordinates[coordinate_base + coordinate];
      const float state_before0_raw = apply_factor(generators, representation,
          coordinate, row, state_after0, -angle, factors);
      const float state_before1_raw = apply_factor(generators, representation,
          coordinate, row, state_after1, -angle, factors);
      const float state_before0 = active ? state_before0_raw : 0.0f;
      const float state_before1 = active ? state_before1_raw : 0.0f;
      __syncwarp();
      const float derivative0 = factor_derivative(generators, representation,
          coordinate, row, state_before0, angle, factors);
      const float derivative1 = factor_derivative(generators, representation,
          coordinate, row, state_before1, angle, factors);
      const float angle_gradient = warp_sum(
          active ? adjoint0 * derivative0 + adjoint1 * derivative1 : 0.0f);
      if (lane == 0) {
        coordinate_gradient[coordinate_base + coordinate] = angle_gradient;
      }
      const float inverse0 = apply_factor(generators, representation,
          coordinate, row, adjoint0, -angle, factors);
      const float inverse1 = apply_factor(generators, representation,
          coordinate, row, adjoint1, -angle, factors);
      if (active) {
        adjoint0 = inverse0;
        adjoint1 = inverse1;
        state_after0 = state_before0;
        state_after1 = state_before1;
      }
      __syncwarp();
    }
    carry0 = adjoint0;
    carry1 = adjoint1;
  }
  if (active) {
    initial_gradient[initial_base + lane] = carry0;
    initial_gradient[initial_base + 24 + lane] = carry1;
  }
}

// General block-affine recurrence with one independently controlled Spin
// action per multiplicity copy.  The scalar 2x2 left action mixes the rotated
// copies and preserves the contraction bound, while one warp still owns the
// complete 2 x (8v + 8+ + 8-) state.
__global__ void independent_block_forward_kernel(
    const float* coordinates, const float* generators, const float* left,
    const float* drive, const float* initial, float* output, int length,
    int factors) {
  const int lane = threadIdx.x;
  const int batch = blockIdx.x;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = batch * 48;
  float state0 = active ? initial[initial_base + lane] : 0.0f;
  float state1 = active ? initial[initial_base + 24 + lane] : 0.0f;

  for (int position = 0; position < length; ++position) {
    const int step = batch * length + position;
    const int coordinate_base0 = step * 2 * factors;
    const int coordinate_base1 = coordinate_base0 + factors;
#pragma unroll 4
    for (int coordinate = 0; coordinate < factors; ++coordinate) {
      const float next0 = apply_factor(generators, representation, coordinate,
          row, state0, coordinates[coordinate_base0 + coordinate], factors);
      const float next1 = apply_factor(generators, representation, coordinate,
          row, state1, coordinates[coordinate_base1 + coordinate], factors);
      if (active) {
        state0 = next0;
        state1 = next1;
      }
      __syncwarp();
    }
    if (active) {
      const int left_base = step * 4;
      const int output_base = step * 48;
      const float rotated0 = state0;
      const float rotated1 = state1;
      state0 = left[left_base] * rotated0 + left[left_base + 1] * rotated1 +
          drive[output_base + lane];
      state1 = left[left_base + 2] * rotated0 + left[left_base + 3] * rotated1 +
          drive[output_base + 24 + lane];
      output[output_base + lane] = state0;
      output[output_base + 24 + lane] = state1;
    }
    __syncwarp();
  }
}

// Occupancy-oriented lowering of the same recurrence.  Triality makes the
// representation index block diagonal, so each warp can own one 8-dimensional
// representation while retaining both coupled multiplicity copies.  This is
// exactly the same map as independent_block_forward_kernel; only scheduling
// and the lane placement of the representation coordinates differ.
__global__ void independent_block_isotypic_forward_kernel(
    const float* coordinates, const float* generators, const float* left,
    const float* drive, const float* initial, float* output, int length,
    int factors) {
  const int lane = threadIdx.x;
  const int representation = blockIdx.x % 3;
  const int batch = blockIdx.x / 3;
  const bool active = lane < 8;
  const int row = active ? lane : 0;
  const int initial_base = batch * 48;
  const int representation_offset = representation * 8;
  float state0 = active
      ? initial[initial_base + representation_offset + lane]
      : 0.0f;
  float state1 = active
      ? initial[initial_base + 24 + representation_offset + lane]
      : 0.0f;

  for (int position = 0; position < length; ++position) {
    const int step = batch * length + position;
    const int coordinate_base0 = step * 2 * factors;
    const int coordinate_base1 = coordinate_base0 + factors;
#pragma unroll 4
    for (int coordinate = 0; coordinate < factors; ++coordinate) {
      const float next0 = isotypic_apply_factor(generators, representation,
          coordinate, row, state0,
          coordinates[coordinate_base0 + coordinate], factors);
      const float next1 = isotypic_apply_factor(generators, representation,
          coordinate, row, state1,
          coordinates[coordinate_base1 + coordinate], factors);
      if (active) {
        state0 = next0;
        state1 = next1;
      }
      __syncwarp();
    }
    if (active) {
      const int left_base = step * 4;
      const int output_base = step * 48 + representation_offset + lane;
      const float rotated0 = state0;
      const float rotated1 = state1;
      state0 = left[left_base] * rotated0 + left[left_base + 1] * rotated1 +
          drive[output_base];
      state1 = left[left_base + 2] * rotated0 + left[left_base + 3] * rotated1 +
          drive[output_base + 24];
      output[output_base] = state0;
      output[output_base + 24] = state1;
    }
    __syncwarp();
  }
}

__global__ void independent_block_backward_kernel(
    const float* coordinates, const float* generators, const float* left,
    const float* drive, const float* initial, const float* output,
    const float* output_gradient, float* coordinate_gradient,
    float* left_gradient, float* drive_gradient, float* initial_gradient,
    int length, int factors) {
  const int lane = threadIdx.x;
  const int batch = blockIdx.x;
  const bool active = lane < 24;
  const int representation = active ? lane >> 3 : 0;
  const int row = lane & 7;
  const int initial_base = batch * 48;
  float carry0 = 0.0f;
  float carry1 = 0.0f;

  for (int reverse_position = 0; reverse_position < length; ++reverse_position) {
    const int position = length - 1 - reverse_position;
    const int step = batch * length + position;
    const int coordinate_base0 = step * 2 * factors;
    const int coordinate_base1 = coordinate_base0 + factors;
    const int left_base = step * 4;
    const int output_base = step * 48;
    const float direct0 = active
        ? output_gradient[output_base + lane] + carry0
        : 0.0f;
    const float direct1 = active
        ? output_gradient[output_base + 24 + lane] + carry1
        : 0.0f;
    const float l00 = left[left_base];
    const float l01 = left[left_base + 1];
    const float l10 = left[left_base + 2];
    const float l11 = left[left_base + 3];
    const float determinant = fmaf(l00, l11, -l01 * l10);
    float rotated0 = 0.0f;
    float rotated1 = 0.0f;
    if (fabsf(determinant) > 1.0e-7f) {
      // output - drive = L * rotated.  The model constructs
      // L = diag(scale) * Q, so this is the ordinary positive-retention path.
      // The explicit 2x2 inverse removes one complete replay of both Spin
      // actions from backward.
      const float right0 = active
          ? output[output_base + lane] - drive[output_base + lane]
          : 0.0f;
      const float right1 = active
          ? output[output_base + 24 + lane] - drive[output_base + 24 + lane]
          : 0.0f;
      rotated0 = active
          ? (l11 * right0 - l01 * right1) / determinant
          : 0.0f;
      rotated1 = active
          ? (l00 * right1 - l10 * right0) / determinant
          : 0.0f;
    } else {
      // Preserve the exact semantic route at zero retention or any nearly
      // singular user-supplied left action.
      rotated0 = active
          ? (position == 0 ? initial[initial_base + lane]
                           : output[output_base - 48 + lane])
          : 0.0f;
      rotated1 = active
          ? (position == 0 ? initial[initial_base + 24 + lane]
                           : output[output_base - 24 + lane])
          : 0.0f;
#pragma unroll 4
      for (int coordinate = 0; coordinate < factors; ++coordinate) {
        const float next0 = apply_factor(generators, representation, coordinate,
            row, rotated0, coordinates[coordinate_base0 + coordinate], factors);
        const float next1 = apply_factor(generators, representation, coordinate,
            row, rotated1, coordinates[coordinate_base1 + coordinate], factors);
        if (active) {
          rotated0 = next0;
          rotated1 = next1;
        }
        __syncwarp();
      }
    }

    const float left00 = warp_sum(active ? direct0 * rotated0 : 0.0f);
    const float left01 = warp_sum(active ? direct0 * rotated1 : 0.0f);
    const float left10 = warp_sum(active ? direct1 * rotated0 : 0.0f);
    const float left11 = warp_sum(active ? direct1 * rotated1 : 0.0f);
    if (lane == 0) {
      left_gradient[left_base] = left00;
      left_gradient[left_base + 1] = left01;
      left_gradient[left_base + 2] = left10;
      left_gradient[left_base + 3] = left11;
    }
    if (active) {
      drive_gradient[output_base + lane] = direct0;
      drive_gradient[output_base + 24 + lane] = direct1;
    }
    float adjoint0 = active ? l00 * direct0 + l10 * direct1 : 0.0f;
    float adjoint1 = active ? l01 * direct0 + l11 * direct1 : 0.0f;
    float state_after0 = rotated0;
    float state_after1 = rotated1;

#pragma unroll 4
    for (int reverse_coordinate = 0; reverse_coordinate < factors;
         ++reverse_coordinate) {
      const int coordinate = factors - 1 - reverse_coordinate;
      const float angle0 = coordinates[coordinate_base0 + coordinate];
      const float angle1 = coordinates[coordinate_base1 + coordinate];
      const float state_before0_raw = apply_factor(generators, representation,
          coordinate, row, state_after0, -angle0, factors);
      const float state_before1_raw = apply_factor(generators, representation,
          coordinate, row, state_after1, -angle1, factors);
      const float state_before0 = active ? state_before0_raw : 0.0f;
      const float state_before1 = active ? state_before1_raw : 0.0f;
      __syncwarp();
      const float derivative0 = factor_derivative(generators, representation,
          coordinate, row, state_before0, angle0, factors);
      const float derivative1 = factor_derivative(generators, representation,
          coordinate, row, state_before1, angle1, factors);
      const float angle_gradient0 = warp_sum(
          active ? adjoint0 * derivative0 : 0.0f);
      const float angle_gradient1 = warp_sum(
          active ? adjoint1 * derivative1 : 0.0f);
      if (lane == 0) {
        coordinate_gradient[coordinate_base0 + coordinate] = angle_gradient0;
        coordinate_gradient[coordinate_base1 + coordinate] = angle_gradient1;
      }
      const float inverse0 = apply_factor(generators, representation,
          coordinate, row, adjoint0, -angle0, factors);
      const float inverse1 = apply_factor(generators, representation,
          coordinate, row, adjoint1, -angle1, factors);
      if (active) {
        adjoint0 = inverse0;
        adjoint1 = inverse1;
        state_after0 = state_before0;
        state_after1 = state_before1;
      }
      __syncwarp();
    }
    carry0 = adjoint0;
    carry1 = adjoint1;
  }
  if (active) {
    initial_gradient[initial_base + lane] = carry0;
    initial_gradient[initial_base + 24 + lane] = carry1;
  }
}

void check_controller_inputs(
    const torch::Tensor& features, const torch::Tensor& weight,
    const torch::Tensor& bias, const torch::Tensor& generators,
    const torch::Tensor& scale, const torch::Tensor& initial,
    const torch::Tensor& gate) {
  TORCH_CHECK(features.is_cuda(), "controller tensors must be CUDA");
  TORCH_CHECK(features.scalar_type() == torch::kFloat32,
      "raw controller backend requires float32");
  for (const auto& tensor : {weight, bias, generators, scale, initial, gate}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32,
        "controller tensors must be CUDA float32");
    TORCH_CHECK(tensor.device() == features.device(), "devices must match");
    TORCH_CHECK(tensor.is_contiguous(), "controller tensors must be contiguous");
  }
  TORCH_CHECK(features.is_contiguous(), "features must be contiguous");
  TORCH_CHECK(features.dim() == 3, "features must be (B,L,F)");
  const auto batch = features.size(0), length = features.size(1);
  const auto input_size = features.size(2);
  TORCH_CHECK(scale.dim() == 3 && scale.size(0) == batch && scale.size(1) == length,
      "scale must be (B,L,C)");
  const auto channels = scale.size(2);
  TORCH_CHECK(weight.sizes() == torch::IntArrayRef({channels * 28, input_size}),
      "weight must be (C*28,F)");
  TORCH_CHECK(bias.numel() == channels * 28, "bias must be (C*28)");
  TORCH_CHECK(generators.sizes() == torch::IntArrayRef({3, 28, 8, 8}),
      "generators must be (3,28,8,8)");
  TORCH_CHECK(initial.sizes() == torch::IntArrayRef({batch, channels, 3, 8}),
      "initial must be (B,C,3,8)");
  TORCH_CHECK(gate.sizes() == torch::IntArrayRef({batch, length}),
      "gate must be (B,L)");
}

void check_coordinate_inputs(
    const torch::Tensor& coordinates, const torch::Tensor& generators,
    const torch::Tensor& scale, const torch::Tensor& initial) {
  TORCH_CHECK(coordinates.is_cuda() && coordinates.scalar_type() == torch::kFloat32,
      "raw coordinate backend requires CUDA float32");
  for (const auto& tensor : {generators, scale, initial}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32,
        "coordinate tensors must be CUDA float32");
    TORCH_CHECK(tensor.device() == coordinates.device(), "devices must match");
    TORCH_CHECK(tensor.is_contiguous(), "coordinate tensors must be contiguous");
  }
  TORCH_CHECK(coordinates.is_contiguous(), "coordinates must be contiguous");
  TORCH_CHECK(coordinates.dim() == 4 && coordinates.size(3) >= 1 &&
      coordinates.size(3) <= 28, "coordinates must be (B,L,C,F), 1 <= F <= 28");
  const auto batch = coordinates.size(0), length = coordinates.size(1);
  const auto channels = coordinates.size(2);
  TORCH_CHECK(scale.sizes() == torch::IntArrayRef({batch, length, channels}),
      "scale must be (B,L,C)");
  const auto factors = coordinates.size(3);
  TORCH_CHECK(generators.sizes() == torch::IntArrayRef({3, factors, 8, 8}),
      "generators must be (3,F,8,8)");
  TORCH_CHECK(initial.sizes() == torch::IntArrayRef({batch, channels, 3, 8}),
      "initial must be (B,C,3,8)");
}

void check_coupled_inputs(
    const torch::Tensor& coordinates, const torch::Tensor& generators,
    const torch::Tensor& left, const torch::Tensor& drive,
    const torch::Tensor& initial) {
  for (const auto& tensor : {coordinates, generators, left, drive, initial}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32,
        "coupled tensors must be CUDA float32");
    TORCH_CHECK(tensor.device() == coordinates.device(), "devices must match");
    TORCH_CHECK(tensor.is_contiguous(), "coupled tensors must be contiguous");
  }
  TORCH_CHECK(coordinates.dim() == 3 && coordinates.size(2) >= 1 &&
      coordinates.size(2) <= 28, "coordinates must be (B,L,F), 1 <= F <= 28");
  const auto batch = coordinates.size(0), length = coordinates.size(1);
  const auto factors = coordinates.size(2);
  TORCH_CHECK(generators.sizes() == torch::IntArrayRef({3, factors, 8, 8}),
      "generators must be (3,F,8,8)");
  TORCH_CHECK(left.sizes() == torch::IntArrayRef({batch, length, 2, 2}),
      "left must be (B,L,2,2)");
  TORCH_CHECK(drive.sizes() == torch::IntArrayRef({batch, length, 2, 3, 8}),
      "drive must be (B,L,2,3,8)");
  TORCH_CHECK(initial.sizes() == torch::IntArrayRef({batch, 2, 3, 8}),
      "initial must be (B,2,3,8)");
}

void check_independent_block_inputs(
    const torch::Tensor& coordinates, const torch::Tensor& generators,
    const torch::Tensor& left, const torch::Tensor& drive,
    const torch::Tensor& initial) {
  for (const auto& tensor : {coordinates, generators, left, drive, initial}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32,
        "independent-block tensors must be CUDA float32");
    TORCH_CHECK(tensor.device() == coordinates.device(), "devices must match");
    TORCH_CHECK(tensor.is_contiguous(),
        "independent-block tensors must be contiguous");
  }
  TORCH_CHECK(coordinates.dim() == 4 && coordinates.size(2) == 2 &&
      coordinates.size(3) >= 1 && coordinates.size(3) <= 28,
      "coordinates must be (B,L,2,F), 1 <= F <= 28");
  const auto batch = coordinates.size(0), length = coordinates.size(1);
  const auto factors = coordinates.size(3);
  TORCH_CHECK(generators.sizes() == torch::IntArrayRef({3, factors, 8, 8}),
      "generators must be (3,F,8,8)");
  TORCH_CHECK(left.sizes() == torch::IntArrayRef({batch, length, 2, 2}),
      "left must be (B,L,2,2)");
  TORCH_CHECK(drive.sizes() == torch::IntArrayRef({batch, length, 2, 3, 8}),
      "drive must be (B,L,2,3,8)");
  TORCH_CHECK(initial.sizes() == torch::IntArrayRef({batch, 2, 3, 8}),
      "initial must be (B,2,3,8)");
}

}  // namespace

torch::Tensor controller_factorized_forward_cuda(
    torch::Tensor features, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor generators, torch::Tensor scale, torch::Tensor drive,
    torch::Tensor initial, torch::Tensor gate) {
  check_controller_inputs(features, weight, bias, generators, scale, initial, gate);
  const int batch = features.size(0), length = features.size(1);
  const int input_size = features.size(2), channels = scale.size(2);
  TORCH_CHECK(drive.is_cuda() && drive.scalar_type() == torch::kFloat32 &&
      drive.is_contiguous() && drive.device() == features.device() &&
      drive.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "drive must be contiguous CUDA float32 with shape (B,L,C,3,8)");
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(features.device());
  controller_factorized_forward_kernel<<<batch * channels, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      features.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
      generators.data_ptr<float>(), scale.data_ptr<float>(), drive.data_ptr<float>(),
      initial.data_ptr<float>(), gate.data_ptr<float>(), output.data_ptr<float>(),
      length, channels, input_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> controller_factorized_backward_cuda(
    torch::Tensor features, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor generators, torch::Tensor scale, torch::Tensor drive,
    torch::Tensor initial,
    torch::Tensor gate, torch::Tensor output, torch::Tensor output_gradient) {
  check_controller_inputs(features, weight, bias, generators, scale, initial, gate);
  const int batch = features.size(0), length = features.size(1);
  const int input_size = features.size(2), channels = scale.size(2);
  TORCH_CHECK(drive.is_cuda() && drive.scalar_type() == torch::kFloat32 &&
      drive.is_contiguous() && drive.device() == features.device() &&
      drive.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "drive must be contiguous CUDA float32 with shape (B,L,C,3,8)");
  TORCH_CHECK(output.is_contiguous() && output_gradient.is_contiguous(),
      "output tensors must be contiguous");
  auto feature_gradient = torch::zeros_like(features);
  auto weight_gradient = torch::zeros_like(weight);
  auto bias_gradient = torch::zeros_like(bias);
  auto scale_gradient = torch::empty_like(scale);
  auto drive_gradient = torch::empty_like(output);
  auto initial_gradient = torch::empty_like(initial);
  const c10::cuda::CUDAGuard guard(features.device());
  controller_factorized_backward_kernel<<<batch * channels, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      features.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
      generators.data_ptr<float>(), scale.data_ptr<float>(), drive.data_ptr<float>(),
      initial.data_ptr<float>(), gate.data_ptr<float>(), output.data_ptr<float>(),
      output_gradient.data_ptr<float>(),
      feature_gradient.data_ptr<float>(), weight_gradient.data_ptr<float>(),
      bias_gradient.data_ptr<float>(), scale_gradient.data_ptr<float>(),
      drive_gradient.data_ptr<float>(), initial_gradient.data_ptr<float>(),
      length, channels, input_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {feature_gradient, weight_gradient, bias_gradient, scale_gradient,
      drive_gradient, initial_gradient};
}

torch::Tensor coordinate_factorized_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial) {
  check_coordinate_inputs(coordinates, generators, scale, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int channels = coordinates.size(2);
  const int factors = coordinates.size(3);
  TORCH_CHECK(drive.is_cuda() && drive.scalar_type() == torch::kFloat32 &&
      drive.is_contiguous() && drive.device() == coordinates.device() &&
      drive.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "drive must be contiguous CUDA float32 with shape (B,L,C,3,8)");
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  coordinate_factorized_forward_kernel<<<batch * channels, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      scale.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), length, channels, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> coordinate_factorized_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient) {
  check_coordinate_inputs(coordinates, generators, scale, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int channels = coordinates.size(2);
  const int factors = coordinates.size(3);
  TORCH_CHECK(drive.is_cuda() && drive.scalar_type() == torch::kFloat32 &&
      drive.is_contiguous() && drive.device() == coordinates.device() &&
      drive.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "drive must be contiguous CUDA float32 with shape (B,L,C,3,8)");
  TORCH_CHECK(output.is_cuda() && output_gradient.is_cuda() &&
      output.is_contiguous() && output_gradient.is_contiguous() &&
      output.device() == coordinates.device() &&
      output_gradient.device() == coordinates.device(),
      "output tensors must be contiguous and on the coordinate device");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}) &&
      output_gradient.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "output tensors must be (B,L,C,3,8)");
  auto coordinate_gradient = torch::empty_like(coordinates);
  auto scale_gradient = torch::empty_like(scale);
  auto drive_gradient = torch::empty_like(output);
  auto initial_gradient = torch::empty_like(initial);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  coordinate_factorized_backward_kernel<<<batch * channels, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      scale.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), output_gradient.data_ptr<float>(),
      coordinate_gradient.data_ptr<float>(),
      scale_gradient.data_ptr<float>(), drive_gradient.data_ptr<float>(),
      initial_gradient.data_ptr<float>(), length, channels, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {coordinate_gradient, scale_gradient, drive_gradient, initial_gradient};
}

torch::Tensor isotypic_coordinate_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial) {
  check_coordinate_inputs(coordinates, generators, scale, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int channels = coordinates.size(2);
  const int factors = coordinates.size(3);
  TORCH_CHECK(drive.is_cuda() && drive.scalar_type() == torch::kFloat32 &&
      drive.is_contiguous() && drive.device() == coordinates.device() &&
      drive.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "drive must be contiguous CUDA float32 with shape (B,L,C,3,8)");
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  isotypic_coordinate_forward_kernel<<<batch * channels * 3, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      scale.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), length, channels, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> isotypic_coordinate_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient) {
  check_coordinate_inputs(coordinates, generators, scale, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int channels = coordinates.size(2);
  const int factors = coordinates.size(3);
  TORCH_CHECK(drive.is_cuda() && drive.scalar_type() == torch::kFloat32 &&
      drive.is_contiguous() && drive.device() == coordinates.device() &&
      drive.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "drive must be contiguous CUDA float32 with shape (B,L,C,3,8)");
  TORCH_CHECK(output.is_cuda() && output_gradient.is_cuda() &&
      output.is_contiguous() && output_gradient.is_contiguous() &&
      output.device() == coordinates.device() &&
      output_gradient.device() == coordinates.device(),
      "output tensors must be contiguous and on the coordinate device");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}) &&
      output_gradient.sizes() == torch::IntArrayRef({batch, length, channels, 3, 8}),
      "output tensors must be (B,L,C,3,8)");
  const int coordinate_elements = coordinates.numel();
  const int scale_elements = scale.numel();
  auto coordinate_gradient_by_representation = torch::empty(
      {3, coordinate_elements}, coordinates.options());
  auto scale_gradient_by_representation = torch::empty(
      {3, scale_elements}, scale.options());
  auto drive_gradient = torch::empty_like(output);
  auto initial_gradient = torch::empty_like(initial);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  isotypic_coordinate_backward_kernel<<<batch * channels * 3, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      scale.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), output_gradient.data_ptr<float>(),
      coordinate_gradient_by_representation.data_ptr<float>(),
      scale_gradient_by_representation.data_ptr<float>(),
      drive_gradient.data_ptr<float>(), initial_gradient.data_ptr<float>(),
      length, channels, factors, coordinate_elements, scale_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto coordinate_gradient =
      coordinate_gradient_by_representation.sum(0).view_as(coordinates);
  auto scale_gradient = scale_gradient_by_representation.sum(0).view_as(scale);
  return {coordinate_gradient, scale_gradient, drive_gradient, initial_gradient};
}

torch::Tensor coupled_coordinate_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial) {
  check_coupled_inputs(coordinates, generators, left, drive, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int factors = coordinates.size(2);
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  coupled_coordinate_forward_kernel<<<batch, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      left.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), length, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> coupled_coordinate_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient) {
  check_coupled_inputs(coordinates, generators, left, drive, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int factors = coordinates.size(2);
  TORCH_CHECK(output.is_cuda() && output_gradient.is_cuda() &&
      output.is_contiguous() && output_gradient.is_contiguous() &&
      output.device() == coordinates.device() &&
      output_gradient.device() == coordinates.device(),
      "output tensors must be contiguous and on the coordinate device");
  TORCH_CHECK(output.sizes() == drive.sizes() &&
      output_gradient.sizes() == drive.sizes(),
      "output tensors must be (B,L,2,3,8)");
  auto coordinate_gradient = torch::empty_like(coordinates);
  auto left_gradient = torch::empty_like(left);
  auto drive_gradient = torch::empty_like(drive);
  auto initial_gradient = torch::empty_like(initial);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  coupled_coordinate_backward_kernel<<<batch, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      left.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), output_gradient.data_ptr<float>(),
      coordinate_gradient.data_ptr<float>(), left_gradient.data_ptr<float>(),
      drive_gradient.data_ptr<float>(), initial_gradient.data_ptr<float>(),
      length, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {coordinate_gradient, left_gradient, drive_gradient, initial_gradient};
}

torch::Tensor independent_block_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial) {
  check_independent_block_inputs(coordinates, generators, left, drive, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int factors = coordinates.size(3);
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  independent_block_forward_kernel<<<batch, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      left.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), length, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor independent_block_isotypic_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial) {
  check_independent_block_inputs(coordinates, generators, left, drive, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int factors = coordinates.size(3);
  auto output = torch::empty_like(drive);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  independent_block_isotypic_forward_kernel<<<batch * 3, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      left.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), length, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> independent_block_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient) {
  check_independent_block_inputs(coordinates, generators, left, drive, initial);
  const int batch = coordinates.size(0), length = coordinates.size(1);
  const int factors = coordinates.size(3);
  TORCH_CHECK(output.is_cuda() && output_gradient.is_cuda() &&
      output.is_contiguous() && output_gradient.is_contiguous() &&
      output.device() == coordinates.device() &&
      output_gradient.device() == coordinates.device(),
      "output tensors must be contiguous and on the coordinate device");
  TORCH_CHECK(output.sizes() == drive.sizes() &&
      output_gradient.sizes() == drive.sizes(),
      "output tensors must be (B,L,2,3,8)");
  auto coordinate_gradient = torch::empty_like(coordinates);
  auto left_gradient = torch::empty_like(left);
  auto drive_gradient = torch::empty_like(drive);
  auto initial_gradient = torch::empty_like(initial);
  const c10::cuda::CUDAGuard guard(coordinates.device());
  independent_block_backward_kernel<<<batch, 32, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      coordinates.data_ptr<float>(), generators.data_ptr<float>(),
      left.data_ptr<float>(), drive.data_ptr<float>(), initial.data_ptr<float>(),
      output.data_ptr<float>(), output_gradient.data_ptr<float>(),
      coordinate_gradient.data_ptr<float>(), left_gradient.data_ptr<float>(),
      drive_gradient.data_ptr<float>(), initial_gradient.data_ptr<float>(),
      length, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {coordinate_gradient, left_gradient, drive_gradient, initial_gradient};
}
