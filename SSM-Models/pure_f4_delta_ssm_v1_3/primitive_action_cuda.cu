#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

constexpr int VALUE_DIM = 27;
constexpr int BLOCK_DIM = 3;
constexpr int BLOCK_COUNT = 9;

__device__ __forceinline__ float compact_component(
    const float* state,
    const float* local,
    const float* local_square,
    const int* permutation,
    int state_base,
    int block,
    int row,
    float angle,
    float frequency) {
  float first = 0.0f;
  float second = 0.0f;
#pragma unroll
  for (int source = 0; source < BLOCK_DIM; ++source) {
    const int index = permutation[block * BLOCK_DIM + source];
    const float value = state[state_base + index];
    first = fmaf(local[(block * BLOCK_DIM + row) * BLOCK_DIM + source], value, first);
    second = fmaf(
        local_square[(block * BLOCK_DIM + row) * BLOCK_DIM + source],
        value,
        second);
  }
  const int index = permutation[block * BLOCK_DIM + row];
  const float value = state[state_base + index];
  if (frequency < 1.0e-12f) {
    return value + angle * first + 0.5f * angle * angle * second;
  }
  float sine;
  float cosine;
  sincosf(angle * frequency, &sine, &cosine);
  return value + (sine / frequency) * first
      + ((1.0f - cosine) / (frequency * frequency)) * second;
}

__device__ __forceinline__ float symmetric_component(
    const float* state,
    const float* eigenvectors,
    const float* eigenvalues,
    const int* permutation,
    int state_base,
    int block,
    int row,
    float angle) {
  float output = 0.0f;
#pragma unroll
  for (int mode = 0; mode < BLOCK_DIM; ++mode) {
    float modal = 0.0f;
#pragma unroll
    for (int source = 0; source < BLOCK_DIM; ++source) {
      const int index = permutation[block * BLOCK_DIM + source];
      modal = fmaf(
          eigenvectors[(block * BLOCK_DIM + source) * BLOCK_DIM + mode],
          state[state_base + index],
          modal);
    }
    output = fmaf(
        eigenvectors[(block * BLOCK_DIM + row) * BLOCK_DIM + mode]
            * expf(angle * eigenvalues[block * BLOCK_DIM + mode]),
        modal,
        output);
  }
  return output;
}

__device__ __forceinline__ float generator_component(
    const float* state,
    const float* local,
    const int* permutation,
    int state_base,
    int block,
    int row) {
  float output = 0.0f;
#pragma unroll
  for (int source = 0; source < BLOCK_DIM; ++source) {
    const int index = permutation[block * BLOCK_DIM + source];
    output = fmaf(
        local[(block * BLOCK_DIM + row) * BLOCK_DIM + source],
        state[state_base + index],
        output);
  }
  return output;
}

__global__ void primitive_forward_kernel(
    const float* values,
    const float* coordinates,
    const int* permutations,
    const int* kinds,
    const float* local_generators,
    const float* local_generator_squares,
    const float* frequencies,
    const float* eigenvectors,
    const float* eigenvalues,
    float* output,
    int copies,
    int factors) {
  extern __shared__ float state[];
  const int linear = threadIdx.x;
  const int width = copies * VALUE_DIM;
  const int item = blockIdx.x;
  if (linear < width) {
    state[linear] = values[item * width + linear];
  }
  __syncthreads();

  const bool active = linear < width;
  const int copy = active ? linear / VALUE_DIM : 0;
  const int slot = active ? linear - copy * VALUE_DIM : 0;
  const int block = slot / BLOCK_DIM;
  const int row = slot - block * BLOCK_DIM;
  for (int factor = 0; factor < factors; ++factor) {
    float next = 0.0f;
    int destination = 0;
    if (active) {
      const int* permutation = permutations + factor * VALUE_DIM;
      const float* local = local_generators
          + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
      const float* local_square = local_generator_squares
          + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
      const float* vectors = eigenvectors
          + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
      const float* spectrum = eigenvalues
          + factor * BLOCK_COUNT * BLOCK_DIM;
      const float angle = coordinates[item * factors + factor];
      destination = copy * VALUE_DIM + permutation[slot];
      next = kinds[factor] == 0
          ? compact_component(
                state,
                local,
                local_square,
                permutation,
                copy * VALUE_DIM,
                block,
                row,
                angle,
                frequencies[factor * BLOCK_COUNT + block])
          : symmetric_component(
                state,
                vectors,
                spectrum,
                permutation,
                copy * VALUE_DIM,
                block,
                row,
                angle);
    }
    __syncthreads();
    if (active) {
      state[destination] = next;
    }
    __syncthreads();
  }
  if (linear < width) {
    output[item * width + linear] = state[linear];
  }
}

__global__ void primitive_backward_kernel(
    const float* coordinates,
    const float* output,
    const float* output_gradient,
    const int* permutations,
    const int* kinds,
    const float* local_generators,
    const float* local_generator_squares,
    const float* frequencies,
    const float* eigenvectors,
    const float* eigenvalues,
    float* value_gradient,
    float* coordinate_gradient,
    int copies,
    int factors) {
  extern __shared__ float shared[];
  const int linear = threadIdx.x;
  const int threads = blockDim.x;
  const int width = copies * VALUE_DIM;
  const int item = blockIdx.x;
  float* state = shared;
  float* gradient = shared + threads;
  float* reduction = shared + 2 * threads;
  if (linear < width) {
    state[linear] = output[item * width + linear];
    gradient[linear] = output_gradient[item * width + linear];
  } else {
    state[linear] = 0.0f;
    gradient[linear] = 0.0f;
  }
  __syncthreads();

  for (int reverse = 0; reverse < factors; ++reverse) {
    const int factor = factors - 1 - reverse;
    const int* permutation = permutations + factor * VALUE_DIM;
    const float* local = local_generators
        + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
    const float* local_square = local_generator_squares
        + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
    const float* vectors = eigenvectors
        + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
    const float* spectrum = eigenvalues
        + factor * BLOCK_COUNT * BLOCK_DIM;
    const float angle = coordinates[item * factors + factor];

    float contribution = 0.0f;
    float previous_state = 0.0f;
    float previous_gradient = 0.0f;
    int destination = 0;
    if (linear < width) {
      const int copy = linear / VALUE_DIM;
      const int slot = linear - copy * VALUE_DIM;
      const int block = slot / BLOCK_DIM;
      const int row = slot - block * BLOCK_DIM;
      destination = copy * VALUE_DIM + permutation[slot];
      contribution = gradient[destination]
          * generator_component(
                state, local, permutation, copy * VALUE_DIM, block, row);
      previous_state = kinds[factor] == 0
          ? compact_component(
                state,
                local,
                local_square,
                permutation,
                copy * VALUE_DIM,
                block,
                row,
                -angle,
                frequencies[factor * BLOCK_COUNT + block])
          : symmetric_component(
                state,
                vectors,
                spectrum,
                permutation,
                copy * VALUE_DIM,
                block,
                row,
                -angle);
      previous_gradient = kinds[factor] == 0
          ? compact_component(
                gradient,
                local,
                local_square,
                permutation,
                copy * VALUE_DIM,
                block,
                row,
                -angle,
                frequencies[factor * BLOCK_COUNT + block])
          : symmetric_component(
                gradient,
                vectors,
                spectrum,
                permutation,
                copy * VALUE_DIM,
                block,
                row,
                angle);
    }
    reduction[linear] = contribution;
    __syncthreads();
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
      if (linear < offset) {
        reduction[linear] += reduction[linear + offset];
      }
      __syncthreads();
    }
    if (linear == 0) {
      coordinate_gradient[item * factors + factor] = reduction[0];
    }
    if (linear < width) {
      state[destination] = previous_state;
      gradient[destination] = previous_gradient;
    }
    __syncthreads();
  }
  if (linear < width) {
    value_gradient[item * width + linear] = gradient[linear];
  }
}

int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

void validate_common(
    const torch::Tensor& values,
    const torch::Tensor& coordinates,
    const torch::Tensor& permutations,
    const torch::Tensor& kinds,
    const torch::Tensor& local_generators,
    const torch::Tensor& local_generator_squares,
    const torch::Tensor& frequencies,
    const torch::Tensor& eigenvectors,
    const torch::Tensor& eigenvalues) {
  TORCH_CHECK(values.is_cuda() && coordinates.is_cuda(), "values and coordinates must be CUDA");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 && coordinates.scalar_type() == torch::kFloat32,
      "values and coordinates must be float32");
  TORCH_CHECK(values.dim() == 3 && values.size(2) == VALUE_DIM,
      "values must be (items,copies,27)");
  TORCH_CHECK(coordinates.dim() == 2 && coordinates.size(0) == values.size(0),
      "coordinates must be (items,factors)");
  TORCH_CHECK(values.size(1) * VALUE_DIM <= 1024, "copies*27 must not exceed 1024 threads");
  TORCH_CHECK(permutations.scalar_type() == torch::kInt32 && kinds.scalar_type() == torch::kInt32,
      "permutations and kinds must be int32");
  TORCH_CHECK(permutations.size(0) == coordinates.size(1) && permutations.size(1) == VALUE_DIM,
      "permutation shape mismatch");
  TORCH_CHECK(kinds.numel() == coordinates.size(1), "kind shape mismatch");
  TORCH_CHECK(local_generators.sizes() == local_generator_squares.sizes(), "local generator shape mismatch");
  TORCH_CHECK(local_generators.size(0) == coordinates.size(1), "local generator factor mismatch");
  TORCH_CHECK(frequencies.size(0) == coordinates.size(1), "frequency factor mismatch");
  TORCH_CHECK(eigenvectors.size(0) == coordinates.size(1), "eigenvector factor mismatch");
  TORCH_CHECK(eigenvalues.size(0) == coordinates.size(1), "eigenvalue factor mismatch");
  TORCH_CHECK(values.is_contiguous() && coordinates.is_contiguous()
      && permutations.is_contiguous() && kinds.is_contiguous()
      && local_generators.is_contiguous() && local_generator_squares.is_contiguous()
      && frequencies.is_contiguous() && eigenvectors.is_contiguous()
      && eigenvalues.is_contiguous(), "all primitive tensors must be contiguous");
}

__global__ void primitive_delta_forward_kernel(
    const float* retention,
    const float* write_key,
    const float* erase_key,
    const float* write_value,
    const float* initial_state,
    const float* query,
    const float* event_coordinates,
    const int* permutations,
    const int* kinds,
    const float* local_generators,
    const float* local_generator_squares,
    const float* frequencies,
    const float* eigenvectors,
    const float* eigenvalues,
    float* reads,
    float* states,
    int length,
    int heads,
    int rank,
    int factors,
    int events,
    int event_stride,
    int first_event_local,
    bool transport_enabled) {
  extern __shared__ float state[];
  const int linear = threadIdx.x;
  const int width = heads * VALUE_DIM;
  const int batch = blockIdx.x;
  const bool active = linear < width;
  const int head = active ? linear / VALUE_DIM : 0;
  const int value_index = active ? linear - head * VALUE_DIM : 0;
  if (active) {
    state[linear] = initial_state[batch * width + linear];
  }
  __syncthreads();

  for (int position = 0; position < length; ++position) {
    const int token = batch * length + position;
    float next = 0.0f;
    if (active) {
      next = retention[token * heads + head] * state[linear];
      for (int update = 0; update < rank; ++update) {
        float projection = 0.0f;
        for (int source_head = 0; source_head < heads; ++source_head) {
          const float retained = retention[token * heads + source_head]
              * state[source_head * VALUE_DIM + value_index];
          projection = fmaf(
              erase_key[(token * rank + update) * heads + source_head],
              retained,
              projection);
        }
        next -= write_key[(token * rank + update) * heads + head] * projection;
      }
    }
    __syncthreads();
    if (active) {
      state[linear] = next;
    }
    __syncthreads();

    const bool is_event = position >= first_event_local
        && ((position - first_event_local) % event_stride == 0);
    const int event = is_event
        ? (position - first_event_local) / event_stride
        : 0;
    if (is_event && transport_enabled) {
      const int slot = active ? value_index : 0;
      const int block = slot / BLOCK_DIM;
      const int row = slot - block * BLOCK_DIM;
      for (int factor = 0; factor < factors; ++factor) {
        float transported = 0.0f;
        int destination = 0;
        if (active) {
          const int* permutation = permutations + factor * VALUE_DIM;
          const float* local = local_generators
              + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
          const float* local_square = local_generator_squares
              + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
          const float* vectors = eigenvectors
              + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
          const float* spectrum = eigenvalues
              + factor * BLOCK_COUNT * BLOCK_DIM;
          const float angle = event_coordinates[
              (batch * events + event) * factors + factor];
          destination = head * VALUE_DIM + permutation[slot];
          transported = kinds[factor] == 0
              ? compact_component(
                    state, local, local_square, permutation,
                    head * VALUE_DIM, block, row, angle,
                    frequencies[factor * BLOCK_COUNT + block])
              : symmetric_component(
                    state, vectors, spectrum, permutation,
                    head * VALUE_DIM, block, row, angle);
        }
        __syncthreads();
        if (active) {
          state[destination] = transported;
        }
        __syncthreads();
      }
    }

    if (active) {
      next = state[linear];
      for (int update = 0; update < rank; ++update) {
        next = fmaf(
            write_key[(token * rank + update) * heads + head],
            write_value[(token * rank + update) * VALUE_DIM + value_index],
            next);
      }
    }
    __syncthreads();
    if (active) {
      state[linear] = next;
      states[(token * heads + head) * VALUE_DIM + value_index] = next;
    }
    __syncthreads();
    if (linear < VALUE_DIM) {
      float read = 0.0f;
      for (int source_head = 0; source_head < heads; ++source_head) {
        read = fmaf(
            query[token * heads + source_head],
            state[source_head * VALUE_DIM + linear],
            read);
      }
      reads[token * VALUE_DIM + linear] = read;
    }
    __syncthreads();
  }
}

__global__ void primitive_delta_backward_kernel(
    const float* retention,
    const float* write_key,
    const float* erase_key,
    const float* write_value,
    const float* initial_state,
    const float* query,
    const float* event_coordinates,
    const float* states,
    const float* read_gradient,
    const float* final_gradient,
    const int* permutations,
    const int* kinds,
    const float* local_generators,
    const float* local_generator_squares,
    const float* frequencies,
    const float* eigenvectors,
    const float* eigenvalues,
    float* retention_gradient,
    float* write_key_gradient,
    float* erase_key_gradient,
    float* write_value_gradient,
    float* initial_gradient,
    float* query_gradient,
    float* coordinate_gradient,
    int length,
    int heads,
    int rank,
    int factors,
    int events,
    int event_stride,
    int first_event_local,
    bool transport_enabled) {
  extern __shared__ float shared[];
  const int linear = threadIdx.x;
  const int threads = blockDim.x;
  const int width = heads * VALUE_DIM;
  const int batch = blockIdx.x;
  const bool active = linear < width;
  const int head = active ? linear / VALUE_DIM : 0;
  const int value_index = active ? linear - head * VALUE_DIM : 0;
  float* action_state = shared;
  float* gradient = shared + threads;
  float* reduction = shared + 2 * threads;
  action_state[linear] = 0.0f;
  gradient[linear] = active ? final_gradient[batch * width + linear] : 0.0f;
  __syncthreads();

  for (int reverse_position = 0; reverse_position < length; ++reverse_position) {
    const int position = length - 1 - reverse_position;
    const int token = batch * length + position;
    if (active) {
      gradient[linear] += query[token * heads + head]
          * read_gradient[token * VALUE_DIM + value_index];
    }
    __syncthreads();
    if (linear < heads) {
      float result = 0.0f;
      for (int component = 0; component < VALUE_DIM; ++component) {
        result = fmaf(
            read_gradient[token * VALUE_DIM + component],
            states[(token * heads + linear) * VALUE_DIM + component],
            result);
      }
      query_gradient[token * heads + linear] = result;
    }
    if (linear < rank * VALUE_DIM) {
      const int update = linear / VALUE_DIM;
      const int component = linear - update * VALUE_DIM;
      float result = 0.0f;
      for (int source_head = 0; source_head < heads; ++source_head) {
        result = fmaf(
            write_key[(token * rank + update) * heads + source_head],
            gradient[source_head * VALUE_DIM + component],
            result);
      }
      write_value_gradient[(token * rank + update) * VALUE_DIM + component] = result;
    }
    if (linear < rank * heads) {
      const int update = linear / heads;
      const int target_head = linear - update * heads;
      float result = 0.0f;
      for (int component = 0; component < VALUE_DIM; ++component) {
        result = fmaf(
            gradient[target_head * VALUE_DIM + component],
            write_value[(token * rank + update) * VALUE_DIM + component],
            result);
      }
      write_key_gradient[(token * rank + update) * heads + target_head] = result;
    }
    if (active) {
      float before_write = states[(token * heads + head) * VALUE_DIM + value_index];
      for (int update = 0; update < rank; ++update) {
        before_write -= write_key[(token * rank + update) * heads + head]
            * write_value[(token * rank + update) * VALUE_DIM + value_index];
      }
      action_state[linear] = before_write;
    }
    __syncthreads();

    const bool is_event = position >= first_event_local
        && ((position - first_event_local) % event_stride == 0);
    const int event = is_event
        ? (position - first_event_local) / event_stride
        : 0;
    if (is_event && transport_enabled) {
      const int slot = active ? value_index : 0;
      const int block = slot / BLOCK_DIM;
      const int row = slot - block * BLOCK_DIM;
      for (int reverse_factor = 0; reverse_factor < factors; ++reverse_factor) {
        const int factor = factors - 1 - reverse_factor;
        float previous_state = 0.0f;
        float previous_gradient = 0.0f;
        float contribution = 0.0f;
        int destination = 0;
        if (active) {
          const int* permutation = permutations + factor * VALUE_DIM;
          const float* local = local_generators
              + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
          const float* local_square = local_generator_squares
              + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
          const float* vectors = eigenvectors
              + factor * BLOCK_COUNT * BLOCK_DIM * BLOCK_DIM;
          const float* spectrum = eigenvalues
              + factor * BLOCK_COUNT * BLOCK_DIM;
          const float angle = event_coordinates[
              (batch * events + event) * factors + factor];
          destination = head * VALUE_DIM + permutation[slot];
          contribution = gradient[destination]
              * generator_component(
                    action_state, local, permutation,
                    head * VALUE_DIM, block, row);
          previous_state = kinds[factor] == 0
              ? compact_component(
                    action_state, local, local_square, permutation,
                    head * VALUE_DIM, block, row, -angle,
                    frequencies[factor * BLOCK_COUNT + block])
              : symmetric_component(
                    action_state, vectors, spectrum, permutation,
                    head * VALUE_DIM, block, row, -angle);
          previous_gradient = kinds[factor] == 0
              ? compact_component(
                    gradient, local, local_square, permutation,
                    head * VALUE_DIM, block, row, -angle,
                    frequencies[factor * BLOCK_COUNT + block])
              : symmetric_component(
                    gradient, vectors, spectrum, permutation,
                    head * VALUE_DIM, block, row, angle);
        }
        reduction[linear] = contribution;
        __syncthreads();
        for (int offset = threads / 2; offset > 0; offset >>= 1) {
          if (linear < offset) {
            reduction[linear] += reduction[linear + offset];
          }
          __syncthreads();
        }
        if (linear == 0) {
          coordinate_gradient[(batch * events + event) * factors + factor]
              = reduction[0];
        }
        if (active) {
          action_state[destination] = previous_state;
          gradient[destination] = previous_gradient;
        }
        __syncthreads();
      }
    }

    if (linear < rank * heads) {
      const int update = linear / heads;
      const int target_head = linear - update * heads;
      float key_result = 0.0f;
      float erase_result = 0.0f;
      for (int component = 0; component < VALUE_DIM; ++component) {
        float projection = 0.0f;
        float action_cotangent = 0.0f;
        for (int source_head = 0; source_head < heads; ++source_head) {
          const float previous = position == 0
              ? initial_state[(batch * heads + source_head) * VALUE_DIM + component]
              : states[((batch * length + position - 1) * heads + source_head)
                  * VALUE_DIM + component];
          const float retained = retention[token * heads + source_head] * previous;
          projection = fmaf(
              erase_key[(token * rank + update) * heads + source_head],
              retained,
              projection);
          action_cotangent = fmaf(
              write_key[(token * rank + update) * heads + source_head],
              gradient[source_head * VALUE_DIM + component],
              action_cotangent);
        }
        const float target_previous = position == 0
            ? initial_state[(batch * heads + target_head) * VALUE_DIM + component]
            : states[((batch * length + position - 1) * heads + target_head)
                * VALUE_DIM + component];
        const float target_retained = retention[token * heads + target_head]
            * target_previous;
        key_result -= gradient[target_head * VALUE_DIM + component] * projection;
        erase_result -= action_cotangent * target_retained;
      }
      write_key_gradient[(token * rank + update) * heads + target_head] += key_result;
      erase_key_gradient[(token * rank + update) * heads + target_head] = erase_result;
    }

    float previous_gradient = 0.0f;
    if (active) {
      float retained_gradient = gradient[linear];
      for (int update = 0; update < rank; ++update) {
        float action_cotangent = 0.0f;
        for (int source_head = 0; source_head < heads; ++source_head) {
          action_cotangent = fmaf(
              write_key[(token * rank + update) * heads + source_head],
              gradient[source_head * VALUE_DIM + value_index],
              action_cotangent);
        }
        retained_gradient -= erase_key[(token * rank + update) * heads + head]
            * action_cotangent;
      }
      previous_gradient = retained_gradient * retention[token * heads + head];
    }
    if (linear < heads) {
      float result = 0.0f;
      for (int component = 0; component < VALUE_DIM; ++component) {
        float retained_gradient = gradient[linear * VALUE_DIM + component];
        for (int update = 0; update < rank; ++update) {
          float action_cotangent = 0.0f;
          for (int source_head = 0; source_head < heads; ++source_head) {
            action_cotangent = fmaf(
                write_key[(token * rank + update) * heads + source_head],
                gradient[source_head * VALUE_DIM + component],
                action_cotangent);
          }
          retained_gradient -= erase_key[(token * rank + update) * heads + linear]
              * action_cotangent;
        }
        const float previous = position == 0
            ? initial_state[(batch * heads + linear) * VALUE_DIM + component]
            : states[((batch * length + position - 1) * heads + linear)
                * VALUE_DIM + component];
        result = fmaf(retained_gradient, previous, result);
      }
      retention_gradient[token * heads + linear] = result;
    }
    __syncthreads();
    if (active) {
      gradient[linear] = previous_gradient;
    }
    __syncthreads();
  }
  if (active) {
    initial_gradient[batch * width + linear] = gradient[linear];
  }
}

}  // namespace

torch::Tensor primitive_action_forward_cuda(
    torch::Tensor values,
    torch::Tensor coordinates,
    torch::Tensor permutations,
    torch::Tensor kinds,
    torch::Tensor local_generators,
    torch::Tensor local_generator_squares,
    torch::Tensor frequencies,
    torch::Tensor eigenvectors,
    torch::Tensor eigenvalues) {
  validate_common(values, coordinates, permutations, kinds, local_generators,
      local_generator_squares, frequencies, eigenvectors, eigenvalues);
  const c10::cuda::CUDAGuard guard(values.device());
  const int copies = values.size(1);
  const int factors = coordinates.size(1);
  const int threads = next_power_of_two(copies * VALUE_DIM);
  auto output = torch::empty_like(values);
  primitive_forward_kernel<<<values.size(0), threads, copies * VALUE_DIM * sizeof(float),
      at::cuda::getCurrentCUDAStream()>>>(
          values.data_ptr<float>(), coordinates.data_ptr<float>(),
          permutations.data_ptr<int>(), kinds.data_ptr<int>(),
          local_generators.data_ptr<float>(), local_generator_squares.data_ptr<float>(),
          frequencies.data_ptr<float>(), eigenvectors.data_ptr<float>(),
          eigenvalues.data_ptr<float>(), output.data_ptr<float>(), copies, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> primitive_action_backward_cuda(
    torch::Tensor coordinates,
    torch::Tensor output,
    torch::Tensor output_gradient,
    torch::Tensor permutations,
    torch::Tensor kinds,
    torch::Tensor local_generators,
    torch::Tensor local_generator_squares,
    torch::Tensor frequencies,
    torch::Tensor eigenvectors,
    torch::Tensor eigenvalues) {
  validate_common(output, coordinates, permutations, kinds, local_generators,
      local_generator_squares, frequencies, eigenvectors, eigenvalues);
  TORCH_CHECK(output_gradient.sizes() == output.sizes()
      && output_gradient.scalar_type() == torch::kFloat32
      && output_gradient.is_cuda() && output_gradient.is_contiguous(),
      "output gradient must be contiguous CUDA float32 matching output");
  const c10::cuda::CUDAGuard guard(output.device());
  const int copies = output.size(1);
  const int factors = coordinates.size(1);
  const int threads = next_power_of_two(copies * VALUE_DIM);
  auto value_gradient = torch::empty_like(output);
  auto coordinate_gradient = torch::empty_like(coordinates);
  const size_t shared_bytes = 3 * threads * sizeof(float);
  primitive_backward_kernel<<<output.size(0), threads, shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
          coordinates.data_ptr<float>(), output.data_ptr<float>(),
          output_gradient.data_ptr<float>(), permutations.data_ptr<int>(),
          kinds.data_ptr<int>(), local_generators.data_ptr<float>(),
          local_generator_squares.data_ptr<float>(), frequencies.data_ptr<float>(),
          eigenvectors.data_ptr<float>(), eigenvalues.data_ptr<float>(),
          value_gradient.data_ptr<float>(), coordinate_gradient.data_ptr<float>(),
          copies, factors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {value_gradient, coordinate_gradient};
}

std::vector<torch::Tensor> primitive_delta_forward_cuda(
    torch::Tensor retention,
    torch::Tensor write_key,
    torch::Tensor erase_key,
    torch::Tensor write_value,
    torch::Tensor initial_state,
    torch::Tensor query,
    torch::Tensor event_coordinates,
    torch::Tensor permutations,
    torch::Tensor kinds,
    torch::Tensor local_generators,
    torch::Tensor local_generator_squares,
    torch::Tensor frequencies,
    torch::Tensor eigenvectors,
    torch::Tensor eigenvalues,
    int64_t event_stride,
    int64_t first_event_local,
    bool transport_enabled) {
  TORCH_CHECK(retention.is_cuda() && write_key.is_cuda() && erase_key.is_cuda()
      && write_value.is_cuda() && initial_state.is_cuda() && query.is_cuda()
      && event_coordinates.is_cuda(), "all recurrence tensors must be CUDA");
  TORCH_CHECK(retention.scalar_type() == torch::kFloat32
      && write_key.scalar_type() == torch::kFloat32
      && erase_key.scalar_type() == torch::kFloat32
      && write_value.scalar_type() == torch::kFloat32
      && initial_state.scalar_type() == torch::kFloat32
      && query.scalar_type() == torch::kFloat32
      && event_coordinates.scalar_type() == torch::kFloat32,
      "all recurrence tensors must be float32");
  TORCH_CHECK(retention.dim() == 3, "retention must be (B,L,H)");
  TORCH_CHECK(write_key.dim() == 4 && erase_key.sizes() == write_key.sizes(),
      "keys must have matching shape (B,L,R,H)");
  TORCH_CHECK(write_value.dim() == 4 && write_value.size(3) == VALUE_DIM,
      "write value must be (B,L,R,27)");
  TORCH_CHECK(initial_state.dim() == 3 && initial_state.size(2) == VALUE_DIM,
      "initial state must be (B,H,27)");
  TORCH_CHECK(query.sizes() == retention.sizes(), "query must match retention");
  TORCH_CHECK(event_coordinates.dim() == 3, "event coordinates must be (B,E,F)");
  const int batch = retention.size(0);
  const int length = retention.size(1);
  const int heads = retention.size(2);
  const int rank = write_key.size(2);
  const int events = event_coordinates.size(1);
  const int factors = event_coordinates.size(2);
  TORCH_CHECK(length > 0 && heads > 0 && rank > 0 && factors > 0,
      "recurrence dimensions must be positive");
  TORCH_CHECK(write_key.size(0) == batch && write_key.size(1) == length
      && write_key.size(3) == heads, "key shape mismatch");
  TORCH_CHECK(write_value.size(0) == batch && write_value.size(1) == length
      && write_value.size(2) == rank, "write value shape mismatch");
  TORCH_CHECK(initial_state.size(0) == batch && initial_state.size(1) == heads,
      "initial state shape mismatch");
  TORCH_CHECK(event_coordinates.size(0) == batch, "event batch mismatch");
  TORCH_CHECK(permutations.size(0) == factors && kinds.numel() == factors,
      "primitive metadata factor mismatch");
  TORCH_CHECK(event_stride > 0 && first_event_local >= 0,
      "event stride must be positive and first event local nonnegative");
  const int expected_events = first_event_local >= length
      ? 0
      : 1 + (length - 1 - first_event_local) / event_stride;
  TORCH_CHECK(events == expected_events, "event coordinate count mismatch");
  TORCH_CHECK(retention.is_contiguous() && write_key.is_contiguous()
      && erase_key.is_contiguous() && write_value.is_contiguous()
      && initial_state.is_contiguous() && query.is_contiguous()
      && event_coordinates.is_contiguous(), "recurrence tensors must be contiguous");
  const int work = std::max(
      std::max(heads * VALUE_DIM, rank * VALUE_DIM), rank * heads);
  const int threads = next_power_of_two(work);
  TORCH_CHECK(threads <= 1024, "recurrence shape requires more than 1024 threads");
  const c10::cuda::CUDAGuard guard(retention.device());
  auto reads = torch::empty({batch, length, VALUE_DIM}, retention.options());
  auto states = torch::empty({batch, length, heads, VALUE_DIM}, retention.options());
  primitive_delta_forward_kernel<<<batch, threads, threads * sizeof(float),
      at::cuda::getCurrentCUDAStream()>>>(
          retention.data_ptr<float>(), write_key.data_ptr<float>(),
          erase_key.data_ptr<float>(), write_value.data_ptr<float>(),
          initial_state.data_ptr<float>(), query.data_ptr<float>(),
          event_coordinates.data_ptr<float>(), permutations.data_ptr<int>(),
          kinds.data_ptr<int>(), local_generators.data_ptr<float>(),
          local_generator_squares.data_ptr<float>(), frequencies.data_ptr<float>(),
          eigenvectors.data_ptr<float>(), eigenvalues.data_ptr<float>(),
          reads.data_ptr<float>(), states.data_ptr<float>(), length, heads, rank,
          factors, events, event_stride, first_event_local, transport_enabled);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto final_state = states.select(1, length - 1).contiguous();
  return {reads, final_state, states};
}

std::vector<torch::Tensor> primitive_delta_backward_cuda(
    torch::Tensor retention,
    torch::Tensor write_key,
    torch::Tensor erase_key,
    torch::Tensor write_value,
    torch::Tensor initial_state,
    torch::Tensor query,
    torch::Tensor event_coordinates,
    torch::Tensor states,
    torch::Tensor read_gradient,
    torch::Tensor final_gradient,
    torch::Tensor permutations,
    torch::Tensor kinds,
    torch::Tensor local_generators,
    torch::Tensor local_generator_squares,
    torch::Tensor frequencies,
    torch::Tensor eigenvectors,
    torch::Tensor eigenvalues,
    int64_t event_stride,
    int64_t first_event_local,
    bool transport_enabled) {
  const int batch = retention.size(0);
  const int length = retention.size(1);
  const int heads = retention.size(2);
  const int rank = write_key.size(2);
  const int events = event_coordinates.size(1);
  const int factors = event_coordinates.size(2);
  TORCH_CHECK(states.dim() == 4 && states.size(0) == batch
      && states.size(1) == length && states.size(2) == heads
      && states.size(3) == VALUE_DIM, "saved state shape mismatch");
  TORCH_CHECK(read_gradient.dim() == 3 && read_gradient.size(0) == batch
      && read_gradient.size(1) == length
      && read_gradient.size(2) == VALUE_DIM, "read gradient shape mismatch");
  TORCH_CHECK(final_gradient.sizes() == initial_state.sizes(),
      "final gradient shape mismatch");
  const int work = std::max(
      std::max(heads * VALUE_DIM, rank * VALUE_DIM), rank * heads);
  const int threads = next_power_of_two(work);
  const c10::cuda::CUDAGuard guard(retention.device());
  auto retention_gradient = torch::empty_like(retention);
  auto write_key_gradient = torch::empty_like(write_key);
  auto erase_key_gradient = torch::empty_like(erase_key);
  auto write_value_gradient = torch::empty_like(write_value);
  auto initial_gradient = torch::empty_like(initial_state);
  auto query_gradient = torch::empty_like(query);
  auto coordinate_gradient = transport_enabled
      ? torch::empty_like(event_coordinates)
      : torch::zeros_like(event_coordinates);
  primitive_delta_backward_kernel<<<batch, threads, 3 * threads * sizeof(float),
      at::cuda::getCurrentCUDAStream()>>>(
          retention.data_ptr<float>(), write_key.data_ptr<float>(),
          erase_key.data_ptr<float>(), write_value.data_ptr<float>(),
          initial_state.data_ptr<float>(), query.data_ptr<float>(),
          event_coordinates.data_ptr<float>(), states.data_ptr<float>(),
          read_gradient.data_ptr<float>(), final_gradient.data_ptr<float>(),
          permutations.data_ptr<int>(), kinds.data_ptr<int>(),
          local_generators.data_ptr<float>(), local_generator_squares.data_ptr<float>(),
          frequencies.data_ptr<float>(), eigenvectors.data_ptr<float>(),
          eigenvalues.data_ptr<float>(), retention_gradient.data_ptr<float>(),
          write_key_gradient.data_ptr<float>(), erase_key_gradient.data_ptr<float>(),
          write_value_gradient.data_ptr<float>(), initial_gradient.data_ptr<float>(),
          query_gradient.data_ptr<float>(), coordinate_gradient.data_ptr<float>(),
          length, heads, rank, factors, events, event_stride, first_event_local,
          transport_enabled);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {retention_gradient, write_key_gradient, erase_key_gradient,
      write_value_gradient, initial_gradient, query_gradient, coordinate_gradient};
}
