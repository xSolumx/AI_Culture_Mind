#include <torch/extension.h>

torch::Tensor primitive_action_forward_cuda(
    torch::Tensor values,
    torch::Tensor coordinates,
    torch::Tensor permutations,
    torch::Tensor kinds,
    torch::Tensor local_generators,
    torch::Tensor local_generator_squares,
    torch::Tensor frequencies,
    torch::Tensor eigenvectors,
    torch::Tensor eigenvalues);

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
    torch::Tensor eigenvalues);

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
    bool transport_enabled);

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
    bool transport_enabled);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &primitive_action_forward_cuda,
      "Canonical primitive exceptional action forward (SM75 CUDA)");
  module.def("backward", &primitive_action_backward_cuda,
      "Canonical primitive exceptional action backward (SM75 CUDA)");
  module.def("delta_forward", &primitive_delta_forward_cuda,
      "Sparse-event primitive Delta recurrence forward (SM75 CUDA)");
  module.def("delta_backward", &primitive_delta_backward_cuda,
      "Sparse-event primitive Delta recurrence backward (SM75 CUDA)");
}
