#include <torch/extension.h>

torch::Tensor spin8_scan_cuda(torch::Tensor action, torch::Tensor scale, torch::Tensor drive, torch::Tensor initial);
torch::Tensor controller_factorized_forward_cuda(
    torch::Tensor features, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor generators, torch::Tensor scale, torch::Tensor drive,
    torch::Tensor initial, torch::Tensor gate);
std::vector<torch::Tensor> controller_factorized_backward_cuda(
    torch::Tensor features, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor generators, torch::Tensor scale, torch::Tensor drive,
    torch::Tensor initial,
    torch::Tensor gate, torch::Tensor output, torch::Tensor output_gradient);
torch::Tensor coordinate_factorized_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial);
std::vector<torch::Tensor> coordinate_factorized_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &spin8_scan_cuda, "Spin(8) recurrent scan forward (CUDA)");
  module.def("controller_forward", &controller_factorized_forward_cuda,
      "Fused Spin(8) controller-factor recurrence forward (CUDA)");
  module.def("controller_backward", &controller_factorized_backward_cuda,
      "Fused Spin(8) controller-factor recurrence backward (CUDA)");
  module.def("coordinate_forward", &coordinate_factorized_forward_cuda,
      "Spin(8) coordinate-factor recurrence forward (CUDA)");
  module.def("coordinate_backward", &coordinate_factorized_backward_cuda,
      "Spin(8) coordinate-factor recurrence backward (CUDA)");
}
