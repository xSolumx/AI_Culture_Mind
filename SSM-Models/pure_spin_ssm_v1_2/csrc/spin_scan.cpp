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
torch::Tensor isotypic_coordinate_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial);
std::vector<torch::Tensor> isotypic_coordinate_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor scale,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient);
torch::Tensor coupled_coordinate_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial);
std::vector<torch::Tensor> coupled_coordinate_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial, torch::Tensor output,
    torch::Tensor output_gradient);
torch::Tensor independent_block_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial);
torch::Tensor independent_block_isotypic_forward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
    torch::Tensor drive, torch::Tensor initial);
std::vector<torch::Tensor> independent_block_backward_cuda(
    torch::Tensor coordinates, torch::Tensor generators, torch::Tensor left,
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
  module.def("isotypic_coordinate_forward", &isotypic_coordinate_forward_cuda,
      "Isotypic-split Spin(8) coordinate recurrence forward (CUDA)");
  module.def("isotypic_coordinate_backward", &isotypic_coordinate_backward_cuda,
      "Isotypic-split Spin(8) coordinate recurrence backward (CUDA)");
  module.def("coupled_coordinate_forward", &coupled_coordinate_forward_cuda,
      "Coupled-isotypic Spin(8) coordinate recurrence forward (CUDA)");
  module.def("coupled_coordinate_backward", &coupled_coordinate_backward_cuda,
      "Coupled-isotypic Spin(8) coordinate recurrence backward (CUDA)");
  module.def("independent_block_forward", &independent_block_forward_cuda,
      "Independent-action block-coupled Spin(8) recurrence forward (CUDA)");
  module.def("independent_block_isotypic_forward",
      &independent_block_isotypic_forward_cuda,
      "Isotypic-split independent-action block recurrence forward (CUDA)");
  module.def("independent_block_backward", &independent_block_backward_cuda,
      "Independent-action block-coupled Spin(8) recurrence backward (CUDA)");
}
