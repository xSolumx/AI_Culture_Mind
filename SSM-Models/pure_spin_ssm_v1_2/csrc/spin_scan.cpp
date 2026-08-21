#include <torch/extension.h>

torch::Tensor spin8_scan_cuda(torch::Tensor action, torch::Tensor scale, torch::Tensor drive, torch::Tensor initial);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &spin8_scan_cuda, "Spin(8) recurrent scan forward (CUDA)");
}
