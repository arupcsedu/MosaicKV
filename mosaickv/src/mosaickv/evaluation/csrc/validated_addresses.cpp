#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <cstdint>

namespace py = pybind11;

py::tuple validated_addresses(py::tuple args, int device) {
  if (args.empty()) throw py::value_error("complete live tensor arguments required");
  py::tuple result(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    PyObject* object = args[i].ptr();
    if (!THPVariable_Check(object))
      throw py::value_error("complete live tensor arguments required");
    const auto& tensor = THPVariable_Unpack(object);
    const auto actual = tensor.device();
    const bool correct_device = device < 0 ? actual.is_cpu()
        : actual.is_cuda() && actual.index() == device;
    if (!correct_device || tensor.requires_grad())
      throw py::value_error("private nongrad buffers on one device required");
    const auto address = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
    if (!address && tensor.numel())
      throw py::value_error("NULL only for masked empty buffers");
    result[i] = py::int_(static_cast<uint64_t>(address));
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("addresses", &validated_addresses,
             "Validate every live tensor's device, gradient and pointer metadata");
}
