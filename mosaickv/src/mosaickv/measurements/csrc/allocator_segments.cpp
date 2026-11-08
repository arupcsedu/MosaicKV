#include <pybind11/pybind11.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace py = pybind11;

// Same allocator snapshot and fields as torch.cuda.memory_snapshot(), without
// converting/symbolizing the unrelated, potentially huge device trace history.
py::dict active_blocks() {
  auto snapshot = [] {
    py::gil_scoped_release release;
    return c10::cuda::CUDACachingAllocator::snapshot();
  }();
  py::dict result;
  for (const auto& segment : snapshot.segments) {
    size_t address = segment.address;
    for (const auto& block : segment.blocks) {
      if (block.allocated) {
        result[py::int_(address)] = py::make_tuple(block.requested_size, block.size);
      }
      address += block.size;
    }
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("active_blocks", &active_blocks);
}
