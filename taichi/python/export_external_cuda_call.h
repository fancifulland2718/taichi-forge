#pragma once
#include "taichi/python/export.h"

namespace taichi {
void export_external_cuda_call(py::module &m);
}  // namespace taichi
