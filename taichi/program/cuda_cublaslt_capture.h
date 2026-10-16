#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

// Private bridge for an already prepared, host-scalar f32 cuBLASLt plan.
// The Python recording retains the common adapter plan and workspace lease.
// This adds no CUDA Toolkit header/link dependency to the portable runtime.
struct CudaCublasLtCapturePlan {
  std::uint64_t execute_address{0};
  std::uint64_t error_address{0};
  std::uint64_t plan_handle{0};
  std::vector<std::vector<int>> shapes;
  std::size_t workspace_bytes{0};
};

std::shared_ptr<aot::CudaGraphCaptureCommand>
make_cuda_cublaslt_capture_command(Program *program,
                                   const CudaCublasLtCapturePlan &plan,
                                   const std::vector<aot::Arg> &arguments);

}  // namespace taichi::lang
