#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

std::size_t cuda_scan_capture_workspace_bytes(int num_items, int value_type);
std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_scan_capture_command(
    Program *program,
    const aot::Arg &values,
    const aot::Arg &workspace,
    int num_items,
    int value_type);

}  // namespace taichi::lang
