#pragma once

#include "taichi/aot/graph_data.h"

namespace taichi::lang {

// Internal C-ABI transport for a prepared source addon. The callback consumes
// an immutable argument block and returns a uint32 status. All device-pointer
// fields and the stream field are relocated only when recording a Graph.
// Python's recording owns the library lease; no Python callback runs here.
std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_addon_capture_command(
    Program *program,
    std::uint64_t invoke_address,
    const std::string &payload,
    std::size_t stream_offset,
    const std::vector<aot::Arg> &arguments,
    const std::vector<std::size_t> &pointer_offsets,
    const std::vector<std::size_t> &scalar_counts,
    const std::vector<bool> &writable,
    std::uint64_t error_address);

}  // namespace taichi::lang
