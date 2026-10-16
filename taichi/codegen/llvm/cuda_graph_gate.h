#pragma once

namespace taichi::lang {
struct LLVMCompiledKernel;

namespace LLVM {
// Clone already-lowered code and uniformly guard every offloaded entry with
// GraphExecutionGateBinding. Ordinary code, metadata and its context owner stay
// intact. The returned variant requires a gated argument prefix at launch;
// callers must retain that contract when exporting or registering the result.
// This producer transform needs no Program, launcher or active CUDA device.
LLVMCompiledKernel make_cuda_graph_gated_kernel(
    const LLVMCompiledKernel &source);
}  // namespace LLVM
}  // namespace taichi::lang
