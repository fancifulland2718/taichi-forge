#pragma once

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/kernel_compile_request.h"
#include "taichi/codegen/compiled_kernel_data.h"

namespace taichi::lang {

class KernelCompiler {
 public:
  using IRNodePtr = std::unique_ptr<IRNode>;
  using CKDPtr = std::unique_ptr<CompiledKernelData>;

  // AST -> CHI IR
  virtual IRNodePtr compile(const KernelCompileRequest &request,
                            const Kernel &kernel_def,
                            GraphKernelMetadata *graph_metadata) const = 0;

  // CHI IR -> CompiledKernelData
  virtual CKDPtr compile(const KernelCompileRequest &request,
                         const Kernel &kernel_def,
                         IRNode &chi_ir) const = 0;

  virtual ~KernelCompiler() = default;
};

}  // namespace taichi::lang
