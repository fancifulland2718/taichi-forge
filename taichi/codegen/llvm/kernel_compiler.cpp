#include "taichi/codegen/codegen.h"
#include "taichi/analysis/gather_snode_tree_dependencies.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"

#include "taichi/codegen/llvm/kernel_compiler.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"

namespace taichi::lang {
namespace LLVM {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const Kernel &kernel_def,
    GraphKernelMetadata *graph_metadata) const {
  auto ir = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.llvm.clone_ir");
    return irpass::analysis::clone(kernel_def.ir.get());
  }();
  bool verbose = compile_config.print_ir;
  if (kernel_def.is_accessor && !compile_config.print_accessor_ir) {
    verbose = false;
  }
  {
    TI_COMPILE_PROFILER("cpp.compile.llvm.compile_to_offloads");
    irpass::compile_to_offloads(ir.get(), compile_config, &kernel_def,
                                /*verbose=*/verbose,
                                /*autodiff_mode=*/kernel_def.autodiff_mode,
                                /*ad_use_stack=*/true,
                                /*start_from_ast=*/kernel_def.ir_is_ast(),
                                graph_metadata);
  }
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  LLVM::CompiledKernelData::InternalData data;
  auto codegen = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.llvm.codegen_create");
    return KernelCodeGen::create(compile_config, device_caps, &kernel_def,
                                 &chi_ir, *config_.tlctx);
  }();
  data.compiled_data = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.llvm.emit_module");
    return codegen->compile_kernel_to_module();
  }();
  data.used_snode_tree_ids =
      irpass::analysis::gather_snode_tree_dependencies(chi_ir);
  data.args.reserve(kernel_def.nested_parameters.size());
  for (const auto &p : kernel_def.nested_parameters)
    data.args.push_back(p);
  data.rets = kernel_def.rets;
  data.args_type = kernel_def.args_type;
  data.args_size = kernel_def.args_size;
  data.ret_type = kernel_def.ret_type;
  data.ret_size = kernel_def.ret_size;
  return std::make_unique<LLVM::CompiledKernelData>(compile_config.arch, data);
}

}  // namespace LLVM
}  // namespace taichi::lang
