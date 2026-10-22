#include "taichi/codegen/codegen.h"
#include "taichi/analysis/gather_snode_tree_dependencies.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"

#include "taichi/codegen/llvm/kernel_compiler.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/llvm_module_options.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"

namespace taichi::lang {
namespace LLVM {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const KernelCompileRequest &request,
    const Kernel &kernel_def,
    GraphKernelMetadata *graph_metadata) const {
  const auto &compile_config = request.config();
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
    const KernelCompileRequest &request,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  const auto &compile_config = request.config();
  const auto &device_caps = request.device_caps();
  LLVM::CompiledKernelData::InternalData data;
  // Preserve runtime-only dependencies (e.g. listgen/GC), then merge the roots
  // introduced by final task lowering without another IR traversal.
  auto dependencies =
      irpass::analysis::gather_kernel_runtime_dependencies(chi_ir);
  data.used_snode_tree_ids = std::move(dependencies.tree_ids);
  auto codegen = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.llvm.codegen_create");
    return KernelCodeGen::create(compile_config, device_caps, &kernel_def,
                                 &chi_ir, *config_.tlctx);
  }();
  data.compiled_data = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.llvm.emit_module");
    return codegen->compile_kernel_to_module(data.used_snode_tree_ids);
  }();
  LLVMModuleOptions::from_config(compile_config)
      .write(*data.compiled_data.module);
  declare_runtime_context_abi(*data.compiled_data.module);
  // Dense struct-for may become an ordinary range during final task lowering.
  // Consult emitted task records, not its pre-lowering statement kind. This
  // also covers explicit dense-list mode without another instruction walk.
  for (const auto &task : data.compiled_data.tasks)
    if (task.task_type == OffloadedTaskType::struct_for ||
        task.task_type == OffloadedTaskType::listgen ||
        task.task_type == OffloadedTaskType::gc || task.sparse_list_op ||
        task.may_mutate_sparse_topology)
      dependencies.runtime_state |= runtime_state_sparse;
  declare_runtime_state_use(
      *data.compiled_data.module,
      dependencies.runtime_state |
          (compile_config.debug ? runtime_state_error : 0u));
  if (kernel_def.get_offload_execution_plan().has_value()) {
    // Entry-specific maxnreg annotations own task-plan caps. Do not add a
    // module-wide CU_JIT_MAX_REGISTERS limit that would collapse them.
    data.compiled_data.cuda_max_registers = 0;
  } else if (const auto &spec = kernel_def.get_kernel_optimization_spec();
             spec.has_value()) {
    data.compiled_data.cuda_max_registers = spec->cuda_max_registers;
  }
  data.may_trigger_hash_overflow = dependencies.may_trigger_hash_overflow;
  data.args.reserve(kernel_def.nested_parameters.size());
  for (const auto &p : kernel_def.nested_parameters)
    data.args.push_back(p);
  data.rets = kernel_def.rets;
  data.args_type = kernel_def.args_type;
  data.args_size = kernel_def.args_size;
  data.ret_type = kernel_def.ret_type;
  data.ret_size = kernel_def.ret_size;
  auto result =
      std::make_unique<LLVM::CompiledKernelData>(compile_config.arch, data);
  result->initialize_generation_bound_snode_relocation_descriptor(
      true, dependencies.structures);
  return result;
}

}  // namespace LLVM
}  // namespace taichi::lang
