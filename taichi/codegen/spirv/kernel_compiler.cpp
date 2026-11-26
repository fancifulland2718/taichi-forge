#include "taichi/codegen/spirv/kernel_compiler.h"
#include <algorithm>
#include <unordered_set>
#include "taichi/analysis/gather_snode_tree_dependencies.h"
#include "taichi/system/profiler.h"

#include "taichi/ir/analysis.h"
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/statements.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/codegen/spirv/array_layout_abi.h"

namespace taichi::lang {
namespace spirv {

namespace {

// Offloading gives loop-carried scalar returns private byte ranges in the
// global temporary buffer. Inspect only statically large straight-line tasks.
// Every temporary access must be a single, unobserved f32/f64 add to its own
// aligned slot. Field atomics, including indexed scatters, keep their semantics.
void group_temporary_sums(OffloadedStmt *task) {
  if (task->task_type != OffloadedTaskType::range_for || !task->const_begin ||
      !task->const_end || int64_t(task->end_value) - task->begin_value < 32768 ||
      task->block_dim < 32 || !task->body)
    return;
  const auto &stmts = task->body->statements;
  if (std::none_of(stmts.begin(), stmts.end(), [](const auto &entry) {
        return entry->template is<GlobalTemporaryStmt>();
      }))
    return;
  std::vector<AtomicOpStmt *> sums;
  std::unordered_set<std::size_t> words;
  for (const auto &stmt : stmts) {
    if (stmt->is_container_statement() || stmt->is<FuncCallStmt>() ||
        stmt->is<ExternalFuncCallStmt>())
      return;
    auto *add = stmt->cast<AtomicOpStmt>();
    for (int i = 0; i < stmt->num_operands(); ++i) {
      auto *operand = stmt->operand(i);
      if (!operand)
        continue;
      if (auto *previous = operand->cast<AtomicOpStmt>();
          previous && previous->dest->is<GlobalTemporaryStmt>())
        return;
      auto *slot = operand->cast<GlobalTemporaryStmt>();
      if (!slot)
        continue;
      if (!add || add->op_type != AtomicOpType::add || add->dest != slot ||
          (add->ret_type != PrimitiveType::f32 &&
           add->ret_type != PrimitiveType::f64))
        return;
      const auto bytes = data_type_size(add->ret_type);
      if (slot->offset % bytes != 0)
        return;
      for (int b = 0; b < bytes; b += 4)
        if (!words.insert(slot->offset + b).second)
          return;
      sums.push_back(add);
    }
  }
  for (auto *add : sums) {
    auto *sum = add->insert_before_me(Stmt::make<InternalFuncStmt>(
        "subgroupAdd", std::vector<Stmt *>{add->val}, add->ret_type, false));
    auto *leader = add->insert_before_me(Stmt::make<InternalFuncStmt>(
        "subgroupElect", std::vector<Stmt *>{}, PrimitiveType::i32, false));
    auto branch = Stmt::make<IfStmt>(leader);
    auto body = std::make_unique<Block>();
    auto *write = body->push_back<AtomicOpStmt>(add->op_type, add->dest, sum);
    write->ret_type = add->ret_type;
    branch->as<IfStmt>()->set_true_statements(std::move(body));
    add->replace_with(std::move(branch), /*replace_usages=*/false);
  }
}

}  // namespace

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const KernelCompileRequest &request,
    const Kernel &kernel_def,
    GraphKernelMetadata *graph_metadata) const {
  const auto &compile_config = request.config();
  auto ir = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.spirv.clone_ir");
    return irpass::analysis::clone(kernel_def.ir.get());
  }();
  {
    TI_COMPILE_PROFILER("cpp.compile.spirv.compile_to_executable");
    irpass::compile_to_executable(ir.get(), compile_config, &kernel_def,
                                  kernel_def.autodiff_mode,
                                  /*ad_use_stack=*/false,
                                  compile_config.print_ir,
                                  /*lower_global_access=*/true,
                                  /*make_thread_local=*/false,
                                  /*make_block_local=*/false,
                                  /*start_from_ast=*/kernel_def.ir_is_ast(),
                                  graph_metadata);
  }
  // A constant negative array index is invalid in logical SPIR-V even if its
  // access is in a dead branch. Fast compilation skips the optional CFG passes;
  // legalize this case using the existing indexed reaching-definition analysis.
  // Other kernels retain their fast schedule. Live invalid accesses remain
  // invalid; do not clamp them or relax validation to conceal the problem.
  auto has_negative_array_index = [&]() {
    return irpass::analysis::any_statement(ir.get(), [](Stmt *stmt) {
      auto *ptr = stmt->cast<MatrixPtrStmt>();
      if (!ptr || !ptr->origin->is<AllocaStmt>())
        return false;
      auto *offset = ptr->offset->cast<ConstStmt>();
      return offset && is_signed(offset->ret_type) && offset->val.val_int() < 0;
    });
  };
  if (!compile_config.advanced_optimization && has_negative_array_index()) {
    TI_COMPILE_PROFILER("cpp.compile.spirv.constant_dead_branches");
    do {
      auto cfg = irpass::analysis::build_cfg(ir.get());
      cfg->simplify_graph();
      bool changed = cfg->forward_constant_local_loads(ir.get());
      changed = irpass::constant_fold(ir.get()) || changed;
      changed = irpass::unreachable_code_elimination(ir.get()) || changed;
      irpass::die(ir.get());
      if (!changed)
        break;
    } while (has_negative_array_index());
  }
  const auto &caps = request.device_caps();
  if (compile_config.arch == Arch::vulkan && compile_config.fast_math &&
      compile_config.opt_level > 0 && !compile_config.debug &&
      caps.get(DeviceCapability::spirv_version) >= 0x10300 &&
      caps.get(DeviceCapability::spirv_has_subgroup_basic) &&
      caps.get(DeviceCapability::spirv_has_subgroup_arithmetic)) {
    if (auto *block = ir->cast<Block>()) {
      for (const auto &stmt : block->statements)
        if (auto *task = stmt->cast<OffloadedStmt>())
          group_temporary_sums(task);
    } else if (auto *task = ir->cast<OffloadedStmt>()) {
      group_temporary_sums(task);
    }
  }
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(
    const KernelCompileRequest &request,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  const auto &compile_config = request.config();
  const auto &device_caps = request.device_caps();
  TI_TRACE("VK codegen for Taichi kernel={}", kernel_def.name);
  KernelCodegen::Params params;
  params.ti_kernel_name = kernel_def.name;
  params.kernel = &kernel_def;
  params.ir_root = &chi_ir;
  params.compiled_structs = *config_.compiled_struct_data;
  params.arch = compile_config.arch;
  params.caps = device_caps;
  params.omit_value_names = compile_config.spirv_omit_value_names;
  // P1.d + V1: compile_tier-aware SPIR-V opt level.
  //   "fast"     — skip the spvtools optimizer entirely (level 0, 0 passes).
  //                V1 revision: previously capped at level 1 (3 cheap passes),
  //                but DCE/DeadBranchElim are cheap and their savings do not
  //                materialise over pass registration + Run() overhead on
  //                SPV-bound kernels. Going straight to 0 gives the largest
  //                compile-time win for the "fast" tier. Correctness cleanup
  //                and driver-safe private-assignment selection belong to
  //                lowering/codegen, independently of optional optimization.
  //                Validator acceptance alone does not qualify driver execution.
  //   "balanced" — no change vs legacy (default external_optimization_level=3,
  //                23 passes). Preserves default runtime perf.
  //   "full"     — no change vs legacy (23 passes).
  // Different tiers produce different artifacts and are segregated by the
  // P2.c offline-cache key (compile_tier is already serialized into the
  // hash; see taichi/analysis/offline_cache_util.cpp).
  int spv_level = compile_config.external_optimization_level;
  if (compile_config.compile_tier == "fast") {
    spv_level = 0;
  }
  params.enable_spv_opt = spv_level > 0;
  params.spv_opt_level = spv_level;
  // V2: opt-in parallel per-task SPIR-V codegen + spvtools::Optimizer::Run.
  // Default false in CompileConfig so behaviour is unchanged for users
  // who do not set ti.init(spirv_parallel_codegen=True).
  params.parallel_codegen = compile_config.spirv_parallel_codegen;
  params.num_compile_threads = std::max(1, compile_config.num_compile_threads);
  // V6: opt-in skip of CreateLoopUnrollPass at level 3.
  params.skip_loop_unroll = compile_config.spirv_skip_loop_unroll;
  // B2: user-facing fine-grained pass disable list. Empty by default.
  params.disabled_passes = compile_config.spirv_disabled_passes;
  // G-6: opt-in task-level optimizer skip for listgen / clear-list / small
  // SPIR-V tasks. Default false preserves the global pass chain.
  params.spirv_adaptive_opt = compile_config.spirv_adaptive_opt;
  params.spirv_adaptive_opt_threshold =
      compile_config.spirv_adaptive_opt_threshold;
  // V8.b: anti-double-pool — when the surrounding compile_kernels outer
  // pool is active, suppress this inner per-task fan-out.
  params.compile_dag_scheduler = compile_config.compile_dag_scheduler;
  // G11-A (2026-05): bitmasked deactivate clears data slot opt-in.
  params.bitmasked_clear_data_on_deactivate =
      compile_config.bitmasked_clear_data_on_deactivate;
  params.listgen_subgroup_ballot =
      compile_config.spirv_listgen_subgroup_ballot;
  params.vulkan_listgen_reuse = compile_config.vulkan_listgen_reuse;
  params.vulkan_spv_stats = compile_config.vulkan_spv_stats;
  params.vulkan_spv_stats_filter = compile_config.vulkan_spv_stats_filter;
  params.vulkan_spv_stats_capacity = compile_config.vulkan_spv_stats_capacity;
  params.vulkan_spv_stats_to_stderr =
      compile_config.vulkan_spv_stats_to_stderr;
  spirv::CompiledKernelData::InternalData internal_data;
  {
    TI_COMPILE_PROFILER("cpp.compile.spirv.codegen_run");
    spirv::KernelCodegen codegen(std::move(params));
    codegen.run(internal_data.metadata.kernel_attribs,
                internal_data.src.spirv_src);
  }
  internal_data.metadata.num_snode_trees = config_.compiled_struct_data->size();
  const auto dependencies = irpass::analysis::gather_kernel_runtime_dependencies(
      chi_ir, /*spirv_function_abi=*/true);
  internal_data.metadata.used_snode_tree_ids = dependencies.tree_ids;
  internal_data.metadata.kernel_attribs.runtime_affine_arrays = dependencies.runtime_affine_arrays;
  for (auto &code : internal_data.src.spirv_src)
    declare_runtime_affine_arrays(code, dependencies.runtime_affine_arrays);
  auto result = std::make_unique<spirv::CompiledKernelData>(
      compile_config.arch, std::move(internal_data));
  result->initialize_generation_bound_snode_relocation_descriptor(
      true, dependencies.structures);
  return result;
}

}  // namespace spirv
}  // namespace taichi::lang
