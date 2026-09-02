#include "taichi/ir/ir.h"
#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/pass.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/transforms/make_external_shared_staged.h"
#include "taichi/util/lang_util.h"

// Keep the private M0 recipe in one reviewable unit while compiling it from
// the established split-runtime link-closure root.
#include "taichi/transforms/make_external_shared_staged.inc"

#include <string>
#include <vector>

namespace taichi::lang {

namespace irpass {

namespace {

struct TaskLaunchPolicyApplication {
  bool active{false};
  bool changed_block_dim{false};
};

TaskLaunchPolicyApplication apply_task_launch_policy(
    IRNode *ir,
    const CompileConfig &config,
    const Kernel *kernel,
    bool start_from_ast) {
  const auto &policy = kernel->get_task_launch_policy();
  if (!policy.has_value()) {
    return {};
  }

  TI_ERROR_IF(!start_from_ast,
              "TaskLaunchPolicy is supported only for direct JIT kernels");
  TI_ERROR_IF(config.arch != Arch::cuda && config.arch != Arch::vulkan,
              "TaskLaunchPolicy block control is unavailable on backend {}",
              arch_name(config.arch));
  TI_ERROR_IF(!ir->is<Block>(),
              "TaskLaunchPolicy expected a frontend kernel block");

  std::vector<FrontendForStmt *> parallel_loops;
  for (const auto &stmt : ir->as<Block>()->statements) {
    if (auto *loop = stmt->cast<FrontendForStmt>();
        loop != nullptr && !loop->strictly_serialized) {
      parallel_loops.push_back(loop);
    }
  }
  TI_ERROR_IF(
      parallel_loops.size() != 1,
      "TaskLaunchPolicy requires exactly one top-level parallel range-for; "
      "found {} parallel loops",
      parallel_loops.size());

  auto *loop = parallel_loops.front();
  TI_ERROR_IF(
      loop->snode != nullptr || loop->external_tensor || loop->mesh != nullptr,
      "TaskLaunchPolicy currently supports only a range-for task; "
      "struct-for, ndarray iteration, and mesh-for remain read-only");

  TaskLaunchPolicyApplication result;
  result.active = true;
  if (policy->injected_block_dim) {
    loop->block_dim = policy->block_dim;
    result.changed_block_dim = true;
  } else if (loop->block_dim == policy->block_dim) {
    result.changed_block_dim = false;
  } else if (policy->mode == Kernel::TaskLaunchPolicyMode::require) {
    TI_ERROR("TaskLaunchPolicy require(block_dim={}) conflicts with the "
             "kernel's explicit ti.loop_config(block_dim={})",
             policy->block_dim, loop->block_dim);
  }
  // A hint never overrides an explicit source-level loop_config. This keeps
  // SharedArray indexing and block-collective assumptions owned by the kernel.
  return result;
}

class BlockSensitiveOperationFinder final : public BasicStmtVisitor {
 public:
  BlockSensitiveOperationFinder() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(AllocaStmt *stmt) override {
    if (stmt->is_shared && reason_.empty()) {
      reason_ = "block-local SharedArray storage";
    }
  }

  void visit(InternalFuncStmt *stmt) override {
    const auto &name = stmt->func_name;
    const bool sensitive =
        name == "linear_thread_idx" || name == "grid_memfence" ||
        name.rfind("block_barrier", 0) == 0 ||
        name.rfind("workgroup", 0) == 0 ||
        name.rfind("localInvocation", 0) == 0 ||
        name.rfind("globalInvocation", 0) == 0 || name.rfind("cuda_", 0) == 0 ||
        name.rfind("warp_", 0) == 0 || name.rfind("subgroup", 0) == 0;
    if (sensitive && reason_.empty()) {
      reason_ = "block-sensitive intrinsic " + name;
    }
  }

  static std::string run(IRNode *ir) {
    BlockSensitiveOperationFinder finder;
    ir->accept(&finder);
    return finder.reason_;
  }

 private:
  std::string reason_;
};

class SerialPreambleSafetyChecker final : public BasicStmtVisitor {
 public:
  SerialPreambleSafetyChecker() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(GlobalStoreStmt *stmt) override {
    if (!stmt->dest->is<GlobalTemporaryStmt>()) {
      safe_ = false;
    }
  }

  void visit(Stmt *stmt) override {
    if (stmt->has_global_side_effect()) {
      safe_ = false;
    }
  }

  static bool run(Block *body) {
    SerialPreambleSafetyChecker checker;
    body->accept(&checker);
    return checker.safe_;
  }

 private:
  bool safe_{true};
};

Stmt *offload_phase_pointer_origin(Stmt *stmt) {
  while (auto *matrix = stmt ? stmt->cast<MatrixPtrStmt>() : nullptr) {
    stmt = matrix->origin;
  }
  return stmt;
}

bool offload_phase_thread_private_pointer(Stmt *stmt) {
  auto *origin = offload_phase_pointer_origin(stmt);
  auto *allocation = origin ? origin->cast<AllocaStmt>() : nullptr;
  return allocation != nullptr && !allocation->is_shared;
}

class ExactPointwiseOffloadChecker final : public BasicStmtVisitor {
 public:
  explicit ExactPointwiseOffloadChecker(OffloadedStmt *task) : task_(task) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(GlobalLoadStmt *stmt) override {
    check_pointer(stmt->src);
  }

  void visit(GlobalStoreStmt *stmt) override {
    check_pointer(stmt->dest);
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!offload_phase_thread_private_pointer(stmt->dest)) {
      reject("atomic or cross-thread read-modify-write effect");
    }
  }

  void visit(Stmt *stmt) override {
    if (stmt->has_global_side_effect()) {
      reject("unsupported global side effect");
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (!stmt->is<IfStmt>()) {
      reject("nested control flow");
    }
  }

  bool qualified() const {
    return reason_.empty();
  }

  const std::string &reason() const {
    return reason_;
  }

 private:
  bool exact_leading_index(const std::vector<Stmt *> &indices, int ndim) const {
    if (ndim < 1 || indices.size() < static_cast<std::size_t>(ndim)) {
      return false;
    }
    auto *index = indices.front()->cast<LoopIndexStmt>();
    return index != nullptr && index->loop == task_ && index->index == 0;
  }

  void check_pointer(Stmt *pointer) {
    pointer = offload_phase_pointer_origin(pointer);
    if (auto *external = pointer ? pointer->cast<ExternalPtrStmt>() : nullptr) {
      if (!exact_leading_index(external->indices, external->ndim)) {
        reject("non-pointwise external access");
      }
      return;
    }
    if (auto *global = pointer ? pointer->cast<GlobalPtrStmt>() : nullptr) {
      auto *index = global->indices.size() == 1
                        ? global->indices.front()->cast<LoopIndexStmt>()
                        : nullptr;
      if (global->snode == nullptr || !global->snode->is_path_all_dense ||
          (global->activate && !global->snode->is_path_all_dense) ||
          index == nullptr || index->loop != task_ || index->index != 0) {
        reject("non-pointwise or sparse field access");
      }
      return;
    }
    if (!offload_phase_thread_private_pointer(pointer)) {
      reject("unsupported pointer origin");
    }
  }

  void reject(const std::string &reason) {
    if (reason_.empty()) {
      reason_ = reason;
    }
  }

  OffloadedStmt *task_{nullptr};
  std::string reason_;
};

class OffloadLoopIndexRebaser final : public BasicStmtVisitor {
 public:
  OffloadLoopIndexRebaser(OffloadedStmt *source, OffloadedStmt *destination)
      : source_(source), destination_(destination) {
    allow_undefined_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->loop == source_) {
      stmt->loop = destination_;
    }
  }

 private:
  OffloadedStmt *source_{nullptr};
  OffloadedStmt *destination_{nullptr};
};

bool empty_offload_auxiliary_blocks(const OffloadedStmt *task) {
  auto empty = [](const std::unique_ptr<Block> &block) {
    return block == nullptr || block->statements.empty();
  };
  return empty(task->tls_prologue) && empty(task->tls_epilogue) &&
         empty(task->bls_prologue) && empty(task->bls_epilogue) &&
         empty(task->mesh_prologue);
}

std::string offload_phase_fusion_blocker(
    const std::vector<OffloadedStmt *> &tasks,
    const std::vector<int> &group) {
  const auto *first = tasks[group.front()];
  for (const int index : group) {
    const auto *task = tasks[index];
    if (task->task_type != OffloadedStmt::TaskType::range_for) {
      return "source task is not range_for";
    }
    if (!task->const_begin || !task->const_end || task->end_stmt != nullptr ||
        task->reversed || task->is_bit_vectorized || task->one_to_one ||
        task->external_shared_staged) {
      return "source task has an unsupported range execution mode";
    }
    if (task->begin_value != first->begin_value ||
        task->end_value != first->end_value ||
        task->block_dim != first->block_dim ||
        task->grid_dim != first->grid_dim ||
        !empty_offload_auxiliary_blocks(task)) {
      return "source tasks do not share one physical constant range";
    }
    const std::string sensitive =
        BlockSensitiveOperationFinder::run(const_cast<OffloadedStmt *>(task));
    if (!sensitive.empty()) {
      return sensitive;
    }
    ExactPointwiseOffloadChecker checker(const_cast<OffloadedStmt *>(task));
    task->body->accept(&checker);
    if (!checker.qualified()) {
      return checker.reason();
    }
  }
  return "";
}

void apply_exact_pointwise_offload_fusion(
    Block *root,
    const Kernel::OffloadExecutionPlan &plan,
    const std::vector<OffloadedStmt *> &source_tasks) {
  for (const auto &group : plan.fusion_groups) {
    const std::string blocker =
        offload_phase_fusion_blocker(source_tasks, group);
    TI_ERROR_IF(!blocker.empty(),
                "offload phase fusion rejected source tasks {}..{}: {}",
                group.front(), group.back(), blocker);
  }

  for (auto group_it = plan.fusion_groups.rbegin();
       group_it != plan.fusion_groups.rend(); ++group_it) {
    const auto &group = *group_it;
    auto *destination = source_tasks[group.front()];
    for (std::size_t member = 1; member < group.size(); ++member) {
      auto *source = source_tasks[group[member]];
      OffloadLoopIndexRebaser rebaser(source, destination);
      source->body->accept(&rebaser);
      while (!source->body->statements.empty()) {
        destination->body->insert(source->body->extract(0));
      }
      root->extract(source);
    }
  }
}

void validate_task_launch_policy_body(
    IRNode *ir,
    const TaskLaunchPolicyApplication &application) {
  if (!application.changed_block_dim) {
    return;
  }
  const std::string reason = BlockSensitiveOperationFinder::run(ir);
  TI_ERROR_IF(
      !reason.empty(),
      "TaskLaunchPolicy cannot change block_dim for a kernel containing {}; "
      "keep ti.loop_config(block_dim=...) in the kernel source instead",
      reason);
}

void validate_task_launch_policy_offloads(
    IRNode *ir,
    const TaskLaunchPolicyApplication &application) {
  if (!application.active) {
    return;
  }
  TI_ERROR_IF(!ir->is<Block>(),
              "TaskLaunchPolicy expected an offloaded kernel block");
  std::vector<OffloadedStmt *> tasks;
  for (const auto &stmt : ir->as<Block>()->statements) {
    if (auto *task = stmt->cast<OffloadedStmt>(); task != nullptr) {
      tasks.push_back(task);
    }
  }
  std::string task_types;
  std::size_t range_tasks = 0;
  bool safe_task_shape = true;
  for (const auto *task : tasks) {
    if (!task_types.empty()) {
      task_types += ",";
    }
    task_types +=
        fmt::format("{}[{}]", OffloadedStmt::task_type_name(task->task_type),
                    task->body->statements.size());
    if (task->task_type == OffloadedStmt::TaskType::range_for) {
      range_tasks += 1;
    } else if (task->task_type != OffloadedStmt::TaskType::serial ||
               !SerialPreambleSafetyChecker::run(task->body.get())) {
      safe_task_shape = false;
    }
  }
  TI_ERROR_IF(
      range_tasks != 1 || !safe_task_shape,
      "TaskLaunchPolicy requires one physical parallel range task plus only "
      "compiler-generated serial bound setup; the compiled kernel produced "
      "{} offloaded task(s): {}",
      tasks.size(), task_types);
}

void apply_offload_execution_plan(IRNode *ir,
                                  const CompileConfig &config,
                                  const Kernel *kernel,
                                  bool start_from_ast) {
  const auto &plan = kernel->get_offload_execution_plan();
  if (!plan.has_value()) {
    return;
  }
  TI_ERROR_IF(!start_from_ast,
              "offload execution plans support direct JIT kernels only");
  TI_ERROR_IF(config.arch != Arch::cuda,
              "offload execution plans require the CUDA backend, got {}",
              arch_name(config.arch));
  TI_ERROR_IF(!ir->is<Block>(),
              "offload execution plan expected an offloaded kernel block");

  std::vector<OffloadedStmt *> tasks;
  for (const auto &stmt : ir->as<Block>()->statements) {
    if (auto *task = stmt->cast<OffloadedStmt>(); task != nullptr) {
      tasks.push_back(task);
    }
  }
  TI_ERROR_IF(tasks.size() != plan->source_tasks.size(),
              "offload execution plan topology mismatch: expected {} task(s), "
              "lowering produced {}",
              plan->source_tasks.size(), tasks.size());
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    auto *task = tasks[index];
    const auto &spec = plan->source_tasks[index];
    const auto actual_kind = OffloadedStmt::task_type_name(task->task_type);
    TI_ERROR_IF(spec.task_index != index || spec.task_kind != actual_kind,
                "offload execution plan source topology mismatch at task {}: "
                "expected {}, got {}",
                index, spec.task_kind, actual_kind);
    if (spec.workgroup_size == 0 ||
        spec.workgroup_size == task->block_dim) {
      continue;
    }
    TI_ERROR_IF(task->task_type != OffloadedStmt::TaskType::range_for,
                "offload execution plan workgroup control applies only to "
                "range_for tasks");
    TI_ERROR_IF(task->source_block_dim_explicit,
                "offload execution plan cannot replace the source-owned "
                "block_dim={} contract on task {} with {}",
                task->block_dim, index, spec.workgroup_size);
    const std::string reason = BlockSensitiveOperationFinder::run(task);
    TI_ERROR_IF(!reason.empty(),
                "offload execution plan cannot change block_dim for task {} "
                "containing {}",
                index, reason);
    task->block_dim = spec.workgroup_size;
  }
  if (!plan->fusion_groups.empty()) {
    apply_exact_pointwise_offload_fusion(ir->as<Block>(), *plan, tasks);
  }
}

}  // namespace

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast,
                         GraphKernelMetadata *graph_metadata,
                         bool stop_before_offload) {
  TI_AUTO_PROF;
  TI_COMPILE_PROFILER("cpp.ir.compile_to_offloads");

  const auto task_launch_policy =
      apply_task_launch_policy(ir, config, kernel, start_from_ast);

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info,
                                 kernel->get_name(), ir);
  print("Initial IR");

  if (!verbose && config.print_preprocessed_ir && start_from_ast) {
    TI_INFO("[{}] {}:", kernel->get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  }

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  if (start_from_ast) {
    {
      TI_COMPILE_PROFILER("cpp.ir.frontend_type_check");
      irpass::frontend_type_check(ir);
    }
    {
      TI_COMPILE_PROFILER("cpp.ir.lower_ast");
      irpass::lower_ast(ir);
    }
    print("Lowered");
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.compile_functions");
    irpass::compile_taichi_functions(ir, config,
                                     Function::IRStage::BeforeLowerAccess);
    irpass::analysis::gather_func_store_dests(ir);
    irpass::compile_taichi_functions(ir, config,
                                     Function::IRStage::OptimizedIR);
    irpass::analysis::gather_func_store_dests(ir);
  }
  validate_task_launch_policy_body(ir, task_launch_policy);

  {
    TI_COMPILE_PROFILER("cpp.ir.validate_shared_array_scope");
    irpass::validate_shared_array_scope(ir);
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.eliminate_immutable_local_vars");
    irpass::eliminate_immutable_local_vars(ir);
  }
  print("Immutable local vars eliminated");

  {
    TI_COMPILE_PROFILER("cpp.ir.type_check.initial");
    irpass::type_check(ir, config);
  }
  print("Typechecked");
  irpass::analysis::verify(ir);

  // P9.A-3 (F3): IR-level reverse fallback for auto_real_function-promoted
  // @ti.func calls. Default budget=0 -> visitor short-circuits (no IR walk
  // overhead beyond a single accept()); positive budget enables selective
  // inline-back of small callees so frequent-call hot-paths don't pay a
  // perma-FuncCallStmt cost. LLVM-only by gate (FuncCallStmt visitor lives
  // only in codegen_llvm.cpp; non-LLVM auto_real_function is rejected at
  // F2's _can_auto_promote, so no FuncCallStmt should reach here for them).
  if (config.auto_real_function_inline_budget != 0 &&
      arch_uses_llvm(config.arch)) {
    TI_COMPILE_PROFILER("cpp.ir.auto_real_function_inline");
    InliningPass::Args inl_args;
    inl_args.budget = config.auto_real_function_inline_budget;
    if (irpass::inlining(ir, config, inl_args)) {
      irpass::type_check(ir, config);
      print("Auto-real-function partial inlined back");
      irpass::analysis::verify(ir);
    }
  }

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda ||
      config.arch == Arch::amdgpu) {
    TI_COMPILE_PROFILER("cpp.ir.bit_loop_vectorize");
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    print("Bit Loop Vectorized");
    irpass::analysis::verify(ir);
  }

  // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
  {
    TI_COMPILE_PROFILER("cpp.ir.lower_matrix_ptr");
    irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
  }
  print("Matrix ptr lowered");

  if (config.force_scalarize_matrix) {
    TI_COMPILE_PROFILER("cpp.ir.scalarize.force_matrix");
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);

    irpass::die(ir);
    print("Scalarized");
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.full_simplify.I");
    irpass::full_simplify(
        ir, config,
        {false, /*autodiff_enabled*/ autodiff_mode != AutodiffMode::kNone,
         kernel->get_name(), verbose});
  }
  print("Simplified I");
  irpass::analysis::verify(ir);

  // Track whether any IR-mutating pass has run since "Simplified I". If not,
  // "Simplified II" below is a no-op (full_simplify on an already-simplified
  // IR converges in one round) and can be skipped — saving one full pass over
  // every kernel that has no external arrays / no autodiff / no debug checks.
  // (P2.a: full_simplify dirty-flag dedupe.)
  bool dirty_since_simplify_i = [&]() {
    TI_COMPILE_PROFILER("cpp.ir.external_ptr_boundary");
    return irpass::handle_external_ptr_boundary(ir, config);
  }();
  print("External ptr boundary processed");

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::analysis::gather_meshfor_relation_types(ir);
  }

  if (config.debug && autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    // Check whether the kernel obeys the autodiff limitation e.g., gloabl data
    // access rule
    // This check should be performed in the forward kernel i.e., autodiff_mode
    // == AutodiffMode::kCheckAutodiffValid
    {
      TI_COMPILE_PROFILER("cpp.ir.autodiff.validation");
      irpass::demote_atomics(ir, config);
      irpass::differentiation_validation_check(ir, config, kernel->get_name());
    }
    irpass::analysis::verify(ir);
    // demote_atomics + differentiation_validation_check both mutate IR.
    dirty_since_simplify_i = true;
  }

  if (autodiff_mode == AutodiffMode::kReverse ||
      autodiff_mode == AutodiffMode::kForward) {
    // Remove local atomics here so that we don't have to handle their gradients
    {
      TI_COMPILE_PROFILER("cpp.ir.autodiff.transform");
      irpass::demote_atomics(ir, config);

      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ true, kernel->get_name(), verbose});
      irpass::auto_diff(ir, config, autodiff_mode, ad_use_stack);
      // TODO: Be carefull with the full_simplify when do high-order autodiff
      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
    }
    print("Gradient");
    irpass::analysis::verify(ir);
    // The two full_simplify calls above already left the IR in a fixed-point
    // state, so the post-flag_access full_simplify below is also redundant
    // for this branch. Keep the dirty flag false.
    dirty_since_simplify_i = false;
  }

  if (config.check_out_of_bound) {
    {
      TI_COMPILE_PROFILER("cpp.ir.check_out_of_bound");
      irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    }
    print("Bound checked");
    irpass::analysis::verify(ir);
    dirty_since_simplify_i = true;
  }

  irpass::flag_access(ir);
  print("Access flagged I");
  irpass::analysis::verify(ir);
  // flag_access only mutates GlobalPtrStmt::activate metadata; full_simplify
  // ignores that field entirely. So flag_access does NOT make the IR dirty
  // for the purposes of simplification.

  if (dirty_since_simplify_i) {
    {
      TI_COMPILE_PROFILER("cpp.ir.full_simplify.II");
      irpass::full_simplify(ir, config,
                            {false, /*autodiff_enabled*/ false,
                             kernel->get_name(), verbose});
    }
    print("Simplified II");
    irpass::analysis::verify(ir);
  } else {
    print("Simplified II (skipped: IR unchanged since Simplified I)");
  }

  if (graph_metadata != nullptr) {
    *graph_metadata = analyze_graph_kernel_metadata(ir, kernel);
  }
  if (stop_before_offload) {
    return;
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.offload");
    irpass::offload(ir, config);
  }
  validate_task_launch_policy_offloads(ir, task_launch_policy);
  print("Offloaded");
  irpass::analysis::verify(ir);
  // NOTE: There was an additional CFG pass here, removed in
  // https://github.com/taichi-dev/taichi/pull/8691
  irpass::flag_access(ir);
  print("Access flagged II");

  {
    TI_COMPILE_PROFILER("cpp.ir.full_simplify.III");
    irpass::full_simplify(
        ir, config,
        {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
  }
  print("Simplified III");
  irpass::analysis::verify(ir);

  // The execution-plan topology is defined over physical tasks that survive
  // post-offload simplification.  Offloading may emit an empty serial bound
  // preamble which Simplified III removes; binding before this point would
  // make a plan reconstructed from the compiled task manifest stale by
  // construction.  Applying the plan here is still before TLS/BLS and LLVM
  // lowering, so task-owned compilation controls remain effective.
  apply_offload_execution_plan(ir, config, kernel, start_from_ast);
  print("Offload execution plan applied");
  irpass::analysis::verify(ir);
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  TI_AUTO_PROF;
  TI_COMPILE_PROFILER("cpp.ir.offload_to_executable");

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info,
                                 kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify(ir);

  // P-Compile-1 cleanup (2026-05): the old driver-level full_simplify skip
  // path is retired. Current measurements show zero skip hits on the heavy
  // worker and no compile-time benefit, while the always-run path is the
  // stable behavior. Passes below may still return `bool` for their own local
  // contracts (e.g. conditional type_check), but offload_to_executable no
  // longer uses a dirty flag to skip simplifies.

  if (config.detect_read_only) {
    TI_COMPILE_PROFILER("cpp.ir.exec.detect_read_only");
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
    // detect_read_only is a pure analysis pass — no IR mutation.
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.exec.demote_atomics.I");
    irpass::demote_atomics(ir, config);
  }
  print("Atomics demoted I");
  irpass::analysis::verify(ir);

  const auto &offload_plan = kernel->get_offload_execution_plan();
  const bool has_shared_staged_task =
      offload_plan.has_value() &&
      std::any_of(offload_plan->tasks.begin(), offload_plan->tasks.end(),
                  [](const auto &task) {
                    return task.memory_strategy == "shared_staged_1d" ||
                           task.memory_strategy == "shared_staged_2d";
                  });
  if (has_shared_staged_task) {
    TI_COMPILE_PROFILER("cpp.ir.exec.make_external_shared_staged");
    irpass::make_external_shared_staged(ir, config, kernel);
    irpass::type_check(ir, config);
    print("Make external shared-staged range");
    irpass::analysis::verify(ir);
  }

  if (config.cache_loop_invariant_global_vars) {
    TI_COMPILE_PROFILER("cpp.ir.exec.cache_loop_invariant");
    irpass::cache_loop_invariant_global_vars(ir, config);
    print("Cache loop-invariant global vars");
  }

  if (config.demote_dense_struct_fors) {
    TI_COMPILE_PROFILER("cpp.ir.exec.demote_dense_struct_fors");
    if (irpass::demote_dense_struct_fors(ir)) {
      irpass::type_check(ir, config);
    }
    print("Dense struct-for demoted");
    irpass::analysis::verify(ir);
  }

  if (config.make_cpu_multithreading_loop && arch_is_cpu(config.arch)) {
    TI_COMPILE_PROFILER("cpp.ir.exec.cpu_multithread_range_for");
    irpass::make_cpu_multithreaded_range_for(ir, config);
    irpass::type_check(ir, config);
    print("Make CPU multithreaded range-for");
    irpass::analysis::verify(ir);
  }

  if (is_extension_supported(config.arch, Extension::mesh) &&
      config.demote_no_access_mesh_fors) {
    irpass::demote_no_access_mesh_fors(ir);
    irpass::type_check(ir, config);
    print("No-access mesh-for demoted");
    irpass::analysis::verify(ir);
  }

  if (make_thread_local) {
    TI_COMPILE_PROFILER("cpp.ir.exec.make_thread_local");
    irpass::make_thread_local(ir, config);
    print("Make thread local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    TI_COMPILE_PROFILER("cpp.ir.exec.make_mesh_thread_local");
    irpass::make_mesh_thread_local(ir, config, {kernel->get_name()});
    print("Make mesh thread local");
    if (config.make_mesh_block_local && config.arch == Arch::cuda) {
      irpass::make_mesh_block_local(ir, config, {kernel->get_name()});
      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
      print("Simplified X");
    }
  }

  if (make_block_local) {
    TI_COMPILE_PROFILER("cpp.ir.exec.make_block_local");
    irpass::make_block_local(
        ir, config, {kernel->get_name(), verbose, kernel->program});
    print("Make block local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::demote_mesh_statements(ir, config, {kernel->get_name()});
    print("Demote mesh statements");
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.exec.demote_atomics.II");
    irpass::demote_atomics(ir, config);
  }
  print("Atomics demoted II");
  irpass::analysis::verify(ir);

  if (is_extension_supported(config.arch, Extension::quant) &&
      config.quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

  irpass::remove_loop_unique(ir);
  print("Remove loop_unique");
  irpass::analysis::verify(ir);

  if (lower_global_access) {
    {
      TI_COMPILE_PROFILER("cpp.ir.exec.full_simplify.before_lower_access");
      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
    }
    print("Simplified before lower access");

    {
      TI_COMPILE_PROFILER("cpp.ir.exec.lower_access");
      irpass::lower_access(ir, config, {kernel->no_activate, true});
    }
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    // flag_access mutates GlobalPtrStmt::activate. The downstream
    // "Simplified IV" pass is always run after the fused-pass cleanup.
    print("Access flagged III");
    irpass::analysis::verify(ir);
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.exec.demote_operations");
    irpass::demote_operations(ir, config);
  }
  print("Operations demoted");

  {
    TI_COMPILE_PROFILER("cpp.ir.exec.full_simplify.IV");
    irpass::full_simplify(
        ir, config,
        {lower_global_access, /*autodiff_enabled*/ false, kernel->get_name(),
         verbose});
  }
  print("Simplified IV");

  if (determine_ad_stack_size) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  bool half2_optimization_enabled =
      (config.arch == Arch::cuda && config.half2_vectorization &&
       !get_custom_cuda_library_path().empty());
  if (config.real_matrix_scalarize) {
    TI_COMPILE_PROFILER("cpp.ir.exec.scalarize.real_matrix");
    if (irpass::scalarize(ir, half2_optimization_enabled)) {
      irpass::die(ir);
      print("DIE");

      // Remove redundant MatrixInitStmt inserted during scalarization
      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
      print("Scalarized");
    }
  }

  // Final field registration correctness & type checking
  {
    TI_COMPILE_PROFILER("cpp.ir.exec.type_check.final");
    irpass::type_check(ir, config);
  }
  irpass::analysis::verify(ir);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast,
                           GraphKernelMetadata *graph_metadata) {
  TI_AUTO_PROF;
  TI_COMPILE_PROFILER("cpp.ir.compile_to_executable");

  compile_to_offloads(ir, config, kernel, verbose, autodiff_mode, ad_use_stack,
                      start_from_ast, graph_metadata);

  offload_to_executable(
      ir, config, kernel, verbose,
      /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse &&
          ad_use_stack,
      lower_global_access, make_thread_local, make_block_local);
}

void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      Function::IRStage target_stage) {
  TI_AUTO_PROF;

  auto current_stage = func->ir_stage();
  auto print = make_pass_printer(verbose, config.print_ir_dbg_info,
                                 func->get_name(), ir);
  print("Initial IR");

  if (target_stage >= Function::IRStage::BeforeLowerAccess &&
      current_stage < Function::IRStage::BeforeLowerAccess) {
    if (autodiff_mode == AutodiffMode::kReverse) {
      irpass::reverse_segments(ir);
      print("Segment reversed (for autodiff)");
    }

    if (current_stage < Function::IRStage::InitialIR) {
      irpass::frontend_type_check(ir);
      irpass::lower_ast(ir);
      print("Lowered");
    }

    // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
    irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
    print("Matrix ptr lowered");

    irpass::demote_atomics(ir, config);
    print("Atomics demoted");
    irpass::associate_continue_scope(ir, config);
    print("Associated continue scope");
    func->set_ir_stage(Function::IRStage::BeforeLowerAccess);
  }

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
  }

  if (target_stage >= Function::IRStage::OptimizedIR &&
      current_stage < Function::IRStage::OptimizedIR) {
    irpass::lower_access(ir, config, {{}, true});
    print("Access lowered");
    irpass::analysis::verify(ir);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify(ir);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify(ir);

    irpass::type_check(ir, config);
    print("Typechecked");

    irpass::demote_operations(ir, config);
    print("Operations demoted");

    if (config.real_matrix_scalarize) {
      if (irpass::scalarize(ir)) {
        // Remove redundant MatrixInitStmt inserted during scalarization
        irpass::die(ir);
        print("Scalarized");
      }
    }

    irpass::full_simplify(ir, config,
                          {true, autodiff_mode != AutodiffMode::kNone,
                           func->get_name(), verbose});
    print("Simplified");
    irpass::analysis::verify(ir);
    func->set_ir_stage(Function::IRStage::OptimizedIR);
  }
}

}  // namespace irpass

}  // namespace taichi::lang
