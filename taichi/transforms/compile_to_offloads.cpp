#include "taichi/ir/ir.h"
#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/system/profiler.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/pass.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/util/lang_util.h"


namespace taichi::lang {

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast,
                         GraphKernelMetadata *graph_metadata) {
  TI_AUTO_PROF;
  TI_COMPILE_PROFILER("cpp.ir.compile_to_offloads");

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
      irpass::full_simplify(
          ir, config,
          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose});
    }
    print("Simplified II");
    irpass::analysis::verify(ir);
  } else {
    print("Simplified II (skipped: IR unchanged since Simplified I)");
  }

  if (graph_metadata != nullptr) {
    *graph_metadata = analyze_graph_kernel_metadata(ir, kernel);
  }

  {
    TI_COMPILE_PROFILER("cpp.ir.offload");
    irpass::offload(ir, config);
  }
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
      irpass::full_simplify(ir, config,
                            {false, /*autodiff_enabled*/ false,
                             kernel->get_name(), verbose});
      print("Simplified X");
    }
  }

  if (make_block_local) {
    TI_COMPILE_PROFILER("cpp.ir.exec.make_block_local");
    irpass::make_block_local(ir, config, {kernel->get_name(), verbose});
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
