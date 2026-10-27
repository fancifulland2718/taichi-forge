// Shared LLVM New-PassManager optimization pipeline used by Taichi JIT
// backends (CPU, CUDA/PTX, AMDGPU/HSACO, DX12/DXIL).
//
// LLVM 17 removed `llvm::PassManagerBuilder` and its legacy IPO pipeline
// populator in favour of the New Pass Manager (`llvm::PassBuilder`).
// This file centralises the replacement so each backend only has to
// describe its tuning parameters (opt level, vectorize flags, target
// machine) instead of duplicating ~30 lines of boilerplate.
//
// The code path here is fully New-PM and therefore remains valid
// through at least LLVM 22 — only the codegen emission step
// (`TargetMachine::addPassesToEmitFile`) still requires the legacy PM,
// which each backend keeps isolated at the end of its pipeline.
#pragma once

#include <string>

#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Target/TargetMachine.h"

namespace taichi::lang {

/// Map an integer llvm_opt_level (0-3) to the matching llvm::OptimizationLevel.
/// Values out of range are clamped: negative → O0, >3 → O3.
inline llvm::OptimizationLevel llvm_opt_level_from_int(int level) {
  switch (level) {
    case 0:  return llvm::OptimizationLevel::O0;
    case 1:  return llvm::OptimizationLevel::O1;
    case 2:  return llvm::OptimizationLevel::O2;
    default: return llvm::OptimizationLevel::O3;
  }
}

/// Effective opt level after considering `compile_tier`.
///
/// `compile_tier == "fast"` forces the opt level down so the full
/// D2 tier semantics line up -- Vulkan already uses
/// `spv_opt_level` 0 in that tier (see `kernel_compiler.cpp`), and
/// this mirrors that policy for CPU / CUDA / AMDGPU / DX12.
///
/// `min_level` is a backend-specific floor: CPU uses O1 to avoid repeated
/// unoptimized address/aggregate work; DX12 retains O0. NVPTX (CUDA) and
/// AMDGCN codegen both rely on
/// mid-level legalization passes (e.g. StackSave/StackRestore lowering,
/// addrspace canonicalization) that are skipped at O0 — those backends
/// must pass `min_level=1` to stay correct under `tier=fast`.
///
/// Callers select the floor through LLVMModuleOptions, except DX12's
/// separate pipeline. Other tiers return `level` unchanged (default O3),
/// including explicitly selected O0 for compiler diagnostics.
inline int effective_llvm_opt_level(int level,
                                    const std::string &tier,
                                    int min_level = 0) {
  if (tier == "fast") {
    return min_level > 0 ? min_level : 0;
  }
  return level;
}

struct LLVMOptPipelineOptions {
  // Optimization level. Mirrors the historical `PassManagerBuilder::OptLevel = 3`.
  llvm::OptimizationLevel opt_level = llvm::OptimizationLevel::O3;

  // Vectorization knobs. Forwarded to `PipelineTuningOptions` and hence
  // control whether LoopVectorize / SLPVectorize are run inside the
  // default pipeline built by `buildPerModuleDefaultPipeline`.
  bool loop_vectorize = true;
  bool slp_vectorize = true;

  // Re-run LoopStrengthReduce + IndVarSimplify +
  // SeparateConstOffsetFromGEP + EarlyCSE after the default pipeline.
  // This is a Taichi-specific tweak (see taichi-dev/taichi#5472) that
  // measurably improves GEP lowering for the GPU backends; CPU benefits
  // from it as well.
  bool run_post_gep_passes = true;
};

// Run the module-level optimization pipeline.
//
// `target_machine` may be null (in which case the pipeline runs without
// target-specific tuning), but every in-tree caller passes a valid
// machine so that `PassBuilder` sees an accurate `TargetIRAnalysis`.
void run_module_opt_pipeline(llvm::Module &module,
                             llvm::TargetMachine *target_machine,
                             const LLVMOptPipelineOptions &opts);

}  // namespace taichi::lang
