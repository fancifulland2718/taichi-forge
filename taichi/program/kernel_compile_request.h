#pragma once

#include <optional>
#include <string>

#include "taichi/program/compile_config.h"
#include "taichi/rhi/device_capability.h"

namespace taichi::lang {

enum class CompilePurpose { jit, aot };
enum class CompileTierSource { base_config, kernel_override, export_override };
enum class CompileTargetSource { capability_snapshot, current_device, aot_default, explicit_target };

// The capability projection is also used by backend lowering. CUDA resolution
// happens once at the entry boundary, never in a worker's task code generator.
struct ResolvedKernelTarget {
  DeviceCapabilityConfig capabilities;
  CompileTargetSource source{CompileTargetSource::capability_snapshot};
};

ResolvedKernelTarget resolve_kernel_target(
    Arch arch,
    const DeviceCapabilityConfig &caps,
    CompilePurpose purpose,
    std::optional<int> current_cuda_compute_capability = std::nullopt);

struct KernelCompileOptions {
  std::optional<std::string> compile_tier;
  std::optional<bool> advanced_optimization;
  std::optional<int> llvm_opt_level;
  std::optional<int> external_optimization_level;
  std::optional<bool> fast_math;
  std::optional<int> full_simplify_global_iter_cap;
};

// Diagnostic provenance, not code identity. The current resolver preserves
// legacy defaults; purpose does not yet choose different JIT/AOT policies.
struct CompileRequestProvenance {
  CompilePurpose purpose;
  CompileTierSource tier_source;
  bool normalized_full_simplify_cap;
  std::array<CompileOptionSource, static_cast<size_t>(CompileOption::count)>
      option_sources{};
};

// An owned snapshot for one compilation request. Resolving a kernel override
// must not temporarily mutate the Program or a caller's target capabilities.
// This captures configuration only; IR, ABI and resource lifetimes retain their
// existing owners and must also participate in cache/materialization checks.
class KernelCompileRequest {
 public:
  const CompileConfig &config() const {
    return config_;
  }
  const DeviceCapabilityConfig &device_caps() const {
    return target_.capabilities;
  }
  const ResolvedKernelTarget &target() const {
    return target_;
  }
  const CompileRequestProvenance &provenance() const {
    return provenance_;
  }

 private:
  KernelCompileRequest(CompileConfig config,
                       ResolvedKernelTarget target,
                       CompileRequestProvenance provenance);

  CompileConfig config_;
  ResolvedKernelTarget target_;
  CompileRequestProvenance provenance_;

  friend KernelCompileRequest resolve_kernel_compile_request(
      const CompileConfig &,
      const DeviceCapabilityConfig &,
      const std::optional<std::string> &,
      CompilePurpose,
      const KernelCompileOptions &);
};

KernelCompileRequest resolve_kernel_compile_request(
    const CompileConfig &base_config,
    const DeviceCapabilityConfig &caps,
    const std::optional<std::string> &kernel_tier = std::nullopt,
    CompilePurpose purpose = CompilePurpose::jit,
    const KernelCompileOptions &export_options = {});

}  // namespace taichi::lang
