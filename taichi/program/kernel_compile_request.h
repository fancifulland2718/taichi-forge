#pragma once

#include <optional>
#include <string>

#include "taichi/program/compile_config.h"
#include "taichi/rhi/device_capability.h"

namespace taichi::lang {

enum class CompilePurpose { jit, aot };
enum class CompileTierSource { base_config, kernel_override };

// Diagnostic provenance, not code identity. The current resolver preserves
// legacy defaults; purpose does not yet choose different JIT/AOT policies.
struct CompileRequestProvenance {
  CompilePurpose purpose;
  CompileTierSource tier_source;
  bool normalized_full_simplify_cap;
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
    return caps_;
  }
  const CompileRequestProvenance &provenance() const {
    return provenance_;
  }

 private:
  KernelCompileRequest(CompileConfig config,
                       DeviceCapabilityConfig caps,
                       CompileRequestProvenance provenance);

  CompileConfig config_;
  DeviceCapabilityConfig caps_;
  CompileRequestProvenance provenance_;

  friend KernelCompileRequest resolve_kernel_compile_request(
      const CompileConfig &,
      const DeviceCapabilityConfig &,
      const std::optional<std::string> &,
      CompilePurpose);
};

KernelCompileRequest resolve_kernel_compile_request(
    const CompileConfig &base_config,
    const DeviceCapabilityConfig &caps,
    const std::optional<std::string> &kernel_tier = std::nullopt,
    CompilePurpose purpose = CompilePurpose::jit);

}  // namespace taichi::lang
