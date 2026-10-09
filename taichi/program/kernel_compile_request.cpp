#include "taichi/program/kernel_compile_request.h"

#include <utility>

namespace taichi::lang {

KernelCompileRequest::KernelCompileRequest(CompileConfig config,
                                           DeviceCapabilityConfig caps,
                                           CompileRequestProvenance provenance)
    : config_(std::move(config)),
      caps_(std::move(caps)),
      provenance_(provenance) {
}

KernelCompileRequest resolve_kernel_compile_request(
    const CompileConfig &base_config,
    const DeviceCapabilityConfig &caps,
    const std::optional<std::string> &kernel_tier,
    CompilePurpose purpose) {
  CompileConfig config = base_config;
  CompileRequestProvenance provenance{purpose,
                                      kernel_tier
                                          ? CompileTierSource::kernel_override
                                          : CompileTierSource::base_config,
                                      false};
  if (kernel_tier) {
    config.compile_tier = *kernel_tier;
  }
  if (config.compile_tier != "fast" && config.compile_tier != "balanced" &&
      config.compile_tier != "full") {
    TI_ERROR("compile_tier must be one of fast, balanced, full; got {}",
             config.compile_tier);
  }

  // Preserve Program's existing normalization. The legacy CompileConfig does
  // not distinguish an explicitly supplied cap of 1 from its default value;
  // record the derivation without claiming that distinction has been resolved.
  constexpr int kLegacyDefaultFullSimplifyGlobalIterCap = 1;
  if (config.compile_tier == "full" &&
      config.full_simplify_global_iter_cap ==
          kLegacyDefaultFullSimplifyGlobalIterCap) {
    config.full_simplify_global_iter_cap = 0;
    provenance.normalized_full_simplify_cap = true;
  }
  return KernelCompileRequest(std::move(config), caps, provenance);
}

}  // namespace taichi::lang
