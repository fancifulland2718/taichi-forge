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
    CompilePurpose purpose,
    const KernelCompileOptions &export_options) {
  CompileConfig config = base_config;
  CompileRequestProvenance provenance{purpose,
                                      kernel_tier
                                          ? CompileTierSource::kernel_override
                                          : CompileTierSource::base_config,
                                      false};
  if (kernel_tier) {
    config.compile_tier = *kernel_tier;
    config.set_option_source(CompileOption::compile_tier,
                             CompileOptionSource::kernel_override);
  }
#define APPLY_EXPORT_OPTION(field)                                      \
  if (export_options.field.has_value()) {                               \
    TI_ERROR_IF(purpose != CompilePurpose::aot,                          \
                "Export compile options require an AOT request.");     \
    config.field = *export_options.field;                               \
    config.set_option_source(CompileOption::field,                      \
                             CompileOptionSource::export_override);    \
  }
  APPLY_EXPORT_OPTION(compile_tier)
  APPLY_EXPORT_OPTION(advanced_optimization)
  APPLY_EXPORT_OPTION(llvm_opt_level)
  APPLY_EXPORT_OPTION(external_optimization_level)
  APPLY_EXPORT_OPTION(fast_math)
  APPLY_EXPORT_OPTION(full_simplify_global_iter_cap)
#undef APPLY_EXPORT_OPTION
  if (export_options.compile_tier) {
    provenance.tier_source = CompileTierSource::export_override;
  }
  TI_ERROR_IF(export_options.llvm_opt_level &&
                  (*export_options.llvm_opt_level < 0 || *export_options.llvm_opt_level > 3),
              "AOT llvm_opt_level must be in [0, 3].");
  TI_ERROR_IF(export_options.external_optimization_level &&
                  (*export_options.external_optimization_level < 0 ||
                   *export_options.external_optimization_level > 3),
              "AOT external_optimization_level must be in [0, 3].");
  if (config.compile_tier != "fast" && config.compile_tier != "balanced" &&
      config.compile_tier != "full") {
    TI_ERROR("compile_tier must be one of fast, balanced, full; got {}",
             config.compile_tier);
  }

  // Only defaults may be expanded to a full fixed point. An explicit 1 is a
  // deliberate compilation budget, just as explicit False and 0 are values.
  constexpr int kLegacyDefaultFullSimplifyGlobalIterCap = 1;
  const auto cap_source = config.option_source(CompileOption::full_simplify_global_iter_cap);
  if (config.compile_tier == "full" &&
      (cap_source == CompileOptionSource::unspecified ||
       cap_source == CompileOptionSource::frontend_default ||
       cap_source == CompileOptionSource::derived) &&
      config.full_simplify_global_iter_cap ==
          kLegacyDefaultFullSimplifyGlobalIterCap) {
    config.full_simplify_global_iter_cap = 0;
    provenance.normalized_full_simplify_cap = true;
    config.set_option_source(CompileOption::full_simplify_global_iter_cap,
                             CompileOptionSource::derived);
  }
  provenance.option_sources = config.option_sources;
  return KernelCompileRequest(std::move(config), caps, provenance);
}

}  // namespace taichi::lang
