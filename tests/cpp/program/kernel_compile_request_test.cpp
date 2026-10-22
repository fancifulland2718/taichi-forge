#include "gtest/gtest.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/program/kernel_compile_request.h"

namespace taichi::lang {
namespace {

TEST(KernelCompileRequest, OwnsConfigAndTargetWithoutChangingTheCaller) {
  CompileConfig base;
  base.arch = Arch::cuda;
  base.compile_tier = "fast";
  base.advanced_optimization = false;
  base.fast_math = false;
  base.llvm_opt_level = 0;
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::cuda_compute_capability, 86);
  const auto request = resolve_kernel_compile_request(base, caps, "balanced");
  EXPECT_EQ(request.config().compile_tier, "balanced");
  EXPECT_FALSE(request.config().advanced_optimization);
  EXPECT_FALSE(request.config().fast_math);
  EXPECT_EQ(request.config().llvm_opt_level, 0);
  EXPECT_EQ(request.provenance().tier_source,
            CompileTierSource::kernel_override);
  EXPECT_EQ(base.compile_tier, "fast");
  base.fast_math = true;
  caps.set(DeviceCapability::cuda_compute_capability, 89);
  EXPECT_FALSE(request.config().fast_math);
  EXPECT_EQ(
      request.device_caps().get(DeviceCapability::cuda_compute_capability), 86);
}

TEST(KernelCompileRequest, ResolvesCudaTargetsWithoutBorrowingProducerCapabilities) {
  for (int producer : {86, 120, 122}) {
    const auto deployment = resolve_kernel_target(Arch::cuda, {}, CompilePurpose::aot, producer);
    EXPECT_EQ(deployment.source, CompileTargetSource::aot_default);
    EXPECT_EQ(deployment.capabilities.get(DeviceCapability::cuda_compute_capability), 80);
    EXPECT_EQ(deployment.capabilities.get(DeviceCapability::cuda_ptx_version), 70);
    const auto current = resolve_kernel_target(Arch::cuda, {}, CompilePurpose::jit, producer);
    EXPECT_EQ(current.source, CompileTargetSource::current_device);
    EXPECT_EQ(current.capabilities.get(DeviceCapability::cuda_compute_capability), producer == 86 ? 86 : 120);
  }
  DeviceCapabilityConfig explicit_caps;
  explicit_caps.set(DeviceCapability::cuda_compute_capability, 60);
  const auto target = resolve_kernel_target(Arch::cuda, explicit_caps, CompilePurpose::aot, 120);
  EXPECT_EQ(target.source, CompileTargetSource::explicit_target);
  EXPECT_EQ(target.capabilities.get(DeviceCapability::cuda_ptx_version), 60);
  EXPECT_FALSE(explicit_caps.contains(DeviceCapability::cuda_ptx_version));
  explicit_caps.set(DeviceCapability::cuda_ptx_version, 50);
  EXPECT_ANY_THROW(resolve_kernel_target(Arch::cuda, explicit_caps, CompilePurpose::aot, 120));
  explicit_caps.set(DeviceCapability::cuda_compute_capability, 122);
  EXPECT_ANY_THROW(resolve_kernel_target(Arch::cuda, explicit_caps, CompilePurpose::aot, 120));
  // Non-CUDA snapshots do not acquire CUDA requirements or device state.
  for (auto arch : {Arch::x64, Arch::vulkan}) {
    const auto other = resolve_kernel_target(arch, {}, CompilePurpose::jit, 122);
    EXPECT_TRUE(other.capabilities.devcaps.empty());
    EXPECT_EQ(other.source, CompileTargetSource::capability_snapshot);
  }
}

TEST(KernelCompileRequest, PreservesLegacyFullTierNormalization) {
  CompileConfig base;
  base.compile_tier = "balanced";
  base.full_simplify_global_iter_cap = 1;
  const auto request = resolve_kernel_compile_request(base, {}, "full");
  EXPECT_EQ(request.config().full_simplify_global_iter_cap, 0);
  EXPECT_TRUE(request.provenance().normalized_full_simplify_cap);
  EXPECT_EQ(base.full_simplify_global_iter_cap, 1);
  for (int cap : {0, 3}) {
    base.full_simplify_global_iter_cap = cap;
    const auto tuned = resolve_kernel_compile_request(base, {}, "full");
    EXPECT_EQ(tuned.config().full_simplify_global_iter_cap, cap);
    EXPECT_FALSE(tuned.provenance().normalized_full_simplify_cap);
  }
}

TEST(KernelCompileRequest, PurposeAndEquivalentOriginsDoNotChangeCodeIdentity) {
  CompileConfig base;
  base.arch = Arch::vulkan;
  base.compile_tier = "fast";
  base.advanced_optimization = false;
  DeviceCapabilityConfig caps;
  const auto jit = resolve_kernel_compile_request(base, caps);
  const auto aot =
      resolve_kernel_compile_request(base, caps, "fast", CompilePurpose::aot);
  EXPECT_EQ(jit.provenance().purpose, CompilePurpose::jit);
  EXPECT_EQ(aot.provenance().purpose, CompilePurpose::aot);
  EXPECT_EQ(jit.provenance().tier_source, CompileTierSource::base_config);
  EXPECT_EQ(aot.provenance().tier_source, CompileTierSource::kernel_override);
  EXPECT_EQ(aot.config().compile_tier, "fast");
  EXPECT_FALSE(aot.config().advanced_optimization);
  EXPECT_EQ(get_hashed_offline_cache_key_context(jit.config(),
                                                 jit.device_caps(), nullptr),
            get_hashed_offline_cache_key_context(aot.config(),
                                                 aot.device_caps(), nullptr));
}

TEST(KernelCompileRequest, ValidatesTheResolvedTier) {
  CompileConfig base;
  base.compile_tier = "invalid";
  EXPECT_ANY_THROW(resolve_kernel_compile_request(base, {}));
  EXPECT_NO_THROW(resolve_kernel_compile_request(base, {}, "fast"));
  base.compile_tier = "balanced";
  EXPECT_ANY_THROW(resolve_kernel_compile_request(base, {}, "invalid"));
}

TEST(KernelCompileRequest, PreservesExplicitCapAndIgnoresOriginsInCodeIdentity) {
  CompileConfig base;
  base.compile_tier = "full";
  base.full_simplify_global_iter_cap = 1;
  for (auto source : {CompileOptionSource::init_keyword,
                      CompileOptionSource::environment,
                      CompileOptionSource::direct_assignment}) {
    base.set_option_source(CompileOption::full_simplify_global_iter_cap, source);
    const auto request = resolve_kernel_compile_request(base, {});
    EXPECT_EQ(request.config().full_simplify_global_iter_cap, 1);
    EXPECT_FALSE(request.provenance().normalized_full_simplify_cap);
  }
  base.compile_tier = "balanced";
  const auto explicit_request = resolve_kernel_compile_request(base, {});
  base.set_option_source(CompileOption::full_simplify_global_iter_cap,
                         CompileOptionSource::unspecified);
  const auto implicit_request = resolve_kernel_compile_request(base, {});
  EXPECT_EQ(get_hashed_offline_cache_key_context(explicit_request.config(), {}, nullptr),
            get_hashed_offline_cache_key_context(implicit_request.config(), {}, nullptr));
}

TEST(KernelCompileRequest, ExportOverridesAreFrozenValuesWithoutJitMutation) {
  CompileConfig base;
  base.compile_tier = "fast";
  base.advanced_optimization = true;
  KernelCompileOptions options;
  options.compile_tier = "full";
  options.advanced_optimization = false;
  options.fast_math = false;
  options.llvm_opt_level = 0;
  options.external_optimization_level = 0;
  options.full_simplify_global_iter_cap = 1;
  const auto request = resolve_kernel_compile_request(
      base, {}, "balanced", CompilePurpose::aot, options);
  EXPECT_EQ(request.config().compile_tier, "full");
  EXPECT_FALSE(request.config().advanced_optimization);
  EXPECT_FALSE(request.config().fast_math);
  EXPECT_EQ(request.config().llvm_opt_level, 0);
  EXPECT_EQ(request.config().external_optimization_level, 0);
  EXPECT_EQ(request.config().full_simplify_global_iter_cap, 1);
  EXPECT_FALSE(request.provenance().normalized_full_simplify_cap);
  EXPECT_EQ(request.provenance().tier_source, CompileTierSource::export_override);
  EXPECT_EQ(base.compile_tier, "fast");
  EXPECT_TRUE(base.advanced_optimization);
  EXPECT_ANY_THROW(resolve_kernel_compile_request(base, {}, {}, CompilePurpose::jit, options));
  options.compile_tier = "fast";
  EXPECT_EQ(request.config().compile_tier, "full");
  options.llvm_opt_level = -1;
  EXPECT_ANY_THROW(resolve_kernel_compile_request(base, {}, {}, CompilePurpose::aot, options));
}

}  // namespace
}  // namespace taichi::lang
