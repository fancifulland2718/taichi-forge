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

}  // namespace
}  // namespace taichi::lang
