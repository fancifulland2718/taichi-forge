#include "gtest/gtest.h"

#include <future>

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/statements.h"
#include "taichi/program/kernel.h"
#include "taichi/program/kernel_compile_request.h"
#include "taichi/program/program.h"

namespace taichi::lang {
namespace {

TEST(KernelCacheContext, SeparatesConfigsTargetsAndRecoversPreviousIdentity) {
  Program program(Arch::x64);
  Kernel kernel(program, [] {}, "context_ast");
  CompileConfig config;
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::cuda_compute_capability, 86);
  const auto first = kernel.get_or_create_kernel_key_for_cache(config, caps);
  EXPECT_EQ(first, get_hashed_offline_cache_key(config, caps, &kernel));
  config.fast_math = !config.fast_math;
  const auto second = kernel.get_or_create_kernel_key_for_cache(config, caps);
  EXPECT_NE(first, second);
  EXPECT_EQ(second, get_hashed_offline_cache_key(config, caps, &kernel));
  caps.set(DeviceCapability::cuda_compute_capability, 89);
  const auto third = kernel.get_or_create_kernel_key_for_cache(config, caps);
  EXPECT_NE(second, third);
  config.fast_math = !config.fast_math;
  caps.set(DeviceCapability::cuda_compute_capability, 86);
  EXPECT_EQ(kernel.get_or_create_kernel_key_for_cache(config, caps), first);
  EXPECT_EQ(kernel.get_cached_kernel_key(), first);
  kernel.invalidate_kernel_key_for_cache();
  EXPECT_TRUE(kernel.get_cached_kernel_key().empty());
  EXPECT_EQ(kernel.get_or_create_kernel_key_for_cache(config, caps), first);
}

TEST(KernelCacheContext, LoweredKernelsAlsoValidateContext) {
  Program program(Arch::x64);
  Kernel kernel(program, std::make_unique<Block>(), "context_lowered");
  CompileConfig config;
  const auto first = kernel.get_or_create_kernel_key_for_cache(config, {});
  EXPECT_EQ(first,
            "N" + get_hashed_offline_cache_key_context(config, {}, &kernel) +
                "_context_lowered");
  config.llvm_opt_level = config.llvm_opt_level == 0 ? 3 : 0;
  EXPECT_NE(first, kernel.get_or_create_kernel_key_for_cache(config, {}));
}

TEST(KernelCacheContext, ConcurrentRequestsReturnTheirOwnKeys) {
  Program program(Arch::x64);
  Kernel kernel(program, [] {}, "context_concurrent");
  CompileConfig base;
  std::vector<std::future<std::string>> results;
  for (int i = 0; i < 8; ++i) {
    auto config = base;
    config.fast_math = (i % 2) != 0;
    results.push_back(std::async(std::launch::async, [&, config] {
      return kernel.get_or_create_kernel_key_for_cache(config, {});
    }));
  }
  // The initial body serialization is shared, while each result owns the key
  // for its request regardless of which worker updates the last-key slot.
  for (int i = 0; i < 8; ++i) {
    auto config = base;
    config.fast_math = (i % 2) != 0;
    EXPECT_EQ(results[i].get(),
              get_hashed_offline_cache_key(config, {}, &kernel));
  }
}

TEST(KernelCacheContext, RetiredDefinitionRequiresThePreservedRequest) {
  Program program(Arch::x64);
  Kernel kernel(program, [] {}, "context_retired");
  CompileConfig config;
  const auto first = kernel.get_or_create_kernel_key_for_cache(config, {});
  kernel.retire_definition(/*preserve_relocatable_abi=*/true);
  EXPECT_EQ(kernel.get_or_create_kernel_key_for_cache(config, {}), first);
  config.fast_math = !config.fast_math;
  EXPECT_ANY_THROW(kernel.get_or_create_kernel_key_for_cache(config, {}));
  EXPECT_EQ(kernel.get_cached_kernel_key(), first);
  kernel.retire_definition();
  EXPECT_TRUE(kernel.get_cached_kernel_key().empty());
  EXPECT_ANY_THROW(kernel.get_or_create_kernel_key_for_cache(config, {}));
}

TEST(KernelCacheContext, EquivalentResolvedRequestsShareIdentity) {
  Program program(Arch::x64);
  Kernel kernel(program, [] {}, "context_resolved");
  CompileConfig base;
  base.compile_tier = "fast";
  base.full_simplify_global_iter_cap = 1;
  const auto jit = resolve_kernel_compile_request(base, {}, "full");
  base.compile_tier = "full";
  base.full_simplify_global_iter_cap = 0;
  const auto aot = resolve_kernel_compile_request(base, {}, std::nullopt,
                                                  CompilePurpose::aot);
  EXPECT_EQ(kernel.get_or_create_kernel_key_for_cache(jit.config(), {}),
            kernel.get_or_create_kernel_key_for_cache(aot.config(), {}));
}

TEST(KernelCacheContext, SnapshotIncludesAbiAndCanonicalizesChangedOrdering) {
  Program program(Arch::x64);
  Kernel kernel(program, [] {}, "context_abi");
  CompileConfig config;
  config.spirv_disabled_passes = {"A", "B"};
  const auto first = kernel.get_or_create_kernel_key_for_cache(config, {});
  config.spirv_disabled_passes = {"B", "A"};
  EXPECT_EQ(first, kernel.get_or_create_kernel_key_for_cache(config, {}));
  EXPECT_EQ(first, kernel.get_or_create_kernel_key_for_cache(config, {}));
  kernel.parameter_list.emplace_back(PrimitiveType::i32);
  const auto parameter = kernel.get_or_create_kernel_key_for_cache(config, {});
  EXPECT_NE(first, parameter);
  kernel.rets.emplace_back(PrimitiveType::f32);
  const auto result = kernel.get_or_create_kernel_key_for_cache(config, {});
  EXPECT_NE(parameter, result);
  kernel.rets[0].dt = PrimitiveType::i32;
  EXPECT_NE(result, kernel.get_or_create_kernel_key_for_cache(config, {}));
}

TEST(KernelCacheContext, ConfigProjectionPreservesFloatingPointBits) {
  CompileConfig first;
  first.arch = Arch::vulkan;
  first.vulkan_pointer_pool_fraction = 0.0;
  auto second = first;
  EXPECT_TRUE(same_offline_cache_compile_config(first, second));
  second.vulkan_pointer_pool_fraction = -0.0;
  EXPECT_FALSE(same_offline_cache_compile_config(first, second));
  EXPECT_NE(get_hashed_offline_cache_key_context(first, {}, nullptr),
            get_hashed_offline_cache_key_context(second, {}, nullptr));
  second = first;
  second.random_seed += 1;
  EXPECT_TRUE(same_offline_cache_compile_config(first, second));
  EXPECT_EQ(get_hashed_offline_cache_key_context(first, {}, nullptr),
            get_hashed_offline_cache_key_context(second, {}, nullptr));
}

}  // namespace
}  // namespace taichi::lang
