#include "gtest/gtest.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/program/compile_config.h"
#include "taichi/rhi/device_capability.h"

namespace taichi::lang {
namespace {

std::string compile_config_cache_key(const CompileConfig &config) {
  return get_hashed_offline_cache_key_context(
      config, DeviceCapabilityConfig{}, /*kernel=*/nullptr);
}

template <typename Mutator>
void expect_distinct_key(const CompileConfig &baseline, Mutator mutate) {
  CompileConfig variant = baseline;
  mutate(variant);
  EXPECT_NE(compile_config_cache_key(baseline),
            compile_config_cache_key(variant));
}

TEST(OfflineCache, FingerprintsIrChangingCompileConfigInputs) {
  CompileConfig cpu;
  cpu.arch = Arch::x64;

  expect_distinct_key(cpu, [](CompileConfig &config) {
    config.cache_loop_invariant_global_vars =
        !config.cache_loop_invariant_global_vars;
  });
  expect_distinct_key(cpu, [](CompileConfig &config) {
    config.make_cpu_multithreading_loop =
        !config.make_cpu_multithreading_loop;
  });
  expect_distinct_key(cpu, [](CompileConfig &config) {
    config.quant_opt_store_fusion = !config.quant_opt_store_fusion;
  });
  expect_distinct_key(cpu, [](CompileConfig &config) {
    config.quant_opt_atomic_demotion = !config.quant_opt_atomic_demotion;
  });

  CompileConfig spirv;
  spirv.arch = Arch::vulkan;
  spirv.max_block_dim = 1024;
  expect_distinct_key(spirv, [](CompileConfig &config) {
    config.spirv_skip_loop_unroll = !config.spirv_skip_loop_unroll;
  });
  expect_distinct_key(spirv,
                      [](CompileConfig &config) { config.max_block_dim /= 2; });
}

}  // namespace
}  // namespace taichi::lang
