#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "tests/cpp/aot/value_parameter_check.h"

#ifdef TI_WITH_CUDA

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

#endif

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {

static void run_return_tests(aot::Module *mod,
                             LlvmRuntimeExecutor *exec,
                             uint64 *result_buffer) {
  aot::Kernel *k_ret = mod->get_kernel("test_ret");
  EXPECT_FALSE(k_ret->value_parameter_types.has_value());

  LaunchContextBuilder builder(k_ret);
  k_ret->launch(builder);
  exec->synchronize();
  EXPECT_EQ(builder.get_struct_ret_int({0, 0}), 1);
  EXPECT_EQ(builder.get_struct_ret_float({0, 1, 0}), 2);
  EXPECT_EQ(builder.get_struct_ret_float({0, 1, 1}), 3);
  EXPECT_EQ(builder.get_struct_ret_float({0, 1, 2}), 4);

  LaunchContextBuilder discarded(k_ret);
  discarded.discard_return_values();
  auto *host_result =
      reinterpret_cast<unsigned char *>(discarded.get_context().result_buffer);
  std::fill_n(host_result, discarded.result_buffer_size, 0xa5);
  k_ret->launch(discarded);
  exec->synchronize();
  if (exec->get_config().arch == Arch::cuda) {
    // Device results still have their compiled storage, but no asynchronous
    // copy may target the caller's discarded host buffer.
    for (std::size_t i = 0; i < discarded.result_buffer_size; ++i)
      EXPECT_EQ(host_result[i], 0xa5);
  }
  EXPECT_ANY_THROW(discarded.get_struct_ret_int({0, 0}));
  EXPECT_ANY_THROW(discarded.get_ret<int64>(0));
  aot_test_utils::check_value_parameters(mod, [&] { exec->synchronize(); });
  // Check assertion error from ti.kernel
  exec->check_runtime_error(result_buffer);
}

TEST(LlvmAotTest, CpuReturn) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmRuntimeExecutor exec{cfg, kNoProfiler};

  // Must have handled all the arch fallback logic by this point.
  uint64 *result_buffer{nullptr};
  exec.materialize_runtime(kNoProfiler, &result_buffer);

  LLVM::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  aot_params.executor_ = &exec;
  aot_params.kernel_launcher =
      std::make_unique<cpu::KernelLauncher>(cpu::KernelLauncher::Config{&exec});
  std::unique_ptr<aot::Module> mod =
      LLVM::make_aot_module(std::move(aot_params));

  run_return_tests(mod.get(), &exec, result_buffer);
}

TEST(LlvmAotTest, CudaReturn) {
#ifdef TI_WITH_CUDA
  if (is_cuda_api_available()) {
    CompileConfig cfg;
    cfg.arch = Arch::cuda;
    cfg.kernel_profiler = false;
    constexpr KernelProfilerBase *kNoProfiler = nullptr;
    LlvmRuntimeExecutor exec{cfg, kNoProfiler};

    // Must have handled all the arch fallback logic by this point.
    uint64 *result_buffer{nullptr};
    exec.materialize_runtime(kNoProfiler, &result_buffer);

    LLVM::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.executor_ = &exec;
    aot_params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
        cuda::KernelLauncher::Config{&exec});
    auto mod = LLVM::make_aot_module(std::move(aot_params));

    run_return_tests(mod.get(), &exec, result_buffer);
  }
#endif
}

}  // namespace taichi::lang
