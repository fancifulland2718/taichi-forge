#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/jit/jit_module.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi::lang {

TEST(LlvmRuntimeInitialization, OrdinaryWorkspaceMatchesRuntimeAndRetires) {
  CompileConfig config;
  config.arch = host_arch();
  LlvmRuntimeExecutor executor(config, nullptr);
  EXPECT_THROW(executor.get_runtime_temporary_ptr(), std::invalid_argument);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  auto *workspace = executor.get_runtime_temporary_ptr();
  ASSERT_NE(workspace, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(workspace) % taichi_page_size, 0);
  using Getter = void *(*)(void *);
  auto getter = reinterpret_cast<Getter>(
      executor.get_runtime_jit_module()->lookup_function("LLVMRuntime_get_temporaries"));
  ASSERT_NE(getter, nullptr);
  EXPECT_EQ(getter(executor.get_llvm_runtime()), workspace);
  executor.finalize();
  EXPECT_THROW(executor.get_runtime_temporary_ptr(), std::invalid_argument);
}

}  // namespace taichi::lang
#endif
