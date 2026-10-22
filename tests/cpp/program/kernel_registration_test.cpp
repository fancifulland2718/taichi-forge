#include "gtest/gtest.h"

#include "taichi/runtime/kernel_execution_handle.h"
#include "tests/cpp/program/test_program.h"
#include "tests/cpp/ir/ndarray_kernel.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif
#ifdef TI_WITH_LLVM
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

namespace taichi::lang {

namespace {
void check_shared_execution_payload(Arch arch) {
  TestProgram test;
  test.setup(arch);
  auto *program = test.prog();
  auto kernel = setup_kernel1(program);
  auto original = program->compile_kernel_execution_handle(
      program->compile_config(), program->get_device_caps(), *kernel);
  auto payload = original->payload_lease();
  KernelExecutionHandle sibling(100, payload);
  ASSERT_EQ(&original->compiled(), &sibling.compiled());
  auto *array = program->create_ndarray(PrimitiveType::i32, {10});
  array->write_int({0}, 2);
  array->write_int({2}, 40);
  auto launch = [&](const KernelExecutionHandle &execution) {
    auto context = kernel->make_launch_context();
    context.set_arg_ndarray({0}, *array);
    program->launch_kernel(execution, context);
    program->synchronize();
  };
  launch(*original);
  ASSERT_EQ(array->read_int({2}), 42);
  launch(sibling);
  ASSERT_EQ(array->read_int({2}), 44);
  ASSERT_TRUE(original->registration().get_handle());
  ASSERT_TRUE(sibling.registration().get_handle());
  EXPECT_NE(original->registration().get_handle()->get_launch_id(),
            sibling.registration().get_handle()->get_launch_id());
#ifdef TI_WITH_CUDA
  if (arch == Arch::cuda) {
    auto &launcher =
        dynamic_cast<cuda::KernelLauncher &>(program->get_kernel_launcher());
    const auto &compiled =
        dynamic_cast<const LLVM::CompiledKernelData &>(*payload);
    auto masked = launcher.register_llvm_kernel_graph_gated(
        compiled, sibling.registration());
    EXPECT_NE(masked.get_launch_id(),
              sibling.registration().get_handle()->get_launch_id());
    EXPECT_EQ(masked.get_launch_id(), launcher
                                          .register_llvm_kernel_graph_gated(
                                              compiled, sibling.registration())
                                          .get_launch_id());
    EXPECT_FALSE(original->registration().get_graph_masked_handle());
  }
#endif
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    auto *executor =
        dynamic_cast<LlvmProgramImpl *>(program->get_program_impl())
            ->get_runtime_executor();
    cpu::KernelLauncher other(LLVM::KernelLauncher::Config{executor});
    KernelRegistrationState private_registration;
    const auto &compiled =
        dynamic_cast<const LLVM::CompiledKernelData &>(*payload);
    other.register_llvm_kernel(compiled, private_registration);
    EXPECT_ANY_THROW(
        other.register_llvm_kernel(compiled, sibling.registration()));
    other.retire_all_kernels();
    EXPECT_ANY_THROW(
        other.register_llvm_kernel(compiled, private_registration));
  }
#endif
  // Retire the old owner with work in flight (through the LLVM cache where
  // its internal test access is available). The launcher's command/module
  // ownership and the independent execution lease remain valid.
  auto pending = kernel->make_launch_context();
  pending.set_arg_ndarray({0}, *array);
  program->launch_kernel(sibling, pending);
  if (arch_uses_llvm(arch)) {
    program->get_program_impl()->get_kernel_compilation_manager().clear();
  } else {
    original->retire();
  }
  program->synchronize();
  EXPECT_EQ(array->read_int({2}), 46);
  EXPECT_FALSE(original->active());
  EXPECT_ANY_THROW(launch(*original));
  launch(sibling);
  EXPECT_EQ(array->read_int({2}), 48);
  EXPECT_TRUE(sibling.active());
  EXPECT_EQ(payload.get(), sibling.payload_lease().get());
  program->delete_ndarray(array);
}
}  // namespace

TEST(KernelRegistrationOwnership, CPUSharedPayload) {
  check_shared_execution_payload(Arch::x64);
}
#ifdef TI_WITH_CUDA
TEST(KernelRegistrationOwnership, CUDASharedPayload) {
  check_shared_execution_payload(Arch::cuda);
}
#endif
#ifdef TI_WITH_VULKAN
TEST(KernelRegistrationOwnership, VulkanSharedPayload) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  check_shared_execution_payload(Arch::vulkan);
}
#endif

#ifdef TI_WITH_LLVM
TEST(KernelRegistrationFailure, MissingSymbolDoesNotPublishContext) {
  TestProgram test;
  test.setup(Arch::x64);
  auto *program = test.prog();
  auto kernel = setup_kernel1(program);
  const auto &compiled =
      dynamic_cast<const LLVM::CompiledKernelData &>(program->compile_kernel(
          program->compile_config(), program->get_device_caps(), *kernel));
  auto *executor = dynamic_cast<LlvmProgramImpl *>(program->get_program_impl())
                       ->get_runtime_executor();
  cpu::KernelLauncher launcher(LLVM::KernelLauncher::Config{executor});
  auto broken_data = compiled.get_internal_data();
  ASSERT_FALSE(broken_data.compiled_data.tasks.empty());
  broken_data.compiled_data.tasks.front().name = "missing_registration_entry";
  LLVM::CompiledKernelData broken(Arch::x64, std::move(broken_data));
  KernelRegistrationState failed, valid;
  EXPECT_ANY_THROW(launcher.register_llvm_kernel(broken, failed));
  EXPECT_FALSE(failed.get_handle());
  EXPECT_EQ(launcher.debug_registered_kernel_count(), 0);
  launcher.register_llvm_kernel(compiled, valid);
  EXPECT_TRUE(valid.get_handle());
  EXPECT_EQ(launcher.debug_registered_kernel_count(), 1);
  launcher.retire_all_kernels();
}
#endif

}  // namespace taichi::lang
