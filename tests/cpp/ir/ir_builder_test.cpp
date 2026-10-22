#include "gtest/gtest.h"

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
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
  const auto &compiled = dynamic_cast<const LLVM::CompiledKernelData &>(
      program->compile_kernel(program->compile_config(),
                              program->get_device_caps(), *kernel));
  auto *executor =
      dynamic_cast<LlvmProgramImpl *>(program->get_program_impl())
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

TEST(IRBuilder, Basic) {
  IRBuilder builder;
  auto *lhs = builder.get_int32(40);
  auto *rhs = builder.get_int32(2);
  auto *add = builder.create_add(lhs, rhs);
  ASSERT_TRUE(add->is<BinaryOpStmt>());
  auto *addc = add->cast<BinaryOpStmt>();
  EXPECT_EQ(addc->lhs, lhs);
  EXPECT_EQ(addc->rhs, rhs);
  EXPECT_EQ(addc->op_type, BinaryOpType::add);
  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  EXPECT_EQ(ir->as<Block>()->size(), 3);
}

TEST(IRBuilder, Print) {
  IRBuilder builder;
  auto *one = builder.get_int32(1);
  ASSERT_TRUE(one->is<ConstStmt>());
  std::string message = "message";
  auto *result = builder.create_print(one, message, one);
  ASSERT_TRUE(result->is<PrintStmt>());
  auto *print = result->cast<PrintStmt>();
  EXPECT_EQ(print->contents.size(), 3);
  ASSERT_TRUE(std::holds_alternative<Stmt *>(print->contents[0]));
  EXPECT_EQ(std::get<Stmt *>(print->contents[0]), one);
  ASSERT_TRUE(std::holds_alternative<std::string>(print->contents[1]));
  EXPECT_EQ(std::get<std::string>(print->contents[1]), message);
  ASSERT_TRUE(std::holds_alternative<Stmt *>(print->contents[2]));
  EXPECT_EQ(std::get<Stmt *>(print->contents[2]), one);
}

TEST(IRBuilder, RangeFor) {
  IRBuilder builder;
  auto *zero = builder.get_int32(0);
  auto *ten = builder.get_int32(10);
  auto *loop = builder.create_range_for(zero, ten);
  Stmt *index;
  {
    auto _ = builder.get_loop_guard(loop);
    index = builder.get_loop_index(loop, 0);
  }
  [[maybe_unused]] auto *ret = builder.create_return(zero);
  EXPECT_EQ(zero->parent->size(), 4);
  ASSERT_TRUE(loop->is<RangeForStmt>());
  auto *loopc = loop->cast<RangeForStmt>();
  EXPECT_EQ(loopc->body->size(), 1);
  EXPECT_EQ(loopc->body->statements[0].get(), index);
}

TEST(IRBuilder, LoopGuard) {
  IRBuilder builder;
  auto *zero = builder.get_int32(0);
  auto *ten = builder.get_int32(10);
  auto *loop = builder.create_range_for(zero, ten);
  Stmt *two;
  Stmt *one;
  Stmt *sum;
  {
    auto _ = builder.get_loop_guard(loop);
    one = builder.get_int32(1);
    builder.set_insertion_point_to_before(loop);
    two = builder.get_int32(2);
    builder.set_insertion_point_to_after(one);
    sum = builder.create_add(one, two);
  }
  // The insertion point should be after the loop now.
  auto *print = builder.create_print(two);
  EXPECT_EQ(zero->parent->size(), 5);
  EXPECT_EQ(zero->parent->statements[2].get(), two);
  EXPECT_EQ(zero->parent->statements[3].get(), loop);
  EXPECT_EQ(zero->parent->statements[4].get(), print);
  EXPECT_EQ(loop->body->size(), 2);
  EXPECT_EQ(loop->body->statements[0].get(), one);
  EXPECT_EQ(loop->body->statements[1].get(), sum);
}

TEST(IRBuilder, ExternalPtr) {
  TestProgram test_prog;
  test_prog.setup();

  IRBuilder builder;
  const int size = 10;
  auto array = std::make_unique<int[]>(size);
  array[0] = 2;
  array[2] = 40;
  auto *arg = builder.create_ndarray_arg_load(/*arg_id=*/{0},
                                              get_data_type<int>(), 1, 0);
  auto *zero = builder.get_int32(0);
  auto *one = builder.get_int32(1);
  auto *two = builder.get_int32(2);
  auto *a1ptr = builder.create_external_ptr(arg, {one});
  builder.create_global_store(a1ptr, one);  // a[1] = 1
  auto *a0 =
      builder.create_global_load(builder.create_external_ptr(arg, {zero}));
  auto *a2ptr = builder.create_external_ptr(arg, {two});
  auto *a2 = builder.create_global_load(a2ptr);
  auto *a0plusa2 = builder.create_add(a0, a2);
  builder.create_global_store(a2ptr, a0plusa2);  // a[2] = a[0] + a[2]
  auto block = builder.extract_ir();
  auto ker = std::make_unique<Kernel>(*test_prog.prog(), std::move(block));
  ker->insert_ndarray_param(get_data_type<int>(), /*total_dim=*/1);
  ker->finalize_params();
  auto launch_ctx = ker->make_launch_context();
  launch_ctx.set_arg_external_array_with_shape(
      /*arg_id=*/{0}, (uint64)array.get(), size, {size});
  auto *prog = test_prog.prog();
  const auto &compiled_kernel_data = prog->compile_kernel_execution_handle(
      prog->compile_config(), prog->get_device_caps(), *ker);
  prog->launch_kernel(*compiled_kernel_data, launch_ctx);
  EXPECT_EQ(array[0], 2);
  EXPECT_EQ(array[1], 1);
  EXPECT_EQ(array[2], 42);
}

TEST(IRBuilder, Ndarray) {
  TestProgram test_prog;
#ifdef TI_WITH_VULKAN
  Arch arch = taichi::lang::vulkan::is_vulkan_api_available() ? Arch::vulkan
                                                              : Arch::x64;
#else
  Arch arch = Arch::x64;
#endif
  test_prog.setup(arch);
  auto *prog = test_prog.prog();

  IRBuilder builder1;
  int size = 10;

  auto *array = prog->create_ndarray(PrimitiveType::i32, {size});
  array->write_int({0}, 2);
  array->write_int({2}, 40);
  auto ker1 = setup_kernel1(test_prog.prog());
  auto launch_ctx1 = ker1->make_launch_context();
  launch_ctx1.set_arg_ndarray(/*arg_id=*/{0}, *array);
  const auto &compiled_kernel_data = prog->compile_kernel_execution_handle(
      prog->compile_config(), prog->get_device_caps(), *ker1);
  prog->launch_kernel(*compiled_kernel_data, launch_ctx1);
  EXPECT_EQ(array->read_int({0}), 2);
  EXPECT_EQ(array->read_int({1}), 1);
  EXPECT_EQ(array->read_int({2}), 42);

  auto ker2 = setup_kernel2(test_prog.prog());
  auto launch_ctx2 = ker2->make_launch_context();
  launch_ctx2.set_arg_ndarray(/*arg_id=*/{0}, *array);
  launch_ctx2.set_arg_int(/*arg_id=*/{1}, 3);
  const auto &compiled_kernel_data2 = prog->compile_kernel_execution_handle(
      prog->compile_config(), prog->get_device_caps(), *ker2);
  prog->launch_kernel(*compiled_kernel_data2, launch_ctx2);
  EXPECT_EQ(array->read_int({0}), 2);
  EXPECT_EQ(array->read_int({1}), 3);
  EXPECT_EQ(array->read_int({2}), 42);
  prog->delete_ndarray(array);
}

TEST(IRBuilder, AtomicOp) {
  TestProgram test_prog;
  test_prog.setup();

  IRBuilder builder;
  const int size = 10;
  auto array = std::make_unique<int[]>(size);
  array[0] = 2;
  array[2] = 40;
  auto *arg = builder.create_ndarray_arg_load(/*arg_id=*/{0},
                                              get_data_type<int>(), 1, 0);
  auto *zero = builder.get_int32(0);
  auto *one = builder.get_int32(1);
  auto *a0ptr = builder.create_external_ptr(arg, {zero});
  builder.create_atomic_add(a0ptr, one);  // a[0] += 1
  auto block = builder.extract_ir();
  auto ker = std::make_unique<Kernel>(*test_prog.prog(), std::move(block));
  ker->insert_ndarray_param(get_data_type<int>(), /*total_dim=*/1);
  ker->finalize_params();
  auto launch_ctx = ker->make_launch_context();
  launch_ctx.set_arg_external_array_with_shape(
      /*arg_id=*/{0}, (uint64)array.get(), size, {size});
  auto *prog = test_prog.prog();
  const auto &compiled_kernel_data = prog->compile_kernel_execution_handle(
      prog->compile_config(), prog->get_device_caps(), *ker);
  prog->launch_kernel(*compiled_kernel_data, launch_ctx);
  EXPECT_EQ(array[0], 3);
}
}  // namespace taichi::lang
