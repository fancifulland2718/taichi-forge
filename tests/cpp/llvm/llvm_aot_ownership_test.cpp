#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/aot_graph_data.h"

namespace taichi::lang {
namespace {

TEST(LLVMCompiledKernelOwnership, MoveAssignmentRetiresModuleBeforeItsContext) {
  auto make_data = [](const std::string &name) {
    LLVMCompiledKernel result;
    result.module_context_owner = std::make_shared<llvm::LLVMContext>();
    result.module = std::make_unique<llvm::Module>(
        name, *result.module_context_owner);
    result.tasks.emplace_back(name, 32, 64);
    result.cuda_max_registers = 48;
    return result;
  };
  auto destination = make_data("old");
  auto source = make_data("replacement");
  std::weak_ptr<llvm::LLVMContext> old_context =
      destination.module_context_owner;
  std::weak_ptr<llvm::LLVMContext> new_context = source.module_context_owner;
  const auto *module = source.module.get();
  destination = std::move(source);
  EXPECT_TRUE(old_context.expired());
  EXPECT_FALSE(new_context.expired());
  EXPECT_EQ(destination.module.get(), module);
  EXPECT_EQ(source.module, nullptr);
  EXPECT_EQ(destination.cuda_max_registers, 48);
  ASSERT_EQ(destination.tasks.size(), 1);
  EXPECT_EQ(destination.tasks.front().name, "replacement");
  auto *self = &destination;
  destination = std::move(*self);
  EXPECT_EQ(destination.module.get(), module);
  EXPECT_FALSE(new_context.expired());
  destination = LLVMCompiledKernel{};
  EXPECT_EQ(destination.module, nullptr);
  EXPECT_TRUE(new_context.expired());
}

class LoadedEntryLauncher : public LLVM::KernelLauncher {
 public:
  LoadedEntryLauncher() : LLVM::KernelLauncher(Config{}) {
  }

  Handle register_llvm_kernel(const LLVM::CompiledKernelData &) override {
    ++registrations;
    return make_handle();
  }

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override {
    ++launches;
    launched_id = handle.get_launch_id();
    launched_context = &ctx;
  }

  int registrations{0};
  int launches{0};
  int launched_id{-1};
  LaunchContextBuilder *launched_context{nullptr};
};

TEST(LLVMAotOwnership, TransfersModuleAndRetainsOnlyCallableAbi) {
  LoadedEntryLauncher launcher;
  KernelLaunchHandle handle;
  handle.set_launch_id(7);
  LlvmOfflineCache::KernelCacheData cache;
  auto context = std::make_shared<llvm::LLVMContext>();
  std::weak_ptr<llvm::LLVMContext> context_lifetime = context;
  cache.compiled_data.module =
      std::make_unique<llvm::Module>("kernel", *context);
  cache.compiled_data.module_context_owner = std::move(context);
  const auto *module_identity = cache.compiled_data.module.get();
  cache.compiled_data.tasks.emplace_back("run", 64, 128);
  cache.compiled_data.cuda_max_registers = 32;
  cache.args.emplace_back(std::vector<int>{0},
                          Callable::Parameter(PrimitiveType::i32, false));
  cache.rets.emplace_back(PrimitiveType::f32);
  cache.used_snode_tree_ids = {2, 5};
  auto *args_type = TypeFactory::get_instance()
                        .get_struct_type({{PrimitiveType::i32, "value"}})
                        ->as<StructType>();
  auto *ret_type = TypeFactory::get_instance()
                       .get_struct_type({{PrimitiveType::f32, "result"}})
                       ->as<StructType>();
  cache.args_type = args_type;
  cache.args_size = 4;
  cache.ret_type = ret_type;
  cache.ret_size = 4;
  std::unique_ptr<llvm_aot::KernelImpl> callable;
  {
    auto data = std::move(cache).take_llvm_ckd_data();
    EXPECT_EQ(data.compiled_data.module.get(), module_identity);
    EXPECT_EQ(cache.compiled_data.module, nullptr);
    EXPECT_FALSE(context_lifetime.expired());
    ASSERT_EQ(data.compiled_data.tasks.size(), 1);
    EXPECT_EQ(data.compiled_data.tasks[0].name, "run");
    EXPECT_EQ(data.compiled_data.cuda_max_registers, 32);
    EXPECT_EQ(data.used_snode_tree_ids, (std::vector<int>{2, 5}));
    callable = std::make_unique<llvm_aot::KernelImpl>(
        &launcher, handle, Arch::cuda, "exported_kernel", data);
  }
  // The callable has copied its ABI; it no longer pins the IR module/context.
  EXPECT_TRUE(context_lifetime.expired());
  EXPECT_EQ(callable->name, "exported_kernel");
  EXPECT_EQ(callable->nested_parameters.at({0}).get_dtype(),
            PrimitiveType::i32);
  ASSERT_EQ(callable->rets.size(), 1);
  EXPECT_EQ(callable->rets[0].dt, PrimitiveType::f32);
  EXPECT_EQ(callable->args_type, args_type);
  EXPECT_EQ(callable->ret_type, ret_type);
  EXPECT_EQ(callable->args_size, 4);
  EXPECT_EQ(callable->ret_size, 4);
  EXPECT_EQ(callable->arch, Arch::cuda);
  EXPECT_EQ(&callable->launcher(), &launcher);
  EXPECT_EQ(callable->handle().get_launch_id(), 7);

  LaunchContextBuilder context_builder(callable.get());
  callable->launch(context_builder);
  callable->launch(context_builder);
  EXPECT_EQ(launcher.registrations, 0);
  EXPECT_EQ(launcher.launches, 2);
  EXPECT_EQ(launcher.launched_id, 7);
  EXPECT_EQ(launcher.launched_context, &context_builder);
}

TEST(LLVMAotOwnership, LoadedEntryRequiresLauncherAndRegistration) {
  LoadedEntryLauncher launcher;
  LLVM::CompiledKernelData::InternalData data;
  KernelLaunchHandle handle;
  EXPECT_ANY_THROW(
      llvm_aot::KernelImpl(&launcher, handle, Arch::x64, "invalid", data));
  handle.set_launch_id(0);
  EXPECT_ANY_THROW(
      llvm_aot::KernelImpl(nullptr, handle, Arch::x64, "invalid", data));
  EXPECT_ANY_THROW(
      llvm_aot::KernelImpl(&launcher, handle, Arch::vulkan, "invalid", data));
}

}  // namespace
}  // namespace taichi::lang
#endif
