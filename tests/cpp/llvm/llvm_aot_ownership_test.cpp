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

TEST(LLVMAotOwnership, CompilerContractSurvivesPackagingAndIndependentCopies) {
  LLVM::CompiledKernelData::InternalData compiled;
  compiled.compiled_data.module_context_owner =
      std::make_shared<llvm::LLVMContext>();
  compiled.compiled_data.module = std::make_unique<llvm::Module>(
      "source", *compiled.compiled_data.module_context_owner);
  compiled.compiled_data.tasks.emplace_back("task", 64, 128);
  compiled.compiled_data.cuda_max_registers = 48;
  compiled.args.emplace_back(std::vector<int>{0, 1},
                             Callable::Parameter(PrimitiveType::i32, false));
  compiled.rets.emplace_back(PrimitiveType::f32);
  compiled.args_type = TypeFactory::get_instance()
                           .get_struct_type({{PrimitiveType::i32, "arg"}})
                           ->as<StructType>();
  compiled.ret_type = TypeFactory::get_instance()
                          .get_struct_type({{PrimitiveType::f32, "ret"}})
                          ->as<StructType>();
  compiled.args_size = compiled.ret_size = 4;
  compiled.used_snode_tree_ids = {2, 5};
  compiled.graph_metadata.available = true;
  compiled.graph_metadata.opaque = false;
  compiled.graph_metadata.blocker.clear();
  GraphKernelResourceEffect effect;
  effect.resource_kind = "snode";
  effect.snode_tree_id = 5;
  effect.access = "write";
  compiled.graph_metadata.effects.push_back(effect);
  auto *module = compiled.compiled_data.module.get();
  const auto *args_type = compiled.args_type;
  const auto *ret_type = compiled.ret_type;
  auto check = [&](const LLVM::KernelCallContract &contract) {
    ASSERT_EQ(contract.args.size(), 1);
    EXPECT_EQ(contract.args[0].first, (std::vector<int>{0, 1}));
    EXPECT_EQ(contract.args[0].second.get_dtype(), PrimitiveType::i32);
    ASSERT_EQ(contract.rets.size(), 1);
    EXPECT_EQ(contract.rets[0].dt, PrimitiveType::f32);
    EXPECT_EQ(contract.args_type, args_type);
    EXPECT_EQ(contract.ret_type, ret_type);
    EXPECT_EQ(contract.args_size, 4);
    EXPECT_EQ(contract.ret_size, 4);
    EXPECT_EQ(contract.used_snode_tree_ids, (std::vector<int>{2, 5}));
    EXPECT_TRUE(contract.graph_metadata.available);
    EXPECT_FALSE(contract.graph_metadata.opaque);
    EXPECT_TRUE(contract.graph_metadata.blocker.empty());
    ASSERT_EQ(contract.graph_metadata.effects.size(), 1);
    EXPECT_EQ(contract.graph_metadata.effects[0].snode_tree_id, 5);
    EXPECT_EQ(contract.graph_metadata.effects[0].access, "write");
  };

  // Packaging consumes compiler facts, with no frontend Kernel to consult.
  LlvmOfflineCache::KernelCacheData cache(std::move(compiled));
  check(cache);
  EXPECT_EQ(cache.compiled_data.module.get(), module);
  EXPECT_EQ(compiled.compiled_data.module, nullptr);
  auto metadata = cache.clone_metadata();
  check(metadata);
  EXPECT_EQ(metadata.compiled_data.module, nullptr);
  auto copied = cache.convert_to_llvm_ckd_data();
  check(copied);
  EXPECT_NE(copied.compiled_data.module.get(), module);
  EXPECT_EQ(cache.compiled_data.module.get(), module);
  // Deployment relocation and contract edits cannot alter the producer.
  copied.used_snode_tree_ids[0] = 7;
  copied.graph_metadata.effects[0].snode_tree_id = 9;
  copied.compiled_data.tasks[0].name = "relocated";
  check(cache);
  EXPECT_EQ(cache.compiled_data.tasks[0].name, "task");
  auto consumed = std::move(cache).take_llvm_ckd_data();
  check(consumed);
  EXPECT_EQ(consumed.compiled_data.module.get(), module);
  EXPECT_EQ(cache.compiled_data.module, nullptr);
  ASSERT_EQ(consumed.compiled_data.tasks.size(), 1);
  EXPECT_EQ(consumed.compiled_data.tasks[0].name, "task");
  EXPECT_EQ(consumed.compiled_data.cuda_max_registers, 48);
}

class LoadedEntryLauncher : public LLVM::KernelLauncher {
 public:
  LoadedEntryLauncher() : LLVM::KernelLauncher(Config{}) {
  }

  Handle register_llvm_kernel(const LLVM::CompiledKernelData &,
                              const KernelRegistrationState &) override {
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
