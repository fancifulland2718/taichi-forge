#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/aot_graph_data.h"

namespace taichi::lang {
namespace {

TEST(LLVMAotOwnership, TransfersModuleAndRetainsOnlyCallableAbi) {
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
        [](LaunchContextBuilder &) {}, "exported_kernel", data);
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
}

}  // namespace
}  // namespace taichi::lang
#endif
