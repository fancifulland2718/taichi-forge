#include "gtest/gtest.h"

#if defined(TI_WITH_CUDA)
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/cuda/cuda_artifact_provider.h"
#include "taichi/runtime/cuda/runtime_module.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include <future>
#include <cstdlib>
#include <fstream>
#include <set>

namespace taichi::lang {
namespace {

llvm::Function *entry(llvm::Module &module, const char *name) {
  auto &context = module.getContext();
  auto *function = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), {}, false),
      llvm::GlobalValue::ExternalLinkage, name, module);
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(context, "body", function));
  builder.CreateRetVoid();
  module.getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDNode::get(
          context, {llvm::ValueAsMetadata::get(function),
                    llvm::MDString::get(context, "kernel"),
                    llvm::ConstantAsMetadata::get(builder.getInt32(1))}));
  return function;
}

std::vector<std::string> kernels(const llvm::Module &module) {
  std::vector<std::string> result;
  for (auto *node : module.getNamedMetadata("nvvm.annotations")->operands()) {
    auto *value = llvm::cast<llvm::ValueAsMetadata>(node->getOperand(0));
    auto *key = llvm::cast<llvm::MDString>(node->getOperand(1));
    if (key->getString() == "kernel")
      result.push_back(value->getValue()->getName().str());
  }
  return result;
}

}  // namespace

TEST(CUDARuntimeModule, RetainsTransitiveCallsWithoutExportingOtherGroups) {
  llvm::LLVMContext context;
  llvm::Module module("runtime_groups", context);
  auto *core = entry(module, "runtime_initialize");
  auto *sparse = entry(module, "runtime_initialize_snode_element_list");
  entry(module, "runtime_sparse_tree_statistics_reset");
  // The core entry uses a function which also has its own deferred entry.
  // It must remain as a device function in core, not a second kernel export.
  llvm::IRBuilder<> builder(core->getEntryBlock().getTerminator());
  builder.CreateCall(sparse);
  ASSERT_TRUE(cuda_runtime_can_partition(module));
  auto core_module = llvm::CloneModule(module);
  retain_cuda_runtime_group(*core_module, CUDARuntimeGroup::core);
  EXPECT_EQ(kernels(*core_module),
            std::vector<std::string>{"runtime_initialize"});
  auto *dependency =
      core_module->getFunction("runtime_initialize_snode_element_list");
  ASSERT_NE(dependency, nullptr);
  EXPECT_TRUE(dependency->hasLocalLinkage());
  EXPECT_EQ(core_module->getFunction("runtime_sparse_tree_statistics_reset"),
            nullptr);
  EXPECT_FALSE(llvm::verifyModule(*core_module));

  auto sparse_module = llvm::CloneModule(module);
  retain_cuda_runtime_group(*sparse_module, CUDARuntimeGroup::sparse);
  EXPECT_EQ(kernels(*sparse_module),
            std::vector<std::string>{"runtime_initialize_snode_element_list"});
  EXPECT_EQ(sparse_module->getFunction("runtime_initialize"), nullptr);
  EXPECT_FALSE(llvm::verifyModule(*sparse_module));
  // Planning never mutates the source used for the other groups.
  EXPECT_EQ(kernels(module).size(), 3u);
}

TEST(CUDARuntimeModule, PreservesUnknownEntriesInCore) {
  llvm::LLVMContext context;
  llvm::Module module("future_runtime", context);
  entry(module, "runtime_future_operation");
  entry(module, "runtime_sparse_snode_statistics_collect");
  retain_cuda_runtime_group(module, CUDARuntimeGroup::core);
  EXPECT_EQ(kernels(module),
            std::vector<std::string>{"runtime_future_operation"});
  EXPECT_EQ(cuda_runtime_group("runtime_retrieve_and_reset_error_code"),
            CUDARuntimeGroup::core);
  EXPECT_EQ(cuda_runtime_group("runtime_unregister_snode_tree_state"),
            CUDARuntimeGroup::core);
}

TEST(CUDARuntimeModule, KeepsSharedStateAndSymbolIdentityMonolithic) {
  llvm::LLVMContext context;
  llvm::Module module("shared_state", context);
  entry(module, "runtime_initialize");
  auto *global = new llvm::GlobalVariable(
      module, llvm::Type::getInt32Ty(context), false,
      llvm::GlobalValue::InternalLinkage,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0), "state");
  EXPECT_FALSE(cuda_runtime_can_partition(module));
  global->setConstant(true);
  EXPECT_FALSE(cuda_runtime_can_partition(module));
  global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  EXPECT_TRUE(cuda_runtime_can_partition(module));
  global->eraseFromParent();
  auto *function = module.getFunction("runtime_initialize");
  auto *pointer = new llvm::GlobalVariable(module, function->getType(), true,
                                           llvm::GlobalValue::InternalLinkage,
                                           function, "callback");
  pointer->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  EXPECT_FALSE(cuda_runtime_can_partition(module));
}

TEST(CUDARuntimeModule, SeparateContextsAndModulesShareExplicitDeviceState) {
  if (!is_cuda_api_available() ||
      !CUDADriver::get_instance().nvidia_extensions_available())
    GTEST_SKIP();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  auto &cuda = CUDAContext::get_instance();
  auto context_guard = cuda.get_guard();
  auto &driver = CUDADriver::get_instance();
  const CUDACodegenTarget target{cuda.get_mcpu(), cuda.get_mattrs()};
  CompileConfig config;
  config.arch = Arch::cuda;
  config.llvm_opt_level = 3;
  std::string bitcode;
  {
    // Destroy the originating LLVMContext before either deferred group is
    // prepared. Each compiled module below owns a fresh, local context.
    llvm::LLVMContext context;
    llvm::Module module("runtime_state_fixture", context);
    module.setTargetTriple("nvptx64-nvidia-cuda");
    const char *names[] = {"runtime_initialize",
                           "runtime_NodeAllocator_initialize_ex",
                           "runtime_sparse_tree_statistics_reset"};
    for (int index = 0; index < 3; ++index) {
      auto *function = llvm::Function::Create(
          llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                  {llvm::PointerType::get(context, 1)}, false),
          llvm::GlobalValue::ExternalLinkage, names[index], module);
      llvm::IRBuilder<> builder(
          llvm::BasicBlock::Create(context, "body", function));
      auto *previous =
          builder.CreateLoad(builder.getInt32Ty(), function->getArg(0));
      auto *value = index == 0 ? static_cast<llvm::Value *>(builder.getInt32(7))
                    : index == 1
                        ? builder.CreateAdd(previous, builder.getInt32(5))
                        : builder.CreateMul(previous, builder.getInt32(2));
      builder.CreateStore(value, function->getArg(0));
      builder.CreateRetVoid();
      module.getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvm::MDNode::get(
              context, {llvm::ValueAsMetadata::get(function),
                        llvm::MDString::get(context, "kernel"),
                        llvm::ConstantAsMetadata::get(builder.getInt32(1))}));
    }
    ASSERT_TRUE(cuda_runtime_can_partition(module));
    llvm::raw_string_ostream output(bitcode);
    llvm::WriteBitcodeToFile(module, output);
  }
  JITSessionCUDA session(nullptr, config, llvm::DataLayout(""));
  auto compile_group = [&](CUDARuntimeGroup group,
                           const CUDACodegenTarget &selected) {
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(bitcode, "runtime"), context);
    EXPECT_TRUE(bool(parsed));
    auto module = std::move(*parsed);
    retain_cuda_runtime_group(*module, group);
    auto artifact = compile_cuda_kernel_artifact(module, config, selected, 0,
                                                 JITModuleRole::runtime);
    EXPECT_EQ(artifact.entry_names.size(), 1u);
    return session.add_artifact(artifact);
  };
  auto *core = compile_group(CUDARuntimeGroup::core, target);
  // A failed preparation has no driver artifact; the same immutable snapshot
  // remains usable for a later attempt, including on a different host thread.
  EXPECT_ANY_THROW(
      compile_group(CUDARuntimeGroup::sparse, {"sm_invalid", target.mattrs}));
  auto *sparse = std::async(std::launch::async, [&] {
                   return compile_group(CUDARuntimeGroup::sparse, target);
                 }).get();
  auto *statistics = compile_group(CUDARuntimeGroup::statistics, target);
  void *state = nullptr;
  driver.malloc(&state, sizeof(int));
  auto release = [&](void *ptr) { driver.mem_free(ptr); };
  std::unique_ptr<void, decltype(release)> allocation(state, release);
  core->call<void *>("runtime_initialize", state);
  sparse->call<void *>("runtime_NodeAllocator_initialize_ex", state);
  statistics->call<void *>("runtime_sparse_tree_statistics_reset", state);
  int result = 0;
  driver.memcpy_device_to_host(&result, state, sizeof(result));
  EXPECT_EQ(result, 24);
  // Only ordinary session-owned modules are involved; no executor or runtime
  // proxy is installed by this independent compiler/driver contract test.
  EXPECT_TRUE(session.remove_module(statistics));
  EXPECT_TRUE(session.remove_module(sparse));
  EXPECT_TRUE(session.remove_module(core));
}

TEST(CUDARuntimeModule, SessionDefersGroupsAndRetriesWithIndependentContexts) {
  if (!is_cuda_api_available() ||
      !CUDADriver::get_instance().stream_get_capture_info_v2.available())
    GTEST_SKIP();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  auto &cuda_context = CUDAContext::get_instance();
  auto guard = cuda_context.get_guard();
  auto &driver = CUDADriver::get_instance();
  CompileConfig config;
  config.arch = Arch::cuda;
  JITSessionCUDA session(nullptr, config, llvm::DataLayout(""));
  const auto loaded = [] {
    return cuda::get_cuda_artifact_provider_telemetry_snapshot()
        .entry_points_loaded;
  };
  const auto before = loaded();
  JITModule *runtime = nullptr;
  {
    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("deferred_runtime", context);
    module->setTargetTriple("nvptx64-nvidia-cuda");
    const char *names[] = {"runtime_initialize",
                           "runtime_NodeAllocator_initialize_ex",
                           "runtime_sparse_tree_statistics_reset"};
    for (int index = 0; index < 3; ++index) {
      auto *function = llvm::Function::Create(
          llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                  {llvm::PointerType::get(context, 1)}, false),
          llvm::GlobalValue::ExternalLinkage, names[index], *module);
      llvm::IRBuilder<> builder(
          llvm::BasicBlock::Create(context, "body", function));
      auto *old = builder.CreateLoad(builder.getInt32Ty(), function->getArg(0));
      auto *value = index == 0 ? static_cast<llvm::Value *>(builder.getInt32(7))
                    : index == 1 ? builder.CreateAdd(old, builder.getInt32(5))
                                 : builder.CreateMul(old, builder.getInt32(2));
      builder.CreateStore(value, function->getArg(0));
      builder.CreateRetVoid();
      module->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvm::MDNode::get(
              context, {llvm::ValueAsMetadata::get(function),
                        llvm::MDString::get(context, "kernel"),
                        llvm::ConstantAsMetadata::get(builder.getInt32(1))}));
    }
    runtime = session.add_module(std::move(module), 0, JITModuleRole::runtime);
  }
  ASSERT_EQ(loaded() - before, 1u);
  EXPECT_ANY_THROW(runtime->lookup_function("runtime_missing"));
  EXPECT_EQ(loaded() - before, 1u);
  void *state = nullptr;
  driver.malloc(&state, sizeof(int));
  auto release = [&](void *ptr) { driver.mem_free(ptr); };
  std::unique_ptr<void, decltype(release)> allocation(state, release);
  runtime->call<void *>("runtime_initialize", state);
  // Force the existing provider identity check to fail during a deferred
  // preparation. Restoring the original environment must allow retry.
  const char *key = "TI_CUDA_PTXAS_MODE";
  const char *env = std::getenv(key);
  const bool had_env = env != nullptr;
  const std::string previous = env ? env : "";
  const auto set_env = [&](const char *value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value)
      setenv(key, value, 1);
    else
      unsetenv(key);
#endif
  };
  set_env(previous == "external" ? "driver" : "external");
  EXPECT_ANY_THROW(
      runtime->lookup_function("runtime_NodeAllocator_initialize_ex"));
  set_env(had_env ? previous.c_str() : nullptr);
  EXPECT_EQ(loaded() - before, 1u);
  auto *sparse = std::async(std::launch::async, [&] {
                   return runtime->lookup_function(
                       "runtime_NodeAllocator_initialize_ex");
                 }).get();
  EXPECT_NE(sparse, nullptr);
  EXPECT_EQ(loaded() - before, 2u);
  EXPECT_EQ(runtime->lookup_function("runtime_NodeAllocator_initialize_ex"),
            sparse);
  runtime->call<void *>("runtime_NodeAllocator_initialize_ex", state);
  runtime->call<void *>("runtime_sparse_tree_statistics_reset", state);
  EXPECT_EQ(loaded() - before, 3u);
  int result = 0;
  driver.memcpy_device_to_host(&result, state, sizeof(result));
  EXPECT_EQ(result, 24);
  // All prepared physical modules retire with their existing session owner.
}

TEST(CUDARuntimeModule, ActualRuntimePartitionsCoverEveryEntry) {
  const char *output = std::getenv("TI_CUDA_RUNTIME_PARTITION_TEST_OUTPUT");
  if (!output)
    GTEST_SKIP() << "Set TI_LIB_DIR and TI_CUDA_RUNTIME_PARTITION_TEST_OUTPUT "
                    "for the built runtime fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  TaichiLLVMContext context(config, Arch::cuda);
  auto module = context.clone_runtime_module();
  context.init_runtime_module(module.get());
  ASSERT_TRUE(cuda_runtime_can_partition(*module));
  const auto expected = kernels(*module);
  std::set<std::string> covered;
  const CUDACodegenTarget target{"sm_80", "+ptx70"};
  std::size_t grouped_bytes = 0;
  for (auto group : {CUDARuntimeGroup::core, CUDARuntimeGroup::sparse,
                     CUDARuntimeGroup::statistics}) {
    auto part = llvm::CloneModule(*module);
    retain_cuda_runtime_group(*part, group);
    for (const auto &name : kernels(*part))
      EXPECT_TRUE(covered.insert(name).second) << name;
    EXPECT_FALSE(llvm::verifyModule(*part));
    auto artifact = compile_cuda_kernel_artifact(part, config, target, 0,
                                                 JITModuleRole::runtime);
    std::ofstream file(
        std::string(output) + "/" + cuda_runtime_group_name(group) + ".ptx",
        std::ios::binary);
    file.write(artifact.payload.data(), artifact.code_size());
    ASSERT_TRUE(file.good());
    grouped_bytes += artifact.code_size();
    TI_INFO("Runtime group {}: {} entries, {} bytes",
            cuda_runtime_group_name(group), artifact.entry_names.size(),
            artifact.code_size());
  }
  EXPECT_EQ(covered, (std::set<std::string>{expected.begin(), expected.end()}));
  auto artifact = compile_cuda_kernel_artifact(module, config, target, 0,
                                               JITModuleRole::runtime);
  std::ofstream file(std::string(output) + "/full.ptx", std::ios::binary);
  file.write(artifact.payload.data(), artifact.code_size());
  ASSERT_TRUE(file.good());
  TI_INFO("Runtime all: {} entries, {} bytes; grouped total {} bytes",
          artifact.entry_names.size(), artifact.code_size(), grouped_bytes);
}

}  // namespace taichi::lang
#endif
