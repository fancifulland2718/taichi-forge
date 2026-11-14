#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>

#include "llvm/IR/IRBuilder.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/compilation_manager/kernel_compilation_manager.h"
#include "taichi/program/program.h"
#include "taichi/runtime/llvm/array_layout_abi.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/llvm_aot_metadata.h"
#include "taichi/runtime/llvm/llvm_aot_module_builder.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"

namespace taichi::lang {
namespace {

// No backend compiler or device is needed: count entry into the existing
// compilation manager and provide a valid, small LLVM payload with contracts.
class ExportTestCompiler : public KernelCompiler {
 public:
  IRNodePtr compile(const KernelCompileRequest &, const Kernel &,
                    GraphKernelMetadata *metadata) const override {
    metadata->available = true;
    metadata->opaque = false;
    GraphKernelResourceEffect effect;
    effect.resource_kind = "argument";
    effect.arg_id = {0};
    effect.access = "write";
    metadata->effects.push_back(std::move(effect));
    return std::make_unique<Block>();
  }

  CKDPtr compile(const KernelCompileRequest &request, const Kernel &,
                 IRNode &) const override {
    ++calls;
    LLVM::CompiledKernelData::InternalData data;
    auto context = std::make_shared<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("export_test", *context);
    auto *function = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context), false),
        llvm::GlobalValue::ExternalLinkage, "task", *module);
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(*context, "", function));
    builder.CreateRetVoid();
    LLVM::declare_runtime_state_use(*module, runtime_state_known | runtime_state_random);
    LLVM::declare_runtime_affine_arrays(*module, {{0}});
    module->addModuleFlag(llvm::Module::Error, LLVM::kSparseRuntimeViewABIFlag,
                          LLVM::kSparseRuntimeViewABI);
    data.compiled_data.module_context_owner = std::move(context);
    data.compiled_data.module = std::move(module);
    data.compiled_data.tasks.emplace_back("task", 1, 1);
    return std::make_unique<LLVM::CompiledKernelData>(request.config().arch,
                                                    std::move(data));
  }

  mutable int calls{0};
};

class LlvmAotExportReuseTest : public testing::Test {
 protected:
  void SetUp() override {
    path = std::filesystem::temp_directory_path() /
           ("taichi-export-reuse-" + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }
  void TearDown() override {
    std::filesystem::remove_all(path);
  }
  std::filesystem::path path;
};

TEST_F(LlvmAotExportReuseTest, AliasSkipsReleasedManagerAndPreservesContracts) {
  Program program(Arch::x64);
  CompileConfig config;
  config.arch = Arch::x64;
  config.offline_cache = true;
  auto compiler = std::make_unique<ExportTestCompiler>();
  auto *observed = compiler.get();
  KernelCompilationManager manager({path.u8string(), std::move(compiler)});
  manager.set_executable_lifecycle_telemetry_enabled(true);
  Kernel first(program, [] {}, "export_first");
  Kernel different(program, [] {}, "export_different");
  different.set_compile_tier_override("fast");
  LlvmAotModuleBuilder builder(manager, config, nullptr, {});
  builder.set_ir_cleanup_for_testing(true);
  builder.add("first", &first);
  builder.seal_kernel_export("first");
  manager.release_build_cache();
  EXPECT_EQ(manager.executable_lifecycle_statistics(false).resident_templates, 0);
  const auto before = manager.executable_lifecycle_statistics(false);
  builder.add("alias", &first);
  EXPECT_EQ(observed->calls, 1);
  const auto after = manager.executable_lifecycle_statistics(false);
  EXPECT_EQ(after.disk_loads, before.disk_loads);
  EXPECT_EQ(after.memory_cache_hits, before.memory_cache_hits);
  EXPECT_EQ(after.resident_templates, 0);
  EXPECT_EQ(builder.kernel_resource_effects("alias").effects.at(0).arg_id,
            std::vector<int>{0});
  EXPECT_EQ(builder.kernel_resource_effects("alias").effects.at(0).access, "write");
  EXPECT_ANY_THROW(builder.add("first", &first));

  const auto read_metadata = [&](const std::string &directory) {
    LLVM::LlvmAotMetadata result;
    const auto text = LLVM::read_aot_artifact(
        (path / directory / LLVM::kLlvmAotMetadataFilename).u8string());
    liong::json::deserialize(liong::json::parse(text), result, true);
    return result;
  };
  builder.dump((path / "first_save").u8string(), "");
  builder.dump((path / "repeat_save").u8string(), "");
  const auto metadata = read_metadata("first_save");
  EXPECT_EQ(metadata.kernel_aliases.at("alias"), "first");
  EXPECT_EQ(metadata.kernel_runtime_state.at("first"),
            runtime_state_known | runtime_state_random);
  EXPECT_EQ(metadata.kernel_runtime_affine_arrays.at("first"),
            (std::vector<std::vector<int>>{{0}}));
  EXPECT_EQ(metadata.kernel_sparse_runtime_view_abis.at("first"),
            LLVM::kSparseRuntimeViewABI);
  EXPECT_EQ(liong::json::print(liong::json::serialize(metadata)),
            liong::json::print(liong::json::serialize(read_metadata("repeat_save"))));
  builder.add("different", &different);
  EXPECT_EQ(observed->calls, 2);
  builder.dump((path / "later_save").u8string(), "");
  EXPECT_EQ(read_metadata("later_save").kernel_runtime_state.size(), 2);
}

TEST_F(LlvmAotExportReuseTest, ReleasedMetadataSurvivesReloadAndFinalFlush) {
  Program program(Arch::x64);
  CompileConfig config;
  config.arch = Arch::x64;
  config.offline_cache = true;
  const auto request = resolve_kernel_compile_request(config, {});
  Kernel first(program, [] {}, "cache_first");
  Kernel second(program, [] {}, "cache_second");
  second.parameter_list.emplace_back(PrimitiveType::i32);
  auto compiler = std::make_unique<ExportTestCompiler>();
  auto *observed = compiler.get();
  KernelCompilationManager manager({path.u8string(), std::move(compiler)});
  manager.set_executable_lifecycle_telemetry_enabled(true);
  const auto key = manager.load_or_compile(request, first).kernel_identity();
  manager.release_build_cache();
  EXPECT_EQ(manager.find_cached_execution_handle(key, first, Arch::x64, true), nullptr);
  manager.load_or_compile(request, second);
  manager.release_build_cache();
  manager.release_build_cache();  // No residents: preserve pending metadata.
  manager.load_or_compile(request, first);
  EXPECT_EQ(observed->calls, 2);
  EXPECT_EQ(manager.executable_lifecycle_statistics(false).disk_loads, 1);
  manager.release_build_cache();
  manager.dump();
  CacheData index;
  const auto filename = fmt::format(KernelCompilationManager::kMetadataFilenameFormat,
                                     "x64", kOfflineCacheSchemaVersion);
  ASSERT_EQ(offline_cache::load_metadata_with_checking(index, (path / filename).u8string()),
            offline_cache::LoadMetadataError::kNoError);
  EXPECT_EQ(index.kernels.size(), 2);
  std::size_t total = 0;
  for (const auto &[_, kernel] : index.kernels)
    total += kernel.size;
  EXPECT_EQ(index.size, total);
  EXPECT_EQ(manager.executable_lifecycle_statistics(false).resident_templates, 0);
  manager.load_or_compile(request, first);
  manager.clear();
  manager.release_build_cache();
  manager.dump();
}

}  // namespace
}  // namespace taichi::lang
