#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>

#include "llvm/IR/IRBuilder.h"
#include "taichi/analysis/gather_snode_tree_dependencies.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/statements.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/compilation_manager/kernel_compilation_manager.h"
#include "taichi/program/program.h"
#include "taichi/program/function.h"
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

class FunctionArrayLayoutTest : public testing::Test {
 protected:
  Program program{Arch::x64};
  std::vector<std::unique_ptr<Function>> functions;

  Function *function(int ndim = 1, bool gradient = false,
                     bool compound_prefix = false, bool access = true,
                     std::vector<int> element_shape = {}) {
    auto owned = std::make_unique<Function>(
        &program, FunctionKey("layout_helper", int(functions.size()), 0));
    auto *func = owned.get();
    functions.push_back(std::move(owned));
    if (compound_prefix) {
      auto *tensor = TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
      auto *record = TypeFactory::get_instance().get_struct_type({{tensor, "pair"}});
      func->insert_scalar_param(record);
    }
    DataType dtype = PrimitiveType::i32;
    if (!element_shape.empty())
      dtype = TypeFactory::get_instance().get_tensor_type(element_shape, dtype);
    func->insert_ndarray_param(dtype, ndim, "array", gradient);
    std::vector<AbstractDictionaryMember> fields;
    for (const auto &parameter : func->parameter_list)
      fields.push_back({parameter.get_dtype(), parameter.name});
    func->args_type = TypeFactory::get_instance().get_struct_type(fields)->as<StructType>();
    func->ir = std::make_unique<Block>();
    if (access)
      array_access(func->ir->as<Block>(), {compound_prefix ? 1 : 0}, ndim, gradient,
                   element_shape);
    return func;
  }

  static ExternalPtrStmt *array_access(Block *block, std::vector<int> id,
                                       int ndim = 1, bool gradient = false,
                                       std::vector<int> element_shape = {}) {
    DataType dtype = PrimitiveType::i32;
    if (!element_shape.empty())
      dtype = TypeFactory::get_instance().get_tensor_type(element_shape, dtype);
    auto *descriptor = TypeFactory::get_instance().get_ndarray_struct_type(
        dtype, ndim, gradient);
    auto *argument = block->push_back<ArgLoadStmt>(
        id, TypeFactory::get_instance().get_pointer_type((Type *)descriptor), true, false, 0);
    std::vector<Stmt *> indices;
    for (int axis = 0; axis < ndim; ++axis)
      indices.push_back(block->push_back<ConstStmt>(TypedConstant(0)));
    return block->push_back<ExternalPtrStmt>(argument, indices, ndim,
                                            element_shape)->as<ExternalPtrStmt>();
  }

  static std::vector<Stmt *> descriptor(Block *block, std::vector<int> id,
                                        int ndim = 1, bool gradient = false,
                                        std::vector<int> element_shape = {}) {
    std::vector<Stmt *> args;
    if (ndim == 0)
      args.push_back(block->push_back<ConstStmt>(TypedConstant(0)));
    for (int axis = 0; axis < ndim; ++axis) {
      auto *shape = block->push_back<ExternalTensorShapeAlongAxisStmt>(axis, id);
      shape->ret_type = PrimitiveType::i32;
      args.push_back(shape);
    }
    DataType dtype = PrimitiveType::i32;
    if (!element_shape.empty())
      dtype = TypeFactory::get_instance().get_tensor_type(element_shape, dtype);
    auto *type = TypeFactory::get_instance().get_ndarray_struct_type(dtype, ndim, gradient);
    auto *source = block->push_back<ArgLoadStmt>(
        id, TypeFactory::get_instance().get_pointer_type((Type *)type), true, false, 0);
    for (int field = TypeFactory::ndarray_shape_member_count(ndim);
         field <= TypeFactory::affine_mode_pos_in_ndarray(ndim); ++field) {
      auto *value = block->push_back<GetElementStmt>(
          source, std::vector<int>{TypeFactory::SHAPE_POS_IN_NDARRAY, field});
      value->ret_type = PrimitiveType::i32;
      args.push_back(value);
    }
    for (int is_grad = 0; is_grad <= int(gradient); ++is_grad) {
      auto *pointer = block->push_back<ExternalTensorBasePtrStmt>(id, bool(is_grad));
      pointer->ret_type = TypeFactory::get_instance().get_pointer_type(PrimitiveType::i32);
      args.push_back(pointer);
    }
    return args;
  }

  static std::vector<std::vector<int>> affine(Block &block) {
    return irpass::analysis::gather_kernel_runtime_dependencies(block).runtime_affine_arrays;
  }
};

TEST_F(FunctionArrayLayoutTest, MapsCompoundPrefixAndRepeatedNestedCalls) {
  auto *leaf = function();
  leaf->ir->as<Block>()->push_back<RandStmt>(PrimitiveType::i32);
  auto *middle = function(1, false, true, false);
  auto forwarded = descriptor(middle->ir->as<Block>(), {1});
  middle->ir->as<Block>()->push_back<FuncCallStmt>(leaf, forwarded);
  Block root;
  for (int input : {3, 7}) {
    std::vector<Stmt *> args{
        root.push_back<ConstStmt>(TypedConstant(10)),
        root.push_back<ConstStmt>(TypedConstant(20))};
    auto array = descriptor(&root, {input});
    args.insert(args.end(), array.begin(), array.end());
    root.push_back<FuncCallStmt>(middle, args);
  }
  const auto result = irpass::analysis::gather_kernel_runtime_dependencies(root);
  EXPECT_EQ(result.runtime_affine_arrays, (std::vector<std::vector<int>>{{3}, {7}}));
  EXPECT_NE(result.runtime_state & runtime_state_random, 0u);
}

TEST_F(FunctionArrayLayoutTest, PreservesRankZeroGradientDescriptor) {
  auto *callee = function(0, true);
  Block root;
  root.push_back<FuncCallStmt>(callee, descriptor(&root, {4}, 0, true));
  EXPECT_EQ(affine(root), (std::vector<std::vector<int>>{{4}}));
}

TEST_F(FunctionArrayLayoutTest, PreservesExactTensorElementDescriptor) {
  auto *callee = function(1, true, false, true, {3});
  Block root;
  root.push_back<FuncCallStmt>(callee, descriptor(&root, {4}, 1, true, {3}));
  EXPECT_EQ(affine(root), (std::vector<std::vector<int>>{{4}}));

  Block wrong_element;
  wrong_element.push_back<FuncCallStmt>(callee, descriptor(&wrong_element, {4}, 1, true, {2}));
  EXPECT_TRUE(affine(wrong_element).empty());
}

TEST_F(FunctionArrayLayoutTest, RejectsMixedOriginsAndChangedStrideOrMode) {
  auto *callee = function();
  for (int changed = 0; changed < 4; ++changed) {
    Block root;
    auto args = descriptor(&root, {4});
    auto other = descriptor(&root, {5});
    args[changed] = other[changed];
    root.push_back<FuncCallStmt>(callee, args);
    EXPECT_TRUE(affine(root).empty());
  }
  for (int changed : {1, 2}) {
    Block root;
    auto args = descriptor(&root, {4});
    args[changed] = root.push_back<ConstStmt>(TypedConstant(0));
    root.push_back<FuncCallStmt>(callee, args);
    EXPECT_TRUE(affine(root).empty());
  }
}

TEST_F(FunctionArrayLayoutTest, RejectsAdditionalRawPointerUseAndCompactCallee) {
  auto *callee = function();
  Block root;
  auto args = descriptor(&root, {4});
  root.push_back<FuncCallStmt>(callee, args);
  root.push_back<GlobalLoadStmt>(args.back());
  EXPECT_TRUE(affine(root).empty());

  auto *compact = function(1, false, false, false);
  array_access(compact->ir->as<Block>(), {0})->runtime_affine = false;
  Block compact_root;
  compact_root.push_back<FuncCallStmt>(compact, descriptor(&compact_root, {4}));
  EXPECT_TRUE(affine(compact_root).empty());
}

TEST_F(FunctionArrayLayoutTest, ScalarHelperDoesNotEraseCallerProof) {
  auto *scalar = function(1, false, false, false);
  scalar->parameter_list.clear();
  scalar->nested_parameters.clear();
  scalar->args_type = nullptr;
  scalar->ir->as<Block>()->push_back<RandStmt>(PrimitiveType::i32);
  Block root;
  array_access(&root, {6});
  root.push_back<FuncCallStmt>(scalar, std::vector<Stmt *>{});
  EXPECT_EQ(affine(root), (std::vector<std::vector<int>>{{6}}));
}

TEST_F(FunctionArrayLayoutTest, RejectsUnknownCalleeEffectsAndRecursiveProof) {
  for (int case_id = 0; case_id < 3; ++case_id) {
    auto *callee = function();
    auto *body = callee->ir->as<Block>();
    if (case_id == 0) {
      body->push_back<InternalFuncStmt>("unknown_pointer_consumer", std::vector<Stmt *>{});
    } else if (case_id == 1) {
      auto *raw = body->push_back<ExternalTensorBasePtrStmt>(std::vector<int>{0}, false);
      body->push_back<GlobalLoadStmt>(raw);
    } else {
      body->push_back<FuncCallStmt>(callee, descriptor(body, {0}));
    }
    Block root;
    root.push_back<FuncCallStmt>(callee, descriptor(&root, {4}));
    const auto result = irpass::analysis::gather_kernel_runtime_dependencies(root);
    EXPECT_TRUE(result.runtime_affine_arrays.empty());
    if (case_id == 0)
      EXPECT_NE(result.runtime_state & runtime_state_external, 0u);
  }
}

}  // namespace
}  // namespace taichi::lang
