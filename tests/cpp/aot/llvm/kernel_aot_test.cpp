#include "gtest/gtest.h"

#include <atomic>
#include <thread>

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/aot_snode_resources.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/codegen/llvm/cuda_root_binding.h"
#include "llvm/IR/Verifier.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/dx12/aot_module_loader_impl.h"

#ifdef TI_WITH_CUDA

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

#endif

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {

TEST(LlvmAotMetadataTest, CudaCompactRootsUseFinalLoweringDependencies) {
  // The same early-known root changes from a direct pointer to slot 1 when
  // final mesh lowering adds an earlier root. No placeholder call may survive.
  for (const auto &roots : {std::vector<int>{7}, std::vector<int>{2, 7, 9}}) {
    llvm::LLVMContext context;
    llvm::Module module("compact-roots", context);
    auto *pointer = llvm::PointerType::get(context, 0);
    auto *entry = llvm::Function::Create(
        llvm::FunctionType::get(pointer, {pointer}, false),
        llvm::GlobalValue::ExternalLinkage, "test", module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(context, "entry", entry));
    auto *result = builder.CreateRet(LLVM::emit_cuda_root_binding(
        module, builder, entry->getArg(0), 7));
    EXPECT_ANY_THROW(LLVM::resolve_cuda_root_bindings(module, {2}));
    LLVM::resolve_cuda_root_bindings(module, roots);
    EXPECT_EQ(module.getFunction("__ti_cuda_root_binding_7"), nullptr);
    EXPECT_FALSE(llvm::verifyModule(module));
    if (roots.size() == 1) {
      EXPECT_EQ(result->getReturnValue(), entry->getArg(0));
      EXPECT_EQ(entry->front().size(), 1);
    } else {
      auto *load = llvm::dyn_cast<llvm::LoadInst>(result->getReturnValue());
      ASSERT_NE(load, nullptr);
      auto *slot = llvm::cast<llvm::GetElementPtrInst>(load->getPointerOperand());
      EXPECT_EQ(slot->getPointerOperand(), entry->getArg(0));
      EXPECT_EQ(llvm::cast<llvm::ConstantInt>(slot->getOperand(1))->getZExtValue(),
                1);
      EXPECT_NE(load->getMetadata(llvm::LLVMContext::MD_invariant_load), nullptr);
    }
  }
}

TEST(LlvmAotMetadataTest,
     RelocationSymbolsCompactAndResolveWithoutNumericSearch) {
  llvm::LLVMContext context;
  llvm::Module module("relocation", context);
  auto *type =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), {}, false);
  auto *function = llvm::Function::Create(
      type, llvm::GlobalValue::ExternalLinkage, "test", module);
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(context, "entry", function));
  auto *first = LLVM::emit_aot_tree_id(module, builder, 1, false);
  auto *second = LLVM::emit_aot_tree_id(module, builder, 2, false);
  auto *sum = builder.CreateAdd(first, second);
  builder.CreateRet(sum);
  LLVM::remap_aot_tree_id_symbols(module, {{1, 0}, {2, 1}});
  EXPECT_NE(module.getNamedGlobal("__ti_aot_tree_id_0"), nullptr);
  EXPECT_NE(module.getNamedGlobal("__ti_aot_tree_id_1"), nullptr);
  EXPECT_ANY_THROW(LLVM::resolve_aot_tree_ids(module, {{0, 5}}));
  EXPECT_EQ(module.global_size(), 2);
  LLVM::resolve_aot_tree_ids(module, {{0, 5}, {1, 9}});
  EXPECT_TRUE(module.global_empty());
  EXPECT_EQ(llvm::cast<llvm::ConstantInt>(
                llvm::cast<llvm::Instruction>(sum)->getOperand(0))
                ->getSExtValue(),
            5);
  EXPECT_EQ(llvm::cast<llvm::ConstantInt>(
                llvm::cast<llvm::Instruction>(sum)->getOperand(1))
                ->getSExtValue(),
            9);
}

TEST(LlvmAotMetadataTest, ValidatesCudaComputeAndPtxRequirements) {
  LLVM::LlvmAotMetadata metadata;
  metadata.schema_version = LLVM::kLlvmAotSchemaVersion;
  metadata.kernel_format = "ll";
  metadata.required_caps["cuda_compute_capability"] = 60;
  metadata.required_caps["cuda_ptx_version"] = 50;

  EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  EXPECT_NO_THROW(LLVM::validate_cuda_aot_metadata(metadata, 60, 50));
  EXPECT_NO_THROW(LLVM::validate_cuda_aot_metadata(metadata, 90, 78));
  EXPECT_ANY_THROW(LLVM::validate_cuda_aot_metadata(metadata, 59, 50));
  EXPECT_ANY_THROW(LLVM::validate_cuda_aot_metadata(metadata, 60, 49));

  LLVM::LlvmAotMetadata missing_schema;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(missing_schema));
  auto old_graph_layout = metadata;
  old_graph_layout.schema_version = 1;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(old_graph_layout));

  LLVM::LlvmAotMetadata missing_ptx;
  missing_ptx.schema_version = LLVM::kLlvmAotSchemaVersion;
  missing_ptx.required_caps["cuda_compute_capability"] = 60;
  EXPECT_ANY_THROW(LLVM::validate_cuda_aot_metadata(missing_ptx, 90, 78));
}

TEST(LlvmAotMetadataTest, DeclaredRepresentationAndLegacyText) {
  LLVM::LlvmAotMetadata legacy;
  const auto json = liong::json::parse(
      R"({"schema_version":2,"required_caps":[]})");
  liong::json::deserialize(json, legacy);
  EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(legacy));
  EXPECT_TRUE(legacy.kernel_format.empty());
  legacy.kernel_format = "bc";
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(legacy));

  LLVM::LlvmAotMetadata current;
  current.schema_version = LLVM::kLlvmAotSchemaVersion;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(current));
  for (const auto *format : {"ll", "bc"}) {
    current.kernel_format = format;
    LLVM::LlvmAotMetadata decoded;
    liong::json::deserialize(liong::json::serialize(current), decoded);
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(decoded));
    EXPECT_EQ(decoded.kernel_format, format);
  }
  current.kernel_format = "guess";
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(current));
  current.kernel_format = "bc";
  current.schema_version = LLVM::kLlvmAotSchemaVersion + 1;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(current));
}

TEST(LlvmAotMetadataTest, PtxRequiresExplicitSchemaAndAbiDigest) {
  LLVM::LlvmAotMetadata metadata;
  metadata.kernel_format = "ptx";
  metadata.abi_sha256 = std::string(64, 'a');
  metadata.schema_version = 3;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.schema_version = LLVM::kCudaPtxAotSchemaVersion;
  EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.abi_sha256.clear();
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
}

TEST(LlvmAotMetadataTest, DenseRelocationHasAnExplicitFormatContract) {
  LLVM::LlvmAotMetadata metadata;
  metadata.schema_version = LLVM::kRelocatableLlvmAotSchemaVersion;
  metadata.resource_binding = "dense_v1";
  metadata.abi_sha256 = std::string(64, 'b');
  for (const auto *format : {"bc", "ptx"}) {
    metadata.kernel_format = format;
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  }
  metadata.schema_version = LLVM::kCudaPtxAotSchemaVersion;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.schema_version = LLVM::kRelocatableLlvmAotSchemaVersion;
  metadata.resource_binding = "guess";
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
}

TEST(LlvmAotMetadataTest, SparseRelocationDoesNotImplyRootSnapshots) {
  LLVM::LlvmAotMetadata metadata;
  metadata.resource_binding = "sparse_v1";
  metadata.abi_sha256 = std::string(64, 'b');
  for (const auto *binding : {"sparse_v1", "sparse_v2"}) {
    metadata.resource_binding = binding;
    for (const auto *format : {"bc", "ptx", "cubin"}) {
      metadata.kernel_format = format;
      metadata.schema_version = metadata.kernel_format == "cubin"
                                    ? LLVM::kCudaCubinAotSchemaVersion
                                    : LLVM::kRelocatableLlvmAotSchemaVersion;
      EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
    }
  }
  metadata.schema_version = LLVM::kInitialRootDataAotSchemaVersion;
  metadata.initial_root_data.push_back({0, 32, std::string(64, 'b')});
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  EXPECT_TRUE(LLVM::is_relocatable_aot_task(OffloadedTaskType::mesh_for, false));
  EXPECT_TRUE(LLVM::is_relocatable_aot_snode(SNodeType::hash, true));
  EXPECT_FALSE(LLVM::is_relocatable_aot_snode(SNodeType::hash, false));
  for (auto type : {SNodeType::bit_struct, SNodeType::quant_array}) {
    EXPECT_TRUE(LLVM::is_relocatable_aot_snode(type, true));
    EXPECT_FALSE(LLVM::is_relocatable_aot_snode(type, false));
    EXPECT_FALSE(LLVM::is_dense_aot_snode(type));
  }
}

TEST(LlvmAotMetadataTest, SparseParentClosureIsValidatedBeforeAllocation) {
  LlvmOfflineCache::FieldCacheData field;
  field.root_id = 42;
  field.root_runtime_local_id = 2;
  field.snode_metas.resize(3);
  for (int i = 0; i < 3; ++i) {
    auto &node = field.snode_metas[i];
    node.id = 42 + i;
    node.parent_id = i == 0 ? -1 : 41 + i;
    node.runtime_local_id = (i + 2) % 3;
    node.type = i == 0 ? SNodeType::root
                      : i == 1 ? SNodeType::pointer : SNodeType::place;
  }
  EXPECT_NO_THROW(LLVM::validate_relocatable_aot_field(field, true));
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(field, false));
  auto invalid = field;
  invalid.snode_metas[1].parent_id = 43;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(invalid, true));
  invalid = field;
  invalid.snode_metas[1].parent_id = 999;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(invalid, true));
  invalid = field;
  invalid.snode_metas[2].id = 43;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(invalid, true));
  invalid = field;
  invalid.snode_metas[2].runtime_local_id = 2;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(invalid, true));
  invalid = field;
  invalid.root_runtime_local_id = 0;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(invalid, true));
}

TEST(LlvmAotMetadataTest, ListOwnersFollowTasksInsteadOfAllKernelDependencies) {
  LlvmOfflineCache cache;
  for (int id : {0, 1}) {
    auto &field = cache.fields[id];
    field.tree_id = id;
    field.snode_metas.resize(1);
    field.snode_metas[0].id = 10 + id;
    field.snode_metas[0].type = SNodeType::dense;
  }
  auto &kernel = cache.kernels["energy"];
  kernel.used_snode_tree_ids = {0, 1};
  kernel.compiled_data.tasks.emplace_back("loop");
  auto &task = kernel.compiled_data.tasks.back();
  task.task_type = OffloadedTaskType::struct_for;
  task.sparse_list_snode_id = 10;
  EXPECT_EQ(LLVM::aot_element_list_tree_ids(cache, true),
            (std::unordered_set<int>{0}));
  task.task_type = OffloadedTaskType::listgen;
  // CUDA reuse-off still declares list ownership without sparse_list_op.
  EXPECT_EQ(LLVM::aot_element_list_tree_ids(cache, true),
            (std::unordered_set<int>{0}));
  task.task_type = OffloadedTaskType::range_for;
  EXPECT_TRUE(LLVM::aot_element_list_tree_ids(cache, true).empty());
  cache.fields[1].snode_metas[0].type = SNodeType::pointer;
  EXPECT_EQ(LLVM::aot_element_list_tree_ids(cache, true),
            (std::unordered_set<int>{1}));
}

TEST(LlvmAotMetadataTest, HashCapacityRetainsTheCompilerIndexContract) {
  LlvmOfflineCache::FieldCacheData field;
  field.snode_metas.resize(3);
  for (int i = 0; i < 3; ++i) {
    auto &node = field.snode_metas[i];
    node.id = node.runtime_local_id = i;
    node.parent_id = i - 1;
    node.type = i == 0 ? SNodeType::root
                      : i == 1 ? SNodeType::hash : SNodeType::place;
  }
  auto &hash = field.snode_metas[1];
  hash.vk_max_active_hint = 8;
  EXPECT_NO_THROW(LLVM::validate_relocatable_aot_field(field, true));
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(field, false));
  hash.vk_max_active_hint = 3;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(field, true));
  hash.vk_max_active_hint = int64{1} << 31;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(field, true));
  hash.vk_max_active_hint = -1;
  hash.num_cells_per_container = 8;
  EXPECT_NO_THROW(LLVM::validate_relocatable_aot_field(field, true));
  hash.num_cells_per_container = 0;
  EXPECT_ANY_THROW(LLVM::validate_relocatable_aot_field(field, true));
}

TEST(LlvmAotMetadataTest, ListOwnersValidateClosureAndBoundLegacyFallback) {
  LlvmOfflineCache cache;
  for (int id : {0, 1, 2}) {
    auto &field = cache.fields[id];
    field.tree_id = id;
    field.snode_metas.resize(1);
    field.snode_metas[0].id = 10 + id;
    field.snode_metas[0].type = SNodeType::dense;
  }
  auto &kernel = cache.kernels["energy"];
  kernel.used_snode_tree_ids = {0, 1};
  kernel.compiled_data.tasks.emplace_back("loop");
  auto &task = kernel.compiled_data.tasks.back();
  task.task_type = OffloadedTaskType::struct_for;
  EXPECT_ANY_THROW(LLVM::aot_element_list_tree_ids(cache, true));
  EXPECT_EQ(LLVM::aot_element_list_tree_ids(cache, false),
            (std::unordered_set<int>{0, 1}));
  task.sparse_list_snode_id = 12;  // Valid tree, absent from kernel closure.
  EXPECT_ANY_THROW(LLVM::aot_element_list_tree_ids(cache, true));
  EXPECT_ANY_THROW(LLVM::aot_element_list_tree_ids(cache, false));
  task.sparse_list_snode_id = 999;
  EXPECT_ANY_THROW(LLVM::aot_element_list_tree_ids(cache, false));
  task.sparse_list_snode_id = 10;
  kernel.used_snode_tree_ids.push_back(99);
  EXPECT_ANY_THROW(LLVM::aot_element_list_tree_ids(cache, true));
}

TEST(LlvmAotMetadataTest, ExactListBindingPreservesDenseSnapshotFormat) {
  LLVM::LlvmAotMetadata metadata;
  metadata.resource_binding = "list_v1";
  metadata.abi_sha256 = std::string(64, 'a');
  for (const auto *format : {"bc", "ptx", "cubin"}) {
    metadata.kernel_format = format;
    metadata.schema_version = metadata.kernel_format == "cubin"
                                  ? LLVM::kCudaCubinAotSchemaVersion
                                  : LLVM::kRelocatableLlvmAotSchemaVersion;
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
    metadata.schema_version = LLVM::kInitialRootDataAotSchemaVersion;
    metadata.initial_root_data.push_back({0, 32, std::string(64, 'b')});
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
    metadata.initial_root_data.clear();
  }
  metadata.resource_binding = "list_v2";
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
}

TEST(LlvmAotMetadataTest, CubinRequiresNewSchemaAndBinaryCompatibleTarget) {
  LLVM::LlvmAotMetadata metadata;
  metadata.schema_version = LLVM::kCudaCubinAotSchemaVersion;
  metadata.kernel_format = "cubin";
  metadata.abi_sha256 = std::string(64, 'c');
  metadata.required_caps["cuda_compute_capability"] = 86;
  metadata.required_caps["cuda_ptx_version"] = 87;
  EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.resource_binding = "dense_v1";
  EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  // Loading machine code does not parse the producer's PTX version.
  EXPECT_NO_THROW(LLVM::validate_cuda_aot_metadata(metadata, 86, 70));
  EXPECT_NO_THROW(LLVM::validate_cuda_aot_metadata(metadata, 89, 70));
  EXPECT_ANY_THROW(LLVM::validate_cuda_aot_metadata(metadata, 80, 87));
  EXPECT_ANY_THROW(LLVM::validate_cuda_aot_metadata(metadata, 120, 87));
  metadata.schema_version = LLVM::kRelocatableLlvmAotSchemaVersion;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.schema_version = LLVM::kCudaCubinAotSchemaVersion;
  metadata.abi_sha256.clear();
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
}

TEST(LlvmAotMetadataTest,
     InitialRootStateRequiresExplicitSchemaAndDenseBinding) {
  LLVM::LlvmAotMetadata metadata;
  metadata.schema_version = LLVM::kInitialRootDataAotSchemaVersion;
  metadata.resource_binding = "dense_v1";
  metadata.abi_sha256 = std::string(64, 'a');
  metadata.initial_root_data.push_back({0, 32, std::string(64, 'b')});
  for (const auto *format : {"bc", "ptx", "cubin"}) {
    metadata.kernel_format = format;
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  }
  metadata.schema_version = LLVM::kCudaCubinAotSchemaVersion;
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.schema_version = LLVM::kInitialRootDataAotSchemaVersion;
  metadata.resource_binding.clear();
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
  metadata.resource_binding = "dense_v1";
  metadata.initial_root_data.clear();
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
}

TEST(LlvmAotMetadataTest, GatedLaunchAbiExtendsExistingArtifactContracts) {
  LLVM::LlvmAotMetadata metadata;
  metadata.schema_version = LLVM::kKernelLaunchAbiAotSchemaVersion;
  metadata.abi_sha256 = std::string(64, 'a');
  metadata.kernel_launch_abis["body"] = cuda::kGraphGateLaunchABI;
  metadata.launch_abi_sha256 = LLVM::aot_launch_abi_sha256(metadata);
  for (const auto *format : {"bc", "ptx", "cubin"}) {
    metadata.kernel_format = format;
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
    metadata.resource_binding = "dense_v1";
    metadata.initial_root_data = {{0, 32, std::string(64, 'b')}};
    EXPECT_NO_THROW(LLVM::validate_llvm_aot_metadata(metadata));
    metadata.resource_binding.clear();
    EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(metadata));
    metadata.initial_root_data.clear();
  }
  metadata.kernel_format = "bc";
  LLVM::LlvmAotMetadata decoded;
  liong::json::deserialize(liong::json::serialize(metadata), decoded);
  EXPECT_EQ(decoded.kernel_launch_abis, metadata.kernel_launch_abis);
  EXPECT_EQ(decoded.launch_abi_sha256, metadata.launch_abi_sha256);
  EXPECT_EQ(LLVM::aot_launch_abi_sha256(decoded), metadata.launch_abi_sha256);
  decoded.kernel_launch_abis["other"] = cuda::kGraphGateLaunchABI;
  EXPECT_NE(LLVM::aot_launch_abi_sha256(decoded), metadata.launch_abi_sha256);
  decoded = metadata;
  decoded.abi_sha256[0] = 'b';
  EXPECT_NE(LLVM::aot_launch_abi_sha256(decoded), metadata.launch_abi_sha256);
  for (std::uint32_t version = 2; version < 8; ++version) {
    decoded = metadata;
    decoded.schema_version = version;
    EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(decoded));
  }
  decoded = metadata;
  decoded.kernel_launch_abis["body"] = "unknown";
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(decoded));
  decoded = metadata;
  decoded.kernel_launch_abis.clear();
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(decoded));
  decoded = metadata;
  decoded.launch_abi_sha256.clear();
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(decoded));
  decoded = metadata;
  decoded.kernel_format = "ll";
  EXPECT_ANY_THROW(LLVM::validate_llvm_aot_metadata(decoded));
}

TEST(LlvmAotMetadataTest, InitialRootDataValidatesExactStorageAndUniqueIds) {
  const std::vector<size_t> sizes{32, 64};
  std::vector<aot::InitialRootData> entries{{1, 64, std::string(64, 'a')}};
  EXPECT_NO_THROW(aot::validate_initial_root_data(entries, sizes));
  entries.push_back(entries.front());
  EXPECT_ANY_THROW(aot::validate_initial_root_data(entries, sizes));
  entries.pop_back();
  entries[0].tree_id = 2;
  EXPECT_ANY_THROW(aot::validate_initial_root_data(entries, sizes));
  entries[0].tree_id = 0;
  EXPECT_ANY_THROW(aot::validate_initial_root_data(entries, sizes));
  entries[0].size = 32;
  entries[0].sha256[0] = 'g';
  EXPECT_ANY_THROW(aot::validate_initial_root_data(entries, sizes));
}

TEST(LlvmAotMetadataTest, CpuFinalizeRetiresAllocationsBeforeExecutorDestruction) {
  CompileConfig cfg;
  cfg.arch = host_arch();
  LlvmProgramImpl program(cfg, nullptr);
  auto *device = static_cast<cpu::CpuDevice *>(
      program.get_runtime_executor()->llvm_device());
  Device::AllocParams params{};
  params.size = 32;
  DeviceAllocation allocation;
  ASSERT_EQ(device->allocate_memory(params, &allocation), RhiResult::success);
  void *mapped = nullptr;
  ASSERT_EQ(device->map_range_for_cpu_native(allocation.get_ptr(), 32, &mapped),
            RhiResult::success);

  // A Python plan may retain the executor after Program::finalize. Allocation
  // records and the executor-owned host pool must retire at finalization.
  program.finalize();
  EXPECT_EQ(device->map_range_for_cpu_native(allocation.get_ptr(), 32, &mapped),
            RhiResult::invalid_usage);
  EXPECT_EQ(program.get_runtime_executor()
                ->get_host_memory_pool_stats().reserved_bytes, 0u);
  EXPECT_EQ(device->allocate_memory(params, &allocation),
            RhiResult::invalid_usage);
}

TEST(LlvmAotMetadataTest, CpuRuntimePoolSurvivesOtherExecutorFinalization) {
  CompileConfig survivor_config;
  survivor_config.arch = host_arch();
  survivor_config.cpu_max_num_threads = 2;
  LlvmRuntimeExecutor survivor(survivor_config, nullptr);
  uint64 *result_buffer = nullptr;
  survivor.materialize_runtime(nullptr, &result_buffer);
  const auto before = survivor.get_host_memory_pool_stats();
  ASSERT_GT(before.requested_live_bytes, 0u);
  RecordProperty("cpu_host_requested_live_bytes",
                 std::to_string(before.requested_live_bytes));
  RecordProperty("cpu_host_reserved_bytes", std::to_string(before.reserved_bytes));
  RecordProperty("cpu_host_used_bytes", std::to_string(before.used_bytes));

  CompileConfig other_config;
  other_config.arch = host_arch();
  other_config.cpu_max_num_threads = 2;
  LlvmProgramImpl other(other_config, nullptr);
  uint64 *other_result_buffer = nullptr;
  other.get_runtime_executor()->materialize_runtime(nullptr,
                                                   &other_result_buffer);
  EXPECT_NE(result_buffer, other_result_buffer);
  const auto after = survivor.get_host_memory_pool_stats();
  EXPECT_EQ(after.allocate_count, before.allocate_count);
  EXPECT_EQ(after.requested_live_bytes, before.requested_live_bytes);
  std::atomic<bool> observed{false};
  std::atomic<bool> retired{false};
  std::thread observer([&] {
    do {
      other.get_runtime_executor()->get_host_memory_pool_stats();
      observed.store(true);
    } while (!retired.load());
  });
  while (!observed.load()) {
    std::this_thread::yield();
  }
  other.finalize();
  retired.store(true);
  observer.join();
  EXPECT_EQ(other.get_runtime_executor()
                ->get_host_memory_pool_stats().reserved_bytes, 0u);

  // Non-LLVM JIT teardown still resets the legacy global pool. Neither this
  // runtime's result buffer nor its LLVM host callbacks may refer to it.
  HostMemoryPool::get_instance().reset();
  survivor.check_runtime_error(result_buffer);
  auto allocation = survivor.allocate_memory_on_device(32, result_buffer);
  survivor.deallocate_memory_on_device(allocation);
  EXPECT_GT(survivor.get_host_memory_pool_stats().allocate_count,
            before.allocate_count);
}

TEST(LlvmAotMetadataTest, ResourceReservationsPrecedeAllocation) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  LlvmRuntimeExecutor executor(cfg, nullptr);
  const std::vector<int> legacy{1, 2};
  executor.reserve_aot_snode_tree_ids(legacy);
  EXPECT_ANY_THROW(executor.reserve_aot_snode_tree_ids(legacy));
  EXPECT_EQ(executor.reserve_aot_snode_tree_ids(std::size_t(2)),
            (std::vector<int>{0, 3}));
  EXPECT_ANY_THROW(executor.reserve_aot_snode_tree_ids(std::vector<int>{4, 2}));
  EXPECT_FALSE(executor.has_snode_tree_binding(4));
  executor.release_aot_snode_tree_ids(legacy);
  EXPECT_EQ(executor.reserve_aot_snode_tree_ids(std::size_t(2)), legacy);
  executor.release_aot_snode_tree_ids({0, 1, 2, 3});
}

static void run_dense_relocation_test(Arch arch) {
  CompileConfig cfg;
  cfg.arch = arch;
  LlvmRuntimeExecutor executor(cfg, nullptr);
  uint64 *result_buffer = nullptr;
  executor.materialize_runtime(nullptr, &result_buffer);
  const char *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  auto load = [&]() {
    LLVM::AotModuleParams params;
    params.executor_ = &executor;
    params.module_path = folder;
    if (arch_is_cpu(arch)) {
      params.kernel_launcher = std::make_unique<cpu::KernelLauncher>(
          cpu::KernelLauncher::Config{&executor});
    } else {
#ifdef TI_WITH_CUDA
      params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
          cuda::KernelLauncher::Config{&executor});
#endif
    }
    return LLVM::make_aot_module(std::move(params));
  };
  auto materialize = [&](aot::Module &module) {
    auto &llvm_module = static_cast<LLVM::LlvmAotModule &>(module);
    for (int id : llvm_module.get_snode_tree_ids()) {
      LLVM::allocate_aot_snode_tree_type(
          &module, module.get_snode_tree(std::to_string(id)), result_buffer);
    }
  };
  auto evaluate = [&](aot::Module &module, float seed) {
    for (const auto *name : {"initialize", "evaluate", "evaluate_grad"}) {
      auto *kernel = module.get_kernel(name);
      LaunchContextBuilder ctx(kernel);
      if (std::string(name) == "initialize")
        ctx.set_arg({0}, seed);
      kernel->launch(ctx);
    }
  };
  auto value = [&](aot::Module &module) {
    auto *kernel = module.get_kernel("get_scalar");
    LaunchContextBuilder ctx(kernel);
    ctx.set_arg({0}, 0);
    kernel->launch(ctx);
    return ctx.get_ret<float>(0);
  };
  auto first = load();
  auto second = load();  // Reservations must work before either allocation.
  materialize(*first);
  materialize(*second);
  evaluate(*first, 1.0f);
  evaluate(*second, -0.5f);
  EXPECT_FLOAT_EQ(value(*first), 8.0f);
  EXPECT_FLOAT_EQ(value(*second), 0.5f);
  second.reset();
  auto replacement = load();
  materialize(*replacement);
  EXPECT_FLOAT_EQ(value(*replacement), 0.0f);
  evaluate(*replacement, 0.25f);
  EXPECT_FLOAT_EQ(value(*replacement), 4.25f);
  EXPECT_FLOAT_EQ(value(*first), 8.0f);
  replacement.reset();
  first.reset();
  for (int id = 0; id < 4; ++id)
    EXPECT_FALSE(executor.has_snode_tree_binding(id));
}

TEST(LlvmAotTest, DenseRelocationCpu) {
  run_dense_relocation_test(Arch::x64);
}

TEST(LlvmAotTest, DenseRelocationCuda) {
#ifdef TI_WITH_CUDA
  if (!is_cuda_api_available())
    GTEST_SKIP();
  run_dense_relocation_test(Arch::cuda);
#else
  GTEST_SKIP();
#endif
}

static void run_demand_root_test(Arch arch) {
  CompileConfig cfg;
  cfg.arch = arch;
  LlvmRuntimeExecutor executor(cfg, nullptr);
  uint64 *result_buffer = nullptr;
  executor.materialize_runtime(nullptr, &result_buffer);
  const char *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  auto load = [&]() {
    LLVM::AotModuleParams params;
    params.executor_ = &executor;
    params.module_path = folder;
    params.allocation_result_buffer = result_buffer;
    if (arch_is_cpu(arch)) {
      params.kernel_launcher = std::make_unique<cpu::KernelLauncher>(
          cpu::KernelLauncher::Config{&executor});
    } else {
#ifdef TI_WITH_CUDA
      params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
          cuda::KernelLauncher::Config{&executor});
#endif
    }
    return LLVM::make_aot_module(std::move(params));
  };
  auto invoke = [](aot::Module &module, const char *name, bool returns = false) {
    auto *kernel = module.get_kernel(name);
    LaunchContextBuilder context(kernel);
    kernel->launch(context);
    return returns ? context.get_ret<int>(0) : 0;
  };
  auto first = load();
  auto second = load();
  ASSERT_EQ(static_cast<LLVM::LlvmAotModule &>(*first).get_num_snode_trees(), 3u);
  for (int id = 0; id < 6; ++id) {
    EXPECT_TRUE(executor.has_snode_tree_binding(id));
    EXPECT_FALSE(executor.has_snode_tree_runtime_state(id));
  }
  // Metadata queries must not allocate any root. A view can precede kernels.
  EXPECT_EQ(first->get_root_storage_view("right").bytes, 16u);
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(0));
  EXPECT_TRUE(executor.has_snode_tree_runtime_state(1));
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(2));
  EXPECT_EQ(invoke(*first, "peek_right", true), 23);
  invoke(*first, "bump_right");
  first->get_root_storage_view("right");
  EXPECT_EQ(invoke(*first, "peek_right", true), 24);
  invoke(*first, "bump_left");
  invoke(*first, "bump_left");
  EXPECT_EQ(invoke(*first, "peek_left", true), 13);
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(2));
  EXPECT_EQ(invoke(*second, "peek_left", true), 11);
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(4));
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(5));
  first.reset();
  for (int id = 0; id < 3; ++id)
    EXPECT_FALSE(executor.has_snode_tree_binding(id));
  auto replacement = load();
  EXPECT_EQ(invoke(*replacement, "peek_left", true), 11);
  EXPECT_EQ(invoke(*second, "peek_left", true), 11);
  replacement.reset();
  second.reset();
  for (int id = 0; id < 6; ++id) {
    EXPECT_FALSE(executor.has_snode_tree_binding(id));
    EXPECT_FALSE(executor.has_snode_tree_runtime_state(id));
  }
}

TEST(LlvmAotTest, DemandRootsCpu) {
  run_demand_root_test(Arch::x64);
}

TEST(LlvmAotTest, DemandRootsCuda) {
#ifdef TI_WITH_CUDA
  if (!is_cuda_api_available())
    GTEST_SKIP();
  run_demand_root_test(Arch::cuda);
#else
  GTEST_SKIP();
#endif
}

TEST(LlvmAotTest, CpuKernel) {
  CompileConfig cfg;
  cfg.arch = Arch::x64;
  cfg.kernel_profiler = false;
  constexpr KernelProfilerBase *kNoProfiler = nullptr;
  LlvmRuntimeExecutor exec{cfg, kNoProfiler};
  // Must have handled all the arch fallback logic by this point.
  uint64 *result_buffer{nullptr};
  exec.materialize_runtime(kNoProfiler, &result_buffer);

  constexpr int kArrLen = 32;
  constexpr int kArrBytes = kArrLen * sizeof(int32_t);
  auto arr_devalloc = exec.allocate_memory_on_device(kArrBytes, result_buffer);
  Ndarray arr = Ndarray(arr_devalloc, PrimitiveType::i32, {kArrLen});

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

  auto *k_run = mod->get_kernel("run");

  LaunchContextBuilder builder(k_run);
  builder.set_arg({0}, /*v=*/0);
  builder.set_arg_ndarray(/*arg_id=*/{1}, arr);
  std::vector<int> vec = {1, 2, 3};
  for (int i = 0; i < vec.size(); ++i) {
    builder.set_struct_arg(/*arg_indices=*/{2, i}, vec[i]);
  }
  k_run->launch(builder);

  auto *data =
      reinterpret_cast<int32_t *>(exec.get_device_alloc_info_ptr(arr_devalloc));
  for (int i = 0; i < kArrLen; ++i) {
    EXPECT_EQ(data[i], i + vec[0]);
  }
}

TEST(LlvmAotTest, CudaKernel) {
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

    constexpr int kArrLen = 32;
    constexpr int kArrBytes = kArrLen * sizeof(int32_t);
    auto arr_devalloc =
        exec.allocate_memory_on_device(kArrBytes, result_buffer);
    Ndarray arr = Ndarray(arr_devalloc, PrimitiveType::i32, {kArrLen});

    LLVM::AotModuleParams aot_params;
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

    std::stringstream aot_mod_ss;
    aot_mod_ss << folder_dir;
    aot_params.module_path = aot_mod_ss.str();
    aot_params.executor_ = &exec;
    aot_params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
        cuda::KernelLauncher::Config{&exec});
    auto mod = LLVM::make_aot_module(std::move(aot_params));

    auto *k_run = mod->get_kernel("run");
    LaunchContextBuilder builder(k_run);
    builder.set_arg({0}, /*v=*/0);
    builder.set_arg_ndarray(/*arg_id=*/{1}, arr);
    std::vector<int> vec = {1, 2, 3};
    for (int i = 0; i < vec.size(); ++i) {
      builder.set_struct_arg(/*arg_indices=*/{2, i}, vec[i]);
    }
    k_run->launch(builder);

    auto *data = reinterpret_cast<int32_t *>(
        exec.get_device_alloc_info_ptr(arr_devalloc));

    std::vector<int32_t> cpu_data(kArrLen);
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)cpu_data.data(), (void *)data, kArrLen * sizeof(int32_t));

    for (int i = 0; i < kArrLen; ++i) {
      EXPECT_EQ(cpu_data[i], i + vec[0]);
    }
  }
#endif
}

#ifdef TI_WITH_DX12
TEST(LlvmAotTest, DX12Kernel) {
  directx12::AotModuleParams aot_params;
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");

  std::stringstream aot_mod_ss;
  aot_mod_ss << folder_dir;
  aot_params.module_path = aot_mod_ss.str();
  // FIXME: add executor.
  auto mod = directx12::make_aot_module(aot_params, Arch::dx12);
  auto *k_run = mod->get_kernel("run");
  EXPECT_TRUE(k_run);
  // FIXME: launch the kernel and check result.
}
#endif

}  // namespace taichi::lang
