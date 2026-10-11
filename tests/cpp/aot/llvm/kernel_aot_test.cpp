#include "gtest/gtest.h"

#include "taichi/program/kernel_profiler.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
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
