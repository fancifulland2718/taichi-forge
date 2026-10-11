#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

#include <algorithm>
#include <fstream>
#include <iterator>

#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#if defined(TI_WITH_CUDA)
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

namespace taichi::lang {
namespace LLVM {

LlvmAotModule::LlvmAotModule(
    const std::string &module_path,
    LlvmRuntimeExecutor *executor,
    std::unique_ptr<LLVM::KernelLauncher> kernel_launcher)
    : executor_(executor),
      kernel_launcher_(std::move(kernel_launcher)) {
  TI_ASSERT(executor_ != nullptr);

  const std::string metadata_path =
      taichi::join_path(module_path, kLlvmAotMetadataFilename);
  std::ifstream stream(metadata_path, std::ios::in | std::ios::binary);
  TI_ERROR_IF(!stream.is_open(),
              "LLVM AOT artifact is missing required schema metadata at {}. "
              "Rebuild the artifact with the current Forge AOT compiler.",
              metadata_path);
  const std::string contents((std::istreambuf_iterator<char>(stream)),
                             std::istreambuf_iterator<char>());
  module_path_ = module_path;
  auto &metadata = metadata_;
  auto json = liong::json::parse(contents.data(),
                                 contents.data() + contents.size());
  liong::json::deserialize(json, metadata);
  validate_llvm_aot_metadata(metadata);

  if (metadata.kernel_format == "ptx") {
    TI_ERROR_IF(arch() != Arch::cuda, "CUDA PTX artifacts require a CUDA runtime");
    TI_ERROR_IF(aot_artifact_sha256(read_aot_artifact(
                    taichi::join_path(module_path, "metadata.tcb"))) !=
                    metadata.abi_sha256,
                "CUDA PTX AOT launch ABI checksum mismatch");
  }

  // Select exactly the declared representation. Missing/corrupt payloads must
  // not silently fall back to a different generation left in this directory.
  const auto format = metadata.kernel_format == "bc"
                          ? LlvmOfflineCache::Format::BC
                          : (metadata.kernel_format == "ptx"
                                 ? LlvmOfflineCache::Format::PTX
                                 : LlvmOfflineCache::Format::LL);
  cache_reader_ = LlvmOfflineCacheFileReader::make(module_path, format);
  TI_ERROR_IF(cache_reader_ == nullptr,
              "LLVM AOT artifact metadata at {} is corrupt or incompatible. "
              "Rebuild the artifact with the current Forge AOT compiler.",
              module_path);

  if (executor_->get_config().arch == Arch::cuda) {
    for (const auto &[key, value] : metadata.required_caps) {
      required_caps_.set(str2devcap(key), value);
    }

    const auto &cuda_context = CUDAContext::get_instance();
    validate_cuda_aot_metadata(metadata,
                               cuda_context.get_compute_capability(),
                               cuda_context.get_ptx_version());
  }

  snode_tree_ids_ = cache_reader_->get_snode_tree_ids();

  const std::string graph_path = fmt::format("{}/graphs.tcb", module_path);
  read_from_binary_file(graphs_, graph_path);
}

FunctionType LlvmAotModule::convert_module_to_function(
    const LLVM::CompiledKernelData &compiled) {
  auto *launcher = kernel_launcher_.get();
  auto handle = kernel_launcher_->register_llvm_kernel(compiled);
  return [handle, launcher](LaunchContextBuilder &ctx) {
    launcher->launch_llvm_kernel(handle, ctx);
  };
}

LlvmOfflineCache::KernelCacheData LlvmAotModule::load_kernel_from_cache(
    const std::string &name) {
  TI_ASSERT(cache_reader_ != nullptr);
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = metadata_.kernel_format == "ptx"
                ? cache_reader_->take_kernel_metadata(loaded, name)
                : cache_reader_->take_kernel_cache(
                      loaded, name,
                      *executor_->get_llvm_context()->get_this_thread_context());
  if (!ok) {
    throw aot::ArtifactLoadError("Failed to read LLVM AOT kernel: " + name);
  }
  for (int tree_id : loaded.used_snode_tree_ids) {
    TI_ERROR_IF(!std::binary_search(snode_tree_ids_.begin(),
                                     snode_tree_ids_.end(), tree_id),
                "LLVM AOT kernel {} is missing SNodeTree {}", name, tree_id);
  }
  return loaded;
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel(
    const std::string &name) {
  if (metadata_.kernel_format != "ptx") {
    return make_new_kernel_impl(name);
  }
  const auto failed = artifact_failures_.find(name);
  if (failed != artifact_failures_.end()) {
    throw aot::ArtifactLoadError(failed->second);
  }
  std::string diagnostic;
  try {
    return make_new_kernel_impl(name);
  } catch (const std::exception &error) {
    diagnostic = error.what();
  } catch (const std::string &error) {
    // TI_ERROR uses a string exception; preserve it in the same stable
    // artifact failure path as decoding and allocation exceptions.
    diagnostic = error;
  }
  auto message = fmt::format("Failed to load CUDA PTX AOT kernel {}: {}",
                              name, diagnostic);
  artifact_failures_.emplace(name, message);
  throw aot::ArtifactLoadError(message);
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel_impl(
    const std::string &name) {
  if (!cache_reader_->has_kernel(name)) return nullptr;
  auto kernel_cache = load_kernel_from_cache(name);
  const auto kernel_name = std::move(kernel_cache.kernel_key);
  if (metadata_.kernel_format == "ptx") {
#if defined(TI_WITH_CUDA)
    const auto it = metadata_.cuda_artifacts.find(name);
    TI_ERROR_IF(it == metadata_.cuda_artifacts.end(),
                "Missing CUDA PTX AOT manifest for kernel {}", name);
    const auto &item = it->second;
    auto bytes = read_aot_artifact(taichi::join_path(module_path_, name + ".ptx"));
    TI_ERROR_IF(bytes.empty() || bytes.size() != item.size ||
                    aot_artifact_sha256(bytes) != item.sha256 ||
                    bytes.find('\0') != std::string::npos,
                "CUDA PTX AOT payload integrity failure for kernel {}", name);
    const auto target = fmt::format("sm_{}|+ptx{}",
        required_caps_.get(DeviceCapability::cuda_compute_capability),
        required_caps_.get(DeviceCapability::cuda_ptx_version));
    TI_ERROR_IF(item.target_identity != target || item.max_registers < 0 ||
                    item.llvm_opt_level < 0 || item.llvm_opt_level > 3,
                "Invalid CUDA PTX AOT target or compiler contract for {}", name);
    std::vector<std::string> task_names;
    for (const auto &task : kernel_cache.compiled_data.tasks) {
      task_names.push_back(task.name);
    }
    auto entries = item.entry_names;
    std::sort(entries.begin(), entries.end());
    std::sort(task_names.begin(), task_names.end());
    TI_ERROR_IF(entries != task_names ||
                    std::adjacent_find(entries.begin(), entries.end()) != entries.end(),
                "CUDA PTX AOT entry manifest mismatch for {}", name);
    CUDAKernelArtifact artifact;
    artifact.payload.assign(bytes.begin(), bytes.end());
    artifact.payload.push_back('\0');
    artifact.entry_names = item.entry_names;
    artifact.target_identity = item.target_identity;
    artifact.provider_identity = "aot_ptx";
    artifact.max_registers = item.max_registers;
    artifact.fast_math = item.fast_math;
    artifact.llvm_opt_level = item.llvm_opt_level;
    auto data = std::move(kernel_cache).take_llvm_ckd_data();
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(kernel_launcher_.get());
    TI_ASSERT(launcher != nullptr);
    const auto handle = launcher->register_cuda_artifact_kernel(data, artifact);
    FunctionType fn = [handle, launcher](LaunchContextBuilder &ctx) {
      launcher->launch_llvm_kernel(handle, ctx);
    };
    return std::make_unique<llvm_aot::KernelImpl>(std::move(fn), kernel_name, data);
#else
    TI_ERROR("CUDA PTX artifacts require a build with CUDA support");
#endif
  }
  LLVM::CompiledKernelData compiled{
      arch(), std::move(kernel_cache).take_llvm_ckd_data()};
  auto fn = convert_module_to_function(compiled);
  // The launcher owns the executable. The callable retains its parameter/return
  // ABI, not an additional LLVM module for the entire AOT module lifetime.
  return std::make_unique<llvm_aot::KernelImpl>(
      std::move(fn), kernel_name, compiled.get_internal_data());
}

std::unique_ptr<aot::Field> LlvmAotModule::make_new_field(
    const std::string &name) {
  // Check if "name" represents snode_tree_id.
  // Avoid using std::atoi due to its poor error handling.
  char *end;
  int snode_tree_id = static_cast<int>(strtol(name.c_str(), &end, 10 /*base*/));

  TI_ASSERT(end != name.c_str());
  TI_ASSERT(*end == '\0');

  // Load FieldCache
  LlvmOfflineCache::FieldCacheData loaded;
  auto ok = cache_reader_->get_field_cache(loaded, snode_tree_id);
  TI_ERROR_IF(!ok, "Failed to load field with id={}", snode_tree_id);

  return std::make_unique<llvm_aot::FieldImpl>(std::move(loaded));
}

std::unique_ptr<aot::CompiledGraph> LlvmAotModule::get_graph(
    const std::string &name) {
  auto it = graphs_.find(name);
  if (it == graphs_.end()) {
    TI_DEBUG("Cannot find graph {}", name);
    return nullptr;
  }

  std::vector<aot::CompiledDispatch> dispatches;
  for (auto &dispatch : it->second.dispatches) {
    dispatches.push_back({dispatch.kernel_name, {}, dispatch.symbolic_args,
                          get_kernel(dispatch.kernel_name)});
  }

  aot::CompiledGraph graph = aot::CompiledGraph({dispatches});

  return std::make_unique<aot::CompiledGraph>(std::move(graph));
}

void allocate_aot_snode_tree_type(aot::Module *aot_module,
                                  aot::Field *aot_field,
                                  uint64 *result_buffer) {
  auto *llvm_aot_module = dynamic_cast<LlvmAotModule *>(aot_module);
  auto *aot_field_impl = dynamic_cast<llvm_aot::FieldImpl *>(aot_field);

  TI_ASSERT(llvm_aot_module != nullptr);
  TI_ASSERT(aot_field_impl != nullptr);

  auto *runtime_executor = llvm_aot_module->get_runtime_executor();
  const auto &field_cache = aot_field_impl->get_snode_tree_cache();

  int snode_tree_id = field_cache.tree_id;
  if (!llvm_aot_module->is_snode_tree_initialized(snode_tree_id)) {
    runtime_executor->initialize_llvm_runtime_snodes(field_cache,
                                                     result_buffer);
    llvm_aot_module->set_initialized_snode_tree(snode_tree_id);
  }
}

std::unique_ptr<aot::Module> make_aot_module(AotModuleParams mod_params) {
  return std::make_unique<LlvmAotModule>(mod_params.module_path,
                                         mod_params.executor_,
                                         std::move(mod_params.kernel_launcher));
}

}  // namespace LLVM
}  // namespace taichi::lang
