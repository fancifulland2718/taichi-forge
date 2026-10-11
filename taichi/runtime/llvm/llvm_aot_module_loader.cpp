#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

#include <algorithm>

#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#if defined(TI_WITH_CUDA)
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

namespace taichi::lang {
namespace LLVM {
namespace {

std::string read_artifact(const io::VirtualDir &directory,
                          const std::string &name) {
  size_t size = 0;
  TI_ERROR_IF(!directory.get_file_size(name, size),
              "Cannot find AOT artifact {}", name);
  std::string bytes(size, '\0');
  TI_ERROR_IF(directory.load_file(name, bytes.data(), size) != size,
              "Failed to read AOT artifact {}", name);
  return bytes;
}

}  // namespace

LlvmAotModule::LlvmAotModule(
    const std::string &module_path,
    LlvmRuntimeExecutor *executor,
    std::unique_ptr<LLVM::KernelLauncher> kernel_launcher)
    : executor_(executor),
      kernel_launcher_(std::move(kernel_launcher)),
      directory_(io::VirtualDir::open(module_path)) {
  TI_ASSERT(executor_ != nullptr);

  size_t metadata_size = 0;
  TI_ERROR_IF(!directory_ || !directory_->get_file_size(
                                 kLlvmAotMetadataFilename, metadata_size),
              "LLVM AOT artifact is missing required schema metadata at {}. "
              "Rebuild the artifact with the current Forge AOT compiler.",
              taichi::join_path(module_path, kLlvmAotMetadataFilename));
  const auto contents = read_artifact(*directory_, kLlvmAotMetadataFilename);
  auto &metadata = metadata_;
  auto json = liong::json::parse(contents.data(),
                                 contents.data() + contents.size());
  liong::json::deserialize(json, metadata);
  validate_llvm_aot_metadata(metadata);

  if (is_cuda_aot_artifact_format(metadata.kernel_format)) {
    TI_ERROR_IF(arch() != Arch::cuda, "CUDA device artifacts require a CUDA runtime");
  }
  if (is_cuda_aot_artifact_format(metadata.kernel_format) || !metadata.resource_binding.empty()) {
    TI_ERROR_IF(aot_artifact_sha256(read_artifact(
                    *directory_, "metadata.tcb")) != metadata.abi_sha256,
                "LLVM AOT launch ABI checksum mismatch");
  }

  // Select exactly the declared representation. Missing/corrupt payloads must
  // not silently fall back to a different generation left in this directory.
  const auto format = metadata.kernel_format == "bc"
                          ? LlvmOfflineCache::Format::BC
                          : (is_cuda_aot_artifact_format(metadata.kernel_format)
                                 ? LlvmOfflineCache::Format::CUDAArtifact
                                 : LlvmOfflineCache::Format::LL);
  cache_reader_ = LlvmOfflineCacheFileReader::make(directory_, format);
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

  if (!aot::load_struct_graph_artifact(*directory_, graphs_)) {
    size_t graph_size = 0;
    if (directory_->get_file_size("graphs.tcb", graph_size)) {
      const auto bytes = read_artifact(*directory_, "graphs.tcb");
      TI_ERROR_IF(bytes.size() < sizeof(size_t) ||
                      !read_from_binary(graphs_, bytes.data(), bytes.size()),
                  "Invalid LLVM AOT Graph metadata");
    }
  }
  const bool relocatable = metadata.resource_binding == "dense_v1";
  // Validate the complete resource closure before reserving or allocating any
  // tree. Legacy artifacts retain their original ID collision contract.
  for (std::size_t i = 0; i < snode_tree_ids_.size(); ++i) {
    const int id = snode_tree_ids_[i];
    LlvmOfflineCache::FieldCacheData field;
    TI_ERROR_IF(
        !cache_reader_->get_field_cache(field, id) || field.tree_id != id,
        "Invalid LLVM AOT field metadata for tree {}", id);
    if (relocatable) {
      TI_ERROR_IF(id != i ||
                      !executor_->get_config().demote_dense_struct_fors ||
                      field.tree_generation == 0 || field.snode_metas.empty(),
                  "Invalid relocatable LLVM AOT dense tree {}", id);
      std::vector<bool> seen(field.snode_metas.size(), false);
      bool found_root = false;
      for (const auto &node : field.snode_metas) {
        TI_ERROR_IF(
            (node.type != SNodeType::root && node.type != SNodeType::dense &&
             node.type != SNodeType::place) ||
                node.runtime_local_id < 0 ||
                std::size_t(node.runtime_local_id) >= seen.size() ||
                seen[node.runtime_local_id],
            "Invalid relocatable LLVM AOT dense node in tree {}", id);
        seen[node.runtime_local_id] = true;
        found_root |= node.type == SNodeType::root &&
                      node.id == field.root_id &&
                      node.runtime_local_id == field.root_runtime_local_id;
      }
      TI_ERROR_IF(!found_root, "Missing relocatable LLVM AOT root in tree {}",
                  id);
    } else {
      TI_ERROR_IF(executor_->has_snode_tree_binding(id),
                  "LLVM AOT SNode tree id {} conflicts with an active module; "
                  "load this artifact in a separate runtime.",
                  id);
    }
  }
  if (relocatable) {
    reserved_tree_ids_ =
        executor_->reserve_aot_snode_tree_ids(snode_tree_ids_.size());
  } else {
    // Even native callers that defer field allocation must not let a second
    // module claim an ID already owned by this artifact.
    reserved_tree_ids_ = snode_tree_ids_;
    executor_->reserve_aot_snode_tree_ids(reserved_tree_ids_);
  }
  try {
    for (std::size_t i = 0; i < snode_tree_ids_.size(); ++i) {
      tree_bindings_.emplace(snode_tree_ids_[i], relocatable
                                                     ? reserved_tree_ids_[i]
                                                     : snode_tree_ids_[i]);
    }
  } catch (...) {
    executor_->release_aot_snode_tree_ids(reserved_tree_ids_);
    throw;
  }
}

LlvmAotModule::~LlvmAotModule() {
  // C API modules must be destroyed before their runtime. Finish their submitted
  // kernels before dropping executable modules, retained launch buffers or roots.
  try {
    executor_->synchronize();
    kernel_launcher_->retire_all_kernels();
    for (const auto &[logical, physical] : tree_bindings_) {
      if (!is_snode_tree_initialized(physical))
        continue;
      LlvmOfflineCache::FieldCacheData field;
      TI_ASSERT(cache_reader_->get_field_cache(field, logical));
      field.tree_id = physical;
      executor_->destroy_dense_aot_snode_tree(field);
    }
    executor_->release_aot_snode_tree_ids(reserved_tree_ids_);
  } catch (const std::exception &error) {
    TI_WARN("Failed to retire LLVM AOT resources: {}", error.what());
  } catch (const std::string &error) {
    TI_WARN("Failed to retire LLVM AOT resources: {}", error);
  } catch (...) {
    TI_WARN("Failed to retire LLVM AOT resources");
  }
}

LlvmOfflineCache::KernelCacheData LlvmAotModule::load_kernel_from_cache(
    const std::string &name) {
  TI_ASSERT(cache_reader_ != nullptr);
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = is_cuda_aot_artifact_format(metadata_.kernel_format)
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
  if (!metadata_.resource_binding.empty()) {
    TI_ERROR_IF(
        !std::is_sorted(loaded.used_snode_tree_ids.begin(),
                        loaded.used_snode_tree_ids.end()) ||
            std::adjacent_find(loaded.used_snode_tree_ids.begin(),
                               loaded.used_snode_tree_ids.end()) !=
                loaded.used_snode_tree_ids.end(),
        "Relocatable LLVM AOT tree dependencies must be sorted and unique");
    for (const auto &task : loaded.compiled_data.tasks) {
      TI_ERROR_IF(task.task_type != OffloadedTaskType::serial &&
                      task.task_type != OffloadedTaskType::range_for,
                  "Relocatable LLVM AOT requires demoted dense tasks");
    }
    if (loaded.compiled_data.module) {
      std::map<int, int> bindings;
      for (int id : loaded.used_snode_tree_ids)
        bindings.emplace(id, tree_bindings_.at(id));
      resolve_aot_tree_ids(*loaded.compiled_data.module, bindings);
    }
    for (int &id : loaded.used_snode_tree_ids)
      id = tree_bindings_.at(id);
  }
  return loaded;
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel(
    const std::string &name) {
  if (!is_cuda_aot_artifact_format(metadata_.kernel_format) && metadata_.resource_binding.empty()) {
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
  auto message =
      fmt::format("Failed to load LLVM AOT kernel {}: {}", name, diagnostic);
  artifact_failures_.emplace(name, message);
  throw aot::ArtifactLoadError(message);
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel_impl(
    const std::string &name) {
  if (!cache_reader_->has_kernel(name)) return nullptr;
  auto kernel_cache = load_kernel_from_cache(name);
  const auto kernel_name = std::move(kernel_cache.kernel_key);
  if (is_cuda_aot_artifact_format(metadata_.kernel_format)) {
#if defined(TI_WITH_CUDA)
    const bool cubin = metadata_.kernel_format == "cubin";
    if (cubin) {
      const auto provider = metadata_.cuda_cubin_providers.find(name);
      TI_ERROR_IF(provider == metadata_.cuda_cubin_providers.end() ||
                      provider->second.empty(),
                  "Missing CUDA cubin AOT provider identity for kernel {}", name);
    }
    const auto it = metadata_.cuda_artifacts.find(name);
    TI_ERROR_IF(it == metadata_.cuda_artifacts.end(),
                "Missing CUDA AOT manifest for kernel {}", name);
    const auto &item = it->second;
    TI_ERROR_IF((cubin || !metadata_.resource_binding.empty()) &&
                    item.contract_sha256 != aot_cuda_contract_sha256(item),
                "CUDA AOT resource contract checksum mismatch for {}",
                name);
    auto bytes =
        read_artifact(*directory_, name + "." + metadata_.kernel_format);
    TI_ERROR_IF(bytes.empty() || bytes.size() != item.size ||
                    aot_artifact_sha256(bytes) != item.sha256 ||
                    (!cubin && bytes.find('\0') != std::string::npos),
                "CUDA AOT payload integrity failure for kernel {}", name);
    TI_ERROR_IF(cubin && (bytes.size() < 64 || bytes[0] != '\177' ||
                         bytes.compare(1, 3, "ELF") != 0),
                "CUDA cubin AOT payload is not an ELF image for kernel {}", name);
    const auto target = fmt::format("sm_{}|+ptx{}",
        required_caps_.get(DeviceCapability::cuda_compute_capability),
        required_caps_.get(DeviceCapability::cuda_ptx_version));
    TI_ERROR_IF(item.target_identity != target || item.max_registers < 0 ||
                    item.llvm_opt_level < 0 || item.llvm_opt_level > 3,
                "Invalid CUDA AOT target or compiler contract for {}", name);
    std::vector<std::string> task_names;
    for (const auto &task : kernel_cache.compiled_data.tasks) {
      task_names.push_back(task.name);
    }
    auto entries = item.entry_names;
    std::sort(entries.begin(), entries.end());
    std::sort(task_names.begin(), task_names.end());
    TI_ERROR_IF(entries != task_names ||
                    std::adjacent_find(entries.begin(), entries.end()) != entries.end(),
                "CUDA AOT entry manifest mismatch for {}", name);
    CUDAKernelArtifact artifact;
    artifact.kind = cubin ? CUDAArtifactKind::cubin : CUDAArtifactKind::ptx;
    artifact.payload.assign(bytes.begin(), bytes.end());
    if (!cubin)
      artifact.payload.push_back('\0');
    artifact.entry_names = item.entry_names;
    artifact.target_identity = item.target_identity;
    artifact.provider_identity = cubin
        ? "aot_cubin:" + metadata_.cuda_cubin_providers.at(name) : "aot_ptx";
    artifact.max_registers = item.max_registers;
    artifact.fast_math = item.fast_math;
    artifact.llvm_opt_level = item.llvm_opt_level;
    std::map<int, int> bindings;
    for (int id : item.tree_relocations) {
      const auto binding = tree_bindings_.find(id);
      TI_ERROR_IF(
          metadata_.resource_binding.empty() ||
              binding == tree_bindings_.end() ||
              !std::binary_search(kernel_cache.used_snode_tree_ids.begin(),
                                  kernel_cache.used_snode_tree_ids.end(),
                                  binding->second) ||
              !bindings.emplace(id, binding->second).second,
          "Invalid CUDA AOT resource relocation {}", id);
    }
    auto data = std::move(kernel_cache).take_llvm_ckd_data();
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(kernel_launcher_.get());
    TI_ASSERT(launcher != nullptr);
    const auto handle =
        launcher->register_cuda_artifact_kernel(data, artifact, bindings);
    return std::make_unique<llvm_aot::KernelImpl>(
        launcher, handle, arch(), kernel_name, data);
#else
    TI_ERROR("CUDA device artifacts require a build with CUDA support");
#endif
  }
  LLVM::CompiledKernelData compiled{
      arch(), std::move(kernel_cache).take_llvm_ckd_data()};
  const auto handle = kernel_launcher_->register_llvm_kernel(compiled);
  // The launcher owns the executable. The callable retains its parameter/return
  // ABI, not an additional LLVM module for the entire AOT module lifetime.
  return std::make_unique<llvm_aot::KernelImpl>(
      kernel_launcher_.get(), handle, arch(), kernel_name,
      compiled.get_internal_data());
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
  loaded.tree_id = tree_bindings_.at(snode_tree_id);

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
