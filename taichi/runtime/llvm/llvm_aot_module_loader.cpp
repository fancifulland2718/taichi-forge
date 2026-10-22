#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

#include <algorithm>

#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/aot_kernel_contract.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/aot_snode_resources.h"
#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"
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
    std::unique_ptr<LLVM::KernelLauncher> kernel_launcher,
    uint64 *allocation_result_buffer)
    : executor_(executor),
      allocation_result_buffer_(allocation_result_buffer),
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
  if (!metadata.kernel_runtime_state.empty())
    TI_ERROR_IF(
        metadata.runtime_state_sha256 != aot_runtime_state_sha256(metadata),
        "AOT runtime-state inventory checksum mismatch");
  if (!metadata.kernel_launch_abis.empty()) {
    const bool has_gate =
        std::any_of(metadata.kernel_launch_abis.begin(),
                    metadata.kernel_launch_abis.end(), [](const auto &entry) {
                      return cuda::is_graph_gate_launch_abi(entry.second);
                    });
    TI_ERROR_IF(has_gate && arch() != Arch::cuda,
                "AOT CUDA gate entries require a CUDA runtime");
    TI_ERROR_IF(metadata.launch_abi_sha256 != aot_launch_abi_sha256(metadata),
                "AOT kernel launch ABI declaration checksum mismatch");
  }

  if (is_cuda_aot_artifact_format(metadata.kernel_format)) {
    TI_ERROR_IF(arch() != Arch::cuda, "CUDA device artifacts require a CUDA runtime");
  }
  if (is_cuda_aot_artifact_format(metadata.kernel_format) ||
      !metadata.resource_binding.empty() ||
      !metadata.kernel_launch_abis.empty() ||
      !metadata.kernel_runtime_state.empty()) {
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
  for (const auto &[name, abi] : metadata.kernel_launch_abis) {
    TI_ERROR_IF(!cache_reader_->has_kernel(name),
                "AOT launch ABI references an absent kernel: {}", name);
  }
  for (const auto &[name, state] : metadata.kernel_runtime_state)
    TI_ERROR_IF(!cache_reader_->has_kernel(name),
                "AOT runtime-state inventory references an absent kernel: {}",
                name);

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
  const bool relocatable =
      is_relocatable_aot_resource_binding(metadata.resource_binding);
  const bool exact_lists = metadata.resource_binding == "list_v1";
  const bool allows_sparse =
      exact_lists || is_sparse_aot_resource_binding(metadata.resource_binding);
  bool sparse = false;
#if defined(TI_WITH_CUDA)
  if (arch() == Arch::cuda &&
      (metadata.resource_binding == "sparse_v2" || exact_lists)) {
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(kernel_launcher_.get());
    TI_ASSERT(launcher != nullptr);
    launcher->enable_sparse_graph_replay();
  }
#endif
  std::vector<size_t> root_sizes;
  // Validate the complete resource closure before reserving or allocating any
  // tree. Legacy artifacts retain their original ID collision contract.
  for (std::size_t i = 0; i < snode_tree_ids_.size(); ++i) {
    const int id = snode_tree_ids_[i];
    LLVM::SNodeTreeLayout field;
    TI_ERROR_IF(
        !cache_reader_->get_field_cache(field, id) || field.tree_id != id,
        "Invalid LLVM AOT field metadata for tree {}", id);
    root_sizes.push_back(field.root_size);
    for (const auto &node : field.snode_metas)
      sparse |= !is_dense_aot_snode(node.type);
    if (relocatable) {
      TI_ERROR_IF(id != i, "Invalid relocatable LLVM AOT tree {}", id);
      validate_relocatable_aot_field(field, allows_sparse);
    } else {
      TI_ERROR_IF(executor_->has_snode_tree_binding(id),
                  "LLVM AOT SNode tree id {} conflicts with an active module; "
                  "load this artifact in a separate runtime.",
                  id);
    }
  }
  // This single node-to-tree index also rejects duplicate module SNode IDs.
  const auto list_trees = cache_reader_->get_element_list_tree_ids(exact_lists);
  TI_ERROR_IF(metadata.resource_binding == "dense_v1" && !list_trees.empty(),
              "Dense LLVM AOT binding cannot contain list tasks");
  dense_root_storage_ = relocatable && !sparse;
  TI_ERROR_IF(!metadata.initial_root_data.empty() && !dense_root_storage_,
              "AOT initial root state requires relocatable dense trees.");
  aot::validate_initial_root_data(metadata.initial_root_data, root_sizes);
  if (allocation_result_buffer_) {
    // Preserve load-time integrity failures without allocating unused device
    // roots or retaining expanded snapshots. Allocation revalidates the
    // selected payload, including directory-backed artifacts changed later.
    for (const auto &entry : metadata.initial_root_data)
      load_data(entry.sha256, entry.size);
  }
  root_storage_views_ = aot::load_root_storage_views(*directory_);
  TI_ERROR_IF(!root_storage_views_.empty() && (!relocatable || sparse),
              "AOT root storage views require relocatable dense trees.");
  root_storage_sizes_ = std::move(root_sizes);
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
    for (const auto &entry : metadata.initial_root_data) {
      initial_root_data_.emplace(tree_bindings_.at(entry.tree_id), entry);
    }
    for (int id : list_trees)
      element_list_tree_ids_.insert(tree_bindings_.at(id));
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
      LLVM::SNodeTreeLayout field;
      TI_ASSERT(cache_reader_->get_field_cache(field, logical));
      field.tree_id = physical;
      executor_->destroy_snode_tree_runtime(field);
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
      TI_ERROR_IF(!is_relocatable_aot_task(
                      task.task_type, metadata_.resource_binding != "dense_v1"),
                  "Task is incompatible with the LLVM AOT resource binding");
    }
    if (loaded.compiled_data.module) {
      std::map<int, int> bindings;
      for (int id : loaded.used_snode_tree_ids)
        bindings.emplace(id, tree_bindings_.at(id));
      resolve_aot_tree_ids(*loaded.compiled_data.module, bindings);
    }
  }
  for (int &id : loaded.used_snode_tree_ids) {
    ensure_snode_tree_initialized(id);
    if (!metadata_.resource_binding.empty())
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
  const auto declaration = metadata_.kernel_launch_abis.find(name);
  const std::string launch_abi =
      declaration == metadata_.kernel_launch_abis.end() ? "ordinary"
                                                        : declaration->second;
  const bool gated = cuda::is_graph_gate_launch_abi(launch_abi);
  const auto context_abi = cuda::launch_runtime_context_abi(launch_abi);
  const auto state_declaration = metadata_.kernel_runtime_state.find(name);
  const auto runtime_state =
      state_declaration == metadata_.kernel_runtime_state.end()
          ? 0u
          : state_declaration->second;
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
    const auto handle = launcher->register_cuda_artifact_kernel(
        data, artifact, bindings,
        gated ? cuda::KernelLaunchABI::graph_gate_v1
              : cuda::KernelLaunchABI::ordinary,
        context_abi, runtime_state);
    return std::make_unique<llvm_aot::KernelImpl>(
        launcher, handle, arch(), kernel_name, data);
#else
    TI_ERROR("CUDA device artifacts require a build with CUDA support");
#endif
  }
  LLVM::CompiledKernelData compiled{
      arch(), std::move(kernel_cache).take_llvm_ckd_data()};
  TI_ERROR_IF(state_declaration != metadata_.kernel_runtime_state.end() &&
                  LLVM::runtime_state_use(
                      *compiled.get_internal_data().compiled_data.module) !=
                      runtime_state,
              "AOT LLVM code does not match its runtime-state inventory: {}",
              name);
  TI_ERROR_IF(
      LLVM::runtime_context_abi(
          *compiled.get_internal_data().compiled_data.module) != context_abi,
      "AOT LLVM code does not match its RuntimeContext layout: {}", name);
  TI_ERROR_IF(compiled_kernel_launch_abi(
                  *compiled.get_internal_data().compiled_data.module) != launch_abi,
              "AOT LLVM code does not match its kernel launch ABI: {}", name);
  KernelLaunchHandle handle;
  KernelRegistrationState registration;
  if (gated) {
#if defined(TI_WITH_CUDA)
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(kernel_launcher_.get());
    TI_ASSERT(launcher != nullptr);
    handle = launcher->register_llvm_kernel(
        compiled, registration, cuda::KernelLaunchABI::graph_gate_v1);
#else
    TI_ERROR("CUDA gate artifacts require a build with CUDA support");
#endif
  } else {
    handle = kernel_launcher_->register_llvm_kernel(compiled, registration);
  }
  // The launcher owns the executable. The callable retains its parameter/return
  // ABI, not an additional LLVM module for the entire AOT module lifetime.
  return std::make_unique<llvm_aot::KernelImpl>(
      kernel_launcher_.get(), handle, arch(), kernel_name,
      compiled.get_internal_data());
}

void LlvmAotModule::ensure_snode_tree_initialized(int logical_tree_id) {
  TI_ERROR_IF(is_corrupted(), "Cannot initialize an invalid LLVM AOT module.");
  const int physical_tree_id = tree_bindings_.at(logical_tree_id);
  if (!allocation_result_buffer_ ||
      is_snode_tree_initialized(physical_tree_id))
    return;
  auto *field = get_snode_tree(std::to_string(logical_tree_id));
  allocate_aot_snode_tree_type(this, field, allocation_result_buffer_);
}

LlvmAotModule::RootStorage LlvmAotModule::get_root_storage(int tree_id) {
  const auto binding = tree_bindings_.find(tree_id);
  if (!dense_root_storage_ ||
      binding == tree_bindings_.end() || tree_id < 0 ||
      static_cast<std::size_t>(tree_id) >= root_storage_sizes_.size())
    throw std::invalid_argument(
        "AOT root storage requires an initialized module-local dense tree");
  ensure_snode_tree_initialized(tree_id);
  if (!is_snode_tree_initialized(binding->second))
    throw std::invalid_argument(
        "AOT root storage requires an initialized module-local dense tree");
  return {executor_->get_snode_tree_device_ptr(binding->second),
          root_storage_sizes_[tree_id]};
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
  LLVM::SNodeTreeLayout loaded;
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

void LlvmAotModule::initialize_snode_tree(const LLVM::SNodeTreeLayout &field,
                                          uint64 *result_buffer) {
  TI_ERROR_IF(is_corrupted(), "Cannot initialize an invalid LLVM AOT module.");
  if (is_snode_tree_initialized(field.tree_id))
    return;
  executor_->initialize_llvm_runtime_snodes(
      field, result_buffer, element_list_tree_ids_.count(field.tree_id) != 0);
  // Record ownership before upload so a failed payload also retires the root.
  set_initialized_snode_tree(field.tree_id);
  const auto initial = initial_root_data_.find(field.tree_id);
  if (initial != initial_root_data_.end()) {
    try {
      initialize_root_data(initial->second,
                           executor_->get_snode_tree_device_ptr(field.tree_id));
    } catch (...) {
      mark_corrupted();
      throw;
    }
  }
}

void allocate_aot_snode_tree_type(aot::Module *aot_module,
                                  aot::Field *aot_field,
                                  uint64 *result_buffer) {
  auto *llvm_aot_module = dynamic_cast<LlvmAotModule *>(aot_module);
  auto *aot_field_impl = dynamic_cast<llvm_aot::FieldImpl *>(aot_field);

  TI_ASSERT(llvm_aot_module != nullptr);
  TI_ASSERT(aot_field_impl != nullptr);

  const auto &field_cache = aot_field_impl->get_snode_tree_cache();
  llvm_aot_module->initialize_snode_tree(field_cache, result_buffer);
}

std::unique_ptr<aot::Module> make_aot_module(AotModuleParams mod_params) {
  return std::make_unique<LlvmAotModule>(mod_params.module_path,
                                         mod_params.executor_,
                                         std::move(mod_params.kernel_launcher),
                                         mod_params.allocation_result_buffer);
}

}  // namespace LLVM
}  // namespace taichi::lang
