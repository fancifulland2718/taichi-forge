#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

#include <algorithm>
#include <cstring>
#include <climits>

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "taichi/runtime/llvm/llvm_module_options.h"
#include "taichi/aot/specialization.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/ir/type_factory.h"
#include "taichi/ir/type_utils.h"

#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/aot_kernel_contract.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/aot_snode_resources.h"
#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/codegen/llvm/range_dispatch.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"
#include "taichi/runtime/llvm/array_layout_abi.h"
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
  TI_ERROR_IF(metadata.schema_version != kDynamicSharedAotSchemaVersion,
              "LLVM AOT task layout requires schema {}. Rebuild the artifact "
              "with the current Forge AOT compiler.",
              kDynamicSharedAotSchemaVersion);
  {
    bool aliases = false, ranges = false, checksum = false;
    for (const auto &[name, value] : json.fields()) {
      if (name == "kernel_aliases") {
        liong::json::deserialize(value, metadata.kernel_aliases, true);
        aliases = true;
      } else if (name == "device_range_dispatch") {
        aot::decode_device_range_dispatch(value, metadata.device_range_dispatch);
        ranges = true;
      } else if (name == "kernel_entry_sha256") {
        checksum = true;
      }
    }
    TI_ERROR_IF(!aliases || !ranges || !checksum,
                "AOT kernel entry policy fields are incomplete");
  }
  validate_llvm_aot_metadata(metadata);
  TI_ERROR_IF(metadata.kernel_runtime_affine_arrays.empty()
                  ? !metadata.array_layout_sha256.empty()
                  : metadata.abi_sha256.size() != 64 ||
                        metadata.array_layout_sha256 != aot_array_layout_sha256(metadata),
              "AOT ndarray layout qualification checksum mismatch");
  if (!metadata.kernel_entry_sha256.empty())
    TI_ERROR_IF(metadata.kernel_entry_sha256 != aot_kernel_entry_sha256(metadata),
                "AOT kernel entry policy checksum mismatch");
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
      !metadata.kernel_runtime_state.empty() ||
      !metadata.kernel_runtime_affine_arrays.empty() ||
      metadata.schema_version == kDynamicSharedAotSchemaVersion) {
    TI_ERROR_IF(aot_artifact_sha256(read_artifact(
                    *directory_, "metadata.tcb")) != metadata.abi_sha256,
                "LLVM AOT launch ABI checksum mismatch");
  }

  // Select exactly the declared representation. Missing/corrupt payloads must
  // not silently fall back to a different generation left in this directory.
  const auto format = is_llvm_aot_bitcode_format(metadata.kernel_format)
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
  for (const auto &[name, arrays] : metadata.kernel_runtime_affine_arrays) {
    TI_ERROR_IF(!cache_reader_->has_kernel(name) || arrays.empty(),
                "AOT ndarray layout qualification references an absent kernel");
    for (const auto &argument : arrays)
      TI_ERROR_IF(argument.empty() || std::any_of(argument.begin(), argument.end(), [](int i) { return i < 0; }),
                  "AOT ndarray layout qualification has an invalid argument path");
  }

  // Validate the entire entry inventory while the reader still owns all task
  // metadata. take_kernel_* later transfers each physical payload exactly once.
  for (const auto &[name, code] : metadata.kernel_aliases) {
    TI_ERROR_IF(name.empty() || name.find('\0') != std::string::npos ||
                    cache_reader_->has_kernel(name) ||
                    metadata.kernel_aliases.count(code) ||
                    !cache_reader_->has_kernel(code),
                "Invalid LLVM AOT kernel alias: {} -> {}", name, code);
  }
  for (const auto &entry : metadata.device_range_dispatch) {
    const auto alias = metadata.kernel_aliases.find(entry.name);
    const auto &code = alias == metadata.kernel_aliases.end() ? entry.name : alias->second;
    const auto *contract = cache_reader_->kernel_metadata(code);
    const auto abi = metadata.kernel_launch_abis.find(code);
    TI_ERROR_IF(arch() != Arch::cuda || entry.name.empty() ||
                    entry.name.find('\0') != std::string::npos || entry.version != 1 ||
                    entry.tasks.empty() || !contract ||
                    abi == metadata.kernel_launch_abis.end() ||
                    abi->second != cuda::kContextLaunchABI ||
                    !range_selections_.emplace(entry.name, entry.tasks).second,
                "Invalid LLVM AOT range dispatch entry: {}", entry.name);
    validate_device_range_selection(contract->compiled_data.tasks, entry.tasks,
                                     taichi_global_tmp_buffer_size);
  }

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
  kernel_tree_dependencies_ = cache_reader_->get_kernel_tree_dependencies();

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
    const std::string &name, LlvmOfflineCacheFileReader *reader) {
  if (!reader)
    reader = cache_reader_.get();
  TI_ASSERT(reader != nullptr);
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = is_cuda_aot_artifact_format(metadata_.kernel_format)
                ? reader->take_kernel_metadata(loaded, name)
                : reader->take_kernel_cache(
                      loaded, name,
                      *executor_->get_llvm_context()->get_this_thread_context());
  if (!ok) {
    throw aot::ArtifactLoadError("Failed to read LLVM AOT kernel: " + name);
  }
  for (const auto &task : loaded.compiled_data.tasks) {
    TI_ERROR_IF(task.host_block_dim &&
                    (arch() != Arch::cuda || task.task_type != OffloadedTaskType::range_for ||
                     task.block_dim <= 0 || !task.host_block_dim->valid(loaded.args_size)),
                "AOT model block size requires a valid CUDA range argument expression");
    if (!task.dynamic_shared_arrays.empty()) {
      TI_ERROR_IF(arch() != Arch::cuda || task.task_type != OffloadedTaskType::range_for ||
                      task.dynamic_shared_array_bytes != 0,
                  "AOT dynamic shared memory requires a CUDA range task");
      for (const auto &array : task.dynamic_shared_arrays)
        for (const auto &extent : array.extents)
          TI_ERROR_IF(!extent.valid(loaded.args_size),
                      "AOT shared-memory extent references an invalid argument");
    }
    TI_ERROR_IF((task.requested_grid_residency_waves != 0 ||
                 task.requested_range_work_per_thread_target != 1) &&
                    (arch() != Arch::cuda ||
                     metadata_.schema_version != kDynamicSharedAotSchemaVersion),
                "Persisted CUDA task launch policy requires AOT schema {}: {}",
                kDynamicSharedAotSchemaVersion, name);
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

aot::Kernel *LlvmAotModule::specialize_kernel(
    aot::Kernel &kernel, LaunchContextBuilder &context,
    const std::vector<uint32_t> &constant_args) {
  auto *source = dynamic_cast<llvm_aot::KernelImpl *>(&kernel);
  TI_ERROR_IF((!arch_is_cpu(arch()) && arch() != Arch::cuda) || !is_llvm_aot_bitcode_format(metadata_.kernel_format) || !source ||
                  !source->device_range_tasks().empty(),
              "Prepared specialization requires an ordinary CPU/CUDA bitcode kernel");
  const auto alias = metadata_.kernel_aliases.find(kernel.name);
  const auto &code = alias == metadata_.kernel_aliases.end() ? kernel.name : alias->second;
  TI_ERROR_IF(!payload_kernels_.count(code), "Specialization kernel is not owned by this module");

  const auto constants = aot::specialization_constants(kernel, context, constant_args);
  std::string key = aot::specialization_key(code, constants);

  // Recover only relations proven by the already-bound allocation owners.
  // Two wrappers or nearby addresses do not prove a common allocation. The
  // cache records ABI slots and relative offsets, never allocation IDs or
  // device addresses, so identical layouts can share code across instances.
  struct PointerRelation {
    int base_slot;
    int64_t displacement;
  };
  std::map<int, PointerRelation> pointers;
  std::map<std::pair<Device *, DeviceAllocationId>, std::pair<int, uint64_t>> roots;
  for (const auto &[path, parameter] : kernel.nested_parameters) {
    if (parameter.ptype != ParameterType::kNdarray)
      continue;
    const auto kind = context.device_allocation_type.find(path);
    if (kind == context.device_allocation_type.end())
      continue;
    for (int member : {TypeFactory::DATA_PTR_POS_IN_NDARRAY,
                       TypeFactory::GRAD_PTR_POS_IN_NDARRAY}) {
      if (member == TypeFactory::GRAD_PTR_POS_IN_NDARRAY && !parameter.needs_grad)
        continue;
      auto pointer_path = path;
      pointer_path.push_back(member);
      DeviceAllocation allocation{kDeviceNullAllocation};
      uint64_t byte_offset = 0;
      if (kind->second == LaunchContextBuilder::DevAllocType::kDenseStorage) {
        const auto *resolved = context.find_resolved_dense_storage_buffer(pointer_path);
        if (!resolved || !resolved->valid)
          continue;
        allocation = resolved->allocation;
        byte_offset = resolved->byte_offset;
      } else if (kind->second == LaunchContextBuilder::DevAllocType::kNdarray) {
        const auto found = context.array_ptrs.find(pointer_path);
        if (found == context.array_ptrs.end() || !found->second)
          continue;
        allocation = *static_cast<const DeviceAllocation *>(found->second);
      } else {
        continue;
      }
      if (!allocation.device)
        continue;
      const int slot = kernel.args_type->get_element_offset(pointer_path);
      const auto [root, inserted] = roots.emplace(
          std::make_pair(allocation.device, allocation.alloc_id),
          std::make_pair(slot, byte_offset));
      if (inserted)
        continue;
      const auto base_offset = root->second.second;
      const auto distance = byte_offset >= base_offset ? byte_offset - base_offset
                                                     : base_offset - byte_offset;
      // No new input limit: an unrepresentable relation keeps its original
      // pointer load. Signed offsets also support reversed and aliased views.
      if (distance > static_cast<uint64_t>(INT64_MAX))
        continue;
      const auto displacement = byte_offset >= base_offset
          ? static_cast<int64_t>(distance) : -static_cast<int64_t>(distance);
      pointers.emplace(slot, PointerRelation{root->second.first, displacement});
    }
  }
  for (const auto &[slot, relation] : pointers)
    key += fmt::format("|p:{}:{}:{}", slot, relation.base_slot, relation.displacement);
  const auto cached = specialized_kernels_.find(key);
  if (cached != specialized_kernels_.end())
    return cached->second.get();

  // Ordinary loading consumes its reader's IR. Reopen only on a specialization
  // miss instead of retaining every generic module in compiler memory. Reuse
  // the same artifact decoder, task contracts and SNode relocation path.
  TI_ERROR_IF(aot_artifact_sha256(read_artifact(*directory_, "metadata.tcb")) != metadata_.abi_sha256,
              "AOT metadata changed after loading");
  auto reader = LlvmOfflineCacheFileReader::make(directory_, LlvmOfflineCache::Format::BC);
  TI_ERROR_IF(!reader, "Cannot reopen specialization bitcode metadata");
  auto data = load_kernel_from_cache(code, reader.get());
  auto &module = *data.compiled_data.module;
  TI_ERROR_IF(cuda::is_graph_gate_launch_abi(compiled_kernel_launch_abi(module)),
              "Prepared specialization does not accept gated graph kernels");
  // Select tasks from immutable prepared facts before lowering LLVM to device
  // code. A current argument value alone is not a constant: live scalar ranges
  // must remain callable with later values. Exclude TLS/collective/prologue
  // cases; the existing independent-iteration proof establishes that an empty
  // range has no effects outside its iterations.
  auto &tasks = data.compiled_data.tasks;
  const auto original_task_count = tasks.size();
  auto fixed_bound = [&](const HostRangeBound &bound) {
    auto fixed_input = [&](uint64_t offset) {
      if (offset > INT_MAX)
        return false;
      const auto found = constants.find(static_cast<int>(offset));
      return found != constants.end() && found->second.bits == 32;
    };
    for (auto offset : bound.offsets)
      if (!fixed_input(offset))
        return false;
    for (const auto &node : bound.program)
      if (node.is_input() && !fixed_input(node.offset))
        return false;
    return true;
  };
  for (auto it = tasks.begin(); it != tasks.end();) {
    const auto &range = it->range_bounds;
    if (tasks.size() == 1 || it->task_type != OffloadedTaskType::range_for || !range ||
        range->empty_group_elision != EmptyRangeGroupProof::independent_iterations ||
        !range->host_launch || !fixed_bound(range->host_launch->begin) ||
        !fixed_bound(range->host_launch->end)) {
      ++it;
      continue;
    }
    const auto &bounds = *range->host_launch;
    const auto first = bounds.begin.evaluate(context.get_context().arg_buffer, kernel.args_size);
    const auto last = bounds.end.evaluate(context.get_context().arg_buffer, kernel.args_size);
    auto *entry = module.getFunction(it->name);
    if (!first || !last || *last > *first || !entry || !entry->use_empty() ||
        std::count_if(tasks.begin(), tasks.end(), [&](const auto &task) {
          return task.name == it->name;
        }) != 1) {
      ++it;
      continue;
    }
    if (auto *annotations = module.getNamedMetadata("nvvm.annotations")) {
      std::vector<llvm::MDNode *> keep;
      for (auto *node : annotations->operands()) {
        const auto *value = node->getNumOperands()
            ? llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(node->getOperand(0).get()) : nullptr;
        if (!value || value->getValue() != entry)
          keep.push_back(node);
      }
      annotations->clearOperands();
      for (auto *node : keep)
        annotations->addOperand(node);
    }
    entry->eraseFromParent();
    it = tasks.erase(it);
  }
  // Keep one empty task when the entire invocation is empty, preserving the
  // existing registration/launch ABI without introducing another runtime path.
  TI_DEBUG("Prepared specialization {} pruned {} of {} tasks before registration",
           code, original_task_count - tasks.size(), original_task_count);
  const auto &layout = module.getDataLayout();
  std::set<llvm::Function *> contexts;
  for (const auto &task : data.compiled_data.tasks) {
    auto *entry = module.getFunction(task.name);
    TI_ERROR_IF(!entry || entry->arg_empty(), "Specialization task entry is missing");
    contexts.insert(entry);
  }
  if (arch_is_cpu(arch())) {
    for (auto &function : module) {
      if (function.hasLocalLinkage() && !function.isDeclaration() &&
          !function.arg_empty() && function.getArg(0)->getType()->isPointerTy() &&
          function.getAttributes().hasParamAttr(0, "taichi.aot.argument_context"))
        contexts.insert(&function);
    }
  }
  // Generated range callbacks receive exactly the parent's RuntimeContext.
  // Prove every use before including an internal function; a real_func can
  // instead receive a new context and must not inherit this argument layout.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &function : module) {
      if (!function.hasLocalLinkage() || function.isDeclaration() || function.arg_empty() ||
          function.use_empty() || contexts.count(&function))
        continue;
      bool proven = true;
      for (auto &use : function.uses()) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
        if (!call || !contexts.count(call->getFunction()) || call->arg_empty() ||
            call->getArgOperand(0) != call->getFunction()->getArg(0)) {
          proven = false;
          break;
        }
        if (call->getCalledFunction() == &function)
          continue;
        const auto *callee = call->getCalledFunction();
        const auto name = callee ? callee->getName() : llvm::StringRef{};
        const bool gpu_range = name == "gpu_parallel_range_for_with_tls" ||
                               name == "gpu_parallel_range_for";
        const bool cpu_range = name == "cpu_parallel_range_for" ||
                               name == "cpu_parallel_range_for_cancellable" ||
                               name == "cpu_parallel_range_for_bounded" ||
                               name == "cpu_parallel_range_for_bounded_cancellable";
        // CPU callbacks preserve the argument buffer in their thread-local
        // context copy; only cpu_thread_id changes. No other context fields
        // are eligible for folding below.
        const unsigned first_callback = gpu_range ? 3 : 6;
        if (!callee || !call->isArgOperand(&use) || (!gpu_range && !cpu_range) ||
            call->getArgOperandNo(&use) < first_callback ||
            call->getArgOperandNo(&use) > first_callback + 2) {
          proven = false;
          break;
        }
      }
      if (proven) {
        contexts.insert(&function);
        changed = true;
      }
    }
  }
  size_t replacements = 0, pointer_replacements = 0;
  for (auto *entry : contexts) {
    for (auto &block : *entry) {
      for (auto it = block.begin(); it != block.end();) {
        auto *load = llvm::dyn_cast<llvm::LoadInst>(&*it++);
        if (!load || load->isVolatile() || load->isAtomic() ||
            (!load->getType()->isIntegerTy() && !load->getType()->isPointerTy()))
          continue;
        int64_t offset = 0;
        auto *base = llvm::GetPointerBaseWithConstantOffset(load->getPointerOperand(), offset, layout);
        auto *buffer = llvm::dyn_cast<llvm::LoadInst>(base);
        if (!buffer || buffer->isVolatile() || buffer->isAtomic() || !buffer->getType()->isPointerTy())
          continue;
        int64_t context_offset = 0;
        auto *context_base = llvm::GetPointerBaseWithConstantOffset(
            buffer->getPointerOperand(), context_offset, layout);
        // RuntimeContext.arg_buffer is its first member. Do not match arbitrary
        // loads merely because they happen to carry compiler-generated names.
        if (context_base != entry->getArg(0) || context_offset != 0)
          continue;
        if (offset < 0 || offset > INT_MAX)
          continue;
        if (load->getType()->isPointerTy()) {
          const auto relation = pointers.find(static_cast<int>(offset));
          if (relation == pointers.end())
            continue;
          llvm::IRBuilder<> builder(load);
          auto *bytes = builder.getInt8Ty();
          auto *slot = builder.CreateGEP(bytes, buffer,
                                         builder.getInt64(relation->second.base_slot));
          auto *base_pointer = builder.CreateLoad(load->getType(), slot);
          // Non-inbounds GEP: the representative is a field/view pointer,
          // not necessarily the beginning of the retained allocation.
          auto *pointer = builder.CreateGEP(bytes, base_pointer,
              builder.getInt64(relation->second.displacement));
          load->replaceAllUsesWith(pointer);
          load->eraseFromParent();
          ++pointer_replacements;
          continue;
        }
        const auto value = constants.find(static_cast<int>(offset));
        if (value == constants.end() ||
            load->getType()->getIntegerBitWidth() != value->second.bits)
          continue;
        load->replaceAllUsesWith(llvm::ConstantInt::get(load->getType(), value->second.value));
        load->eraseFromParent();
        ++replacements;
      }
    }
  }
  // A library function receives a private argument buffer, not the entry's
  // RuntimeContext layout. Let LLVM expose those stores/loads through a legal
  // inline instead of copying entry offsets into an unrelated callee ABI.
  // Only a single direct use is selected: prepared code does not duplicate a
  // shared implementation at every call site. The persisted library and the
  // ordinary JIT/unspecialized optimization policy remain unchanged.
  if (replacements || pointer_replacements) {
    for (auto &function : module) {
      if (!function.hasLocalLinkage() || function.isDeclaration() ||
          !function.hasFnAttribute("taichi.aot.real_function") ||
          function.hasFnAttribute(llvm::Attribute::NoInline) ||
          function.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
          !function.hasOneUse())
        continue;
      auto *call = llvm::dyn_cast<llvm::CallBase>(*function.user_begin());
      if (call && call->getCalledFunction() == &function &&
          call->getFunction() != &function)
        function.addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }
  TI_DEBUG("Prepared specialization {} folded {} ABI loads and {} storage relations",
           code, replacements, pointer_replacements);
  if (arch_is_cpu(arch())) {
    // CUDA reoptimizes at registration. CPU normally optimizes at export, so
    // rerun its existing pass pipeline after introducing prepared constants.
    auto target = llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());
    const auto effective = LLVMModuleOptions::read(module, executor_->get_config());
    auto &options = target.getOptions();
    options.AllowFPOpFusion = effective.fast_math ? llvm::FPOpFusion::Fast : llvm::FPOpFusion::Strict;
    options.UnsafeFPMath = effective.fast_math;
    options.NoInfsFPMath = effective.fast_math;
    options.NoNaNsFPMath = effective.fast_math;
    auto machine = llvm::cantFail(target.createTargetMachine());
    LLVMOptPipelineOptions passes;
    passes.opt_level = llvm_opt_level_from_int(effective.opt_level);
    passes.loop_vectorize = true;
    passes.slp_vectorize = true;
    passes.run_post_gep_passes = true;
    run_module_opt_pipeline(module, machine.get(), passes);
  }
  CompiledKernelData compiled{arch(), std::move(data).take_llvm_ckd_data()};
  KernelRegistrationState registration;
  auto handle = kernel_launcher_->register_llvm_kernel(compiled, registration);
  auto result = std::make_unique<llvm_aot::KernelImpl>(
      kernel_launcher_.get(), handle, arch(), kernel.name, compiled.get_internal_data(),
      kernel.runtime_affine_arrays);
  static_cast<CallableBase &>(*result) = static_cast<const CallableBase &>(kernel);
  auto *callable = result.get();
  specialized_kernels_.emplace(std::move(key), std::move(result));
  return callable;
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
  const auto alias = metadata_.kernel_aliases.find(name);
  const auto &code = alias == metadata_.kernel_aliases.end() ? name : alias->second;
  const auto failed = payload_failures_.find(code);
  if (failed != payload_failures_.end())
    throw aot::ArtifactLoadError(failed->second);
  auto payload = payload_kernels_.find(code);
  if (payload == payload_kernels_.end()) {
    if (!cache_reader_->has_kernel(code))
      return nullptr;
    try {
      payload = payload_kernels_.emplace(code, load_kernel_payload(code)).first;
    } catch (const std::exception &error) {
      const auto message = fmt::format("Failed to load LLVM AOT code {}: {}", code, error.what());
      payload_failures_.emplace(code, message);
      throw aot::ArtifactLoadError(message);
    } catch (const std::string &error) {
      const auto message = fmt::format("Failed to load LLVM AOT code {}: {}", code, error);
      payload_failures_.emplace(code, message);
      throw aot::ArtifactLoadError(message);
    }
  }
  const auto selected = range_selections_.find(name);
  return std::make_unique<llvm_aot::KernelImpl>(
      *payload->second, name,
      selected == range_selections_.end() ? std::vector<std::uint32_t>{}
                                         : selected->second);
}

std::unique_ptr<llvm_aot::KernelImpl> LlvmAotModule::load_kernel_payload(
    const std::string &name, bool defer_registration) {
  std::unique_ptr<LlvmOfflineCacheFileReader> reopened;
  auto *reader = cache_reader_.get();
  if (!defer_registration) {
    // take_kernel_cache consumes its metadata as well as its IR. A deferred
    // ordinary registration must reopen the frozen artifact, just as prepared
    // specialization does, rather than reusing that consumed reader.
    TI_ERROR_IF(aot_artifact_sha256(read_artifact(*directory_, "metadata.tcb")) != metadata_.abi_sha256,
                "AOT metadata changed after loading");
    reopened = LlvmOfflineCacheFileReader::make(directory_, LlvmOfflineCache::Format::BC);
    TI_ERROR_IF(!reopened || !reopened->has_kernel(name),
                "Cannot reopen deferred AOT kernel: {}", name);
    reader = reopened.get();
  }
  if (!reader->has_kernel(name)) return nullptr;
  auto kernel_cache = load_kernel_from_cache(name, reader);
  const auto array_declaration = metadata_.kernel_runtime_affine_arrays.find(name);
  const auto affine_arrays = array_declaration == metadata_.kernel_runtime_affine_arrays.end()
      ? std::vector<std::vector<int>>{} : array_declaration->second;
  for (const auto &argument : affine_arrays)
    TI_ERROR_IF(std::none_of(kernel_cache.args.begin(), kernel_cache.args.end(),
        [&](const auto &entry) { return entry.first == argument && entry.second.ptype == ParameterType::kNdarray; }),
        "AOT ndarray layout qualification does not match a compiled parameter");
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
  const auto view_declaration =
      metadata_.kernel_sparse_runtime_view_abis.find(name);
  const auto view_abi =
      view_declaration == metadata_.kernel_sparse_runtime_view_abis.end()
          ? 0u
          : view_declaration->second;
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
        context_abi, runtime_state, view_abi);
    return std::make_unique<llvm_aot::KernelImpl>(
        launcher, handle, arch(), kernel_name, data, affine_arrays);
#else
    TI_ERROR("CUDA device artifacts require a build with CUDA support");
#endif
  }
  LLVM::CompiledKernelData compiled{
      arch(), std::move(kernel_cache).take_llvm_ckd_data()};
  TI_ERROR_IF(runtime_affine_arrays(*compiled.get_internal_data().compiled_data.module) != affine_arrays,
              "AOT LLVM code does not match its ndarray layout qualification");
  TI_ERROR_IF(
      LLVM::sparse_runtime_view_abi(
          *compiled.get_internal_data().compiled_data.module) != view_abi,
      "AOT LLVM code does not match its sparse runtime view qualification: {}",
      name);
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
  if (defer_registration && !gated && metadata_.kernel_format == "bc-library") {
    // Contract/dependency validation above still happens at lookup. A prepared
    // consumer can now compile only its specialization; an ordinary consumer
    // registers the generic implementation when it first needs its handle.
    // Reopen through the same validated reader rather than retaining a large
    // LLVM module for a generic implementation that may never be used.
    return std::make_unique<llvm_aot::KernelImpl>(
        kernel_launcher_.get(), KernelLaunchHandle{}, arch(), kernel_name,
        compiled.get_internal_data(), affine_arrays,
        [this, name] { return load_kernel_payload(name, false)->handle(); });
  }
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
      compiled.get_internal_data(), affine_arrays);
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

std::set<int> LlvmAotModule::snode_tree_dependencies(
    const std::set<std::string> *entries) const {
  std::set<int> selected;
  if (entries) {
    // Validate the entire requested closure before materializing any tree.
    for (const auto &name : *entries) {
      const auto alias = metadata_.kernel_aliases.find(name);
      const auto &code = alias == metadata_.kernel_aliases.end() ? name : alias->second;
      const auto found = kernel_tree_dependencies_.find(code);
      if (found == kernel_tree_dependencies_.end())
        throw std::invalid_argument("SNode instance plan references unknown kernel: " + name);
      for (int logical : found->second) {
        if (!std::binary_search(snode_tree_ids_.begin(), snode_tree_ids_.end(), logical))
          throw std::invalid_argument("SNode instance kernel references a missing tree: " + name);
        selected.insert(logical);
      }
    }
  } else {
    selected.insert(snode_tree_ids_.begin(), snode_tree_ids_.end());
  }
  return selected;
}

namespace {
struct FixedSNodeLayoutProjection {
  int tree;
  SNodeTreeLayout layout;
  TI_IO_DEF(tree, layout);
};
}  // namespace

std::string LlvmAotModule::fixed_snode_plan_layouts(
    const std::set<std::string> &entries) const {
  std::vector<FixedSNodeLayoutProjection> result;
  for (int tree : snode_tree_dependencies(&entries)) {
    SNodeTreeLayout layout;
    if (!cache_reader_->get_field_cache(layout, tree))
      throw std::invalid_argument("Missing fixed component tree layout");
    result.push_back({tree, layout.normalized()});
  }
  return liong::json::print(liong::json::serialize(result));
}

std::vector<LlvmSparseRuntimeViewTree>
LlvmAotModule::snode_instance_prototypes(const std::set<std::string> *entries) {
  if ((!arch_is_cpu(arch()) && arch() != Arch::cuda) || !allocation_result_buffer_)
    throw std::invalid_argument(
        "SNode instances require a runtime-owned CPU/CUDA AOT module");
  const auto selected = snode_tree_dependencies(entries);
  std::vector<int> physical;
  for (int logical : selected) {
    ensure_snode_tree_initialized(logical);
    physical.push_back(tree_bindings_.at(logical));
  }
  if (physical.empty())
    throw std::invalid_argument("Selected AOT closure has no fixed SNode trees to instantiate");
  return executor_->runtime_view_trees(physical);
}

SNodeTreeLayout LlvmAotModule::clone_instance_tree(
    const LlvmSparseRuntimeViewTree &prototype,
    int target_tree_id) {
  if (!allocation_result_buffer_ ||
      std::none_of(tree_bindings_.begin(), tree_bindings_.end(),
                   [&](const auto &entry) { return entry.second == prototype.tree_id; }))
    throw std::invalid_argument("SNode instance prototype belongs to another module");
  return executor_->clone_snode_tree_runtime(prototype, target_tree_id,
                                            allocation_result_buffer_);
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
    aot::CompiledDispatch loaded;
    loaded.kernel_name = dispatch.kernel_name;
    loaded.symbolic_args = dispatch.symbolic_args;
    loaded.compiled_kernel = get_kernel(dispatch.kernel_name);
    dispatches.push_back(std::move(loaded));
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
