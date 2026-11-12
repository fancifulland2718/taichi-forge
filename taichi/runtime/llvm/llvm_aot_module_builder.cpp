#include "taichi/runtime/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include <fstream>
#include "llvm/Transforms/Utils/Cloning.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_metadata.h"
#include "taichi/runtime/llvm/aot_kernel_contract.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/codegen/llvm/range_dispatch.h"
#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/program/kernel_compile_request.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"
#include "taichi/runtime/llvm/array_layout_abi.h"
#include "taichi/runtime/llvm/llvm_module_codec.h"
#include "llvm/Support/raw_ostream.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/aot_snode_resources.h"
#include "taichi/runtime/llvm/aot_code_units.h"
#include "taichi/runtime/llvm/function_code_cache.h"
#include "taichi/analysis/offline_cache_util.h"
#if defined(TI_WITH_CUDA)
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/cuda/cuda_artifact_provider.h"
#endif

namespace taichi::lang {

LlvmAotModuleBuilder::LlvmAotModuleBuilder(
    KernelCompilationManager &compilation_manager,
    const CompileConfig &compile_config,
    LlvmProgramImpl *prog,
    const DeviceCapabilityConfig &caps)
    : compilation_manager_(compilation_manager),
      compile_config_(compile_config),
      prog_(prog),
      caps_(caps),
      explicit_cuda_target_(
          caps.contains(DeviceCapability::cuda_compute_capability)) {
  caps_ = resolve_kernel_target(compile_config_.arch, caps, CompilePurpose::aot).capabilities;
}

void LlvmAotModuleBuilder::set_artifact_format(const std::string &format) {
  TI_ERROR_IF(compile_options_frozen_,
              "AOT artifact format must be selected before exporting kernels.");
  TI_ERROR_IF(format != "bc" && format != "bc-library" && !LLVM::is_cuda_aot_artifact_format(format),
              "Unsupported LLVM AOT artifact format '{}'", format);
  TI_ERROR_IF(LLVM::is_cuda_aot_artifact_format(format) &&
                  compile_config_.arch != Arch::cuda,
              "AOT PTX/cubin artifacts require CUDA");
  TI_ERROR_IF(format == "cubin" && !explicit_cuda_target_,
              "AOT cubin requires an explicit cuda_compute_capability in caps");
  artifact_format_ = format;
  TI_ERROR_IF(format == "bc-library" && compile_config_.arch != Arch::cuda,
              "Experimental AOT code libraries currently require CUDA");
  function_code_cache_ = format == "bc-library"
      ? std::make_shared<LLVM::FunctionCodeCache>() : nullptr;
}

void LlvmAotModuleBuilder::enable_streaming_exports() {
  AotModuleBuilder::enable_streaming_exports();
  if (streaming_compilation_manager_)
    return;
  streaming_compilation_manager_ = prog_->make_kernel_compilation_manager();
  streaming_compilation_manager_->set_executable_lifecycle_telemetry_enabled(true);
}

void LlvmAotModuleBuilder::seal_kernel_export_per_backend(
    const std::string &identifier) {
  if (!streaming_compilation_manager_)
    return;
  const auto &name = kernel_names_.at(identifier);
  auto &compiled = cache_.kernels.at(name).compiled_data;
  if (compiled.module) {
    if (artifact_format_ == "bc-library") {
      LLVM::export_code_units(*compiled.module,
          [&](const std::string &unit, const std::string &bytes) {
            const auto [item, inserted] = sealed_code_units_.emplace(unit, bytes);
            TI_ERROR_IF(!inserted && item->second != bytes,
                        "Conflicting AOT code unit {}", unit);
          }, false);
    }
    std::string bytes;
    llvm::raw_string_ostream output(bytes);
    encode_llvm_module(*compiled.module, output, LLVMModuleEncoding::bitcode);
    output.flush();
    sealed_bitcode_.emplace(name, std::move(bytes));
    compiled.module.reset();
    compiled.module_context_owner.reset();
  }
  // Code is already atomically published. Release build-owned IR now, and
  // flush its metadata index once at archive time, avoiding a growing rewrite
  // after every asset. Alias lookup retains the original target identity.
  streaming_compilation_manager_->release_build_cache();
}

std::map<std::string, std::uint64_t>
LlvmAotModuleBuilder::streaming_export_statistics() {
  if (!streaming_compilation_manager_)
    return {};
  const auto stats =
      streaming_compilation_manager_->executable_lifecycle_statistics(false);
  std::uint64_t resident = 0, bytes = 0;
  for (const auto &[_, kernel] : cache_.kernels)
    resident += kernel.compiled_data.module != nullptr;
  for (const auto &[_, bitcode] : sealed_bitcode_)
    bytes += bitcode.size();
  std::uint64_t unit_bytes = 0;
  for (const auto &[_, bitcode] : sealed_code_units_)
    unit_bytes += bitcode.size();
  std::map<std::string, std::uint64_t> result = {{"resident_ir_modules", resident},
          {"serialized_code_bytes", bytes},
          {"serialized_code_unit_bytes", unit_bytes},
          {"shared_code_units", sealed_code_units_.size()},
          {"physical_code_entries", cache_.kernels.size()},
          {"resident_templates", stats.resident_templates},
          {"compiler_invocations", stats.compiler_invocations},
          {"disk_loads", stats.disk_loads},
          {"memory_cache_hits", stats.memory_cache_hits + stats.loaded_cache_hits}};
  if (function_code_cache_)
    result.merge(function_code_cache_->statistics());
  return result;
}

std::unique_ptr<llvm::Module> LlvmAotModuleBuilder::load_kernel_module(
    const std::string &name,
    std::shared_ptr<llvm::LLVMContext> &context_owner) const {
  const auto &compiled = cache_.kernels.at(name).compiled_data;
  if (compiled.module) {
    context_owner = compiled.module_context_owner;
    return llvm::CloneModule(*compiled.module);
  }
  const auto &bytes = sealed_bitcode_.at(name);
  // A private context also releases LLVM's interned types/constants after
  // writing this asset; parsing into the Program context would retain them.
  context_owner = std::make_shared<llvm::LLVMContext>();
  auto decoded = decode_llvm_module(llvm::MemoryBufferRef(bytes, name),
                                    *context_owner, LLVMModuleEncoding::bitcode);
  if (!decoded)
    TI_ERROR("Cannot decode sealed AOT kernel {}: {}", name,
              llvm::toString(decoded.takeError()));
  return std::move(*decoded);
}

void LlvmAotModuleBuilder::dump(const std::string &output_dir,
                                const std::string &filename) const {
  LlvmOfflineCacheFileWriter writer;
  LlvmOfflineCache exported;
  // dump() must preserve the producer for another directory/archive or a later
  // add(). Copy metadata now; clone and transform only the current kernel
  // below.
  exported.fields = cache_.fields;
  for (const auto &[name, kernel] : cache_.kernels)
    exported.kernels.emplace(name, kernel.clone_metadata());
  LLVM::LlvmAotMetadata metadata;
  metadata.schema_version = LLVM::kLlvmAotSchemaVersion;
  metadata.kernel_format = artifact_format_;
  bool relocatable =
      !cache_.fields.empty() &&
      (arch_is_cpu(compile_config_.arch) || compile_config_.arch == Arch::cuda);
  std::map<int, int> tree_mapping;
  bool sparse = false;
  for (const auto &[id, field] : cache_.fields) {
    tree_mapping.emplace(id, id);
    for (const auto &node : field.snode_metas) {
      sparse |= !LLVM::is_dense_aot_snode(node.type);
      relocatable &= LLVM::is_relocatable_aot_snode(node.type, true);
    }
  }
  for (const auto &[name, kernel] : cache_.kernels) {
    for (const auto &task : kernel.compiled_data.tasks) {
      relocatable &= LLVM::is_relocatable_aot_task(task.task_type, true);
    }
  }
  if (relocatable) {
    const auto list_trees = LLVM::aot_element_list_tree_ids(
        cache_, !compile_config_.demote_dense_struct_fors);
    int local_id = 0;
    for (auto &[producer, local] : tree_mapping)
      local = local_id++;
    decltype(exported.fields) fields;
    for (auto &[producer, field] : exported.fields) {
      field.tree_id = tree_mapping.at(producer);
      fields.emplace(field.tree_id, std::move(field));
    }
    exported.fields = std::move(fields);
    // v2 qualifies CUDA's serially latched listgen decision for fixed replay.
    // Old sparse_v1 artifacts remain valid for ordinary execution only.
    if (!compile_config_.demote_dense_struct_fors && !list_trees.empty()) {
      metadata.resource_binding = "list_v1";
    } else if (sparse) {
      metadata.resource_binding =
          compile_config_.arch == Arch::cuda ? "sparse_v2" : "sparse_v1";
    } else {
      metadata.resource_binding = "dense_v1";
    }
  }
  TI_ERROR_IF(!initial_root_data_.empty() && (!relocatable || sparse),
              "AOT initial root state requires relocatable dense trees.");
  TI_ERROR_IF(!root_storage_views_.empty() && (!relocatable || sparse),
              "AOT root storage views require relocatable dense trees.");
  aot::write_root_storage_views(root_storage_views_, tree_mapping, output_dir);
  for (const auto &[producer, entry] : initial_root_data_) {
    const auto field = cache_.fields.find(producer);
    TI_ERROR_IF(
        field == cache_.fields.end() || field->second.root_size != entry.size,
        "AOT initial root state references an absent or incompatible "
        "tree {}.",
        producer);
    auto local = entry;
    local.tree_id = tree_mapping.at(producer);
    metadata.initial_root_data.push_back(std::move(local));
  }
  auto prepare_module = [&](LlvmOfflineCache::KernelCacheData &kernel) {
    auto module = load_kernel_module(kernel.kernel_key,
                                     kernel.compiled_data.module_context_owner);
    if (relocatable) {
      LLVM::remap_aot_kernel_tree_ids(*module, kernel, tree_mapping);
    } else {
      // Unsupported layouts/tasks and legacy targets retain embedded IDs.
      LLVM::resolve_aot_tree_ids(*module, tree_mapping);
    }
    if (artifact_format_ == "bc-library")
      LLVM::export_code_units(*module, output_dir);
    return module;
  };
  if (!sealed_code_units_.empty()) {
    taichi::create_directories(taichi::join_path(output_dir, "code_units"));
    for (const auto &[name, bytes] : sealed_code_units_) {
      std::ofstream output(taichi::join_path(output_dir, name), std::ios::binary);
      output.write(bytes.data(), bytes.size());
      TI_ERROR_IF(!output, "Cannot write sealed AOT code unit {}", name);
    }
  }
  writer.set_data(std::move(exported));
  if (LLVM::is_cuda_aot_artifact_format(artifact_format_)) {
#if defined(TI_WITH_CUDA)
    metadata.schema_version = LLVM::kCudaPtxAotSchemaVersion;
    const CUDACodegenTarget target{
        fmt::format("sm_{}", caps_.get(DeviceCapability::cuda_compute_capability)),
        fmt::format("+ptx{}", caps_.get(DeviceCapability::cuda_ptx_version))};
    writer.dump(
        output_dir, LlvmOfflineCache::Format::CUDAArtifact, false,
        [&](const std::string &prefix,
            LlvmOfflineCache::KernelCacheData &kernel) {
          auto &compiled = kernel.compiled_data;
          const int max_reg = compiled.cuda_max_registers < 0
                                  ? compile_config_.gpu_max_reg
                                  : compiled.cuda_max_registers;
          auto &item = metadata.cuda_artifacts[kernel.kernel_key];
          if (relocatable) {
            LLVM::prepare_ptx_tree_relocations(*compiled.module);
          }
          auto artifact = compile_cuda_kernel_artifact(
              compiled.module, compile_config_, target, max_reg,
              JITModuleRole::user_kernel);
          if (artifact_format_ == "cubin") {
            artifact = cuda::compile_cuda_aot_cubin(std::move(artifact),
                                                    compile_config_);
            metadata.cuda_cubin_providers[kernel.kernel_key] =
                artifact.provider_identity;
          }
          const auto path = prefix + "." + artifact_format_;
          std::ofstream out(path, std::ios::binary | std::ios::trunc);
          TI_ERROR_IF(!out, "Cannot write CUDA AOT artifact {}", path);
          out.write(artifact.payload.data(), artifact.code_size());
          out.close();
          TI_ERROR_IF(!out, "Failed to write CUDA AOT artifact {}", path);
          item.sha256 = LLVM::aot_artifact_sha256(
              llvm::StringRef(artifact.payload.data(), artifact.code_size()));
          item.size = artifact.code_size();
          item.entry_names = artifact.entry_names;
          item.tree_relocations = artifact.aot_tree_relocations;
          item.target_identity = artifact.target_identity;
          item.max_registers = artifact.max_registers;
          item.fast_math = artifact.fast_math;
          item.llvm_opt_level = artifact.llvm_opt_level;
          if (relocatable || artifact_format_ == "cubin")
            item.contract_sha256 = LLVM::aot_cuda_contract_sha256(item);
          return artifact.code_size();
        },
        prepare_module);
    metadata.abi_sha256 = LLVM::aot_artifact_sha256(LLVM::read_aot_artifact(
        taichi::join_path(output_dir, "metadata.tcb")));
#else
    TI_ERROR("PTX export requires a build with CUDA support");
#endif
  } else {
    writer.dump(output_dir, LlvmOfflineCache::Format::BC, false, {},
                prepare_module);
  }
  if (relocatable) {
    metadata.schema_version = LLVM::kRelocatableLlvmAotSchemaVersion;
    metadata.abi_sha256 = LLVM::aot_artifact_sha256(
        LLVM::read_aot_artifact(taichi::join_path(output_dir, "metadata.tcb")));
  }
  if (artifact_format_ == "cubin") {
    metadata.schema_version = LLVM::kCudaCubinAotSchemaVersion;
  }
  if (!metadata.initial_root_data.empty()) {
    metadata.schema_version = LLVM::kInitialRootDataAotSchemaVersion;
  }
  if (compile_config_.arch == Arch::cuda) {
    for (const auto capability : {
             DeviceCapability::cuda_compute_capability,
             DeviceCapability::cuda_ptx_version,
         }) {
      metadata.required_caps[to_string(capability)] = caps_.get(capability);
    }
  }
  if (!kernel_launch_abis_.empty()) {
    metadata.schema_version = LLVM::kKernelLaunchAbiAotSchemaVersion;
    metadata.kernel_launch_abis = kernel_launch_abis_;
    if (metadata.abi_sha256.empty()) {
      metadata.abi_sha256 = LLVM::aot_artifact_sha256(LLVM::read_aot_artifact(
          taichi::join_path(output_dir, "metadata.tcb")));
    }
    metadata.launch_abi_sha256 = LLVM::aot_launch_abi_sha256(metadata);
  }
  for (const auto &[name, kernel] : cache_.kernels) {
    // A sealed export retains compact bitcode; inspect one module at a time.
    std::shared_ptr<llvm::LLVMContext> context_owner;
    std::unique_ptr<llvm::Module> restored;
    auto *module = kernel.compiled_data.module.get();
    if (module == nullptr) {
      restored = load_kernel_module(name, context_owner);
      module = restored.get();
    }
    const auto state = LLVM::runtime_state_use(*module);
    auto arrays = LLVM::runtime_affine_arrays(*module);
    if (!arrays.empty())
      metadata.kernel_runtime_affine_arrays.emplace(name, std::move(arrays));
    if (state)
      metadata.kernel_runtime_state.emplace(name, state);
    const auto view_abi =
        LLVM::sparse_runtime_view_abi(*module);
    if (view_abi)
      metadata.kernel_sparse_runtime_view_abis.emplace(name, view_abi);
  }
  if (!metadata.kernel_runtime_state.empty() || !metadata.kernel_runtime_affine_arrays.empty()) {
    if (metadata.abi_sha256.empty())
      metadata.abi_sha256 = LLVM::aot_artifact_sha256(LLVM::read_aot_artifact(
          taichi::join_path(output_dir, "metadata.tcb")));
    if (!metadata.kernel_runtime_state.empty())
      metadata.runtime_state_sha256 = LLVM::aot_runtime_state_sha256(metadata);
    if (!metadata.kernel_runtime_affine_arrays.empty())
      metadata.array_layout_sha256 = LLVM::aot_array_layout_sha256(metadata);
  }
  const bool task_launch_policy = std::any_of(
      cache_.kernels.begin(), cache_.kernels.end(), [](const auto &entry) {
        const auto &tasks = entry.second.compiled_data.tasks;
        return std::any_of(tasks.begin(), tasks.end(), [](const auto &task) {
          return task.requested_grid_residency_waves != 0 ||
                 task.requested_range_work_per_thread_target != 1;
        });
      });
  if (task_launch_policy) {
    TI_ERROR_IF(compile_config_.arch != Arch::cuda ||
                    metadata.kernel_launch_abis.empty(),
                "AOT task launch policy requires the current CUDA launch ABI");
    metadata.schema_version = LLVM::kTaskLaunchPolicyAotSchemaVersion;
  }
  metadata.schema_version = LLVM::kDynamicSharedAotSchemaVersion;
  if (metadata.abi_sha256.empty())
    metadata.abi_sha256 = LLVM::aot_artifact_sha256(LLVM::read_aot_artifact(
        taichi::join_path(output_dir, "metadata.tcb")));
  for (const auto &[name, code] : kernel_names_) {
    if (name != code)
      metadata.kernel_aliases.emplace(name, code);
  }
  for (const auto &[name, tasks] : range_selections_)
    metadata.device_range_dispatch.push_back({name, 1, tasks});
  if (!metadata.kernel_aliases.empty() || !metadata.device_range_dispatch.empty()) {
    if (metadata.abi_sha256.empty())
      metadata.abi_sha256 = LLVM::aot_artifact_sha256(LLVM::read_aot_artifact(
          taichi::join_path(output_dir, "metadata.tcb")));
    metadata.kernel_entry_sha256 = LLVM::aot_kernel_entry_sha256(metadata);
  }
  const std::string json =
      liong::json::print(liong::json::serialize(metadata));
  std::fstream f(taichi::join_path(output_dir,
                                  LLVM::kLlvmAotMetadataFilename),
                 std::ios::trunc | std::ios::out);
  TI_ERROR_IF(!f.is_open(), "Cannot write LLVM AOT metadata in {}",
              output_dir);
  f.write(json.data(), json.size());

  dump_graph(output_dir);
  aot::write_struct_layout_artifact(ndarray_layouts_, output_dir);
  aot::write_value_parameter_artifact(value_parameters_, output_dir);
  aot::write_argument_origin_artifact(argument_origins_, output_dir);
  exported_tree_bindings_ = tree_mapping;
  if (streaming_compilation_manager_)
    streaming_compilation_manager_->dump();
}

const std::vector<int> &LlvmAotModuleBuilder::kernel_snode_tree_ids(
    const std::string &identifier) const {
  const auto found = kernel_names_.find(identifier);
  TI_ERROR_IF(found == kernel_names_.end(), "Unknown AOT kernel export '{}'.",
              identifier);
  return cache_.kernels.at(found->second).used_snode_tree_ids;
}

void LlvmAotModuleBuilder::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  add_kernel_impl(identifier, kernel, "ordinary");
}

void LlvmAotModuleBuilder::add_with_launch_abi_per_backend(
    const std::string &identifier,
    Kernel *kernel,
    const std::string &launch_abi) {
  TI_ERROR_IF(compile_config_.arch != Arch::cuda ||
                  launch_abi != cuda::kGraphGateLaunchABI,
              "Unsupported AOT kernel launch ABI '{}' for this backend",
              launch_abi);
  add_kernel_impl(identifier, kernel, launch_abi);
}

void LlvmAotModuleBuilder::add_kernel_impl(const std::string &identifier,
                                         Kernel *kernel,
                                         const std::string &launch_abi) {
  const auto &artifact = compile_kernel(kernel);
  const auto &source = artifact.get_internal_data();
  record_kernel_resource_effects(identifier, artifact.graph_metadata());
  LLVMCompiledKernel projected;
  projected.tasks = source.compiled_data.tasks;
  projected.cuda_max_registers = source.compiled_data.cuda_max_registers;
  std::vector<std::uint32_t> range_tasks;
  if (const auto &plan = kernel->get_offload_execution_plan(); plan) {
    LLVM::project_aot_task_launch_policy(projected, *plan, compile_config_.arch);
    range_tasks = plan->device_range_tasks;
  }
  if (!range_tasks.empty()) {
    TI_ERROR_IF(compile_config_.arch != Arch::cuda || launch_abi != "ordinary" ||
                    LLVM::runtime_context_abi(*source.compiled_data.module) !=
                        LLVM::kRuntimeContextABI,
                "AOT device range requires an ordinary CUDA context kernel");
    LLVM::validate_device_range_selection(projected.tasks, range_tasks,
                                          taichi_global_tmp_buffer_size);
  }
  // Compiled identity includes target/options/relocation. The projection adds
  // entry launch facts; range selection intentionally remains outside code.
  TI_ASSERT(!artifact.kernel_identity().empty());
  const auto identity = artifact.kernel_identity() + "\n" + launch_abi + "\n" +
      liong::json::print(liong::json::serialize(projected));
  const auto existing = kernel_identities_.find(identity);
  const auto named = kernel_names_.find(identifier);
  if (named != kernel_names_.end()) {
    const auto selected = range_selections_.find(identifier);
    const std::vector<std::uint32_t> empty;
    TI_ERROR_IF(existing == kernel_identities_.end() ||
                    existing->second != named->second ||
                    (selected == range_selections_.end() ? empty : selected->second) != range_tasks,
                "LLVM AOT export '{}' refers to a different kernel or policy", identifier);
    return;
  }
  if (existing != kernel_identities_.end()) {
    kernel_names_.emplace(identifier, existing->second);
    if (!range_tasks.empty())
      range_selections_.emplace(identifier, std::move(range_tasks));
    return;
  }
  // Clone IR only for a new physical payload; aliases never retain another
  // module. dump() retains its separate relocation/mangling transaction.
  auto compiled = source;
  compiled.compiled_data.tasks = std::move(projected.tasks);
  if (launch_abi == cuda::kGraphGateLaunchABI) {
    TI_ERROR_IF(!compiled.rets.empty() ||
                    std::any_of(compiled.compiled_data.tasks.begin(),
                                compiled.compiled_data.tasks.end(),
                                [](const auto &task) { return task.one_to_one; }),
                "AOT Graph gated code requires void, non-bounded kernels");
    compiled.compiled_data =
        LLVM::make_cuda_graph_gated_kernel(compiled.compiled_data);
  }
  const std::string compiled_launch_abi =
      LLVM::compiled_kernel_launch_abi(*compiled.compiled_data.module);
  LlvmOfflineCache::KernelCacheData kcache(std::move(compiled));
  kcache.kernel_key = identifier;
  // A callable artifact must contain the layouts of every tree it accesses,
  // including gradient trees. add_field() is optional name publication, not a
  // prerequisite for making the kernel's storage dependencies deployable.
  for (int tree_id : kcache.used_snode_tree_ids) {
    if (cache_.fields.find(tree_id) == cache_.fields.end()) {
      cache_.fields.emplace(tree_id, prog_->get_cached_field(tree_id));
    }
  }
  kcache.last_used_at = std::time(nullptr);
  kcache.created_at = std::time(nullptr);
  cache_.kernels[identifier] = std::move(kcache);
  kernel_names_.emplace(identifier, identifier);
  kernel_identities_.emplace(identity, identifier);
  if (!range_tasks.empty())
    range_selections_.emplace(identifier, std::move(range_tasks));
  if (compiled_launch_abi == "ordinary")
    kernel_launch_abis_.erase(identifier);
  else
    kernel_launch_abis_[identifier] = compiled_launch_abi;
}

void LlvmAotModuleBuilder::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  add_per_backend(identifier + "__tmpl__" + key, kernel);
}

void LlvmAotModuleBuilder::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  // Field refers to a leaf node(Place SNode) in a SNodeTree.
  // It makes no sense to just serialize the leaf node or its corresponding
  // branch. Instead, the minimal unit we have to serialize is the entire
  // SNodeTree. Note that SNodeTree's uses snode_tree_id as its identifier,
  // rather than the field's name. (multiple fields may end up referring to the
  // same SNodeTree)

  // 1. Find snode_tree_id
  int snode_tree_id = rep_snode->get_snode_tree_id();

  // 2. Fetch Cache from the Program
  // Kernel compilation is not allowed until all the Fields are finalized,
  // so we finished SNodeTree compilation during AOTModuleBuilder construction.
  //
  // By the time "add_field_per_backend()" is called,
  // SNodeTrees should have already been finalized,
  // with compiled info stored in LlvmProgramImpl::cache_data_.
  TI_ASSERT(prog_ != nullptr);
  LLVM::SNodeTreeLayout field_cache = prog_->get_cached_field(snode_tree_id);

  // 3. Update AOT Cache
  cache_.fields[snode_tree_id] = std::move(field_cache);
}

const LLVM::CompiledKernelData &LlvmAotModuleBuilder::compile_kernel(
    Kernel *kernel) {
  auto config = compile_config_;
  config.llvm_aot_relocatable_roots =
      arch_is_cpu(config.arch) || config.arch == Arch::cuda;
  const auto request = resolve_kernel_compile_request(
      config, caps_, kernel->get_compile_tier_override(), CompilePurpose::aot,
      compile_options_);
  auto &manager = streaming_compilation_manager_ ? *streaming_compilation_manager_
                                                : compilation_manager_;
  LLVM::FunctionCodeScope functions(kernel, function_code_cache_,
      function_code_cache_
          ? get_hashed_offline_cache_key_context(request.config(),
                                                request.device_caps(), nullptr) +
                kernel->optimization_spec_cache_key()
          : "");
  const auto &ckd = manager.load_or_compile(
      request, *kernel);
  TI_ASSERT(arch_uses_llvm(ckd.arch()));
  return dynamic_cast<const LLVM::CompiledKernelData &>(ckd);
}

}  // namespace taichi::lang
