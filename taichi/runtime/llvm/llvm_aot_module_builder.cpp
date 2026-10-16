#include "taichi/runtime/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include <fstream>
#include "llvm/Transforms/Utils/Cloning.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_metadata.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/program/kernel_compile_request.h"
#include "taichi/rhi/cuda/cuda_capability.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
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
  if (compile_config_.arch != Arch::cuda) {
    return;
  }

  // AOT targets a deployable device family rather than the producer's device.
  // Ampere is the default; callers may select an exact target through caps.
  constexpr int kDefaultCudaAotComputeCapability = 80;
  constexpr int kMinimumCudaAotComputeCapability = 60;
  int target = caps_.contains(DeviceCapability::cuda_compute_capability)
                   ? static_cast<int>(caps_.get(
                         DeviceCapability::cuda_compute_capability))
                   : kDefaultCudaAotComputeCapability;
  const auto resolution =
      cuda::detail::resolve_compute_capability_target(target);
  TI_ERROR_IF(
      target < kMinimumCudaAotComputeCapability ||
          resolution.codegen_compute_capability != target,
      "Unsupported CUDA AOT target compute capability {}. Choose an exact "
      "LLVM-supported target at or above 60.",
      target);

  if (caps_.contains(DeviceCapability::cuda_ptx_version)) {
    TI_ERROR_IF(
        caps_.get(DeviceCapability::cuda_ptx_version) !=
            static_cast<uint32_t>(resolution.ptx_version),
        "CUDA AOT target {} requires PTX {}, but caps requested PTX {}.",
        target, resolution.ptx_version,
        caps_.get(DeviceCapability::cuda_ptx_version));
  }
  caps_.set(DeviceCapability::cuda_compute_capability, target);
  caps_.set(DeviceCapability::cuda_ptx_version, resolution.ptx_version);
}

void LlvmAotModuleBuilder::set_artifact_format(const std::string &format) {
  TI_ERROR_IF(compile_options_frozen_,
              "AOT artifact format must be selected before exporting kernels.");
  TI_ERROR_IF(format != "bc" && !LLVM::is_cuda_aot_artifact_format(format),
              "Unsupported LLVM AOT artifact format '{}'", format);
  TI_ERROR_IF(LLVM::is_cuda_aot_artifact_format(format) &&
                  compile_config_.arch != Arch::cuda,
              "AOT PTX/cubin artifacts require CUDA");
  TI_ERROR_IF(format == "cubin" && !explicit_cuda_target_,
              "AOT cubin requires an explicit cuda_compute_capability in caps");
  artifact_format_ = format;
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
      !cache_.fields.empty() && compile_config_.demote_dense_struct_fors &&
      (arch_is_cpu(compile_config_.arch) || compile_config_.arch == Arch::cuda);
  std::map<int, int> tree_mapping;
  for (const auto &[id, field] : cache_.fields) {
    tree_mapping.emplace(id, id);
    for (const auto &node : field.snode_metas) {
      relocatable &= node.type == SNodeType::root ||
                     node.type == SNodeType::dense ||
                     node.type == SNodeType::place;
    }
  }
  for (const auto &[name, kernel] : cache_.kernels) {
    for (const auto &task : kernel.compiled_data.tasks) {
      relocatable &= task.task_type == OffloadedTaskType::serial ||
                     task.task_type == OffloadedTaskType::range_for;
    }
  }
  if (relocatable) {
    int local_id = 0;
    for (auto &[producer, local] : tree_mapping)
      local = local_id++;
    decltype(exported.fields) fields;
    for (auto &[producer, field] : exported.fields) {
      field.tree_id = tree_mapping.at(producer);
      fields.emplace(field.tree_id, std::move(field));
    }
    exported.fields = std::move(fields);
    metadata.resource_binding = "dense_v1";
  }
  TI_ERROR_IF(!initial_root_data_.empty() && !relocatable,
              "AOT initial root state requires relocatable dense trees.");
  TI_ERROR_IF(!root_storage_views_.empty() && !relocatable,
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
    auto module = llvm::CloneModule(
        *cache_.kernels.at(kernel.kernel_key).compiled_data.module);
    if (relocatable) {
      LLVM::remap_aot_tree_id_symbols(*module, tree_mapping);
      for (int &id : kernel.used_snode_tree_ids)
        id = tree_mapping.at(id);
    } else {
      // Sparse/list-managed and legacy targets retain their embedded-ID ABI.
      LLVM::resolve_aot_tree_ids(*module, tree_mapping);
    }
    return module;
  };
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
  const std::string json =
      liong::json::print(liong::json::serialize(metadata));
  std::fstream f(taichi::join_path(output_dir,
                                  LLVM::kLlvmAotMetadataFilename),
                 std::ios::trunc | std::ios::out);
  TI_ERROR_IF(!f.is_open(), "Cannot write LLVM AOT metadata in {}",
              output_dir);
  f.write(json.data(), json.size());

  dump_graph(output_dir);
}

const std::vector<int> &LlvmAotModuleBuilder::kernel_snode_tree_ids(
    const std::string &identifier) const {
  const auto found = cache_.kernels.find(identifier);
  TI_ERROR_IF(found == cache_.kernels.end(), "Unknown AOT kernel export '{}'.",
              identifier);
  return found->second.used_snode_tree_ids;
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
  auto compiled = compile_kernel(kernel);
  if (launch_abi == cuda::kGraphGateLaunchABI) {
    TI_ERROR_IF(!compiled.rets.empty() ||
                    std::any_of(compiled.compiled_data.tasks.begin(),
                                compiled.compiled_data.tasks.end(),
                                [](const auto &task) { return task.one_to_one; }),
                "AOT Graph gated code requires void, non-bounded kernels");
    compiled.compiled_data =
        LLVM::make_cuda_graph_gated_kernel(compiled.compiled_data);
  }
  LlvmOfflineCache::KernelCacheData kcache;
  kcache.kernel_key = identifier;
  kcache.compiled_data = std::move(compiled.compiled_data);
  kcache.used_snode_tree_ids = std::move(compiled.used_snode_tree_ids);
  // A callable artifact must contain the layouts of every tree it accesses,
  // including gradient trees. add_field() is optional name publication, not a
  // prerequisite for making the kernel's storage dependencies deployable.
  for (int tree_id : kcache.used_snode_tree_ids) {
    if (cache_.fields.find(tree_id) == cache_.fields.end()) {
      cache_.fields.emplace(tree_id, prog_->get_cached_field(tree_id));
    }
  }
  kcache.graph_metadata = std::move(compiled.graph_metadata);
  kcache.args.reserve(kernel->nested_parameters.size());
  for (const auto &p : kernel->nested_parameters)
    kcache.args.push_back(p);
  kcache.args_type = kernel->args_type;
  kcache.args_size = kernel->args_size;
  kcache.rets = kernel->rets;
  kcache.ret_size = kernel->ret_size;
  kcache.ret_type = kernel->ret_type;
  kcache.last_used_at = std::time(nullptr);
  kcache.created_at = std::time(nullptr);
  cache_.kernels[identifier] = std::move(kcache);
  if (launch_abi == "ordinary")
    kernel_launch_abis_.erase(identifier);
  else
    kernel_launch_abis_[identifier] = launch_abi;
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
  LlvmOfflineCache::FieldCacheData field_cache =
      prog_->get_cached_field(snode_tree_id);

  // 3. Update AOT Cache
  cache_.fields[snode_tree_id] = std::move(field_cache);
}

LLVM::CompiledKernelData::InternalData LlvmAotModuleBuilder::compile_kernel(
    Kernel *kernel) {
  auto config = compile_config_;
  config.llvm_aot_relocatable_roots =
      arch_is_cpu(config.arch) || config.arch == Arch::cuda;
  const auto request = resolve_kernel_compile_request(
      config, caps_, kernel->get_compile_tier_override(), CompilePurpose::aot,
      compile_options_);
  const auto &ckd = compilation_manager_.load_or_compile(
      request.config(), request.device_caps(), *kernel);
  TI_ASSERT(arch_uses_llvm(ckd.arch()));
  return dynamic_cast<const LLVM::CompiledKernelData &>(ckd)
      .get_internal_data();
}

}  // namespace taichi::lang
