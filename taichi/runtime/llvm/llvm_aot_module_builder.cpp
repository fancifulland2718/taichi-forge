#include "taichi/runtime/llvm/llvm_aot_module_builder.h"

#include <algorithm>
#include <fstream>
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_metadata.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/rhi/cuda/cuda_capability.h"

namespace taichi::lang {

LlvmAotModuleBuilder::LlvmAotModuleBuilder(
    KernelCompilationManager &compilation_manager,
    const CompileConfig &compile_config,
    LlvmProgramImpl *prog,
    const DeviceCapabilityConfig &caps)
    : compilation_manager_(compilation_manager),
      compile_config_(compile_config),
      prog_(prog),
      caps_(caps) {
  if (compile_config_.arch != Arch::cuda) {
    return;
  }

  constexpr int kDefaultCudaAotComputeCapability = 60;
  int target = caps_.contains(DeviceCapability::cuda_compute_capability)
                   ? static_cast<int>(caps_.get(
                         DeviceCapability::cuda_compute_capability))
                   : kDefaultCudaAotComputeCapability;
  const auto resolution =
      cuda::detail::resolve_compute_capability_target(target);
  TI_ERROR_IF(
      target < kDefaultCudaAotComputeCapability ||
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

void LlvmAotModuleBuilder::dump(const std::string &output_dir,
                                const std::string &filename) const {
  LlvmOfflineCacheFileWriter writer;
  writer.set_data(std::move(cache_));
  writer.dump(output_dir);

  if (compile_config_.arch == Arch::cuda) {
    LLVM::LlvmAotMetadata metadata;
    for (const auto capability : {
             DeviceCapability::cuda_compute_capability,
             DeviceCapability::cuda_ptx_version,
         }) {
      metadata.required_caps[to_string(capability)] = caps_.get(capability);
    }
    const std::string json =
        liong::json::print(liong::json::serialize(metadata));
    std::fstream f(
        taichi::join_path(output_dir, LLVM::kLlvmAotMetadataFilename),
                   std::ios::trunc | std::ios::out);
    TI_ERROR_IF(!f.is_open(), "Cannot write CUDA AOT metadata in {}",
                output_dir);
    f.write(json.data(), json.size());
  }

  dump_graph(output_dir);
}

void LlvmAotModuleBuilder::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto compiled = compile_kernel(kernel);
  LlvmOfflineCache::KernelCacheData kcache;
  kcache.kernel_key = identifier;
  kcache.compiled_data = std::move(compiled);
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

LLVMCompiledKernel LlvmAotModuleBuilder::compile_kernel(Kernel *kernel) {
  const auto &ckd =
      compilation_manager_.load_or_compile(compile_config_, caps_, *kernel);
  TI_ASSERT(arch_uses_llvm(ckd.arch()));
  return dynamic_cast<const LLVM::CompiledKernelData &>(ckd)
      .get_internal_data()
      .compiled_data.clone();
}

}  // namespace taichi::lang
