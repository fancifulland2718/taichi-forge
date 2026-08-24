#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

#include <fstream>
#include <iterator>

#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang {
namespace LLVM {

LlvmAotModule::LlvmAotModule(
    const std::string &module_path,
    LlvmRuntimeExecutor *executor,
    std::unique_ptr<LLVM::KernelLauncher> kernel_launcher)
    : executor_(executor),
      kernel_launcher_(std::move(kernel_launcher)),
      cache_reader_(LlvmOfflineCacheFileReader::make(module_path)) {
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
  LlvmAotMetadata metadata;
  auto json = liong::json::parse(contents.data(),
                                 contents.data() + contents.size());
  liong::json::deserialize(json, metadata);
  validate_llvm_aot_metadata(metadata);

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

  const std::string graph_path = fmt::format("{}/graphs.tcb", module_path);
  read_from_binary_file(graphs_, graph_path);
}

FunctionType LlvmAotModule::convert_module_to_function(
    const std::string &name,
    LlvmOfflineCache::KernelCacheData &&loaded) {
  Arch arch = executor_->get_config().arch;
  auto *launcher = kernel_launcher_.get();
  LLVM::CompiledKernelData ckd{arch, loaded.convert_to_llvm_ckd_data()};
  auto handle = kernel_launcher_->register_llvm_kernel(ckd);
  return [handle, launcher](LaunchContextBuilder &ctx) {
    launcher->launch_llvm_kernel(handle, ctx);
  };
}

LlvmOfflineCache::KernelCacheData LlvmAotModule::load_kernel_from_cache(
    const std::string &name) {
  TI_ASSERT(cache_reader_ != nullptr);
  auto *tlctx = executor_->get_llvm_context();
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = cache_reader_->get_kernel_cache(loaded, name,
                                            *tlctx->get_this_thread_context());
  TI_ERROR_IF(!ok, "Failed to load kernel={}", name);
  return loaded;
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel(
    const std::string &name) {
  auto kernel_cache = load_kernel_from_cache(name);
  auto fn = convert_module_to_function(name, kernel_cache.clone());
  return std::make_unique<llvm_aot::KernelImpl>(fn, std::move(kernel_cache));
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
