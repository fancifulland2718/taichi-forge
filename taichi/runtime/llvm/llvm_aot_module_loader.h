#pragma once

#include "taichi/aot/module_loader.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#include "taichi/runtime/llvm/llvm_aot_metadata.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi::lang {
namespace LLVM {

/* TODO(zhanlue) refactor this interface once SNodeTreeType is available
   The "aot::Field" created by "make_new_field()" is a SNodeTree in essense.
   Therefore we're actually initializing the entire SNodeTree.
*/
TI_DLL_EXPORT void allocate_aot_snode_tree_type(aot::Module *aot_module,
                                                aot::Field *aot_field,
                                                uint64 *result_buffer);

class LlvmAotModule final : public aot::Module {
 public:
  explicit LlvmAotModule(const std::string &module_path,
                         LlvmRuntimeExecutor *executor,
                         std::unique_ptr<LLVM::KernelLauncher> kernel_launcher);
  ~LlvmAotModule() override;

  Arch arch() const override {
    return executor_->get_config().arch;
  }

  uint64_t version() const override {
    return 0;
  }

  size_t get_root_size() const override {
    return 0;
  }

  const DeviceCapabilityConfig &get_required_caps() const override {
    return required_caps_;
  }

  LlvmRuntimeExecutor *const get_runtime_executor() {
    return executor_;
  }

  size_t get_num_snode_trees() {
    return snode_tree_ids_.size();
  }

  const std::vector<int> &get_snode_tree_ids() const {
    return snode_tree_ids_;
  }

  void set_initialized_snode_tree(int snode_tree_id) {
    initialized_snode_tree_ids.insert(snode_tree_id);
  }

  bool is_snode_tree_initialized(int snode_tree_id) {
    return initialized_snode_tree_ids.count(snode_tree_id);
  }

  void initialize_snode_tree(const LlvmOfflineCache::FieldCacheData &field,
                             uint64 *result_buffer);

  std::unique_ptr<aot::CompiledGraph> get_graph(
      const std::string &name) override;

 protected:
  RootStorage get_root_storage(int tree_id) override;
  const io::VirtualDir *data_source() const override {
    return directory_.get();
  }

  LlvmOfflineCache::KernelCacheData load_kernel_from_cache(
      const std::string &name);

  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override;
  std::unique_ptr<aot::Kernel> make_new_kernel_impl(const std::string &name);

  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &name) override {
    TI_NOT_IMPLEMENTED;
  }

  /* TODO(zhanlue): replace "make_new_field()" with "make_snode_tree()" once
     SNodeTreeType is available Field is not a standalone data structure - it is
     essentially part of a SNodeTree object. User should always operate on a
     "SNodeTree" instead of a "Field".
  */
  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override;

  LlvmRuntimeExecutor *const executor_{nullptr};
  std::unique_ptr<LLVM::KernelLauncher> kernel_launcher_{nullptr};
  std::unique_ptr<LlvmOfflineCacheFileReader> cache_reader_{nullptr};
  DeviceCapabilityConfig required_caps_;
  std::shared_ptr<const io::VirtualDir> directory_;
  LlvmAotMetadata metadata_;
  std::unordered_map<std::string, std::string> artifact_failures_;
  std::vector<int> snode_tree_ids_;
  std::vector<std::size_t> root_storage_sizes_;
  std::map<int, int> tree_bindings_;
  std::vector<int> reserved_tree_ids_;
  std::map<int, aot::InitialRootData> initial_root_data_;

  // To prevent repeated SNodeTree initialization
  std::unordered_set<int> initialized_snode_tree_ids;
};

struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path;
  LlvmRuntimeExecutor *executor_{nullptr};
  std::unique_ptr<LLVM::KernelLauncher> kernel_launcher{nullptr};
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(
    AotModuleParams mod_params);

}  // namespace LLVM
}  // namespace taichi::lang
