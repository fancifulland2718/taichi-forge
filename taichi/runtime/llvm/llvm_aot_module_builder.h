#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

namespace LLVM {
class FunctionCodeCache;
}

class LlvmAotModuleBuilder : public AotModuleBuilder {
 public:
  explicit LlvmAotModuleBuilder(KernelCompilationManager &compilation_manager,
                                const CompileConfig &compile_config,
                                LlvmProgramImpl *prog,
                                const DeviceCapabilityConfig &caps);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

  void set_artifact_format(const std::string &format) override;
  void enable_streaming_exports() override;
  std::map<std::string, std::uint64_t> streaming_export_statistics() override;
  std::map<int, int> exported_snode_tree_bindings() const override {
    return exported_tree_bindings_;
  }

 protected:
  void seal_kernel_export_per_backend(const std::string &identifier) override;
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;
  void add_with_launch_abi_per_backend(
      const std::string &identifier,
      Kernel *kernel,
      const std::string &launch_abi) override;
  const std::vector<int> &kernel_snode_tree_ids(
      const std::string &identifier) const override;
  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

 private:
  std::unique_ptr<llvm::Module> load_kernel_module(
      const std::string &name,
      std::shared_ptr<llvm::LLVMContext> &context_owner) const;
  const LLVM::CompiledKernelData &compile_kernel(Kernel *kernel);
  void add_kernel_impl(const std::string &identifier,
                        Kernel *kernel,
                        const std::string &launch_abi);

  LlvmOfflineCache cache_;
  mutable std::map<int, int> exported_tree_bindings_;
  std::map<std::string, std::string> kernel_launch_abis_;
  std::map<std::string, std::string> kernel_names_;
  std::map<std::string, std::string> kernel_identities_;
  std::map<std::string, std::vector<std::uint32_t>> range_selections_;
  KernelCompilationManager &compilation_manager_;
  // Reuse the same compiler/cache implementation, with build-local ownership.
  // Sealing never evicts a normal JIT/Graph/Component caller's cache entry.
  std::unique_ptr<KernelCompilationManager> streaming_compilation_manager_;
  std::map<std::string, std::string> sealed_bitcode_;
  std::map<std::string, std::string> sealed_code_units_;
  std::shared_ptr<LLVM::FunctionCodeCache> function_code_cache_;
  const CompileConfig &compile_config_;
  LlvmProgramImpl *prog_ = nullptr;
  DeviceCapabilityConfig caps_;
  bool explicit_cuda_target_{false};
  std::string artifact_format_{"bc"};
};

}  // namespace taichi::lang
