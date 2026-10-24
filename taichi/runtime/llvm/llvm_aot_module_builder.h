#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

class LlvmAotModuleBuilder : public AotModuleBuilder {
 public:
  explicit LlvmAotModuleBuilder(KernelCompilationManager &compilation_manager,
                                const CompileConfig &compile_config,
                                LlvmProgramImpl *prog,
                                const DeviceCapabilityConfig &caps);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

  void set_artifact_format(const std::string &format) override;

 protected:
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
  const LLVM::CompiledKernelData &compile_kernel(Kernel *kernel);
  void add_kernel_impl(const std::string &identifier,
                        Kernel *kernel,
                        const std::string &launch_abi);

  LlvmOfflineCache cache_;
  std::map<std::string, std::string> kernel_launch_abis_;
  std::map<std::string, std::string> kernel_names_;
  std::map<std::string, std::string> kernel_identities_;
  std::map<std::string, std::vector<std::uint32_t>> range_selections_;
  KernelCompilationManager &compilation_manager_;
  const CompileConfig &compile_config_;
  LlvmProgramImpl *prog_ = nullptr;
  DeviceCapabilityConfig caps_;
  bool explicit_cuda_target_{false};
  std::string artifact_format_{"bc"};
};

}  // namespace taichi::lang
