#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/gfx/aot_utils.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/compilation_manager/kernel_compilation_manager.h"

namespace taichi::lang {
namespace gfx {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(
      const std::vector<spirv::CompiledSNodeStructs> &compiled_structs,
      KernelCompilationManager &compilation_manager,
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &caps);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;
  std::map<int, int> exported_snode_tree_bindings() const override;

 private:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

  const std::vector<int> &kernel_snode_tree_ids(
      const std::string &identifier) const override;

  void add_compiled_kernel(const std::string &identifier, Kernel *kernel);

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;

  std::string write_spv_file(const std::string &output_dir,
                             const spirv::TaskAttributes &k,
                             const std::vector<uint32_t> &source_code) const;

  const std::vector<spirv::CompiledSNodeStructs> &compiled_structs_;
  TaichiAotData ti_aot_data_;
  std::unordered_map<std::string, size_t> kernel_identities_;
  std::unordered_map<std::string, size_t> kernel_names_;

  KernelCompilationManager &compilation_manager_;
  const CompileConfig &config_;
  DeviceCapabilityConfig caps_;
};

}  // namespace gfx
}  // namespace taichi::lang
