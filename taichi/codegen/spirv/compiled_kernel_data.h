#pragma once

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi::lang {

namespace spirv {

class CompiledKernelData : public lang::CompiledKernelData {
 public:
  struct InternalData {
    using TaskCode = std::vector<uint32_t>;
    using TasksCode = std::vector<TaskCode>;

    // meta data
    struct Metadata {
      TaichiKernelAttributes kernel_attribs;
      std::size_t num_snode_trees{0};
      std::vector<int> used_snode_tree_ids;
      GraphKernelMetadata graph_metadata;
      TI_IO_DEF(kernel_attribs,
                num_snode_trees,
                used_snode_tree_ids,
                graph_metadata);
    } metadata;
    // source code
    struct Source {
      TasksCode spirv_src;
      TI_IO_DEF(spirv_src);
    } src;
  };

  CompiledKernelData() = default;
  CompiledKernelData(Arch arch, InternalData data);

  Arch arch() const override;
  std::unique_ptr<lang::CompiledKernelData> clone() const override;
  std::vector<int> snode_tree_ids() const override;
  std::size_t task_count() const override;
  std::vector<OffloadedTaskManifest> task_manifest() const override;
  const GraphKernelMetadata &graph_metadata() const override {
    return data_.metadata.graph_metadata;
  }
  void set_graph_metadata(GraphKernelMetadata metadata) override {
    data_.metadata.graph_metadata = std::move(metadata);
  }

  const InternalData &get_internal_data() const {
    return data_;
  }

 protected:
  void refresh_task_identities() override;
  Err load_impl(const CompiledKernelDataFile &file) override;
  Err dump_impl(CompiledKernelDataFile &file) const override;

 private:
  static Err src2str(const InternalData::Source &src, std::string &result);
  static Err str2src(const std::string &str, InternalData::Source &result);

  Arch arch_;
  InternalData data_;
};

}  // namespace spirv

}  // namespace taichi::lang
