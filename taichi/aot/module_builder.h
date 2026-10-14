#pragma once

#include <string>
#include <map>
#include <vector>

#include "taichi/aot/module_data.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/snode.h"
#include "taichi/aot/module_data.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/initial_root_data.h"
#include "taichi/program/kernel_compile_request.h"

namespace taichi::lang {

class Kernel;
class DataType;

class AotModuleBuilder {
 public:
  virtual ~AotModuleBuilder() = default;

  virtual void set_artifact_format(const std::string &format) {
    TI_ERROR("Explicit AOT artifact format '{}' is unsupported by this backend",
             format);
  }

  void set_compile_options(KernelCompileOptions options) {
    TI_ERROR_IF(compile_options_frozen_,
                "AOT compile options must be selected before exporting kernels.");
    compile_options_ = std::move(options);
  }

  void add(const std::string &identifier, Kernel *kernel);

  void add_field(const std::string &identifier,
                 const SNode *rep_snode,
                 bool is_scalar,
                 DataType dt,
                 std::vector<int> shape,
                 int row_num,
                 int column_num);

  void add_kernel_template(const std::string &identifier,
                           const std::string &key,
                           Kernel *kernel);

  virtual void load(const std::string &output_dir);

  virtual void dump(const std::string &output_dir,
                    const std::string &filename) const = 0;

  void add_graph(const std::string &name, const aot::CompiledGraph &graph);

  // Producer IDs are relocated by dump(). Reject conflicting re-snapshots.
  void add_initial_root_data(const std::vector<aot::InitialRootData> &entries);

  // Exact dependencies of selected compiled exports, in producer tree IDs.
  std::vector<int> used_snode_tree_ids(
      const std::vector<std::string> &entries) const;

 protected:
  /**
   * Intended to be overridden by each backend's implementation.
   */
  virtual void add_per_backend(const std::string &identifier,
                               Kernel *kernel) = 0;
  virtual const std::vector<int> &kernel_snode_tree_ids(
      const std::string &identifier) const {
    TI_ERROR("AOT dependency inspection is unsupported by this backend.");
    static const std::vector<int> empty;
    return empty;
  }
  virtual void add_field_per_backend(const std::string &identifier,
                                     const SNode *rep_snode,
                                     bool is_scalar,
                                     DataType dt,
                                     std::vector<int> shape,
                                     int row_num,
                                     int column_num) {
    TI_NOT_IMPLEMENTED;
  }

  virtual void add_ndarray_per_backend(const std::string &identifier,
                                       bool is_scalar,
                                       DataType dt,
                                       std::vector<int> shape,
                                       int row_num,
                                       int column_num) {
    TI_NOT_IMPLEMENTED;
  }

  virtual void add_per_backend_tmpl(const std::string &identifier,
                                    const std::string &key,
                                    Kernel *kernel) {
    TI_NOT_IMPLEMENTED;
  }

  void dump_graph(std::string output_dir) const;

  static bool all_fields_are_dense_in_container(const SNode *container);

 protected:
  std::unordered_map<std::string, aot::CompiledGraph> graphs_;
  std::map<int, aot::InitialRootData> initial_root_data_;
  KernelCompileOptions compile_options_;
  bool compile_options_frozen_{false};
};

}  // namespace taichi::lang
