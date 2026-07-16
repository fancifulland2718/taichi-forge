#include "taichi/runtime/gfx/aot_module_builder_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/aot/module_data.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/runtime/gfx/aot_graph_data.h"

namespace taichi::lang {
namespace gfx {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<spirv::CompiledSNodeStructs> &compiled_structs,
    KernelCompilationManager &compilation_manager,
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps)
    : compiled_structs_(compiled_structs),
      compilation_manager_(compilation_manager),
      config_(compile_config),
      caps_(caps) {
  ti_aot_data_.metadata_version = TaichiAotData::kMetadataVersion;
  for (const auto &pair : caps.to_inner()) {
    ti_aot_data_.required_caps[to_string(pair.first)] = pair.second;
  }
  ti_aot_data_.root_buffer_sizes.reserve(compiled_structs.size());
  for (std::size_t tree_id = 0; tree_id < compiled_structs.size(); ++tree_id) {
    const auto &compiled = compiled_structs[tree_id];
    TI_ERROR_IF(
        compiled.root == nullptr ||
            compiled.root->get_snode_tree_id() != static_cast<int>(tree_id),
        "GFX AOT requires active SNodeTrees with contiguous artifact-local "
        "ids [0, N). Tree slot {} is empty or has a different runtime id; "
        "rebuild the module from a clean runtime after destroyed trees.",
        tree_id);
    ti_aot_data_.root_buffer_sizes.push_back(compiled.root_size);
  }
  if (!ti_aot_data_.root_buffer_sizes.empty()) {
    ti_aot_data_.root_buffer_size = ti_aot_data_.root_buffer_sizes.front();
  }
}

std::string AotModuleBuilderImpl::write_spv_file(
    const std::string &output_dir,
    const TaskAttributes &k,
    const std::vector<uint32_t> &source_code) const {
  const std::string spv_path = fmt::format("{}/{}.spv", output_dir, k.name);
  std::ofstream fs(spv_path, std::ios_base::binary | std::ios::trunc);
  fs.write((char *)source_code.data(), source_code.size() * sizeof(uint32_t));
  fs.close();
  return k.name + ".spv";
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on Unified Device API backends.");

  const auto &spirv_codes = ti_aot_data_.spirv_codes;
  for (int i = 0; i < spirv_codes.size(); ++i) {
    auto &k = ti_aot_data_.kernels[i];
    for (int j = 0; j < spirv_codes[i].size(); ++j) {
      if (!spirv_codes[i][j].empty()) {
        std::string spv_path =
            write_spv_file(output_dir, k.tasks_attribs[j], spirv_codes[i][j]);
      }
    }
  }

  {
    std::string json = liong::json::print(liong::json::serialize(ti_aot_data_));
    std::fstream f(output_dir + "/metadata.json",
                   std::ios::trunc | std::ios::out);
    f.write(json.data(), json.size());
  }

  {
    std::string json = liong::json::print(liong::json::serialize(graphs_));
    std::fstream f(output_dir + "/graphs.json",
                   std::ios::trunc | std::ios::out);
    f.write(json.data(), json.size());
  }
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  const auto &ckd =
      compilation_manager_.load_or_compile(config_, caps_, *kernel);
  const auto &spirv_ckd = dynamic_cast<const spirv::CompiledKernelData &>(ckd);

  auto compiled = spirv_ckd.get_internal_data();
  TI_ERROR_IF(
      compiled.metadata.num_snode_trees !=
          ti_aot_data_.root_buffer_sizes.size(),
      "GFX AOT kernel '{}' was compiled for {} SNodeTrees, but the artifact "
      "contains {} layouts.",
      identifier,
      compiled.metadata.num_snode_trees,
      ti_aot_data_.root_buffer_sizes.size());
  compiled.metadata.kernel_attribs.name = identifier;
  ti_aot_data_.kernels.push_back(compiled.metadata.kernel_attribs);
  ti_aot_data_.kernel_metadata.push_back(
      {compiled.metadata.num_snode_trees,
       compiled.metadata.used_snode_tree_ids});
  ti_aot_data_.spirv_codes.push_back(compiled.src.spirv_src);
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  // Note that currently we only support adding dense fields in AOT for all
  // backends. In opengl backend we only error out when a non dense field is
  // added to the aot module, but in metal backend we error out earlier when
  // constructing aot module. Ideally we will unify this behavior but it doesn't
  // matter too much for now.
  TI_ERROR_IF(!all_fields_are_dense_in_container(rep_snode->parent),
              "AOT: only supports dense field");

  const int tree_id = rep_snode->get_snode_tree_id();
  TI_ERROR_IF(tree_id < 0 ||
                  static_cast<std::size_t>(tree_id) >=
                      compiled_structs_.size() ||
                  compiled_structs_[tree_id].root == nullptr,
              "AOT field '{}' belongs to inactive SNodeTree id {}.",
              identifier,
              tree_id);
  const auto &dense_desc =
      compiled_structs_[tree_id].snode_descriptors.at(rep_snode->parent->id);

  aot::CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.snode_tree_id = tree_id;
  field_data.is_scalar = is_scalar;
  field_data.dtype = static_cast<int>(dt->cast<PrimitiveType>()->type);
  field_data.dtype_name = dt.to_string();
  field_data.shape = shape;
  field_data.mem_offset_in_parent = dense_desc.mem_offset_in_parent_cell;
  if (!is_scalar) {
    field_data.element_shape = {row_num, column_num};
  }
  ti_aot_data_.fields.push_back(field_data);
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  const auto &ckd =
      compilation_manager_.load_or_compile(config_, caps_, *kernel);
  const auto &spirv_ckd = dynamic_cast<const spirv::CompiledKernelData &>(ckd);

  auto compiled = spirv_ckd.get_internal_data();
  TI_ERROR_IF(
      compiled.metadata.num_snode_trees !=
          ti_aot_data_.root_buffer_sizes.size(),
      "GFX AOT kernel template '{}|{}' was compiled for {} SNodeTrees, but "
      "the artifact contains {} layouts.",
      identifier,
      key,
      compiled.metadata.num_snode_trees,
      ti_aot_data_.root_buffer_sizes.size());
  compiled.metadata.kernel_attribs.name = identifier + "|" + key;
  ti_aot_data_.kernels.push_back(compiled.metadata.kernel_attribs);
  ti_aot_data_.kernel_metadata.push_back(
      {compiled.metadata.num_snode_trees,
       compiled.metadata.used_snode_tree_ids});
  ti_aot_data_.spirv_codes.push_back(compiled.src.spirv_src);
}

}  // namespace gfx
}  // namespace taichi::lang
