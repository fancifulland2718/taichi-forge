#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/program/kernel_compile_request.h"

#include <fstream>
#include <set>
#include <type_traits>

#include "taichi/aot/module_data.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/runtime/gfx/aot_graph_data.h"

namespace taichi::lang {
namespace gfx {
namespace {

bool is_dense_tree(const SNode *node) {
  if (node->need_activation()) {
    return false;
  }
  for (const auto &child : node->ch) {
    if (!is_dense_tree(child.get())) {
      return false;
    }
  }
  return true;
}

}  // namespace

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
    // Producer IDs may contain retired slots. Only the exported resource
    // closure is required to be active; dump assigns its compact module IDs.
    ti_aot_data_.root_buffer_sizes.push_back(compiled.root ? compiled.root_size : 0);
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
    // Copy metadata only, never the potentially large SPIR-V payload vectors.
    TaichiAotData artifact;
    artifact.metadata_version = ti_aot_data_.metadata_version;
    artifact.kernels = ti_aot_data_.kernels;
    artifact.kernel_metadata = ti_aot_data_.kernel_metadata;
    artifact.kernel_aliases = ti_aot_data_.kernel_aliases;
    artifact.fields = ti_aot_data_.fields;
    artifact.required_caps = ti_aot_data_.required_caps;
    std::set<int> referenced;
    for (const auto &metadata : artifact.kernel_metadata) {
      referenced.insert(metadata.used_snode_tree_ids.begin(),
                        metadata.used_snode_tree_ids.end());
    }
    for (const auto &field : artifact.fields) {
      referenced.insert(field.snode_tree_id);
    }
    std::map<int, int> module_ids;
    for (int tree_id : referenced) {
      TI_ERROR_IF(tree_id < 0 ||
                      static_cast<size_t>(tree_id) >= compiled_structs_.size() ||
                      !compiled_structs_[tree_id].root,
                  "GFX AOT references inactive SNodeTree {}.", tree_id);
      TI_ERROR_IF(!is_dense_tree(compiled_structs_[tree_id].root),
                  "GFX AOT resource relocation supports dense SNodeTrees; "
                  "SNodeTree {} requires sparse runtime state.", tree_id);
      module_ids.emplace(tree_id, static_cast<int>(artifact.root_buffer_sizes.size()));
      artifact.root_buffer_sizes.push_back(compiled_structs_[tree_id].root_size);
    }
    if (!artifact.root_buffer_sizes.empty()) {
      artifact.root_buffer_size = artifact.root_buffer_sizes.front();
    }
    for (auto &metadata : artifact.kernel_metadata) {
      metadata.num_snode_trees = artifact.root_buffer_sizes.size();
      for (int &id : metadata.used_snode_tree_ids) {
        id = module_ids.at(id);
      }
    }
    for (auto &field : artifact.fields) {
      field.snode_tree_id = module_ids.at(field.snode_tree_id);
    }
    for (auto &kernel : artifact.kernels) {
      for (auto &task : kernel.tasks_attribs) {
        for (auto &bind : task.buffer_binds) {
          if (bind.buffer.type == spirv::TaskAttributes::BufferType::Root) {
            TI_ASSERT(bind.buffer.root_id.size() == 1);
            bind.buffer.root_id[0] = module_ids.at(bind.buffer.root_id[0]);
          }
        }
      }
    }
    std::string json = liong::json::print(liong::json::serialize(artifact));
    std::fstream f(output_dir + "/metadata.json",
                   std::ios::trunc | std::ios::out);
    f.write(json.data(), json.size());
  }

  {
    aot::write_struct_graph_artifact(graphs_, output_dir);
    const aot::GraphMap empty;
    std::string json = liong::json::print(liong::json::serialize(
        aot::has_struct_graph_arguments(graphs_) ? empty : graphs_));
    std::fstream f(output_dir + "/graphs.json",
                   std::ios::trunc | std::ios::out);
    f.write(json.data(), json.size());
  }
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  add_compiled_kernel(identifier, kernel);
}

void AotModuleBuilderImpl::add_compiled_kernel(const std::string &identifier,
                                              Kernel *kernel) {
  const auto request = resolve_kernel_compile_request(
      config_, caps_, kernel->get_compile_tier_override(), CompilePurpose::aot,
      compile_options_);
  const auto &ckd = compilation_manager_.load_or_compile(
      request.config(), request.device_caps(), *kernel);
  const auto &spirv_ckd = dynamic_cast<const spirv::CompiledKernelData &>(ckd);

  const auto &compiled = spirv_ckd.get_internal_data();
  TI_ERROR_IF(
      compiled.metadata.num_snode_trees !=
          ti_aot_data_.root_buffer_sizes.size(),
      "GFX AOT kernel '{}' was compiled for {} SNodeTrees, but the artifact "
      "contains {} layouts.",
      identifier,
      compiled.metadata.num_snode_trees,
      ti_aot_data_.root_buffer_sizes.size());
  const AotKernelMetadata metadata{compiled.metadata.num_snode_trees,
                                   compiled.metadata.used_snode_tree_ids};
  const auto identity = kernel_identity(compiled.metadata.kernel_attribs, metadata);
  const auto existing = kernel_identities_.find(identity);
  const size_t index = existing == kernel_identities_.end()
                           ? ti_aot_data_.kernels.size()
                           : existing->second;
  if (existing != kernel_identities_.end()) {
    TI_ERROR_IF(ti_aot_data_.spirv_codes[index] != compiled.src.spirv_src,
                "GFX AOT kernel '{}' has conflicting shader payloads for the "
                "same task and ABI identity.", identifier);
  }
  const auto [named, inserted] = kernel_names_.emplace(identifier, index);
  TI_ERROR_IF(!inserted && named->second != index,
              "GFX AOT export name '{}' refers to different kernels.", identifier);
  if (!inserted) {
    return;
  }
  if (existing != kernel_identities_.end()) {
    ti_aot_data_.kernel_aliases.push_back({identifier, index});
    return;
  }
  kernel_identities_.emplace(identity, index);
  ti_aot_data_.kernels.push_back(compiled.metadata.kernel_attribs);
  ti_aot_data_.kernels.back().name = identifier;
  ti_aot_data_.kernel_metadata.push_back(metadata);
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
  add_compiled_kernel(identifier + "__tmpl__" + key, kernel);
}

}  // namespace gfx
}  // namespace taichi::lang
