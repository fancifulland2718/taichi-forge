#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/range_dispatch.h"
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
AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<spirv::CompiledSNodeStructs> &compiled_structs,
    KernelCompilationManager &compilation_manager,
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps)
    : compiled_structs_(compiled_structs),
      compilation_manager_(compilation_manager),
      config_(compile_config),
      caps_(caps) {
  // Keep state-free artifacts readable by version-2 runtimes.
  ti_aot_data_.metadata_version = 2;
  for (const auto &pair : caps.to_inner()) {
    ti_aot_data_.required_caps[to_string(pair.first)] = pair.second;
  }
}

std::map<int, int> AotModuleBuilderImpl::exported_snode_tree_bindings() const {
  std::set<int> referenced;
  for (const auto &metadata : ti_aot_data_.kernel_metadata)
    referenced.insert(metadata.used_snode_tree_ids.begin(), metadata.used_snode_tree_ids.end());
  for (const auto &field : ti_aot_data_.fields)
    referenced.insert(field.snode_tree_id);
  std::map<int, int> result;
  for (int tree : referenced) {
    TI_ERROR_IF(tree < 0 || static_cast<std::size_t>(tree) >= compiled_structs_.size() ||
                    !compiled_structs_[tree].root,
                "GFX AOT references inactive SNodeTree {}.", tree);
    result.emplace(tree, static_cast<int>(result.size()));
  }
  return result;
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
    artifact.device_range_dispatch = ti_aot_data_.device_range_dispatch;
    artifact.kernels = ti_aot_data_.kernels;
    artifact.kernel_metadata = ti_aot_data_.kernel_metadata;
    artifact.kernel_aliases = ti_aot_data_.kernel_aliases;
    artifact.fields = ti_aot_data_.fields;
    artifact.required_caps = ti_aot_data_.required_caps;
    const auto module_ids = exported_snode_tree_bindings();
    bool sparse_resources = false;
    for (const auto &[tree_id, local] : module_ids) {
      sparse_resources |= AotSNodeTreeResources::classify_storage(
          compiled_structs_[tree_id].root) == AotSNodeTreeResources::StorageKind::Sparse;
      artifact.root_buffer_sizes.push_back(compiled_structs_[tree_id].root_size);
    }
    if (!artifact.root_buffer_sizes.empty()) {
      artifact.root_buffer_size = artifact.root_buffer_sizes.front();
    }
    if (sparse_resources) {
      TI_ERROR_IF(config_.arch != Arch::vulkan,
                  "Sparse GFX AOT resource reconstruction requires Vulkan");
      artifact.metadata_version = 6;
      for (const auto &[tree_id, local] : module_ids) {
        const auto &compiled = compiled_structs_[tree_id];
        auto entry = AotSNodeTreeResources::from_compiled(tree_id, compiled);
        artifact.snode_tree_resources.push_back(std::move(entry));
      }
    }
    for (const auto &[producer, entry] : initial_root_data_) {
      const auto id = module_ids.find(producer);
      TI_ERROR_IF(id == module_ids.end(),
                  "AOT initial root state references unexported tree {}.",
                  producer);
      TI_ERROR_IF(AotSNodeTreeResources::classify_storage(compiled_structs_[producer].root) !=
                      AotSNodeTreeResources::StorageKind::PlainDense,
                  "AOT initial root state requires plain dense storage.");
      auto local = entry;
      local.tree_id = id->second;
      artifact.initial_root_data.push_back(std::move(local));
    }
    if (!artifact.initial_root_data.empty()) {
      artifact.metadata_version =
          std::max(artifact.metadata_version, uint32_t(3));
      aot::validate_initial_root_data(artifact.initial_root_data,
                                      artifact.root_buffer_sizes);
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
      if (kernel.ctx_attribs.descriptor_array_offsets)
        artifact.metadata_version =
            std::max(artifact.metadata_version, uint32_t(7));
      for (const auto &[indices, argument] : kernel.ctx_attribs.args()) {
        if (argument.ptype == ParameterType::kRWTexture) {
          TI_ERROR_IF(argument.format == BufferFormat::unknown,
                      "AOT RWTexture metadata lacks its format; recompile the "
                      "kernel without the old cache.");
          artifact.metadata_version =
              std::max(artifact.metadata_version, uint32_t(4));
        } else if (argument.ptype == ParameterType::kTextureCollection) {
          artifact.metadata_version =
              std::max(artifact.metadata_version, uint32_t(5));
        }
      }
      for (auto &task : kernel.tasks_attribs) {
        if (!task.cooperative_matrix_requirements.empty())
          artifact.metadata_version =
              std::max(artifact.metadata_version, uint32_t(8));
        for (auto &bind : task.buffer_binds) {
          if (bind.buffer.type == spirv::TaskAttributes::BufferType::Root) {
            TI_ASSERT(bind.buffer.root_id.size() == 1);
            bind.buffer.root_id[0] = module_ids.at(bind.buffer.root_id[0]);
          }
        }
      }
    }
    // Typed range bounds replace the old untagged constant/offset metadata.
    artifact.metadata_version = TaichiAotData::kMetadataVersion;
    std::string json = liong::json::print(liong::json::serialize(artifact));
    aot::write_root_storage_views(root_storage_views_, module_ids, output_dir);
    std::fstream f(output_dir + "/metadata.json",
                   std::ios::trunc | std::ios::out);
    f.write(json.data(), json.size());
  }

  {
    aot::write_struct_graph_artifact(graphs_, output_dir);
    aot::write_struct_layout_artifact(ndarray_layouts_, output_dir);
    aot::write_value_parameter_artifact(value_parameters_, output_dir);
    aot::write_argument_origin_artifact(argument_origins_, output_dir);
    const aot::GraphMap empty;
    std::string json = liong::json::print(liong::json::serialize(
        aot::has_struct_graph_arguments(graphs_) ? empty : graphs_));
    std::fstream f(output_dir + "/graphs.json",
                   std::ios::trunc | std::ios::out);
    f.write(json.data(), json.size());
  }
}

const std::vector<int> &AotModuleBuilderImpl::kernel_snode_tree_ids(
    const std::string &identifier) const {
  const auto found = kernel_names_.find(identifier);
  TI_ERROR_IF(found == kernel_names_.end(), "Unknown AOT kernel export '{}'.",
              identifier);
  return ti_aot_data_.kernel_metadata[found->second].used_snode_tree_ids;
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
      request, *kernel);
  const auto &spirv_ckd = dynamic_cast<const spirv::CompiledKernelData &>(ckd);

  const auto &compiled = spirv_ckd.get_internal_data();
  std::vector<uint32_t> range_tasks;
  if (const auto &plan = kernel->get_offload_execution_plan(); plan) {
    TI_ERROR_IF(plan->tasks.size() != compiled.metadata.kernel_attribs.tasks_attribs.size(),
                "AOT range dispatch task topology differs from compiled code");
    for (std::size_t i = 0; i < plan->tasks.size(); ++i) {
      const auto &task = compiled.metadata.kernel_attribs.tasks_attribs[i];
      TI_ERROR_IF(plan->tasks[i].task_index != i ||
                      plan->tasks[i].task_kind != offloaded_task_type_name(task.task_type),
                  "AOT range dispatch topology differs at task {}", i);
    }
    range_tasks = plan->device_range_tasks;
    RangeDispatch::validate_selection(compiled.metadata.kernel_attribs, range_tasks, kGtmpBufferSize);
  }
  // Global tree counts are compilation-time observations, not dependencies.
  // A module may add fields later or reuse kernels compiled before unrelated
  // layouts were created. Validate its actual closure; dump() assigns compact
  // module-local IDs and the final count without retaining unused layouts.
  for (int tree_id : compiled.metadata.used_snode_tree_ids) {
    TI_ERROR_IF(tree_id < 0 ||
                    static_cast<size_t>(tree_id) >= compiled_structs_.size() ||
                    !compiled_structs_[tree_id].root,
                "GFX AOT kernel '{}' references inactive SNodeTree {}.",
                identifier, tree_id);
  }
  const AotKernelMetadata metadata{/*num_snode_trees=*/0,
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
    const auto selected = std::find_if(ti_aot_data_.device_range_dispatch.begin(),
        ti_aot_data_.device_range_dispatch.end(), [&](const auto &entry) { return entry.name == identifier; });
    const std::vector<uint32_t> empty;
    TI_ERROR_IF((selected == ti_aot_data_.device_range_dispatch.end() ? empty : selected->tasks) != range_tasks,
                "GFX AOT export name '{}' refers to different range dispatch policies", identifier);
    return;
  }
  if (!range_tasks.empty())
    ti_aot_data_.device_range_dispatch.push_back({identifier, 1, std::move(range_tasks)});
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
