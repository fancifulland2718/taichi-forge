#include "taichi/aot/module_builder.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/program/kernel.h"

#include <set>
#include <limits>

namespace taichi::lang {

void AotModuleBuilder::add_root_storage_view(
    const std::string &name,
    const storage::DenseStorageDescriptor &descriptor) {
  const auto &properties = descriptor.properties();
  if (name.empty() || name.find('\0') != std::string::npos ||
      descriptor.owner().kind != storage::StorageOwnerKind::kSNodePayload ||
      !properties.aligned ||
      properties.uniqueness != storage::StorageMappingUniqueness::kProvenUnique ||
      descriptor.byte_offset() < 0)
    throw std::invalid_argument(
        "AOT root storage requires a named aligned affine dense Field view");
  aot::RootStorageView view;
  view.tree_id = descriptor.owner().tree.tree_id;
  view.byte_offset = static_cast<std::uint64_t>(descriptor.byte_offset());
  view.dtype = descriptor.scalar_type()->to_string();
  const auto copy_shape = [](const auto &source, auto &destination) {
    for (const auto size : source) {
      if (size < 0 || size > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(
            "AOT root storage dimensions must fit uint32");
      destination.push_back(static_cast<std::uint32_t>(size));
    }
  };
  copy_shape(descriptor.index_shape(), view.shape);
  copy_shape(descriptor.element_shape(), view.element_shape);
  view.index_strides_bytes = descriptor.index_strides_bytes();
  view.element_strides_bytes = descriptor.element_strides_bytes();
  view.properties();
  const auto [found, inserted] = root_storage_views_.emplace(name, view);
  if (!inserted && !(found->second == view))
    throw std::invalid_argument(
        "AOT root storage name has conflicting layouts");
}

bool AotModuleBuilder::needs_kernel_export(
    const std::string &name,
    Kernel *kernel,
    const std::string &launch_abi) const {
  TI_ERROR_IF(kernel == nullptr, "AOT kernel {} has no compiled identity",
              name);
  const auto found = kernel_exports_.find(name);
  if (found == kernel_exports_.end())
    return true;
  TI_ERROR_IF(
      found->second.kernel != kernel || found->second.launch_abi != launch_abi,
      "AOT kernel name {} is already bound to another specialization or "
      "launch ABI",
      name);
  return false;
}

void AotModuleBuilder::record_kernel_export(const std::string &name,
                                            Kernel *kernel,
                                            const std::string &launch_abi) {
  if (kernel->value_parameter_types) {
    for (const auto &[indices, logical] : *kernel->value_parameter_types)
      aot::validate_value_parameter_type(
          logical, aot::compiled_value_parameter_type(*kernel, indices));
    value_parameters_.emplace(name, *kernel->value_parameter_types);
  }
  kernel_exports_.emplace(name, ExportIdentity{kernel, launch_abi});
  if (launch_abi == "ordinary")
    ordinary_export_names_.emplace(kernel, name);
}

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  if (!needs_kernel_export(identifier, kernel, "ordinary"))
    return;
  compile_options_frozen_ = true;
  add_per_backend(identifier, kernel);
  record_kernel_export(identifier, kernel, "ordinary");
}

void AotModuleBuilder::add_with_launch_abi(const std::string &identifier,
                                           Kernel *kernel,
                                           const std::string &launch_abi) {
  if (launch_abi == "ordinary") {
    add(identifier, kernel);
    return;
  }
  if (!needs_kernel_export(identifier, kernel, launch_abi))
    return;
  add_with_launch_abi_per_backend(identifier, kernel, launch_abi);
  compile_options_frozen_ = true;
  record_kernel_export(identifier, kernel, launch_abi);
}

void AotModuleBuilder::add_field(const std::string &identifier,
                                 const SNode *rep_snode,
                                 bool is_scalar,
                                 DataType dt,
                                 std::vector<int> shape,
                                 int row_num,
                                 int column_num) {
  add_field_per_backend(identifier, rep_snode, is_scalar, dt, shape, row_num,
                        column_num);
}

void AotModuleBuilder::add_kernel_template(const std::string &identifier,
                                           const std::string &key,
                                           Kernel *kernel) {
  const auto name = identifier + "__tmpl__" + key;
  if (!needs_kernel_export(name, kernel, "ordinary"))
    return;
  compile_options_frozen_ = true;
  add_per_backend_tmpl(identifier, key, kernel);
  record_kernel_export(name, kernel, "ordinary");
}

bool AotModuleBuilder::all_fields_are_dense_in_container(
    const SNode *container) {
  for (const auto &ch : container->ch) {
    if (ch->type != SNodeType::place) {
      return false;
    }
  }
  const auto *parent = container->parent;
  if (!parent) {
    return false;
  }
  if (parent->type != SNodeType::root) {
    return false;
  }
  return true;
}

std::vector<int> AotModuleBuilder::used_snode_tree_ids(
    const std::vector<std::string> &entries) const {
  std::set<int> ids;
  for (const auto &name : entries) {
    const auto &dependencies = kernel_snode_tree_ids(name);
    ids.insert(dependencies.begin(), dependencies.end());
  }
  return {ids.begin(), ids.end()};
}

void AotModuleBuilder::add_initial_root_data(
    const std::vector<aot::InitialRootData> &entries) {
  auto merged = initial_root_data_;
  for (const auto &entry : entries) {
    aot::validate_initial_root_data(entry);
    const auto [existing, inserted] = merged.emplace(entry.tree_id, entry);
    TI_ERROR_IF(!inserted && (existing->second.size != entry.size ||
                              existing->second.sha256 != entry.sha256),
                "AOT initial root state conflicts for tree {}; create a new "
                "module.",
                entry.tree_id);
  }
  initial_root_data_ = std::move(merged);
}

void AotModuleBuilder::load(const std::string &output_dir) {
  TI_ERROR("Aot loader not supported");
}

void AotModuleBuilder::add_ndarray_layout(const std::string &name, DataType type) {
  if (name.empty() || name.find('\0') != std::string::npos)
    throw std::invalid_argument("Invalid AOT ndarray layout name");
  const auto found = ndarray_layouts_.find(name);
  if (found != ndarray_layouts_.end()) {
    if (found->second != type)
      throw std::invalid_argument("Conflicting AOT ndarray layout name");
    return;
  }
  aot::StructLayout checked(type);
  ndarray_layouts_.emplace(name, type);
}

void AotModuleBuilder::dump_graph(std::string output_dir) const {
  const std::string graph_file = fmt::format("{}/graphs.tcb", output_dir);
  aot::write_struct_graph_artifact(graphs_, output_dir);
  // Old readers see no exports they cannot represent; ordinary artifacts keep
  // their existing binary representation and remain readable by old loaders.
  const aot::GraphMap empty;
  write_to_binary_file(
      aot::has_struct_graph_arguments(graphs_) ? empty : graphs_, graph_file);
}

std::unordered_map<std::string, Kernel *> AotModuleBuilder::add_graph(
    const std::string &name,
    const aot::CompiledGraph &graph) {
  compile_options_frozen_ = true;
  for (const auto &[arg_name, arg] : graph.args) {
    TI_ERROR_IF(
        arg.tag == aot::ArgKind::kAccelerationStructure,
        "AOT Graph {} contains JIT-only acceleration-structure argument {}",
        name, arg_name);
    TI_ERROR_IF(arg.tag == aot::ArgKind::kTextureCollection,
                "AOT Graph {} contains JIT-only TextureCollection argument {}",
                name, arg_name);
  }
  TI_ERROR_IF(graph.has_indirect_dispatches(),
              "AOT Graph {} contains JIT-only indirect dispatches", name);
  TI_ERROR_IF(graph.has_dispatch_labels(),
              "AOT Graph {} contains JIT-only dispatch labels", name);
  if (graphs_.count(name) != 0) {
    TI_ERROR("Graph {} already exists", name);
  }
  // Resolve the entire closure before compiling anything. Export only a copy:
  // the source Graph may still execute or be exported through another module.
  auto exported = graph;
  std::unordered_map<std::string, Kernel *> kernels;
  std::unordered_map<Kernel *, std::string> pending_names;
  std::vector<std::pair<std::string, Kernel *>> pending;
  for (auto &dispatch : exported.dispatches) {
    auto *kernel = dispatch.ti_kernel;
    const auto published = ordinary_export_names_.find(kernel);
    if (published != ordinary_export_names_.end()) {
      dispatch.kernel_name = published->second;
    } else {
      const auto planned = pending_names.find(kernel);
      if (planned != pending_names.end()) {
        dispatch.kernel_name = planned->second;
      } else {
        needs_kernel_export(dispatch.kernel_name, kernel, "ordinary");
        const auto [found, inserted] =
            kernels.emplace(dispatch.kernel_name, kernel);
        TI_ERROR_IF(!inserted && found->second != kernel,
                    "AOT Graph {} has conflicting kernel name {}", name,
                    dispatch.kernel_name);
        pending_names.emplace(kernel, dispatch.kernel_name);
        pending.emplace_back(dispatch.kernel_name, kernel);
      }
    }
    kernels.emplace(dispatch.kernel_name, kernel);
  }
  for (const auto &e : pending) {
    add(e.first, e.second);
  }
  graphs_.emplace(name, std::move(exported));
  return kernels;
}
}  // namespace taichi::lang
