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
      !properties.compact_contiguous || !properties.ndarray_abi_compatible ||
      !properties.aligned || descriptor.byte_offset() < 0)
    throw std::invalid_argument(
        "AOT root storage requires a named packed dense Field view");
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
  view.bytes();
  const auto [found, inserted] = root_storage_views_.emplace(name, view);
  if (!inserted && !(found->second == view))
    throw std::invalid_argument(
        "AOT root storage name has conflicting layouts");
}

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  compile_options_frozen_ = true;
  add_per_backend(identifier, kernel);
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
  compile_options_frozen_ = true;
  add_per_backend_tmpl(identifier, key, kernel);
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

void AotModuleBuilder::dump_graph(std::string output_dir) const {
  const std::string graph_file = fmt::format("{}/graphs.tcb", output_dir);
  aot::write_struct_graph_artifact(graphs_, output_dir);
  // Old readers see no exports they cannot represent; ordinary artifacts keep
  // their existing binary representation and remain readable by old loaders.
  const aot::GraphMap empty;
  write_to_binary_file(aot::has_struct_graph_arguments(graphs_) ? empty : graphs_,
                       graph_file);
}

void AotModuleBuilder::add_graph(const std::string &name,
                                 const aot::CompiledGraph &graph) {
  compile_options_frozen_ = true;
  for (const auto &[arg_name, arg] : graph.args) {
    TI_ERROR_IF(arg.tag == aot::ArgKind::kAccelerationStructure,
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
  // Handle adding kernels separately.
  std::unordered_map<std::string, lang::Kernel *> kernels;
  for (const auto &dispatch : graph.dispatches) {
    kernels[dispatch.kernel_name] = dispatch.ti_kernel;
  }
  for (auto &e : kernels) {
    add(e.first, e.second);
  }
  graphs_[name] = graph;
}
}  // namespace taichi::lang
