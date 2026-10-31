#include "taichi_core_impl.h"
#include <cstring>
#include <limits>

#include "argument_binding.h"
#ifdef TI_WITH_LLVM
#include "taichi_llvm_impl.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#endif
#include "taichi/struct/dense_snode_schema.h"

namespace {
using namespace taichi::lang;

#ifdef TI_WITH_LLVM
struct GroupAllocation {
  DeviceAllocation allocation{};
  LlvmRuntimeExecutor *executor{nullptr};
  ~GroupAllocation() {
    // A backend failure must not throw from a lease destructor. Failed device
    // retirement is handled by the runtime's existing fault domain.
    try {
      if (executor)
        executor->deallocate_memory_on_device(allocation);
    } catch (...) {
    }
  }
};

#endif

struct LayoutNode {
  int parent{-1};
  int type{0};
  std::string dtype;
  std::vector<int> shape;
  std::size_t cell_size{0};
  std::size_t offset{0};
  TI_IO_DEF(parent, type, dtype, shape, cell_size, offset);
};
struct Layout {
  int version{1};
  std::size_t root_size{0};
  std::vector<LayoutNode> nodes;
  TI_IO_DEF(version, root_size, nodes);
};

}  // namespace

struct capi::DenseSNodeGroup {
  std::unique_ptr<taichi::lang::SNode> root;
  std::shared_ptr<Runtime::BorrowedMemory> storage;
  std::string layout;
};

#ifdef TI_WITH_LLVM
std::size_t capi::LlvmRuntime::compile_dense_snode_layout(
    taichi::lang::SNode &root) {
  using namespace taichi::lang;
  auto *context = executor_->get_llvm_context();
  StructCompilerLLVM compiler(arch, executor_->get_config(), context,
                              context->new_module("component_layout"), -1);
  compiler.run_layout(root);
  return compiler.root_size;
}

std::pair<taichi::lang::DeviceAllocation, std::shared_ptr<void>>
capi::LlvmRuntime::allocate_dense_snode_storage(std::size_t size) {
  auto owner = std::make_shared<GroupAllocation>();
  owner->allocation =
      executor_->allocate_memory_on_device(size, result_buffer, true);
  owner->executor = executor_;
  return {owner->allocation, owner};
}
#endif

std::size_t Runtime::compile_dense_snode_layout(taichi::lang::SNode &) {
  throw std::invalid_argument(
      "Dense SNode groups are unsupported on this backend");
}

std::pair<taichi::lang::DeviceAllocation, std::shared_ptr<void>>
Runtime::allocate_dense_snode_storage(std::size_t) {
  throw std::invalid_argument(
      "Dense SNode storage is unsupported on this backend");
}

TiMemory Runtime::create_dense_snode_group(const std::string &schema_json,
                                           const std::string &dimensions) {
  using namespace taichi::lang;
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  auto schema = DenseSNodeSchema::from_json(schema_json);
  const auto bindings = liong::json::parse_complete(dimensions);
  if (!bindings.is_obj())
    throw std::invalid_argument(
        "Dense dimensions must be an object of integers");
  std::map<std::string, int> extents;
  for (const auto &[name, value] : bindings.fields()) {
    if (value.ty != liong::json::L_JSON_INT || value.num_int <= 0 ||
        value.num_int > std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "Dense dimension must be a positive i32 integer");
    extents.emplace(name, static_cast<int>(value.num_int));
  }
  auto group = std::make_shared<capi::DenseSNodeGroup>();
  group->root = schema.instantiate(extents);
  Layout layout;
  layout.root_size = compile_dense_snode_layout(*group->root);
  std::vector<SNode *> nodes;
  std::function<void(SNode &)> collect = [&](SNode &node) {
    nodes.push_back(&node);
    for (auto &child : node.ch)
      collect(*child);
  };
  collect(*group->root);
  if (!layout.root_size)
    throw std::invalid_argument("Dense group requires nonempty storage");
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = *nodes[i];
    LayoutNode entry;
    entry.parent = schema.nodes.at(i).parent;
    entry.type = static_cast<int>(node.type);
    entry.dtype = node.dt->to_string();
    for (int axis = 0; axis < node.num_active_indices; ++axis)
      entry.shape.push_back(node.shape_along_axis(axis));
    entry.cell_size = node.cell_size_bytes;
    entry.offset = node.offset_bytes_in_parent_cell;
    layout.nodes.push_back(std::move(entry));
  }
  group->layout = liong::json::print(liong::json::serialize(layout));
  group->storage = std::make_shared<Runtime::BorrowedMemory>();
  // The backend owns allocation and zeroing; all aliases share this lease.
  auto [allocation, owner] = allocate_dense_snode_storage(layout.root_size);
  group->storage->owner = std::move(owner);
  const auto handle = register_memory(allocation, layout.root_size, false, 0,
                                      {}, 0, false, group->storage);
  try {
    dense_groups_.emplace(handle, group);
  } catch (...) {
    Runtime::free_memory(handle);
    throw;
  }
  return handle;
}

const std::string &Runtime::dense_snode_group_layout(TiMemory handle) {
  get_memory(handle);
  const auto found = dense_groups_.find(handle);
  if (found == dense_groups_.end())
    throw MemoryBindingError(
        "Memory is not a live dense SNode group of this runtime");
  return found->second->layout;
}

TiNdArray Runtime::dense_snode_group_member(TiMemory handle,
                                            const std::string &text,
                                            TiDataType dtype,
                                            const TiNdShape &element_shape) {
  using namespace taichi::lang;
  dense_snode_group_layout(handle);  // Runtime/owner validation before use.
  const auto &group = dense_groups_.at(handle);
  if (element_shape.dim_count > 2)
    throw MemoryBindingError(
        "Dense member requires a scalar/vector/matrix element");
  TiNdArray array{};
  array.elem_type = dtype;
  array.elem_shape = element_shape;
  // Reuse the ordinary ndarray ABI's extent and checked byte-product rules.
  // The actual tree/placement validates the number of lanes, not a fixed cap.
  const auto element_layout = capi::ndarray_storage_layout(array);
  const uint64_t width = element_layout.alignment;
  const uint64_t lanes = element_layout.element_bytes / width;
  const auto paths_json = liong::json::parse_complete(text);
  if (!paths_json.is_arr())
    throw MemoryBindingError("Dense member paths must be an array");
  std::vector<std::vector<int>> paths;
  for (const auto &path_json : paths_json.elems()) {
    if (!path_json.is_arr())
      throw MemoryBindingError("Dense member path must be an integer array");
    std::vector<int> path;
    for (const auto &index : path_json.elems()) {
      if (index.ty != liong::json::L_JSON_INT || index.num_int < 0 ||
          index.num_int > std::numeric_limits<int>::max())
        throw MemoryBindingError(
            "Dense member path requires nonnegative native child indices");
      path.push_back(static_cast<int>(index.num_int));
    }
    paths.push_back(std::move(path));
  }
  if (lanes != paths.size())
    throw MemoryBindingError(
        "Dense member element shape differs from component paths");
  const auto scalar_type = capi::ndarray_scalar_type(dtype);
  SNode *anchor = nullptr;
  for (std::size_t lane = 0; lane < paths.size(); ++lane) {
    auto *node = group->root.get();
    for (int child : paths[lane]) {
      if (child < 0 || static_cast<std::size_t>(child) >= node->ch.size())
        throw MemoryBindingError(
            "Dense member path is outside the group schema");
      node = node->ch[child].get();
    }
    if (node->type != SNodeType::place || node->dt != scalar_type)
      throw MemoryBindingError(
          "Dense member path/type does not select a declared scalar place");
    if (!anchor)
      anchor = node;
    if (node->parent != anchor->parent ||
        node->offset_bytes_in_parent_cell <
            anchor->offset_bytes_in_parent_cell ||
        node->offset_bytes_in_parent_cell -
                anchor->offset_bytes_in_parent_cell !=
            lane * width)
      throw MemoryBindingError(
          "Dense member components are not contiguous in the same SNode");
  }
  const auto member_layout = dense_snode_member_layout(*anchor);
  const uint64_t offset = member_layout.offset;
  array.shape.dim_count = member_layout.shape.size();
  if (array.shape.dim_count > 16)
    throw MemoryBindingError("Dense member rank exceeds the argument ABI");
  for (uint32_t i = 0; i < array.shape.dim_count; ++i)
    array.shape.dims[i] = member_layout.shape[i];
  capi::ndarray_storage_layout(
      array);  // Validate logical shape/products independently.
  auto layout = std::make_shared<storage::DenseStorageLayoutSpec>();
  layout->scalar_type = scalar_type;
  layout->index_shape.assign(array.shape.dims,
                             array.shape.dims + array.shape.dim_count);
  layout->element_shape.assign(element_shape.dims,
                               element_shape.dims + element_shape.dim_count);
  auto strides = [](const std::vector<int64_t> &shape, uint64_t stride) {
    std::vector<int64_t> result(shape.size());
    for (std::size_t i = shape.size(); i-- > 0;) {
      if (stride > uint64_t(std::numeric_limits<int64_t>::max()) ||
          (shape[i] &&
           stride > uint64_t(std::numeric_limits<int64_t>::max()) / shape[i]))
        throw MemoryBindingError(
            "Dense member stride exceeds native address range");
      result[i] = stride;
      stride *= shape[i];
    }
    return result;
  };
  layout->index_strides_bytes = member_layout.strides;
  layout->element_strides_bytes = strides(layout->element_shape, width);
  const auto analysis = storage::analyze_dense_storage_layout(*layout);
  if (!analysis || !analysis.properties.element_contiguous ||
      !(analysis.properties.single_record_stride_compatible ||
        analysis.properties.compact_contiguous) ||
      analysis.properties.has_negative_stride ||
      analysis.properties.uniqueness != storage::StorageMappingUniqueness::kProvenUnique ||
      analysis.properties.reachable_begin != 0)
    throw MemoryBindingError("Dense member has an unqualified record mapping");
  const auto bytes = uint64_t(analysis.properties.reachable_end);
  const auto source = get_memory(handle);
  if (offset > source.size || bytes > source.size - offset)
    throw MemoryBindingError(
        "Dense member exceeds its compiled root allocation");
  array.memory = register_memory(
      source.allocation, bytes, false, offset, {}, offset, false,
      group->storage,
      analysis.properties.ndarray_abi_compatible ? nullptr : layout);
  return array;
}

void Runtime::free_dense_group(TiMemory handle) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  const auto found = dense_groups_.find(handle);
  if (found == dense_groups_.end())
    return;
  // This first group API closes at a cold boundary, including ordinary calls
  // that are not represented by a prepared user. Deferred instance retirement
  // must later carry an invocation lease instead of weakening this boundary.
  wait();
  // Existing retirement drains every alias's prepared/native users before
  // revoking the family and releasing its actual allocation owner.
  retire_borrowed_memory(found->second->storage);
  dense_groups_.erase(found);
}

void Runtime::retire_dense_groups() {
  while (!dense_groups_.empty())
    free_memory(dense_groups_.begin()->first);
}
void Runtime::abandon_dense_groups() {
  // Keep leases alive if the device could not prove retirement.
  if (!dense_groups_.empty())
    (void)new decltype(dense_groups_)(std::move(dense_groups_));
}

namespace {
Runtime &dense_runtime(TiRuntime handle) {
  return *reinterpret_cast<Runtime *>(handle);
}
}  // namespace

TiMemory ti_create_dense_snode_group_ext(TiRuntime runtime,
                                         const char *schema,
                                         const char *dimensions) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(schema);
  TI_CAPI_ARGUMENT_NULL_RV(dimensions);
  out = dense_runtime(runtime).create_dense_snode_group(schema, dimensions);
  TI_CAPI_TRY_CATCH_END();
  return out;
}

void ti_get_dense_snode_group_layout_ext(TiRuntime runtime,
                                         TiMemory group,
                                         uint64_t *size,
                                         char *json) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(group);
  TI_CAPI_ARGUMENT_NULL(size);
  auto &owner = dense_runtime(runtime);
  std::lock_guard<std::recursive_mutex> lock(owner.submission_mutex());
  const auto &text = owner.dense_snode_group_layout(group);
  const auto capacity = *size;
  *size = text.size() + 1;
  if (json) {
    if (capacity < *size)
      throw std::invalid_argument("Dense layout output buffer is too small");
    std::memcpy(json, text.c_str(), *size);
  }
  TI_CAPI_TRY_CATCH_END();
}

void ti_get_dense_snode_group_member_ext(TiRuntime runtime,
                                         TiMemory group,
                                         const char *paths,
                                         TiDataType dtype,
                                         const TiNdShape *element_shape,
                                         TiNdArray *array) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(group);
  TI_CAPI_ARGUMENT_NULL(paths);
  TI_CAPI_ARGUMENT_NULL(element_shape);
  TI_CAPI_ARGUMENT_NULL(array);
  auto &owner = dense_runtime(runtime);
  std::lock_guard<std::recursive_mutex> lock(owner.submission_mutex());
  *array = owner.dense_snode_group_member(group, paths, dtype, *element_shape);
  TI_CAPI_TRY_CATCH_END();
}
