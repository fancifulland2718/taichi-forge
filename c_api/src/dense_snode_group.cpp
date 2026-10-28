#include "taichi_llvm_impl.h"

#ifdef TI_WITH_LLVM
#include <cstring>
#include <limits>

#include "argument_binding.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/struct/dense_snode_schema.h"

namespace capi {
namespace {
using namespace taichi::lang;

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

liong::json::JsonValue parse_small(const std::string &text) {
  if (text.size() > 4 * 1024 * 1024)
    throw std::invalid_argument("Dense group binding JSON is too large");
  return liong::json::parse_complete(text);
}
}  // namespace

struct DenseSNodeGroup {
  std::unique_ptr<taichi::lang::SNode> root;
  std::shared_ptr<Runtime::BorrowedMemory> storage;
  std::string layout;
};

TiMemory LlvmRuntime::create_dense_snode_group(const std::string &schema_json,
                                               const std::string &dimensions) {
  using namespace taichi::lang;
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (!arch_is_cpu(arch) && arch != taichi::Arch::cuda)
    throw std::invalid_argument("Dense group requires a CPU/CUDA runtime");
  auto schema = DenseSNodeSchema::from_json(schema_json);
  const auto bindings = parse_small(dimensions);
  if (!bindings.is_obj())
    throw std::invalid_argument("Dense dimensions must be an object of integers");
  std::map<std::string, int> extents;
  for (const auto &[name, value] : bindings.fields()) {
    if (value.ty != liong::json::L_JSON_INT || value.num_int <= 0 ||
        value.num_int > std::numeric_limits<int>::max())
      throw std::invalid_argument("Dense dimension must be a positive i32 integer");
    extents.emplace(name, static_cast<int>(value.num_int));
  }
  auto group = std::make_shared<DenseSNodeGroup>();
  group->root = schema.instantiate(extents);
  auto *context = executor_->get_llvm_context();
  StructCompilerLLVM compiler(arch, executor_->get_config(), context,
                              context->new_module("component_layout"), -1);
  // Identical layout pass to normal JIT SNodes. These parameterized dense
  // resources need no fixed-root accessors or LLVM runtime tree directory.
  compiler.run_layout(*group->root);
  Layout layout;
  layout.root_size = compiler.root_size;
  if (!layout.root_size)
    throw std::invalid_argument("Dense group requires nonempty storage");
  for (std::size_t i = 0; i < compiler.snodes.size(); ++i) {
    const auto &node = *compiler.snodes[i];
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
  // Allocate with the same LLVM runtime allocator used by existing managed
  // resources. A single allocation retains the complete compiled dense root.
  auto owner = std::make_shared<GroupAllocation>();
  owner->allocation = executor_->allocate_memory_on_device(layout.root_size, result_buffer, true);
  owner->executor = executor_;
  group->storage->owner = owner;
  const auto handle = register_memory(owner->allocation, layout.root_size, false,
                                       0, {}, 0, false, group->storage);
  try {
    dense_groups_.emplace(handle, group);
  } catch (...) {
    Runtime::free_memory(handle);
    throw;
  }
  return handle;
}

const std::string &LlvmRuntime::dense_snode_group_layout(TiMemory handle) {
  get_memory(handle);
  const auto found = dense_groups_.find(handle);
  if (found == dense_groups_.end())
    throw MemoryBindingError("Memory is not a live dense SNode group of this runtime");
  return found->second->layout;
}

TiNdArray LlvmRuntime::dense_snode_group_member(TiMemory handle,
                                               const std::string &text,
                                               TiDataType dtype,
                                               const TiNdShape &element_shape) {
  using namespace taichi::lang;
  dense_snode_group_layout(handle);  // Runtime/owner validation before use.
  const auto &group = dense_groups_.at(handle);
  const auto paths_json = parse_small(text);
  if (!paths_json.is_arr())
    throw MemoryBindingError("Dense member paths must be an array");
  std::vector<std::vector<int>> paths;
  for (const auto &path_json : paths_json.elems()) {
    if (!path_json.is_arr())
      throw MemoryBindingError("Dense member path must be an integer array");
    std::vector<int> path;
    for (const auto &index : path_json.elems()) {
      if (index.ty != liong::json::L_JSON_INT || index.num_int < 0 || index.num_int > 4096)
        throw MemoryBindingError("Dense member path requires bounded child indices");
      path.push_back(static_cast<int>(index.num_int));
    }
    paths.push_back(std::move(path));
  }
  if (paths.empty() || paths.size() > 4096 || element_shape.dim_count > 2)
    throw MemoryBindingError("Dense member requires scalar/vector/matrix place paths");
  uint64_t lanes = 1;
  for (uint32_t i = 0; i < element_shape.dim_count; ++i) {
    const auto extent = element_shape.dims[i];
    if (!extent || extent > 4096 / lanes)
      throw MemoryBindingError("Dense member element shape exceeds qualified range");
    lanes *= extent;
  }
  if (lanes != paths.size())
    throw MemoryBindingError("Dense member element shape differs from component paths");
  const auto scalar_type = ndarray_scalar_type(dtype);
  const uint64_t width = data_type_size(scalar_type);
  SNode *anchor = nullptr;
  for (std::size_t lane = 0; lane < paths.size(); ++lane) {
    auto *node = group->root.get();
    for (int child : paths[lane]) {
      if (child < 0 || static_cast<std::size_t>(child) >= node->ch.size())
        throw MemoryBindingError("Dense member path is outside the group schema");
      node = node->ch[child].get();
    }
    if (node->type != SNodeType::place || node->dt != scalar_type)
      throw MemoryBindingError("Dense member path/type does not select a declared scalar place");
    if (!anchor)
      anchor = node;
    if (node->parent != anchor->parent ||
        node->offset_bytes_in_parent_cell != anchor->offset_bytes_in_parent_cell + lane * width)
      throw MemoryBindingError("Dense member components are not contiguous in the same SNode");
  }
  auto *parent = anchor->parent;
  TiNdArray array{};
  array.elem_type = dtype;
  array.elem_shape = element_shape;
  uint64_t offset = anchor->offset_bytes_in_parent_cell;
  if (parent->type == SNodeType::dense) {
    if (parent->parent != group->root.get() || parent->cell_size_bytes != lanes * width)
      throw MemoryBindingError("Dense member is not a qualified compact record");
    offset += parent->offset_bytes_in_parent_cell;
    array.shape.dim_count = anchor->num_active_indices;
    if (array.shape.dim_count > 16)
      throw MemoryBindingError("Dense member rank exceeds the argument ABI");
    for (uint32_t i = 0; i < array.shape.dim_count; ++i)
      array.shape.dims[i] = anchor->shape_along_axis(i);
  } else if (parent != group->root.get()) {
    throw MemoryBindingError("Dense member must belong to a simple dense or rank-zero root");
  }
  const auto bytes = ndarray_storage_layout(array).bytes;
  const auto source = get_memory(handle);
  if (offset > source.size || bytes > source.size - offset)
    throw MemoryBindingError("Dense member exceeds its compiled root allocation");
  array.memory = register_memory(source.allocation, bytes, false, offset, {},
                                  offset, false, group->storage);
  return array;
}

void LlvmRuntime::free_dense_group(TiMemory handle) {
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

void LlvmRuntime::retire_dense_groups() {
  while (!dense_groups_.empty())
    free_memory(dense_groups_.begin()->first);
}
}  // namespace capi

namespace {
capi::LlvmRuntime &dense_runtime(TiRuntime handle) {
  auto &runtime = *reinterpret_cast<Runtime *>(handle);
  if (!taichi::arch_is_cpu(runtime.arch) && runtime.arch != taichi::Arch::cuda)
    throw std::invalid_argument("Dense SNode groups require CPU/CUDA");
  return static_cast<capi::LlvmRuntime &>(runtime);
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

void ti_get_dense_snode_group_layout_ext(TiRuntime runtime, TiMemory group,
                                         uint64_t *size, char *json) {
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

void ti_get_dense_snode_group_member_ext(TiRuntime runtime, TiMemory group,
                                         const char *paths, TiDataType dtype,
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
#endif  // TI_WITH_LLVM
