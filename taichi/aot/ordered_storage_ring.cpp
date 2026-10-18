#include "taichi/aot/ordered_storage_ring.h"

#include <limits>
#include <map>
#include <type_traits>
#include "taichi/aot/execution_plan.h"
#include "taichi/common/operation_json.h"

namespace taichi::lang::aot {
namespace {
template <typename Visit>
void operation_slots(ExecutionPlan::Operation &operation, Visit visit) {
  if (auto *kernel = std::get_if<ExecutionPlan::Kernel>(&operation)) {
    if (kernel->indirect_packet || kernel->bounded_range)
      throw std::invalid_argument("Storage rings require fixed dispatches");
    for (auto &slot : kernel->arguments)
      visit(slot);
  } else if (auto *copy = std::get_if<ExecutionPlan::Copy>(&operation)) {
    visit(copy->source);
    visit(copy->destination);
  } else if (auto *fill = std::get_if<ExecutionPlan::Fill>(&operation)) {
    visit(fill->destination);
  } else {
    throw std::invalid_argument("Storage rings require kernel/copy/fill operations");
  }
}

template <typename Visit>
void constraint_slots(ExecutionPlan::Constraint &constraint, Visit visit) {
  std::visit([&](auto &value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, ExecutionPlan::ArrayLayout>) {
      visit(value.array);
    } else if constexpr (std::is_same_v<T, ExecutionPlan::DisjointWrites>) {
      for (auto &slot : value.reads)
        visit(slot);
      for (auto &write : value.writes)
        visit(write.array);
    } else {
      visit(value.left);
      visit(value.right);
    }
  }, constraint);
}

template <typename T>
void reserve_copies(std::vector<T> &values,
                    std::size_t copies,
                    std::size_t width) {
  if (width && copies > (values.max_size() - values.size()) / width)
    throw std::invalid_argument("Storage ring expansion exceeds address space");
  values.reserve(values.size() + copies * width);
}
}  // namespace

void expand_ordered_storage_ring(ExecutionPlan &plan,
                                 const liong::json::JsonValue &recipe,
                                 std::set<std::string> &arguments) {
  using namespace operation_json;
  fields(recipe, {"kind", "version", "capacity", "arrays"});
  const auto capacity = integer(recipe["capacity"]);
  if (name(recipe["kind"]) != "ordered_temporary_ring" ||
      integer(recipe["version"]) != 1 || !capacity ||
      capacity > std::numeric_limits<std::size_t>::max() ||
      !recipe["arrays"].is_arr() || recipe["arrays"].arr.inner.empty())
    throw std::invalid_argument("Invalid ordered storage ring recipe");
  if (plan.cuda_binding_frames || plan.vulkan_submission ||
      !plan.initialization.empty() || plan.operations.empty())
    throw std::invalid_argument(
        "Storage rings require ordinary replay without another execution recipe");

  std::set<std::string> selected, used;
  for (const auto &item : recipe["arrays"].arr.inner)
    if (!selected.insert(name(item)).second)
      throw std::invalid_argument("Duplicate storage ring array");
  for (auto &operation : plan.operations)
    operation_slots(operation, [&](const auto &slot) { used.insert(slot); });
  std::vector<ExecutionResources::Resource> arrays;
  std::set<std::string> declared;
  auto reserved = arguments;
  for (const auto &resource : plan.resources.declarations) {
    if (!declared.insert(resource.name).second)
      throw std::invalid_argument("Duplicate execution resource: " + resource.name);
    reserved.insert(resource.name);
    if (selected.count(resource.name)) {
      if (!std::holds_alternative<ExecutionResources::Array>(resource.value) ||
          !used.count(resource.name))
        throw std::invalid_argument("Storage ring arrays must be used private ndarrays");
      arrays.push_back(resource);
    } else {
      for (const auto &ref : ExecutionResources::dependencies(resource))
        if (selected.count(ref))
          throw std::invalid_argument(
              "Shared resources cannot depend on storage ring arrays");
    }
  }
  if (arrays.size() != selected.size())
    throw std::invalid_argument("Storage ring arrays must be declared privately");
  for (const auto &[primal, gradient] : plan.resources.gradients)
    if (selected.count(primal) || selected.count(gradient))
      throw std::invalid_argument("Storage ring arrays cannot own gradient bindings");

  // Keep shared constraints once, and duplicate only constraints involving a
  // selected array. Kernel code/entry identities are never duplicated here.
  std::vector<ExecutionPlan::Constraint> constraints;
  for (auto &constraint : plan.constraints) {
    bool affected = false;
    constraint_slots(constraint, [&](const auto &slot) {
      affected |= selected.count(slot) != 0;
    });
    if (affected)
      constraints.push_back(constraint);
  }
  const auto operations = plan.operations;
  const auto copies = static_cast<std::size_t>(capacity - 1);
  reserve_copies(plan.operations, copies, operations.size());
  reserve_copies(plan.resources.declarations, copies, arrays.size());
  reserve_copies(plan.constraints, copies, constraints.size());
  std::size_t counter = 0;
  for (std::size_t slot = 0; slot < copies; ++slot) {
    std::map<std::string, std::string> names;
    for (const auto &array : arrays) {
      std::string generated;
      do {
        generated = "__forge_aot_ring_" + std::to_string(counter++);
      } while (!reserved.insert(generated).second);
      names.emplace(array.name, generated);
      arguments.insert(generated);
      plan.replay_bindings.insert(generated);
    }
    const auto rename = [&](std::string &value) {
      if (const auto found = names.find(value); found != names.end())
        value = found->second;
    };
    for (auto array : arrays) {
      rename(array.name);
      for (auto &dimension : std::get<ExecutionResources::Array>(array.value).shape)
        if (auto *extent = std::get_if<ExecutionResources::Extent>(&dimension))
          rename(extent->array);
      plan.resources.declarations.push_back(std::move(array));
    }
    for (auto operation : operations) {
      operation_slots(operation, rename);
      plan.operations.push_back(std::move(operation));
    }
    for (auto constraint : constraints) {
      constraint_slots(constraint, rename);
      plan.constraints.push_back(std::move(constraint));
    }
  }
  plan.storage_ring = OrderedStorageRing{static_cast<std::size_t>(capacity)};
}
}  // namespace taichi::lang::aot
