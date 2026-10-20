#include "taichi/aot/ordered_storage_ring.h"

#include <limits>
#include <map>
#include <algorithm>
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
    if constexpr (std::is_same_v<T, ExecutionPlan::ArrayLayout> ||
                  std::is_same_v<T, ExecutionPlan::StructArrayLayout>) {
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
  if (name(recipe["kind"]) != "ordered_temporary_ring")
    throw std::invalid_argument("Invalid ordered storage ring recipe");
  std::map<std::string, std::size_t> capacities;
  std::size_t capacity = 0;
  const auto add_group = [&](const auto &group) {
    const auto count = integer(group["capacity"]);
    if (!count || count > std::numeric_limits<std::size_t>::max() ||
        !group["arrays"].is_arr() || group["arrays"].arr.inner.empty())
      throw std::invalid_argument("Invalid ordered storage ring group");
    capacity = std::max(capacity, static_cast<std::size_t>(count));
    for (const auto &item : group["arrays"].arr.inner)
      if (!capacities.emplace(name(item), static_cast<std::size_t>(count)).second)
        throw std::invalid_argument("Duplicate storage ring array");
  };
  if (integer(recipe["version"]) == 1) {
    fields(recipe, {"kind", "version", "capacity", "arrays"});
    add_group(recipe);
  } else if (integer(recipe["version"]) == 2) {
    fields(recipe, {"kind", "version", "groups"});
    if (!recipe["groups"].is_arr() || recipe["groups"].arr.inner.empty())
      throw std::invalid_argument("Storage ring groups must be nonempty");
    for (const auto &group : recipe["groups"].arr.inner) {
      fields(group, {"capacity", "arrays"});
      add_group(group);
    }
  } else {
    throw std::invalid_argument("Unsupported storage ring recipe version");
  }
  if (plan.cuda_binding_frames || plan.vulkan_submission ||
      !plan.initialization.empty() || plan.operations.empty())
    throw std::invalid_argument(
        "Storage rings require ordinary replay without another execution recipe");

  std::set<std::string> selected, used;
  for (const auto &[array, count] : capacities)
    selected.insert(array);
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
  reserve_copies(plan.constraints, copies, constraints.size());
  std::size_t array_copies = 0;
  for (const auto &array : arrays) {
    const auto extra = capacities.at(array.name) - 1;
    if (extra > plan.resources.declarations.max_size() - array_copies)
      throw std::invalid_argument("Storage ring resources exceed address space");
    array_copies += extra;
  }
  reserve_copies(plan.resources.declarations, 1, array_copies);
  // Allocate names only for each group's own capacity. The whole execution
  // has max(capacities) frames, not the Cartesian product or least common
  // multiple. This is a declared shared frame schedule, not independent clocks.
  std::map<std::string, std::vector<std::string>> names;
  std::size_t counter = 0;
  for (const auto &array : arrays) {
    auto &slots = names[array.name];
    slots.reserve(capacities.at(array.name));
    slots.push_back(array.name);
    for (std::size_t slot = 1; slot < capacities.at(array.name); ++slot) {
      std::string generated;
      do {
        generated = "__forge_aot_ring_" + std::to_string(counter++);
      } while (!reserved.insert(generated).second);
      slots.push_back(generated);
      arguments.insert(generated);
      plan.replay_bindings.insert(generated);
    }
  }
  const auto rename = [&](std::string &value, std::size_t frame) {
    if (const auto found = names.find(value); found != names.end())
      value = found->second[frame % found->second.size()];
  };
  for (const auto &source : arrays) {
    for (std::size_t slot = 1; slot < capacities.at(source.name); ++slot) {
      auto array = source;
      rename(array.name, slot);
      for (auto &dimension : std::get<ExecutionResources::Array>(array.value).shape)
        if (auto *extent = std::get_if<ExecutionResources::Extent>(&dimension))
          rename(extent->array, slot);
      plan.resources.declarations.push_back(std::move(array));
    }
  }
  for (std::size_t frame = 1; frame < capacity; ++frame) {
    const auto bind = [&](std::string &value) { rename(value, frame); };
    for (auto operation : operations) {
      operation_slots(operation, bind);
      plan.operations.push_back(std::move(operation));
    }
    for (auto constraint : constraints) {
      constraint_slots(constraint, bind);
      plan.constraints.push_back(std::move(constraint));
    }
  }
  plan.storage_ring = OrderedStorageRing{static_cast<std::size_t>(capacity)};
}
}  // namespace taichi::lang::aot
