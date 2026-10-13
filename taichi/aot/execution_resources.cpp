#include "taichi/aot/execution_resources.h"
#include "taichi/aot/execution_plan_json.h"

#include <limits>
#include <unordered_map>

namespace taichi::lang::aot {
namespace {
using namespace execution_json;

std::uint32_t word(const JsonValue &value) {
  const auto result = integer(value);
  if (result > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("Execution resource dimension exceeds uint32");
  return static_cast<std::uint32_t>(result);
}
ExecutionResources::Extent extent(const JsonValue &value) {
  return {name(value["array"]), word(value["axis"])};
}
std::vector<std::string> dependencies(
    const ExecutionResources::Resource &resource) {
  std::vector<std::string> result;
  if (const auto *array =
          std::get_if<ExecutionResources::Array>(&resource.value)) {
    for (const auto &dimension : array->shape)
      if (const auto *ref = std::get_if<ExecutionResources::Extent>(&dimension))
        result.push_back(ref->array);
  } else if (const auto *native =
                 std::get_if<NativeOperationDescription>(&resource.value)) {
    for (const auto &[port, slot] : native->bindings)
      result.push_back(slot);
  } else {
    result.push_back(
        std::get<ExecutionResources::Extent>(resource.value).array);
  }
  return result;
}
}  // namespace

ExecutionResources ExecutionResources::parse(const JsonValue &root) {
  ExecutionResources result;
  if (const auto found = root.obj.inner.find("resources");
      found != root.obj.inner.end()) {
    if (!found->second.is_arr())
      throw std::invalid_argument("Execution resources must be an array");
    for (const auto &value : found->second.arr.inner) {
      const auto kind = name(value["kind"]);
      Resource resource{name(value["name"]), {}, Array{}};
      if (kind == "native") {
        auto description = value;
        description.obj.inner.erase("name");
        resource.value = NativeOperationDescription::parse(description);
        result.declarations.push_back(std::move(resource));
        continue;
      }
      resource.dtype = name(value["dtype"]);
      if (kind == "ndarray") {
        fields(value, {"kind", "name", "dtype", "shape", "element_shape",
                       "zero_initialize"});
        if (!value["shape"].is_arr())
          throw std::invalid_argument(
              "Execution resource shape must be an array");
        auto &array = std::get<Array>(resource.value);
        array.zero_initialize = optional_bool(value, "zero_initialize");
        for (const auto &dimension : value["shape"].arr.inner) {
          if (dimension.is_obj()) {
            fields(dimension, {"array", "axis"});
            array.shape.emplace_back(extent(dimension));
          } else {
            array.shape.emplace_back(word(dimension));
          }
        }
        if (auto shape = value.obj.inner.find("element_shape");
            shape != value.obj.inner.end()) {
          if (!shape->second.is_arr())
            throw std::invalid_argument(
                "Execution element shape must be an array");
          for (const auto &dimension : shape->second.arr.inner) {
            const auto size = word(dimension);
            if (!size)
              throw std::invalid_argument(
                  "Execution element dimensions must be positive");
            array.element_shape.push_back(size);
          }
        }
      } else if (kind == "extent") {
        fields(value, {"kind", "name", "dtype", "array", "axis"});
        if (resource.dtype != "i32" && resource.dtype != "u32" &&
            resource.dtype != "i64" && resource.dtype != "u64")
          throw std::invalid_argument(
              "Execution extent requires a 32/64-bit integer dtype");
        resource.value = extent(value);
      } else {
        throw std::invalid_argument("Unsupported execution resource kind: " +
                                    kind);
      }
      result.declarations.push_back(std::move(resource));
    }
  }
  if (const auto found = root.obj.inner.find("gradients");
      found != root.obj.inner.end()) {
    if (!found->second.is_obj())
      throw std::invalid_argument(
          "Execution gradients must map primal names to gradient names");
    for (const auto &[primal, gradient] : found->second.obj.inner)
      result.gradients.emplace(name(JsonValue(primal)), name(gradient));
  }
  return result;
}

void ExecutionResources::analyze(std::set<std::string> &arguments) {
  if (declarations.empty() && gradients.empty()) {
    order.clear();
    materialize.clear();
    return;
  }
  const auto count = declarations.size();
  std::unordered_map<std::string, std::size_t> indices;
  for (std::size_t i = 0; i < count; ++i)
    if (!indices.emplace(declarations[i].name, i).second)
      throw std::invalid_argument("Duplicate execution resource: " +
                                  declarations[i].name);
  std::vector<std::vector<std::string>> references(count);
  std::vector<std::vector<std::size_t>> dependents(count);
  std::vector<std::size_t> indegree(count), ready;
  for (std::size_t i = 0; i < count; ++i) {
    references[i] = dependencies(declarations[i]);
    for (const auto &ref : references[i]) {
      if (const auto parent = indices.find(ref); parent != indices.end()) {
        if (!std::holds_alternative<NativeOperationDescription>(
                declarations[i].value) &&
            !std::holds_alternative<Array>(declarations[parent->second].value))
          throw std::invalid_argument(
              "Execution shape reference requires an ndarray: " + ref);
        ++indegree[i];
        dependents[parent->second].push_back(i);
      }
    }
    if (!indegree[i])
      ready.push_back(i);
  }
  for (std::size_t head = 0; head < ready.size(); ++head)
    for (const auto child : dependents[ready[head]])
      if (!--indegree[child])
        ready.push_back(child);
  if (ready.size() != count)
    throw std::invalid_argument("Execution resource dependency cycle");

  for (const auto &[primal, gradient] : gradients) {
    for (const auto *ref : {&primal, &gradient})
      if (const auto found = indices.find(*ref);
          found != indices.end() &&
          !std::holds_alternative<Array>(declarations[found->second].value))
        throw std::invalid_argument(
            "Execution gradient reference requires an ndarray: " + *ref);
    arguments.insert(primal);
    arguments.insert(gradient);
  }
  materialize.assign(count, false);
  for (std::size_t i = 0; i < count; ++i)
    materialize[i] = arguments.count(declarations[i].name) != 0;
  std::vector<bool> needed(count, false);
  std::vector<std::string> pending(arguments.begin(), arguments.end());
  arguments.clear();
  for (std::size_t head = 0; head < pending.size(); ++head) {
    const auto found = indices.find(pending[head]);
    if (found == indices.end()) {
      arguments.insert(pending[head]);
    } else if (!needed[found->second]) {
      needed[found->second] = true;
      const auto &refs = references[found->second];
      if (std::holds_alternative<NativeOperationDescription>(
              declarations[found->second].value))
        for (const auto &ref : refs)
          if (const auto parent = indices.find(ref); parent != indices.end())
            materialize[parent->second] = true;
      pending.insert(pending.end(), refs.begin(), refs.end());
    }
  }
  order.clear();
  for (const auto i : ready)
    if (needed[i])
      order.push_back(i);
}
std::set<std::string> ExecutionResources::runtime_bindings(
    std::set<std::string> roots) const {
  std::map<std::string, const Resource *> by_name;
  for (const auto &resource : declarations)
    by_name.emplace(resource.name, &resource);
  std::vector<std::string> pending(roots.begin(), roots.end());
  auto add = [&](const std::string &name) {
    if (roots.insert(name).second)
      pending.push_back(name);
  };
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const auto name = pending[i];
    if (const auto gradient = gradients.find(name); gradient != gradients.end())
      add(gradient->second);
    if (const auto resource = by_name.find(name); resource != by_name.end())
      if (const auto *native =
              std::get_if<NativeOperationDescription>(&resource->second->value))
        for (const auto &[port, slot] : native->bindings)
          add(slot);
  }
  return roots;
}
}  // namespace taichi::lang::aot
