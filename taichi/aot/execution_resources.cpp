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
std::vector<ExecutionResources::Dimension> shape(const JsonValue &value) {
  if (!value.is_arr())
    throw std::invalid_argument("Execution resource shape must be an array");
  std::vector<ExecutionResources::Dimension> result;
  for (const auto &dimension : value.arr.inner) {
    if (dimension.is_obj()) {
      fields(dimension, {"array", "axis"});
      result.emplace_back(extent(dimension));
    } else {
      result.emplace_back(word(dimension));
    }
  }
  return result;
}
bool is_array(const ExecutionResources::Resource &resource) {
  return std::holds_alternative<ExecutionResources::Array>(resource.value) ||
         std::holds_alternative<ExecutionResources::View>(resource.value) ||
         std::holds_alternative<ExecutionResources::Member>(resource.value) ||
         std::holds_alternative<ExecutionResources::ModuleArray>(
             resource.value);
}
ExecutionResources::Scalar scalar(const std::string &dtype,
                                  const JsonValue &value) {
  std::size_t digits = 0;
  if (dtype == "i8" || dtype == "u8")
    digits = 2;
  else if (dtype == "i16" || dtype == "u16")
    digits = 4;
  else if (dtype == "i32" || dtype == "u32" || dtype == "f32" || dtype == "f16")
    // F16 scalar arguments use a float32 payload in the public C API.
    digits = 8;
  else if (dtype == "i64" || dtype == "u64" || dtype == "f64")
    digits = 16;
  return {operation_json::scalar_bits(value, digits)};
}
}  // namespace

std::vector<std::string> ExecutionResources::dependencies(
    const ExecutionResources::Resource &resource,
    bool include_allocator) {
  std::vector<std::string> result;
  const std::vector<ExecutionResources::Dimension> *dimensions = nullptr;
  if (const auto *array =
          std::get_if<ExecutionResources::Array>(&resource.value)) {
    dimensions = &array->shape;
    if (include_allocator && !array->allocator.empty())
      result.push_back(array->allocator);
  } else if (const auto *view =
                 std::get_if<ExecutionResources::View>(&resource.value)) {
    result.push_back(view->array);
    dimensions = &view->shape;
  } else if (const auto *member = std::get_if<Member>(&resource.value)) {
    result.push_back(member->array);
  } else if (const auto *native =
                 std::get_if<NativeOperationDescription>(&resource.value)) {
    for (const auto &[port, slot] : native->bindings)
      result.push_back(slot);
  } else if (const auto *ref =
                 std::get_if<ExecutionResources::Extent>(&resource.value)) {
    result.push_back(ref->array);
  }
  if (dimensions)
    for (const auto &dimension : *dimensions)
      if (const auto *ref = std::get_if<ExecutionResources::Extent>(&dimension))
        result.push_back(ref->array);
  return result;
}
ExecutionResources ExecutionResources::parse(const JsonValue &root) {
  ExecutionResources result;
  if (const auto found = root.obj.inner.find("resources");
      found != root.obj.inner.end()) {
    if (!found->second.is_arr())
      throw std::invalid_argument("Execution resources must be an array");
    for (const auto &value : found->second.arr.inner) {
      const auto kind = name(value["kind"]);
      Resource resource{name(value["name"]), {}, Array{}};
      if (kind == "host_blob") {
        fields(value, {"kind", "name", "data", "byte_size"});
        const auto key = name(value["data"]);
        if (key.size() != 64 ||
            key.find_first_not_of("0123456789abcdef") != std::string::npos)
          throw std::invalid_argument(
              "Execution host blob requires a lowercase SHA-256 key");
        resource.value = HostBlob{key, integer(value["byte_size"])};
        result.declarations.push_back(std::move(resource));
        continue;
      }
      if (kind == "native") {
        auto description = value;
        description.obj.inner.erase("name");
        resource.value = NativeOperationDescription::parse(description);
        result.declarations.push_back(std::move(resource));
        continue;
      }
      if (kind == "ndarray_view") {
        fields(value, {"kind", "name", "array", "offset", "shape"});
        resource.value = View{name(value["array"]), integer(value["offset"]),
                              shape(value["shape"])};
        result.declarations.push_back(std::move(resource));
        continue;
      }
      if (kind == "ndarray_member") {
        fields(value, {"kind", "name", "array", "path"});
        if (integer(root["version"]) < 23)
          throw std::invalid_argument("Array members require execution plan version 23");
        resource.value = Member{name(value["array"]), name(value["path"])};
        result.declarations.push_back(std::move(resource));
        continue;
      }
      if (kind == "module_array") {
        fields(value, {"kind", "name", "view"});
        resource.value = ModuleArray{name(value["view"])};
        result.declarations.push_back(std::move(resource));
        continue;
      }
      const bool structured = kind == "ndarray" && value.obj.inner.count("layout");
      if (!structured)
        resource.dtype = name(value["dtype"]);
      if (kind == "ndarray") {
        fields(value, {"kind", "name", "dtype", "shape", "element_shape",
                       "zero_initialize", "initial_data", "usage",
                       "allocator", "layout"});
        auto &array = std::get<Array>(resource.value);
        if (structured) {
          if (integer(root["version"]) < 23 || value.obj.inner.count("dtype") ||
              value.obj.inner.count("element_shape"))
            throw std::invalid_argument(
                "Structure resources require version 23, layout and no primitive dtype/element_shape");
          array.layout = name(value["layout"]);
        }
        if (const auto allocator = value.obj.inner.find("allocator");
            allocator != value.obj.inner.end()) {
          if (integer(root["version"]) < 19)
            throw std::invalid_argument(
                "Array allocators require execution plan version 19");
          array.allocator = name(allocator->second);
        }
        if (const auto usage = value.obj.inner.find("usage");
            usage != value.obj.inner.end()) {
          if (integer(root["version"]) < 18)
            throw std::invalid_argument(
                "Array usages require execution plan version 18");
          if (!usage->second.is_arr() || usage->second.arr.inner.empty())
            throw std::invalid_argument(
                "Array usage must be a nonempty name array");
          array.usage = AllocUsage::None;
          for (const auto &item : usage->second.arr.inner) {
            const auto label = name(item);
            const auto flag = label == "storage"    ? AllocUsage::Storage
                              : label == "uniform"  ? AllocUsage::Uniform
                              : label == "vertex"   ? AllocUsage::Vertex
                              : label == "index"    ? AllocUsage::Index
                              : label == "indirect" ? AllocUsage::Indirect
                                                    : AllocUsage::None;
            if (flag == AllocUsage::None)
              throw std::invalid_argument("Unknown array usage: " + label);
            array.usage = array.usage | flag;
          }
        }
        array.shape = shape(value["shape"]);
        array.zero_initialize = optional_bool(value, "zero_initialize");
        if (const auto data = value.obj.inner.find("initial_data");
            data != value.obj.inner.end()) {
          array.initial_data = name(data->second);
          if (array.initial_data.size() != 64 ||
              array.initial_data.find_first_not_of("0123456789abcdef") !=
                  std::string::npos ||
              array.zero_initialize)
            throw std::invalid_argument(
                "Execution initial_data requires a lowercase SHA-256 key and "
                "no zero initialization");
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
      } else if (kind == "scalar") {
        fields(value, {"kind", "name", "dtype", "bits"});
        resource.value = scalar(resource.dtype, value["bits"]);
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
    references[i] = dependencies(declarations[i], false);
    for (const auto &ref : references[i])
      if (const auto parent = indices.find(ref); parent != indices.end())
        if (!std::holds_alternative<NativeOperationDescription>(
                declarations[i].value) &&
            !is_array(declarations[parent->second]))
          throw std::invalid_argument(
              "Execution shape reference requires an ndarray: " + ref);
    if (const auto *array = std::get_if<Array>(&declarations[i].value);
        array && !array->allocator.empty()) {
      if (const auto parent = indices.find(array->allocator);
          parent != indices.end()) {
        const auto *native = std::get_if<NativeOperationDescription>(
            &declarations[parent->second].value);
        if (!native || !native->bindings.empty())
          throw std::invalid_argument(
              "Array allocator requires an input-free native resource: " +
              array->allocator);
      }
      references[i].push_back(array->allocator);
    }
    for (const auto &ref : references[i]) {
      if (const auto parent = indices.find(ref); parent != indices.end()) {
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
      if (const auto found = indices.find(*ref); found != indices.end()) {
        const auto &resource = declarations[found->second];
        const auto *array = std::get_if<Array>(&resource.value);
        if (!is_array(resource) || std::holds_alternative<Member>(resource.value) ||
            (array && !array->layout.empty()))
          throw std::invalid_argument(
              "Execution gradient reference requires a primitive ndarray: " + *ref);
      }
    arguments.insert(primal);
    arguments.insert(gradient);
  }
  materialize.assign(count, false);
  for (std::size_t i = 0; i < count; ++i)
    materialize[i] = arguments.count(declarations[i].name) != 0;
  // Propagate storage demand separately from shape-only reachability. Walk in
  // reverse topological order so nested views keep every storage ancestor.
  for (auto it = ready.rbegin(); it != ready.rend(); ++it) {
    if (const auto *array = std::get_if<Array>(&declarations[*it].value))
      if (materialize[*it] && !array->allocator.empty())
        if (const auto parent = indices.find(array->allocator);
            parent != indices.end())
          materialize[parent->second] = true;
    if (const auto *native =
            std::get_if<NativeOperationDescription>(&declarations[*it].value)) {
      if (materialize[*it])
        for (const auto &[port, slot] : native->bindings)
          if (const auto parent = indices.find(slot); parent != indices.end())
            materialize[parent->second] = true;
    } else if (const auto *view = std::get_if<View>(&declarations[*it].value)) {
      if (materialize[*it])
        if (const auto parent = indices.find(view->array);
            parent != indices.end())
          materialize[parent->second] = true;
    } else if (const auto *member = std::get_if<Member>(&declarations[*it].value)) {
      if (materialize[*it])
        if (const auto parent = indices.find(member->array); parent != indices.end())
          materialize[parent->second] = true;
    }
  }
  std::vector<bool> needed(count, false);
  std::vector<std::string> pending(arguments.begin(), arguments.end());
  arguments.clear();
  for (std::size_t head = 0; head < pending.size(); ++head) {
    const auto found = indices.find(pending[head]);
    if (found == indices.end()) {
      arguments.insert(pending[head]);
    } else if (!needed[found->second]) {
      needed[found->second] = true;
      const auto refs = dependencies(declarations[found->second],
                                     materialize[found->second]);
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
    if (const auto resource = by_name.find(name); resource != by_name.end()) {
      if (const auto *array = std::get_if<Array>(&resource->second->value))
        if (!array->allocator.empty())
          add(array->allocator);
      if (const auto *native =
              std::get_if<NativeOperationDescription>(&resource->second->value))
        for (const auto &[port, slot] : native->bindings)
          add(slot);
      if (const auto *view = std::get_if<View>(&resource->second->value))
        add(view->array);
      if (const auto *member = std::get_if<Member>(&resource->second->value))
        add(member->array);
    }
  }
  return roots;
}
}  // namespace taichi::lang::aot
