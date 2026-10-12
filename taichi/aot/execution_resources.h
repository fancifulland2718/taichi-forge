#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace liong::json {
struct JsonValue;
}

namespace taichi::lang::aot {
// Reconstructible cold bindings, independent of device handles and allocation.
// Shape-only dependencies are resolved without materializing device storage.
struct ExecutionResources {
  struct Extent {
    std::string array;
    std::uint32_t axis;
  };
  using Dimension = std::variant<std::uint32_t, Extent>;
  struct Array {
    std::vector<Dimension> shape;
    std::vector<std::uint32_t> element_shape;
  };
  struct Resource {
    std::string name;
    std::string dtype;
    std::variant<Array, Extent> value;
  };

  std::vector<Resource> declarations;
  std::map<std::string, std::string> gradients;
  // Topological order of the required closure, with storage requested only for
  // arrays actually bound to operations/gradients. Unused declarations remain
  // format-validated but require no consumer allocation.
  std::vector<std::size_t> order;
  std::vector<bool> materialize;

  static ExecutionResources parse(const liong::json::JsonValue &root);
  // Replace operation slots by the exact external binding set.
  void analyze(std::set<std::string> &arguments);
};
}  // namespace taichi::lang::aot
