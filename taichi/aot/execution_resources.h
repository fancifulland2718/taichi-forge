#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include "taichi/runtime/native_operation_description.h"

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
    // Cold allocation policy; runs once before explicit initialization/replay.
    bool zero_initialize{false};
    // SHA-256 key of packed initial bytes in the module's data directory.
    // Each materialization receives independent mutable device storage.
    std::string initial_data;
    // Physical buffer uses are fixed at allocation; views inherit them.
    AllocUsage usage{AllocUsage::Storage};
    // Optional cold native allocator. Empty keeps the runtime's ordinary
    // allocation policy. No process-local pool/handle is serialized.
    std::string allocator;
  };
  struct Scalar {
    // Exact C API scalar payload, independent of JSON numeric precision.
    std::uint64_t bits;
  };
  struct View {
    std::string array;
    // Packed records, not scalar components or bytes. Element layout is
    // inherited from the parent; a view never reinterprets its dtype.
    std::uint64_t offset{0};
    std::vector<Dimension> shape;
  };
  struct ModuleArray {
    std::string view;
  };
  struct HostBlob {
    // Immutable bytes in the existing content-addressed module data directory.
    std::string data;
    std::uint64_t byte_size;
  };
  struct Resource {
    std::string name;
    std::string dtype;
    std::variant<Array,
                 Extent,
                 NativeOperationDescription,
                 Scalar,
                 View,
                 ModuleArray,
                 HostBlob>
        value;
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
  std::set<std::string> runtime_bindings(std::set<std::string> roots) const;
};
}  // namespace taichi::lang::aot
