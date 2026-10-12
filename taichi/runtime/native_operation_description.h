#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "taichi/common/json.h"
#include "taichi/ir/type.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

// Persistent provider identity and parameters, with symbolic storage slots.
// Neither addresses nor live provider objects belong in this description.
struct NativeOperationDescription {
  using Key = std::tuple<std::string, std::string, std::uint32_t>;
  std::string provider;
  std::string operation;
  std::uint32_t version{1};
  liong::json::JsonValue parameters;
  std::map<std::string, std::string> bindings;

  Key key() const {
    return {provider, operation, version};
  }
  static NativeOperationDescription parse(const liong::json::JsonValue &value);
};

// Validated owner-local storage metadata. JIT and AOT adapters supply the same
// cold view; providers retain prepared commands, not this symbolic binding map.
// The enclosing executor owns storage leases through completion.
struct NativeArrayBinding {
  DevicePtr pointer;
  std::uint64_t bytes;
  DataType scalar_type;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> element_shape;
  std::uint64_t base_offset{0};
};
using NativeOperationBindings = std::map<std::string, NativeArrayBinding>;

}  // namespace taichi::lang
