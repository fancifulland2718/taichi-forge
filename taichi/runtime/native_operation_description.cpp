#include "taichi/runtime/native_operation_description.h"
#include "taichi/common/operation_json.h"

namespace taichi::lang {
NativeOperationDescription NativeOperationDescription::parse(
    const liong::json::JsonValue &value) {
  using namespace operation_json;
  fields(value, {"kind", "provider", "operation", "version", "parameters",
                 "bindings"});
  if (name(value["kind"]) != "native")
    throw std::invalid_argument("Expected a native operation description");
  const auto version = integer(value["version"]);
  if (!version || version > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument(
        "Native operation version must be a positive uint32");
  NativeOperationDescription result;
  result.provider = name(value["provider"]);
  result.operation = name(value["operation"]);
  result.version = static_cast<std::uint32_t>(version);
  result.parameters = value["parameters"];
  if (!result.parameters.is_obj() || !value["bindings"].is_obj())
    throw std::invalid_argument(
        "Native parameters and bindings must be objects");
  for (const auto &[port, slot] : value["bindings"].obj.inner) {
    name(liong::json::JsonValue(port));
    result.bindings.emplace(port, name(slot));
  }
  return result;
}
}  // namespace taichi::lang
