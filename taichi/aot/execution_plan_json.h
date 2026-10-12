#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include "taichi/common/json.h"

namespace taichi::lang::aot::execution_json {
using liong::json::JsonValue;
inline void fields(const JsonValue &object,
                   std::initializer_list<const char *> allowed) {
  if (!object.is_obj())
    throw std::invalid_argument("Execution plan entries must be objects");
  for (const auto &[name, value] : object.obj.inner)
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
      throw std::invalid_argument("Unknown execution plan field: " + name);
}
inline std::string name(const JsonValue &value) {
  if (!value.is_str() || value.str.empty() ||
      value.str.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "Execution plan names must be nonempty strings without NUL");
  return value.str;
}
inline std::uint64_t integer(const JsonValue &value) {
  if (value.ty != liong::json::L_JSON_INT || value.num_int < 0)
    throw std::invalid_argument(
        "Execution plan sizes and offsets must be nonnegative integers");
  return static_cast<std::uint64_t>(value.num_int);
}
inline std::optional<std::uint64_t> optional_integer(const JsonValue &object,
                                                     const char *key) {
  const auto found = object.obj.inner.find(key);
  return found == object.obj.inner.end()
             ? std::nullopt
             : std::optional(integer(found->second));
}
}  // namespace taichi::lang::aot::execution_json
