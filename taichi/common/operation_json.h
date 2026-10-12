#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include "taichi/common/json.h"

namespace taichi::lang::operation_json {
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
inline float optional_f32(const JsonValue &object,
                          const char *key,
                          float fallback) {
  const auto found = object.obj.inner.find(key);
  if (found == object.obj.inner.end())
    return fallback;
  const auto &value = found->second;
  if (value.ty != liong::json::L_JSON_FLOAT &&
      value.ty != liong::json::L_JSON_INT)
    throw std::invalid_argument("Execution scalars must be finite f32 numbers");
  const double number = value.ty == liong::json::L_JSON_FLOAT
                            ? value.num_float
                            : static_cast<double>(value.num_int);
  if (!std::isfinite(number) ||
      std::abs(number) > (std::numeric_limits<float>::max)())
    throw std::invalid_argument("Execution scalars must be finite f32 numbers");
  return static_cast<float>(number);
}
inline bool optional_bool(const JsonValue &object,
                          const char *key,
                          bool fallback = false) {
  const auto found = object.obj.inner.find(key);
  if (found == object.obj.inner.end())
    return fallback;
  if (found->second.ty != liong::json::L_JSON_BOOLEAN)
    throw std::invalid_argument("Execution flags must be booleans");
  return found->second.b;
}
}  // namespace taichi::lang::operation_json
