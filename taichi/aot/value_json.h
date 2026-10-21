#pragma once

#include <unordered_set>
#include "taichi/aot/value_layout.h"
#include "taichi/common/json_serde.h"

namespace taichi::lang::aot {
// Shared producer/consumer reflection. This describes the compiled layout;
// neither side invents padding or reconstructs logical tensor shapes from slots.
inline liong::json::JsonValue value_layout_tree(DataType type,
                                                const ValueLayout &layout,
                                                std::size_t &leaf) {
  using namespace liong::json;
  if (const auto *structure = type->cast<StructType>()) {
    JsonArray members;
    std::unordered_set<std::string> names;
    for (const auto &member : structure->elements()) {
      if (member.name.empty() || !names.insert(member.name).second)
        throw std::invalid_argument("Invalid value structure member name");
      members.inner.emplace_back(JsonObject{
          {"name", JsonValue(member.name)},
          {"value", value_layout_tree(member.type, layout, leaf)}});
    }
    return JsonValue(JsonObject{{"kind", JsonValue("struct")},
                                {"members", JsonValue(std::move(members))}});
  }
  const auto &member = layout.members.at(leaf++);
  JsonObject result{
      {"kind", JsonValue(member.shape.empty() ? "scalar" : "tensor")},
      {"dtype", JsonValue(member.scalar_type->to_string())},
      {"shape", serialize(member.shape)},
      {"offset", JsonValue(member.offset)},
      {"scalar_size", JsonValue(member.scalar_size)}};
  if (!member.component_offsets.empty())
    result.inner.emplace("component_offsets", serialize(member.component_offsets));
  return JsonValue(std::move(result));
}
}  // namespace taichi::lang::aot
