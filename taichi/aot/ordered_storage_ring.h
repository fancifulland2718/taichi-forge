#pragma once

#include <set>
#include <string>

namespace liong::json {
struct JsonValue;
}
namespace taichi::lang::aot {
struct ExecutionPlan;
// Expand a compact storage template once, before resource analysis/preparation.
void expand_ordered_storage_ring(ExecutionPlan &plan,
                                 const liong::json::JsonValue &recipe,
                                 std::set<std::string> &arguments);
}  // namespace taichi::lang::aot
