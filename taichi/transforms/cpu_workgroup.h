#pragma once

namespace taichi::lang {
class RangeForStmt;
struct CompileConfig;

namespace irpass {
// Returns false if the width or scratch extents still need model binding.
bool materialize_cpu_workgroup(RangeForStmt &range,
                               const CompileConfig &config,
                               bool unbound);
}  // namespace irpass
}  // namespace taichi::lang
