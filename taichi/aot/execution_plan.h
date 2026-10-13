#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include "taichi/aot/execution_resources.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang::aot {
// Versioned persistent operations, independent of live definitions, runtime
// handles, backend commands and storage ownership. Binding consumes this only
// in the cold phase; replay retains no JSON or symbolic argument lookups.
struct ExecutionPlan {
  struct Kernel {
    std::string entry;
    std::vector<std::string> arguments;
    std::string label;
  };
  struct Copy {
    std::string destination;
    std::string source;
    std::uint64_t destination_offset{0};
    std::uint64_t source_offset{0};
    std::optional<std::uint64_t> bytes;
  };
  struct Fill {
    std::string destination;
    std::uint32_t value{0};
    std::uint64_t offset{0};
    std::optional<std::uint64_t> bytes;
  };
  struct Compact {
    struct Column {
      std::string values;
      std::string output;
    };
    std::vector<Column> columns;
    std::string flags;
    std::string count;
  };
  struct Sort {
    std::string keys;
    std::optional<std::string> values;
  };
  using Native = NativeOperationDescription;
  using Operation = std::variant<Kernel, Copy, Fill, Compact, Sort, Native>;
  struct ArrayLayout {
    std::string array;
    NativeArrayLayoutRequirement required;
  };
  struct Disjoint {
    std::string left;
    std::string right;
    bool allow_same_view{false};
  };
  using Constraint = std::variant<ArrayLayout, Disjoint>;
  std::vector<Constraint> constraints;
  std::vector<Operation> operations;
  std::vector<Operation> initialization;
  // Runtime dependencies exclude shape-only references resolved at prepare.
  std::set<std::string> replay_bindings;
  std::vector<std::string> arguments;
  ExecutionResources resources;

  static ExecutionPlan parse(const std::string &json);
};
}  // namespace taichi::lang::aot
