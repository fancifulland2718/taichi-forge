#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace taichi::lang::aot {
// Versioned persistent operations, independent of live definitions, runtime
// handles, backend commands and storage ownership. Binding consumes this only
// in the cold phase; replay retains no JSON or symbolic argument lookups.
struct ExecutionPlan {
  struct Kernel {
    std::string entry;
    std::vector<std::string> arguments;
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
    std::string values;
    std::string flags;
    std::string output;
    std::string count;
  };
  using Operation = std::variant<Kernel, Copy, Fill, Compact>;
  std::vector<Operation> operations;
  std::vector<std::string> arguments;

  static ExecutionPlan parse(const std::string &json);
};
}  // namespace taichi::lang::aot
