#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "taichi/aot/execution_resources.h"

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
  struct Fft {
    std::string input;
    std::string output;
    std::vector<std::uint64_t> dimensions;
    std::uint64_t batch_count{1};
    bool inverse{false};
  };
  struct Gemm {
    std::string a;
    std::string b;
    std::string output;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t inner;
    float alpha{1};
    float beta{0};
    bool transpose_a{false};
    bool transpose_b{false};
  };
  using Operation = std::variant<Kernel, Copy, Fill, Compact, Sort, Fft, Gemm>;
  std::vector<Operation> operations;
  std::vector<std::string> arguments;
  ExecutionResources resources;

  static ExecutionPlan parse(const std::string &json);
};
}  // namespace taichi::lang::aot
