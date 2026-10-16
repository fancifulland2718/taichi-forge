#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include "taichi/aot/execution_resources.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_description.h"
#include "taichi/runtime/vulkan_control_recipe.h"

namespace taichi::lang::aot {
// Versioned persistent operations, independent of live definitions, runtime
// handles, backend commands and storage ownership. Binding consumes this only
// in the cold phase; replay retains no JSON or symbolic argument lookups.
struct ExecutionPlan {
  struct Kernel {
    struct BoundedRange {
      bool cuda{false};
      std::string extent;
      std::uint32_t capacity{0};
      // Nonzero only for a selected CUDA adaptive physical grid recipe.
      std::uint32_t block_dim{0};
      bool grouped_update{false};
    };
    std::string entry;
    std::vector<std::string> arguments;
    std::string label;
    std::optional<std::string> indirect_packet;
    std::optional<BoundedRange> bounded_range;
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
  struct While;
  using Operation = std::variant<Kernel,
                                 Copy,
                                 Fill,
                                 Compact,
                                 Sort,
                                 Native,
                                 std::shared_ptr<const While>>;
  struct While {
    struct Vulkan {
      std::string counter;
      std::optional<std::string> status;
      VulkanControlRecipe recipe;
      std::optional<VulkanNestedControlRecipe<std::string>> nested;
    };
    std::string predicate;
    std::uint64_t max_iterations{0};
    bool continue_while_nonzero{true};
    std::vector<Operation> condition;
    std::vector<Operation> body;
    std::optional<Vulkan> vulkan;
  };
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
