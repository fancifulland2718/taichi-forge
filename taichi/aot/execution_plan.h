#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>
#include "taichi/aot/execution_resources.h"
#include "taichi/runtime/branch_selection.h"
#include "taichi/runtime/cuda_control_recipe.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_description.h"
#include "taichi/runtime/ordered_storage_ring.h"
#include "taichi/runtime/vulkan_control_recipe.h"
#include "taichi/runtime/vulkan_submission_recipe.h"

namespace taichi::lang::aot {
// Versioned persistent operations, independent of live definitions, runtime
// handles, backend commands and storage ownership. Binding consumes this only
// in the cold phase; replay retains no JSON or symbolic argument lookups.
struct ExecutionPlan {
  static constexpr std::uint32_t kMaxVersion = 25;
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
  using Native = NativeOperationDescription;
  struct Branch;
  struct While;
  struct Parallel;
  using Operation = std::variant<Kernel,
                                 Copy,
                                 Fill,
                                 Native,
                                 std::shared_ptr<const Branch>,
                                 std::shared_ptr<const While>,
                                 std::shared_ptr<const Parallel>>;
  // CUDA fork/join; each branch is an ordered, ordinary kernel sequence.
  // Surrounding operations retain the prefix and the join/suffix.
  struct Parallel {
    std::vector<std::vector<Operation>> branches;
  };
  struct Branch {
    enum class Lowering {
      cpu,
      cuda_conditional,
      cuda_host,
      vulkan_host,
      cuda_masked
    };
    Lowering lowering{Lowering::cpu};
    std::string selector;
    BranchSelection selection;
    std::vector<Operation> condition;
    std::vector<std::vector<Operation>> bodies;
  };
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
    std::optional<CudaControlRecipe<std::string>> cuda_control;
  };
  struct ArrayLayout {
    std::string array;
    NativeArrayLayoutRequirement required;
  };
  struct StructArrayLayout {
    std::string array;
    std::string layout;
    std::vector<std::uint64_t> exact_shape;
  };
  struct Disjoint {
    std::string left;
    std::string right;
    bool allow_same_view{false};
  };
  struct DisjointWrites {
    struct Write {
      std::string array;
      std::optional<std::size_t> inplace_read;
    };
    std::vector<std::string> reads;
    std::vector<Write> writes;
  };
  struct ResourceAlias {
    std::string left;
    std::string right;
    std::vector<std::string> left_path;
    std::vector<std::string> right_path;
  };
  using Constraint =
      std::variant<ArrayLayout, StructArrayLayout, Disjoint, DisjointWrites,
                   ResourceAlias>;
  std::vector<Constraint> constraints;
  std::vector<Operation> operations;
  std::vector<Operation> initialization;
  // Runtime dependencies exclude shape-only references resolved at prepare.
  std::set<std::string> replay_bindings;
  std::set<std::string> initialization_bindings;
  std::vector<std::string> arguments;
  ExecutionResources resources;
  bool cuda_binding_frames{false};
  std::optional<VulkanSubmissionRecipe> vulkan_submission;
  std::optional<OrderedStorageRing> storage_ring;

  static ExecutionPlan parse(const std::string &json);
};
}  // namespace taichi::lang::aot
