#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// Compiler-owned classification of the state embedded in an SNode-dependent
// executable.  Runtime reuse decisions consume this type directly; Python
// diagnostics only serialize it and must never infer admission from task
// names or layout fingerprints.
enum class SNodeRelocationClass : std::uint8_t {
  not_applicable = 0,
  fully_relocatable = 1,
  partially_relocatable = 2,
  generation_bound = 3,
};

enum class SNodeRelocationState : std::uint8_t {
  tree_identity = 0,
  root_allocation = 1,
  runtime_state = 2,
  backend_registration = 3,
  sparse_listgen_state = 4,
  sparse_active_list_metadata = 5,
  sparse_allocator_state = 6,
  compiler_embedded_state_unclassified = 7,
  pointer_allocator_and_list_state = 8,
  bitmasked_activity_state = 9,
  dynamic_chunk_and_length_state = 10,
  hash_bucket_tombstone_and_pool_state = 11,
};

// Compiler-side inventory of structural SNode state referenced by one
// executable. This is deliberately a bit mask: a kernel can traverse several
// sparse trees, and losing one category must never silently promote it to the
// dense relocation class.
enum class SNodeRelocationStructure : std::uint32_t {
  none = 0,
  pointer = 1u << 0,
  bitmasked = 1u << 1,
  dynamic = 1u << 2,
  hash = 1u << 3,
};

inline constexpr SNodeRelocationStructure operator|(
    SNodeRelocationStructure lhs,
    SNodeRelocationStructure rhs) noexcept {
  return static_cast<SNodeRelocationStructure>(
      static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

inline SNodeRelocationStructure &operator|=(
    SNodeRelocationStructure &lhs,
    SNodeRelocationStructure rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

inline constexpr bool has_snode_relocation_structure(
    SNodeRelocationStructure value,
    SNodeRelocationStructure expected) noexcept {
  return (static_cast<std::uint32_t>(value) &
          static_cast<std::uint32_t>(expected)) != 0;
}

enum class SNodeRelocationBlocker : std::uint8_t {
  compiler_embedded_state_unclassified = 0,
  executable_and_generation_binding_not_separated = 1,
  in_flight_rebind_not_qualified = 2,
  graph_masked_rebind_not_qualified = 3,
  sparse_state_not_qualified = 4,
  llvm_registration_generation_specific = 5,
  spirv_registration_generation_specific = 6,
};

struct SNodeTaskRelocationDescriptor {
  std::uint32_t task_index{0};
  OffloadedTaskType task_type{OffloadedTaskType::serial};
  SNodeRelocationClass relocation_class{
      SNodeRelocationClass::generation_bound};
  std::vector<SNodeRelocationState> generation_bound_state;
};

struct SNodeRelocationDescriptor {
  static constexpr std::uint32_t kSchemaVersion = 1;

  std::uint32_t schema_version{kSchemaVersion};
  Arch backend{Arch::x64};
  bool compiler_emitted{false};
  bool has_snode_tree_dependencies{false};
  bool compiler_embedded_state_fully_classified{false};
  bool reuse_admitted{false};
  SNodeRelocationClass relocation_class{
      SNodeRelocationClass::generation_bound};
  std::vector<SNodeTaskRelocationDescriptor> tasks;
  std::vector<SNodeRelocationBlocker> blockers;
};

// Conservative promotion applied after compiler codegen has enumerated every
// task. Dense kernels have no allocator/list contents to relocate. Their code
// object is therefore reusable when the backend can supply generation-owned
// roots at registration or launch time; backend registration itself remains
// a per-generation binding.
void qualify_dense_snode_relocation(SNodeRelocationDescriptor &descriptor);

inline std::string_view snode_relocation_class_name(
    SNodeRelocationClass value) noexcept {
  switch (value) {
    case SNodeRelocationClass::not_applicable:
      return "not_applicable";
    case SNodeRelocationClass::fully_relocatable:
      return "fully_relocatable";
    case SNodeRelocationClass::partially_relocatable:
      return "partially_relocatable";
    case SNodeRelocationClass::generation_bound:
      return "generation_bound";
  }
  return "generation_bound";
}

inline std::string_view snode_relocation_state_name(
    SNodeRelocationState value) noexcept {
  switch (value) {
    case SNodeRelocationState::tree_identity:
      return "tree_id_and_generation";
    case SNodeRelocationState::root_allocation:
      return "root_allocation";
    case SNodeRelocationState::runtime_state:
      return "runtime_state";
    case SNodeRelocationState::backend_registration:
      return "backend_registration";
    case SNodeRelocationState::sparse_listgen_state:
      return "sparse_listgen_state";
    case SNodeRelocationState::sparse_active_list_metadata:
      return "active_list_metadata";
    case SNodeRelocationState::sparse_allocator_state:
      return "sparse_allocator_or_active_list";
    case SNodeRelocationState::compiler_embedded_state_unclassified:
      return "compiler_embedded_state_unclassified";
    case SNodeRelocationState::pointer_allocator_and_list_state:
      return "pointer_allocator_and_list_state";
    case SNodeRelocationState::bitmasked_activity_state:
      return "bitmasked_activity_state";
    case SNodeRelocationState::dynamic_chunk_and_length_state:
      return "dynamic_chunk_and_length_state";
    case SNodeRelocationState::hash_bucket_tombstone_and_pool_state:
      return "hash_bucket_tombstone_and_pool_state";
  }
  return "compiler_embedded_state_unclassified";
}

inline std::string_view snode_relocation_blocker_name(
    SNodeRelocationBlocker value) noexcept {
  switch (value) {
    case SNodeRelocationBlocker::compiler_embedded_state_unclassified:
      return "compiler_ir_embedded_state_not_fully_enumerated";
    case SNodeRelocationBlocker::executable_and_generation_binding_not_separated:
      return "layout_module_and_generation_binding_not_separated";
    case SNodeRelocationBlocker::in_flight_rebind_not_qualified:
      return "old_work_in_flight_rebind_not_qualified";
    case SNodeRelocationBlocker::graph_masked_rebind_not_qualified:
      return "graph_masked_handle_rebind_not_qualified";
    case SNodeRelocationBlocker::sparse_state_not_qualified:
      return "sparse_list_and_allocator_relocation_not_qualified";
    case SNodeRelocationBlocker::llvm_registration_generation_specific:
      return "llvm_jit_module_registration_is_generation_specific";
    case SNodeRelocationBlocker::spirv_registration_generation_specific:
      return "spirv_pipeline_registration_is_generation_specific";
  }
  return "compiler_ir_embedded_state_not_fully_enumerated";
}

}  // namespace taichi::lang
