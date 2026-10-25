#pragma once

#include "taichi/analysis/range_for_bounds.h"
#include "taichi/analysis/range_collective.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/snode.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// A code-generation fact, independent of whether a caller selects device range
// dispatch. Keep unknown intrinsics/calls conservative: their physical-thread
// or workspace contract is not described by an ordinary grid-stride loop.
inline EmptyRangeGroupProof describe_empty_range_group_elision(
    OffloadedStmt &task) {
  // CPU's physical range belongs to its worker scheduler. Retain its bounds
  // without walking a large CPU body for a GPU-only transformation.
  if (arch_is_cpu(task.device))
    return EmptyRangeGroupProof::fixed_mapping;
  if (task.task_type != OffloadedTaskType::range_for || task.one_to_one ||
      task.reversed || task.is_bit_vectorized || !task.body)
    return EmptyRangeGroupProof::fixed_mapping;
  if (task.external_shared_staged || task.bls_size)
    return EmptyRangeGroupProof::local_storage;
  if (task.bls_prologue || task.bls_epilogue || task.mesh_prologue)
    return EmptyRangeGroupProof::auxiliary_region;
  RangeReductionSlots reductions;
  const bool has_tls = task.tls_prologue || task.tls_epilogue;
  if (has_tls && !describe_range_reduction_slots(task, reductions))
    return EmptyRangeGroupProof::auxiliary_region;
  auto proof = EmptyRangeGroupProof::independent_iterations;
  irpass::analysis::any_statement(task.body.get(), [&](Stmt *stmt) {
    if (stmt->is<InternalFuncStmt>() || stmt->is<ExternalFuncCallStmt>() ||
        stmt->is<FuncCallStmt>())
      proof = EmptyRangeGroupProof::unproven_call;
    else if (stmt->is<GlobalThreadIndexStmt>() ||
             stmt->is<BlockCornerIndexStmt>() || stmt->is<LoopLinearIndexStmt>() ||
             stmt->is<RandStmt>())
      proof = EmptyRangeGroupProof::physical_thread;
    else if (auto *ptr = stmt->cast<ThreadLocalPtrStmt>()) {
      const auto found = reductions.find(ptr->offset);
      if (found == reductions.end() || found->second.dt != ptr->ret_type.ptr_removed())
        proof = EmptyRangeGroupProof::local_storage;
    } else if (stmt->is<BlockLocalPtrStmt>())
      proof = EmptyRangeGroupProof::local_storage;
    else if (stmt->is<SNodeOpStmt>())
      proof = EmptyRangeGroupProof::sparse_topology;
    if (auto *allocation = stmt->cast<AllocaStmt>())
      if (allocation->is_shared)
        proof = EmptyRangeGroupProof::local_storage;
    if (auto *lookup = stmt->cast<SNodeLookupStmt>()) {
      if (lookup->snode->type != SNodeType::dense &&
          lookup->snode->type != SNodeType::root)
        proof = EmptyRangeGroupProof::sparse_topology;
    }
    return proof != EmptyRangeGroupProof::independent_iterations;
  });
  return has_tls && proof == EmptyRangeGroupProof::independent_iterations
             ? EmptyRangeGroupProof::compiler_tls_reduction
             : proof;
}

inline std::optional<RangeForBounds> describe_launch_range_for_bounds(
    OffloadedStmt &task) {
  auto bounds = describe_range_for_bounds(task);
  if (bounds) {
    bounds->host_launch = task.host_launch_range;
    bounds->empty_group_elision = describe_empty_range_group_elision(task);
  }
  return bounds;
}

inline bool supports_device_range_dispatch(OffloadedStmt &task) {
  const auto bounds = describe_launch_range_for_bounds(task);
  return bounds && bounds->supports_device_dispatch();
}

}  // namespace taichi::lang
