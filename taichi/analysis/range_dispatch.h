#pragma once

#include "taichi/analysis/range_for_bounds.h"
#include "taichi/analysis/range_collective.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/snode.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// These intrinsics communicate only within the original CUDA block/warp.
// In particular, grid-size queries, async copies, unknown calls and physical
// thread identity are not admitted by this list.
inline bool is_cuda_local_collective(const InternalFuncStmt &call) {
  if (call.with_runtime_context)
    return false;
  const auto &name = call.func_name;
  return name == "block_barrier" || name == "block_barrier_and_i32" ||
         name == "block_barrier_or_i32" || name == "block_barrier_count_i32" ||
         name == "warp_barrier" || name == "cuda_ballot_i32" ||
         name == "cuda_all_sync_i32" || name == "cuda_shfl_sync_i32" ||
         name == "cuda_shfl_sync_f32" || name == "cuda_shfl_up_sync_i32" ||
         name == "cuda_shfl_up_sync_f32" || name == "cuda_shfl_down_sync_i32" ||
         name == "cuda_shfl_down_sync_f32" || name == "cuda_shfl_xor_sync_i32";
}

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
  bool cuda_collective = false;
  irpass::analysis::any_statement(task.body.get(), [&](Stmt *stmt) {
    if (auto *call = stmt->cast<InternalFuncStmt>()) {
      if (task.device == Arch::cuda && !has_tls && is_cuda_local_collective(*call))
        cuda_collective = true;
      else
        proof = EmptyRangeGroupProof::unproven_call;
    } else if (stmt->is<ExternalFuncCallStmt>() || stmt->is<FuncCallStmt>())
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
    if (stmt->is<DynamicSharedPtrStmt>())
      proof = EmptyRangeGroupProof::local_storage;
    if (auto *allocation = stmt->cast<AllocaStmt>())
      if (allocation->is_shared) {
        if (task.device == Arch::cuda && !has_tls)
          cuda_collective = true;
        else
          proof = EmptyRangeGroupProof::local_storage;
      }
    if (auto *lookup = stmt->cast<SNodeLookupStmt>()) {
      if (lookup->snode->type != SNodeType::dense &&
          lookup->snode->type != SNodeType::root)
        proof = EmptyRangeGroupProof::sparse_topology;
    }
    return proof != EmptyRangeGroupProof::independent_iterations;
  });
  if (cuda_collective && proof == EmptyRangeGroupProof::independent_iterations)
    return EmptyRangeGroupProof::cuda_block_collective;
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
