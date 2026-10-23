#pragma once

#include <optional>

#include "external/picosha2/picosha2.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang::cuda {

// These requests use existing OffloadedTask wire fields. They describe a
// deployment default, never a producer-device occupancy result or code key.
inline void validate_persisted_task_launch_policy(const OffloadedTask &task,
                                                 std::size_t index) {
  const auto waves = task.requested_grid_residency_waves;
  const auto work = task.requested_range_work_per_thread_target;
  TI_ERROR_IF(waves != 0 && waves != 1 && waves != 2 && waves != 4,
              "Persisted CUDA grid residency waves are invalid at task {}",
              index);
  TI_ERROR_IF(work != 1 && work != 2 && work != 4 && work != 8,
              "Persisted CUDA work-per-thread target is invalid at task {}",
              index);
  if (waves == 0 && work == 1)
    return;
  TI_ERROR_IF(task.task_type != OffloadedTaskType::range_for ||
                  task.one_to_one ||
                  task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
                  task.may_mutate_sparse_topology || task.grid_dim <= 0 ||
                  task.block_dim <= 0 || task.dynamic_shared_array_bytes < 0,
              "Persisted CUDA launch policy requires a dense grid-stride "
              "range at task {}", index);
  TI_ERROR_IF(work != 1 && task.constant_range_size < 0,
              "Persisted CUDA work-per-thread control requires a constant "
              "range at task {}", index);
}

struct PersistedTaskLaunchPolicy {
  std::string identity;
  LaunchContextBuilder::CudaTaskExecutionPlanDigest content_digest{};
  std::vector<std::string> task_kinds;
  std::vector<int> grid_residency_waves;
  std::vector<int> range_work_per_thread_targets;
  std::vector<int> sparse_list_policies;
};

inline std::optional<PersistedTaskLaunchPolicy> make_persisted_task_launch_policy(
    const std::vector<OffloadedTask> &tasks) {
  bool requested = false;
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    const auto &task = tasks[index];
    validate_persisted_task_launch_policy(task, index);
    requested |= task.requested_grid_residency_waves != 0 ||
                 task.requested_range_work_per_thread_target != 1;
  }
  // Existing ordinary and sparse artifacts retain their allocation-free
  // baseline selection, without manufacturing an execution plan.
  if (!requested)
    return std::nullopt;
  PersistedTaskLaunchPolicy policy;
  for (const auto &task : tasks) {
    policy.task_kinds.emplace_back(offloaded_task_type_name(task.task_type));
    policy.grid_residency_waves.push_back(task.requested_grid_residency_waves);
    policy.range_work_per_thread_targets.push_back(
        task.requested_range_work_per_thread_target);
    policy.sparse_list_policies.push_back(0);
  }
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(policy.task_kinds);
  serializer(policy.grid_residency_waves);
  serializer(policy.range_work_per_thread_targets);
  serializer(policy.sparse_list_policies);
  serializer.finalize();
  picosha2::hash256(serializer.data, policy.content_digest);
  policy.identity = "aot-task-launch-v1:" +
                    picosha2::bytes_to_hex_string(policy.content_digest);
  return policy;
}

}  // namespace taichi::lang::cuda
