#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// Backend-neutral, host-only description of one compiled offloaded task.
//
// This deliberately describes launch geometry without controlling it.  N0
// consumers can inspect compiler/backend decisions without allocating a
// device telemetry buffer or changing the launch path; launch overrides are a
// separate, later contract.
struct OffloadedTaskManifest {
  std::string task_id;
  std::string logical_task_id;
  std::string optimization_spec_id;
  std::string task_name;
  Arch arch{Arch::x64};
  std::uint32_t task_index{0};
  OffloadedTaskType task_type{OffloadedTaskType::serial};

  // Values entering backend code generation. A missing value means that the
  // execution model does not expose the corresponding GPU-shaped request.
  std::optional<std::int64_t> requested_grid_size;
  std::optional<std::int64_t> requested_block_size;
  bool source_block_size_explicit{false};
  std::string requested_thread_local_mode{"auto"};
  int requested_cuda_min_blocks_per_sm{2};
  std::optional<int> requested_cuda_max_registers;
  std::optional<int> requested_grid_residency_waves;
  int requested_range_work_per_thread_target{1};

  // Values selected by backend code generation. CPU execution intentionally
  // leaves these empty instead of pretending that its worker scheduler is a
  // GPU grid.
  std::optional<std::int64_t> selected_grid_size;
  std::optional<std::int64_t> selected_block_size;

  // Proven launch geometry for an ordinary direct invocation. Dynamic or
  // invocation-specific launch paths must leave these empty and explain why.
  std::optional<std::int64_t> actual_grid_size;
  std::optional<std::int64_t> actual_block_size;
  std::string actual_geometry_kind;
  std::string actual_geometry_reason;

  // Logical range-to-lane mapping. ``one_to_one`` proves that reducing a
  // dispatch grid reduces the number of visited logical indices.
  // ``device_bounded_grid_stride`` instead keeps the saturation-capped CUDA
  // grid while loading the logical end from a device extent.
  std::string range_mapping;
  std::optional<std::int64_t> constant_range_size;

  std::uint64_t static_shared_bytes{0};
  std::uint64_t dynamic_shared_bytes{0};
  std::uint64_t thread_local_bytes{0};
};

}  // namespace taichi::lang
