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
  std::string task_name;
  Arch arch{Arch::x64};
  std::uint32_t task_index{0};
  OffloadedTaskType task_type{OffloadedTaskType::serial};

  // Values entering backend code generation. A missing value means that the
  // execution model does not expose the corresponding GPU-shaped request.
  std::optional<std::int64_t> requested_grid_size;
  std::optional<std::int64_t> requested_block_size;

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

  std::uint64_t static_shared_bytes{0};
  std::uint64_t dynamic_shared_bytes{0};
};

}  // namespace taichi::lang
