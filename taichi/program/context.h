#pragma once

// Use relative path here for runtime compilation
#include "taichi/inc/constants.h"
#include <cstddef>
#include <cstdint>

#if defined(TI_RUNTIME_HOST)
namespace taichi::lang {
#endif

struct LLVMRuntime;

// Private CUDA Graph execution-mask binding stored immediately before a
// gated capture packet's argument buffer. Keeping this out of RuntimeContext
// preserves the split-runtime ABI: ordinary launches see the historical
// context layout, while graph-gated task variants can recover the binding
// from ``arg_buffer - sizeof(GraphExecutionGateBinding)``.
struct GraphExecutionGateBinding {
  std::uintptr_t gate{0};
  std::uint32_t expected{0};
  std::uint32_t reserved{0};
};

static_assert(sizeof(GraphExecutionGateBinding) == 16);
static_assert(offsetof(GraphExecutionGateBinding, gate) == 0);
static_assert(offsetof(GraphExecutionGateBinding, expected) == 8);

// Private CPU Graph binding stored immediately before RuntimeContext's
// argument buffer. It lets a one-to-one range task derive its scheduler end
// directly from a DeviceExtent without changing the split-runtime context ABI.
struct CpuBoundedRangeBinding {
  std::uintptr_t extent{0};
  std::int32_t capacity{0};
  std::uint32_t reserved{0};
};

static_assert(sizeof(CpuBoundedRangeBinding) == 16);
static_assert(offsetof(CpuBoundedRangeBinding, extent) == 0);
static_assert(offsetof(CpuBoundedRangeBinding, capacity) == 8);

// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  char *arg_buffer{nullptr};

  LLVMRuntime *runtime{nullptr};

  int32_t cpu_thread_id;

  // We move the pointer of result buffer from LLVMRuntime to RuntimeContext
  // because each real function need a place to store its result, but
  // LLVMRuntime is shared among functions. So we moved the pointer to
  // RuntimeContext which each function have one.
  uint64_t *result_buffer;
};

#if defined(TI_RUNTIME_HOST)
}  // namespace taichi::lang
#endif
