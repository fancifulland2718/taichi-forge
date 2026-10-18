#pragma once

#include "taichi/runtime/native_resource.h"

namespace taichi::lang {

// A cold, owner-local allocation policy. Each returned allocation belongs to
// the device's ordinary allocation registry and is freed through that device.
// The allocation record, not this factory's lifetime, protects its backing pool.
class NativeMemoryAllocator : public NativeResource {
 public:
  virtual DeviceAllocation allocate_memory(std::size_t bytes,
                                           AllocUsage usage) = 0;
};

}  // namespace taichi::lang
