#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include "taichi/rhi/device.h"

namespace taichi::lang {

// Prepared backend commands own their scratch and recording state. The
// enclosing execution retains input/output leases and retires submitted work
// before releasing this object.
class NativeOperation {
 public:
  virtual ~NativeOperation() = default;
  virtual Arch arch() const = 0;
  virtual Device &device() const = 0;
  virtual std::size_t workspace_bytes() const = 0;
  virtual void run_cpu() {
    throw std::invalid_argument("Native operation has no CPU implementation");
  }
  virtual void record_cuda(void *stream) {
    throw std::invalid_argument("Native operation has no CUDA recording");
  }
  virtual void record_vulkan(Device *device, CommandList *commands) {
    throw std::invalid_argument("Native operation has no Vulkan recording");
  }
};

struct CompactOperation {
  DevicePtr values;
  DevicePtr flags;
  DevicePtr output;
  DevicePtr count;
  std::size_t size;
  std::size_t item_bytes;
};

// Stable selection by a nonzero i32 predicate. Count is written on every
// replay, including empty input. Preparation submits no mathematical work.
std::shared_ptr<NativeOperation> prepare_compact_operation(
    Arch arch,
    Device &device,
    const CompactOperation &operation);

namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_compact_operation(
    Device &device,
    const CompactOperation &operation);
}

}  // namespace taichi::lang
