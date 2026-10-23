#pragma once

#include "taichi/rhi/public_device.h"

namespace taichi::lang {

// Immutable submission prefix which can be waited on by multiple consumers,
// including after a host completion query. It owns no submitted operands.
// The backend owner must drain all consumers before retiring its native state;
// host query/wait must remain valid after that retirement. Consumable binary
// semaphores intentionally do not implement this contract.
class ReusableStreamDependency : public StreamSemaphoreObject {
 public:
  ~ReusableStreamDependency() override = default;
};

// A legacy completion can expose an additional reusable prefix without
// changing its consumable semaphore or its host fence semantics. Acquisition
// happens when the completion is created, never while replaying a prepared
// operation. The returned prefix must remain queryable after owner retirement.
class ReusableStreamDependencySource {
 public:
  virtual ~ReusableStreamDependencySource() = default;
  virtual std::shared_ptr<ReusableStreamDependency>
  reusable_dependency() const = 0;
};

}  // namespace taichi::lang
