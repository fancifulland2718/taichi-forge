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

}  // namespace taichi::lang
