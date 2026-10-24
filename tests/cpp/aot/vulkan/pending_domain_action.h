#pragma once

#include "tests/cpp/aot/vulkan/timeline_gate.h"
#include "taichi/runtime/runtime_client.h"
#include <chrono>
#include <future>
#include <mutex>
#include <vector>

namespace taichi::lang::test_detail {
// TimelineGate's DeviceHandle owns function tables, not VkDevice. Retire these
// test-only semaphore/fence objects in Program's existing RuntimeClient phase,
// after domain draining and before backend destruction. release and retire
// share a test mutex so the final signal cannot race gate destruction.
class GateOwner final : public RuntimeClient {
 public:
  explicit GateOwner(std::size_t count = 2) : gates(count) {
  }
  std::vector<std::unique_ptr<TimelineGate>> gates;
  void release(std::size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gates[index])
      gates[index]->release();
  }
  void release_all() {
    for (std::size_t i = 0; i < gates.size(); ++i)
      release(i);
  }
  void retire() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &gate : gates)
      gate.reset();
  }

 private:
  std::mutex mutex_;
};

// Failed assertions/exceptions must release the controlled GPU dependency
// before std::future joins the cold close/finalize thread.
class PendingAction {
 public:
  PendingAction(std::shared_ptr<GateOwner> gates, std::function<void()> action)
      : gates_(std::move(gates)),
        result_(
            std::async(std::launch::async, [this, action = std::move(action)] {
              started_.set_value();
              action();
            })) {
    started_.get_future().wait();
  }
  ~PendingAction() {
    gates_->release_all();
    if (result_.valid())
      result_.wait();
  }
  std::future_status status(std::chrono::milliseconds timeout) {
    return result_.wait_for(timeout);
  }
  void finish() {
    result_.get();
  }

 private:
  std::shared_ptr<GateOwner> gates_;
  std::promise<void> started_;
  std::future<void> result_;
};

}  // namespace taichi::lang::test_detail
