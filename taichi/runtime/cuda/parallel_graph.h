#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace taichi::lang::cuda {

// Cold fork/join capture shared by JIT and loaded-code consumers. The caller
// owns the root capture, context/locks, executable and completion. Branch
// streams are capture-only: no mathematical work is submitted by prepare().
// After retiring the executable (or aborting a failed capture), release(true)
// destroys streams/events without another wait. release(false) abandons handles
// when the backend is unsafe. Destruction itself makes no backend calls.
class ParallelGraphCapture {
 public:
  using Recorder = std::function<std::uint32_t(std::size_t, void *)>;

  ParallelGraphCapture() = default;
  ParallelGraphCapture(const ParallelGraphCapture &) = delete;
  ParallelGraphCapture &operator=(const ParallelGraphCapture &) = delete;
  // Reconfiguration requires the preceding executable/capture to be retired.
  void prepare(std::size_t branch_count);
  std::uint32_t capture(void *root_stream, const Recorder &record) const;
  void release(bool backend_safe);
  std::size_t branch_count() const {
    return branches_.size();
  }

 private:
  struct Branch {
    void *stream{nullptr};
    void *tail{nullptr};
  };
  std::vector<Branch> branches_;
  void *fork_{nullptr};
  bool prepared_{false};
};

}  // namespace taichi::lang::cuda
