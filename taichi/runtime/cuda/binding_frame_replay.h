#pragma once

#include <cstdint>
#include <memory>
#include "taichi/program/runtime_completion.h"

namespace taichi::lang::cuda {

// One executable shared by immutable captured argument frames. Independent of
// Program, kernel compilation, argument binding and resource reconstruction.
// Callers serialize every operation under the CUDA submission lock and retain
// their runtime owner. Frame leases own the source graph and all its operands;
// preparation uploads and replay use the default stream.
class BindingFrameReplay {
 public:
  struct Snapshot {
    bool failed;
    std::size_t executables;
    std::size_t pending_frames;
    RuntimeCompletionCudaEventPoolSnapshot events;
  };

  BindingFrameReplay(std::uint64_t owner_domain,
                     std::shared_ptr<RuntimeFaultDomain> fault_domain);
  ~BindingFrameReplay();
  BindingFrameReplay(const BindingFrameReplay &) = delete;
  BindingFrameReplay &operator=(const BindingFrameReplay &) = delete;

  // Preparation changes the selected frame but performs no mathematical work.
  // It also retains host staging until already-enqueued uploads complete.
  void prepare(void *source_graph, const std::shared_ptr<void> &frame);
  void run(void *source_graph, const std::shared_ptr<void> &frame);
  void collect();
  bool is_active(const void *frame) const;
  bool failed() const;
  bool belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const;
  // Cold owner retirement after its stream wait (or an unsafe-backend fault).
  // Other published frames keep the executable and remain independently usable.
  void release_completed_frame(const void *frame, bool backend_safe) noexcept;
  Snapshot snapshot() const;

  // Two-phase retirement: stop device use first, then the caller invalidates
  // all published frames (including inactive ones), then release their leases.
  // The return value governs whether frame-specific backend cleanup is safe.
  bool retire_backend(bool backend_safe) noexcept;
  void release_frame_leases() noexcept;

 private:
  void select(void *source_graph, const std::shared_ptr<void> &frame);
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace taichi::lang::cuda
