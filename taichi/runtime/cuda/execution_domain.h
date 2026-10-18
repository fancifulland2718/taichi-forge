#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#include "taichi/program/runtime_completion.h"

namespace taichi::lang::cuda {

class ExecutionDomainBusy : public std::runtime_error {
 public:
  ExecutionDomainBusy() : std::runtime_error("CUDA execution domain is full") {
  }
};

// Runtime-only state shared by JIT and AOT consumers. No stream, event or
// domain identity belongs in a compiled artifact. The owning runtime retires
// its domains before destroying execution instances and device storage.
class ExecutionDomain {
 public:
  struct Options {
    std::size_t capacity{2};
    int priority{0};
  };
  struct Snapshot {
    std::uint64_t identity{0};
    std::uint64_t owner{0};
    std::uintptr_t context{0};
    std::uintptr_t stream{0};
    int requested_priority{0};
    int effective_priority{0};
    int least_priority{0};
    int greatest_priority{0};
    std::size_t capacity{0};
    std::size_t pending{0};
    bool accepting{false};
    bool closed{false};
  };

  // Cold capability query; never creates a stream or drains the GPU.
  static bool available();
  ExecutionDomain(std::shared_ptr<RuntimeFaultDomain> owner, Options options);
  ~ExecutionDomain();
  ExecutionDomain(const ExecutionDomain &) = delete;
  ExecutionDomain &operator=(const ExecutionDomain &) = delete;

  // Caller has prepared independent executable/argument/scratch state before
  // entering. The owner assigns a unique sequence after successful admission.
  // Every device action in enqueue must use its supplied nonblocking stream.
  // Enqueue may not recursively submit/close this domain or wait on the host.
  // All predecessor validation and capacity checks precede device effects.
  // Capacity exhaustion is immediate; no other domain is polled or drained.
  RuntimeCompletion submit(
      const std::vector<RuntimeCompletion> &dependencies,
      std::shared_ptr<RuntimeCompletionResources> resources,
      const std::function<void(void *)> &enqueue);

  // Completion/resource callbacks execute without holding the admission lock.
  std::size_t collect_ready(std::size_t budget);
  Snapshot snapshot() const;

  // Cold, idempotent drain of this stream only. No domain admission lock or
  // global CUDA driver/submission lock is held across a host backend wait.
  // An external producer must eventually signal already submitted waits.
  void close();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace taichi::lang::cuda
