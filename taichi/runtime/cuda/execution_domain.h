#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/runtime_client.h"

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
    std::size_t instance_capacity{64};
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
  // Optional host-only admission reserves provider transactions after those
  // checks and before any dependency is enqueued. It must not call the GPU,
  // wait, or reenter this domain. An exception leaves the domain usable and
  // consumes no sequence; resources must own rollback of partial reservations.
  // Destruction of rejected resources occurs outside the admission mutex.
  RuntimeCompletion submit(
      const std::vector<RuntimeCompletion> &dependencies,
      std::shared_ptr<RuntimeCompletionResources> resources,
      const std::function<void(void *)> &enqueue,
      const std::function<void()> &admit = {});

  // Completion/resource callbacks execute without holding the admission lock.
  std::size_t collect_ready(std::size_t budget);
  Snapshot snapshot() const;
  RuntimeCompletion latest_completion() const;
  // Cold owner retirement closes admission on every domain before draining
  // any one domain. Existing tickets remain queryable while a drain waits.
  void stop_accepting();
  bool belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const;
  // Reserve bounded ownership before preparation can create backend state.
  // Preparation and failure cleanup run without the admission mutex; close
  // waits for both before draining/releasing instances. The callback must not
  // close this domain. A concurrent close may retire the prepared instance
  // before this method returns, and subsequent submissions are then rejected.
  // Retaining through cold close prevents wrapper/ticket drops from freeing
  // CUDA storage on a completion-query hot path.
  void prepare_instance(const std::shared_ptr<RuntimeClient> &instance,
                        const std::function<void()> &prepare);

  // Cold, idempotent destructive teardown. The explicit drain waits only for
  // this stream without admission/global CUDA locks, but driver or vendor
  // destruction may synchronize the context. To pause while peers are active,
  // stop_accepting() and wait latest_completion(), retaining bounded state.
  // An external producer must eventually signal already submitted waits.
  void close();

 private:
  friend class ExecutionDomainRegistry;
  RuntimeCompletion retirement_completion() const;
  struct State;
  std::unique_ptr<State> state_;
};

// One registry per runtime generation, shared with its borrowed AOT views.
// Retained slots bound both live domains and historical closed-domain storage.
// Registration and retirement are cold; submit never acquires this mutex.
class ExecutionDomainRegistry final : public RuntimeClient {
 public:
  explicit ExecutionDomainRegistry(std::shared_ptr<RuntimeFaultDomain> owner,
                                   std::size_t capacity = 64);
  ~ExecutionDomainRegistry() override;
  std::shared_ptr<ExecutionDomain> create(ExecutionDomain::Options options);
  bool belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const;
  bool accepting() const;
  // Cold explicit runtime synchronization. Snapshot every queue before any
  // wait, so subsequent work cannot extend the requested prefixes.
  void synchronize();
  void retire() override;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace taichi::lang::cuda
