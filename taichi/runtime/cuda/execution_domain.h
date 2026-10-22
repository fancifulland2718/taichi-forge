#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/runtime_client.h"

namespace taichi::lang::cuda {

class ExecutionInstance;

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
    bool gpu_timing{false};
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
  // checks and before any dependency is enqueued. Nonblocking producer status
  // queries are allowed; it must not enqueue work, wait, or reenter this
  // domain. An exception leaves the domain usable and
  // consumes no sequence; resources must own rollback of partial reservations.
  // Destruction of rejected resources occurs outside the admission mutex.
  RuntimeCompletion submit(
      const std::vector<RuntimeCompletion> &dependencies,
      std::shared_ptr<RuntimeCompletionResources> resources,
      const std::function<void(void *)> &enqueue,
      const std::function<void()> &admit = {},
      const std::function<void(void *)> &before_timing = {},
      const std::function<void(void *)> &after_timing = {},
      bool sample_gpu = false);

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

  // Cold owner registration with an explicit submission identity. The handle
  // is usable only after prepare returns. Submission records its exact last
  // prefix under the same admission lock that excludes retirement.
  std::shared_ptr<ExecutionInstance> prepare_owned_instance(
      const std::shared_ptr<RuntimeClient> &owner,
      const std::function<void(const std::shared_ptr<ExecutionInstance> &)> &prepare);
  RuntimeCompletion submit_instance(
      const std::shared_ptr<ExecutionInstance> &instance,
      const std::vector<RuntimeCompletion> &dependencies,
      std::shared_ptr<RuntimeCompletionResources> resources,
      const std::function<void(void *)> &enqueue,
      const std::function<void()> &admit = {},
      const std::function<void(void *)> &before_timing = {},
      const std::function<void(void *)> &after_timing = {},
      bool sample_gpu = false);
  // Stop this instance, wait only its last accepted prefix, then retire its
  // client outside the admission lock. Successful teardown releases its budget
  // even while peers remain pending. Client teardown must not reenter close or
  // retirement; it can inspect/submit unrelated instances. A failed teardown
  // retains its owner and budget, and remains observable on repeated calls.
  // The client's backend/vendor frees may still synchronize other queues;
  // callers must separately qualify those resource-specific teardown paths.
  void retire_instance(const std::shared_ptr<ExecutionInstance> &instance);
  bool instance_retired(const std::shared_ptr<ExecutionInstance> &instance) const;

  // Cold, idempotent destructive teardown. The explicit drain waits only for
  // this stream without admission/global CUDA locks, but driver or vendor
  // destruction may synchronize the context. To pause while peers are active,
  // stop_accepting() and wait latest_completion(), retaining bounded state.
  // An external producer must eventually signal already submitted waits.
  void close();

 private:
  friend class ExecutionDomainRegistry;
  RuntimeCompletion retirement_completion() const;
  RuntimeCompletion submit_impl(
      const std::shared_ptr<ExecutionInstance> &instance,
      const std::vector<RuntimeCompletion> &dependencies,
      std::shared_ptr<RuntimeCompletionResources> resources,
      const std::function<void(void *)> &enqueue,
      const std::function<void()> &admit,
      const std::function<void(void *)> &before_timing,
      const std::function<void(void *)> &after_timing,
      bool sample_gpu);
  struct State;
  std::unique_ptr<State> state_;
};

// One registry per runtime generation, shared with its borrowed AOT views.
// Registration grows on the cold path and reuses closed-domain slots. This
// storage is not an admission budget: each domain controls its own in-flight
// capacity. Submit never acquires the registry mutex.
class ExecutionDomainRegistry final : public RuntimeClient {
 public:
  explicit ExecutionDomainRegistry(std::shared_ptr<RuntimeFaultDomain> owner,
                                   std::size_t initial_capacity = 0);
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
