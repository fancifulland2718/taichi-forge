#pragma once

#ifdef TI_WITH_VULKAN
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "taichi/program/runtime_completion.h"
#include "taichi/rhi/vulkan/vulkan_execution_queue.h"
#include "taichi/runtime/runtime_client.h"

namespace taichi::lang::vulkan {

class ExecutionInstance;

// Runtime ownership around the RHI queue. Descriptions/artifacts contain no
// queue identities. A shared Program/AOT owner retires this registry before
// code, storage, and the Vulkan device. Completion tokens remain common to all
// backends and keep their reusable timeline prefix after host observation.
class ExecutionDomain {
 public:
  struct Options {
    std::size_t capacity{2};
    std::size_t instance_capacity{64};
    bool gpu_timing{false};
  };
  struct Snapshot {
    std::uint64_t identity{0};
    std::uint64_t owner{0};
    std::uint32_t queue_index{0};
    std::size_t capacity{0};
    std::size_t pending{0};
    bool accepting{false};
    bool closed{false};
  };
  using Commands = VulkanExecutionQueue::Commands;
  using CommandSelector = VulkanExecutionQueue::CommandSelector;
  using TimingSelector = std::function<StreamGpuTiming(std::size_t)>;
  using Busy = VulkanExecutionQueue::Busy;
  using Admission =
      std::function<void(std::size_t, const RuntimeCompletion &)>;

  ExecutionDomain(VulkanDevice &device,
                  std::shared_ptr<RuntimeFaultDomain> owner,
                  Options options);
  ~ExecutionDomain();
  ExecutionDomain(const ExecutionDomain &) = delete;
  ExecutionDomain &operator=(const ExecutionDomain &) = delete;

  // Cold recording only. Commands and their operand closure remain immutable;
  // per-submission parameter snapshots use the preallocated admission slot.
  std::unique_ptr<CommandList> new_command_list();
  bool gpu_timing_enabled() const;
  StreamGpuTiming prepare_timing(CommandList &begin, CommandList &end);
  std::shared_ptr<const Commands> prepare_commands(
      const std::vector<CommandList *> &commands,
      std::shared_ptr<void> operands = {});

  // Capacity and every predecessor are checked before the host-only callback.
  // It may fill its preallocated slot and reserve completion publication, but
  // cannot enqueue/wait/reenter or publish the still-unsubmitted ticket.
  // A rejected callback leaves the queue and instance usable. All fallible
  // completion construction precedes the single native queue submission.
  RuntimeCompletion submit(
      std::shared_ptr<const Commands> commands,
      const std::vector<RuntimeCompletion> &dependencies = {},
      std::shared_ptr<RuntimeCompletionResources> resources = {},
      const Admission &admit = {});
  RuntimeCompletion submit_instance(
      const std::shared_ptr<ExecutionInstance> &instance,
      std::shared_ptr<const Commands> commands,
      const std::vector<RuntimeCompletion> &dependencies = {},
      std::shared_ptr<RuntimeCompletionResources> resources = {},
      const Admission &admit = {});
  // Select an immutable variant inside admission, after collection and exact
  // owner/capacity checks. This serializes first-use and per-slot selection
  // without holding an instance mutex while completed operands are released.
  RuntimeCompletion submit_instance_selected(
      const std::shared_ptr<ExecutionInstance> &instance,
      const CommandSelector &select,
      const std::vector<RuntimeCompletion> &dependencies = {},
      std::shared_ptr<RuntimeCompletionResources> resources = {},
      const Admission &admit = {},
      const TimingSelector &timing = {});
  std::size_t collect_ready(std::size_t budget);
  Snapshot snapshot() const;
  RuntimeCompletion latest_completion() const;
  bool belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const;
  void stop_accepting();

  // Reserve the caller-selected instance budget before preparation. Both
  // preparation and cleanup run outside admission locks. Failed cleanup keeps
  // its owner/budget quarantined; ordinary rejection leaves peers usable.
  std::shared_ptr<ExecutionInstance> prepare_owned_instance(
      const std::shared_ptr<RuntimeClient> &client,
      const std::function<void(const std::shared_ptr<ExecutionInstance> &)> &prepare);
  // Stop only this instance and wait its exact last prefix. Callback failure
  // remains observable and does not retire unrelated instances.
  void retire_instance(const std::shared_ptr<ExecutionInstance> &instance);
  bool instance_retired(const std::shared_ptr<ExecutionInstance> &instance) const;
  void close();

 private:
  friend class ExecutionDomainRegistry;
  RuntimeCompletion retirement_completion() const;
  RuntimeCompletion submit_impl(
      const std::shared_ptr<ExecutionInstance> &instance,
      std::shared_ptr<const Commands> commands,
      const std::vector<RuntimeCompletion> &dependencies,
      std::shared_ptr<RuntimeCompletionResources> resources,
      const Admission &admit,
      const CommandSelector &select = {},
      const TimingSelector &timing = {});
  struct State;
  std::unique_ptr<State> state_;
};

// Cold registration grows/reuses closed slots. The actual number of independent
// queues is determined by the selected Vulkan device; no software registry cap.
class ExecutionDomainRegistry final : public RuntimeClient {
 public:
  ExecutionDomainRegistry(VulkanDevice &device,
                          std::shared_ptr<RuntimeFaultDomain> owner);
  ~ExecutionDomainRegistry() override;
  std::shared_ptr<ExecutionDomain> create(ExecutionDomain::Options options);
  bool belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const;
  bool accepting() const;
  void synchronize();
  void retire() override;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace taichi::lang::vulkan
#endif
