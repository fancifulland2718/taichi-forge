#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include "taichi/rhi/device.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/branch_selection.h"
#include "taichi/runtime/cuda_control_recipe.h"
#include "taichi/runtime/vulkan_control_recipe.h"
#include "taichi/runtime/vulkan_submission_recipe.h"
#include "taichi/runtime/ordered_storage_ring.h"

namespace taichi::lang {
class LaunchContextBuilder;
class NativeOperation;
class NativeKernelResourceBinding;
class RuntimeFaultDomain;
class ExternalSynchronizationDomain;
struct NativeRecordingState;
namespace aot {
class Kernel;
}
namespace gfx {
class GfxRuntime;
}
namespace cuda {
class BindingFrameReplay;
class ExecutionDomain;
}

// Cold bindings shared by loaded AOT entries and native runtime adapters. The
// caller owns registered code, storage and submission/retirement serialization.
// Contexts and operation descriptors are borrowed only during preparation.
class FixedExecution {
 public:
  // Runtime-only ownership for independent CUDA capture/ordered recipes. Code
  // and array owners outlive every pending launch; provider objects supplied in
  // operations must have independent mutable state. Never serialized in AOT.
  struct CudaExecutionBinding {
    std::shared_ptr<cuda::ExecutionDomain> domain;
    std::shared_ptr<RuntimeCompletionResources> operands;
  };
  struct Dispatch {
    aot::Kernel *kernel;
    LaunchContextBuilder *context;
    DevicePtr indirect_dispatch{kDeviceNullPtr};
    DevicePtr bounded_extent{kDeviceNullPtr};
    std::uint32_t bounded_capacity{0};
    std::uint32_t adaptive_block_dim{0};
    bool grouped_update{false};
    // Cold-only logical binding identity; never dereferenced or retained.
    const void *bounded_binding_identity{nullptr};
    std::vector<std::shared_ptr<NativeKernelResourceBinding>> resources;
  };
  struct BufferCopy {
    DevicePtr destination;
    DevicePtr source;
    std::size_t bytes;
  };
  struct BufferFill {
    DevicePtr destination;
    std::size_t bytes;
    std::uint32_t value;
  };
  struct While;
  struct Branch;
  struct Parallel;
  using Operation = std::variant<Dispatch,
                                 BufferCopy,
                                 BufferFill,
                                 std::shared_ptr<NativeOperation>,
                                 std::shared_ptr<const Branch>,
                                 std::shared_ptr<const While>,
                                 std::shared_ptr<const Parallel>>;
  struct Parallel {
    std::vector<std::vector<Operation>> branches;
  };
  struct Branch {
    DevicePtr selector{kDeviceNullPtr};
    BranchSelection selection;
    std::vector<Operation> condition;
    std::vector<std::vector<Operation>> bodies;
    BranchExecutionMode mode{BranchExecutionMode::native};
  };
  struct While {
    struct Vulkan {
      DevicePtr counter{kDeviceNullPtr};
      DevicePtr status{kDeviceNullPtr};
      VulkanControlRecipe recipe;
      std::optional<VulkanNestedControlRecipe<DevicePtr>> nested;
    };
    DevicePtr predicate;
    std::uint64_t max_iterations{0};
    bool continue_while_nonzero{true};
    std::vector<Operation> condition;
    std::vector<Operation> body;
    std::optional<Vulkan> vulkan;
    std::optional<CudaControlRecipe<DevicePtr>> cuda_control;
    bool host_control{false};
  };

  // Optional cold facts chain unconditional phases of one prepared execution.
  // Published only after successful recording; they do not execute
  // initialization.
  // A CUDA binding family may supply one shared replay and a lease for this
  // frame's code/storage. Captured frames retain that lease through completion,
  // including after this FixedExecution wrapper is destroyed.
  FixedExecution(Arch arch,
                 Device &device,
                 gfx::GfxRuntime *gfx_runtime,
                 const std::vector<Operation> &operations,
                 NativeRecordingState *sequence_state = nullptr,
                 std::shared_ptr<RuntimeFaultDomain> fault_domain = {},
                 std::shared_ptr<cuda::BindingFrameReplay> binding_replay = {},
                 std::shared_ptr<void> binding_operands = {},
                 const VulkanSubmissionRecipe *submission_recipe = nullptr,
                 const OrderedStorageRing *storage_ring = nullptr,
                 const CudaExecutionBinding *execution_binding = nullptr);
  ~FixedExecution();
  FixedExecution(const FixedExecution &) = delete;
  FixedExecution &operator=(const FixedExecution &) = delete;
  void run();
  RuntimeCompletion submit(
      const std::vector<RuntimeCompletion> &dependencies = {});
  // Cold, once-only binding before publishing this instance to submitters.
  // The whole replay owns one external epoch on its selected CUDA stream.
  // Initialization has its own separately qualified access closure.
  void bind_external_access(
      std::vector<std::shared_ptr<ExternalSynchronizationDomain>> domains);
  // Called by an owner which has completed submitted work before retiring a
  // frame's external allocations. It does not retire sibling bindings.
  void retire_completed_binding(bool backend_safe = true);
  std::uint32_t execution_kind() const;
  std::size_t argument_bytes() const;
  std::size_t workspace_bytes() const;

 private:
  void run_impl();
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang
