#include "taichi/runtime/fixed_execution.h"
#include "taichi/program/runtime_fault.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <limits>
#include <set>
#include <unordered_set>
#include "taichi/aot/graph_data.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_image.h"
#include "taichi/program/launch_context_builder.h"
#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/binding_frame_replay.h"
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/cuda/ordered_submission.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/gfx/aot_graph_data.h"
#endif

namespace taichi::lang {
struct FixedExecution::State {
  std::shared_ptr<RuntimeFaultDomain> fault_domain;
  std::uint32_t kind{0};
  std::size_t bytes{0};
  std::size_t workspace_bytes{0};
  // Destroy backend recordings before their retained workspace owners.
  std::vector<std::shared_ptr<NativeOperation>> native_operations;
  std::vector<std::shared_ptr<NativeImageKernelBinding>> image_bindings;
  std::size_t native_workspace_bytes{0};
  // Host steps retire their native blocks and readback commands before staging
  // and provider owners. All selection storage/commands are prepared cold.
  DeviceAllocationUnique host_staging;
  using HostSteps = std::vector<std::function<void()>>;
  HostSteps host_steps;
  bool host_cuda{false};
  bool host_failed{false};
  static void run_host_steps(const HostSteps &steps) {
    for (const auto &step : steps)
      step();
  }
#ifdef TI_WITH_LLVM
  struct CpuStep {
    cpu::KernelLauncher *launcher{nullptr};
    std::vector<cpu::KernelLauncher::PreparedLaunch> launches;
    std::function<void()> native;
  };
  std::vector<CpuStep> cpu_steps;
  static void run_cpu_steps(std::vector<CpuStep> &steps) {
    for (auto &step : steps) {
      if (step.launcher)
        step.launcher->launch_prepared(step.launches);
      else
        step.native();
    }
  }
#endif
#ifdef TI_WITH_CUDA
  std::unique_ptr<cuda::FixedGraph> cuda_graph;
  std::shared_ptr<cuda::FixedGraph> cuda_frame;
  std::shared_ptr<cuda::BindingFrameReplay> cuda_binding_replay;
  using CudaStep = std::variant<std::unique_ptr<cuda::FixedGraph>,
                                std::shared_ptr<NativeOperation>>;
  std::vector<CudaStep> cuda_steps;
  // Retire submitted native work before releasing graph/provider owners.
  std::unique_ptr<cuda::OrderedSubmissions> cuda_ordered;
  bool cuda_failed{false};
#endif
#ifdef TI_WITH_VULKAN
  std::vector<std::unique_ptr<gfx::GraphReplayRegistration>> vulkan_graphs;
  gfx::GfxRuntime *vulkan_runtime{nullptr};
#endif
};

namespace {
void check_pointer_copy(void *destination, void *source, std::size_t bytes) {
  auto dst = reinterpret_cast<std::uintptr_t>(destination);
  auto src = reinterpret_cast<std::uintptr_t>(source);
  if (bytes && (!dst || !src || (dst <= src ? src - dst : dst - src) < bytes))
    throw std::invalid_argument(
        "Fixed copy pointers must be live and nonoverlapping");
}
#ifdef TI_WITH_LLVM
void *cpu_pointer(DevicePtr ptr, std::size_t bytes) {
  auto *device = dynamic_cast<cpu::CpuDevice *>(ptr.device);
  void *result = nullptr;
  if (!device || device->map_range_for_cpu_native(ptr, bytes, &result) !=
                     RhiResult::success)
    throw std::invalid_argument("Fixed CPU operation requires owned storage");
  return result;
}
#endif
#ifdef TI_WITH_CUDA
void *cuda_pointer(DevicePtr ptr) {
  auto *device = dynamic_cast<cuda::CudaDevice *>(ptr.device);
  if (!device)
    throw std::invalid_argument("Fixed CUDA operation requires CUDA storage");
  auto *base = device->get_memory_addr(ptr);
  return ptr.offset ? static_cast<char *>(base) + ptr.offset : base;
}
#endif
}  // namespace

FixedExecution::FixedExecution(
    Arch arch,
    Device &device,
    gfx::GfxRuntime *gfx_runtime,
    const std::vector<Operation> &operations,
    NativeRecordingState *sequence_state,
    std::shared_ptr<RuntimeFaultDomain> fault_domain,
    std::shared_ptr<cuda::BindingFrameReplay> binding_replay,
    std::shared_ptr<void> binding_operands)
    : state_(std::make_unique<State>()) {
  state_->fault_domain = std::move(fault_domain);
  if (binding_replay) {
    if (arch != Arch::cuda || !binding_operands)
      throw std::invalid_argument(
          "Shared binding replay requires CUDA and owned operands");
#ifdef TI_WITH_CUDA
    if (!binding_replay->belongs_to(state_->fault_domain))
      throw std::invalid_argument(
          "CUDA binding replay belongs to another runtime owner");
    for (const auto &operation : operations) {
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        if (dispatch->bounded_extent != kDeviceNullPtr ||
            dispatch->indirect_dispatch != kDeviceNullPtr ||
            dispatch->adaptive_block_dim)
          throw std::invalid_argument(
              "CUDA binding replay requires fixed dispatches");
      } else if (const auto *native =
                     std::get_if<std::shared_ptr<NativeOperation>>(
                         &operation)) {
        if (!*native ||
            (*native)->execution_mode() != NativeOperationMode::cuda_capture)
          throw std::invalid_argument(
              "CUDA binding replay requires capture-compatible native "
              "operations");
      } else if (!std::holds_alternative<BufferCopy>(operation) &&
                 !std::holds_alternative<BufferFill>(operation)) {
        throw std::invalid_argument(
            "CUDA binding replay requires a flat operation sequence");
      }
    }
#else
    throw std::invalid_argument(
        "CUDA binding replay is unavailable in this runtime");
#endif
  }
  if (state_->fault_domain)
    state_->fault_domain->throw_if_submission_disallowed(
        "prepare fixed execution");
  if (operations.empty())
    throw std::invalid_argument("Fixed execution requires a nonempty batch");
  // Validate every descriptor before preparing backend state or commands.
  std::unordered_set<const NativeOperation *> owners;
  NativeRecordingState recording_state =
      sequence_state ? *sequence_state : NativeRecordingState{};
  std::unordered_set<const void *> active_controls;
  bool has_host_selection = false;
  std::function<void(const std::vector<Operation> &, NativeRecordingState &,
                     bool)>
      validate;
  validate = [&](const std::vector<Operation> &operations,
                 NativeRecordingState &recording_state, bool inside_control) {
    for (const auto &operation : operations) {
      if (const auto *value =
              std::get_if<std::shared_ptr<const Branch>>(&operation)) {
        const auto &branch = *value;
        if (!branch || branch->selector.device != &device ||
            branch->selector.offset % 4 ||
            (!arch_is_cpu(arch) && arch != Arch::cuda &&
             !(arch == Arch::vulkan && branch->mode == BranchExecutionMode::host)))
          throw std::invalid_argument(
              "Fixed branch requires a supported lowering and aligned selector storage");
        const bool host = branch->mode == BranchExecutionMode::host;
        const bool masked = branch->mode == BranchExecutionMode::cuda_masked;
        if (!host && !masked && branch->mode != BranchExecutionMode::native)
          throw std::invalid_argument("Unsupported fixed branch execution mode");
        if (masked) {
          const auto flat = [](const auto &region) {
            return std::all_of(region.begin(), region.end(), [](const auto &op) {
              const auto *dispatch = std::get_if<Dispatch>(&op);
              return dispatch && !dispatch->indirect_dispatch.alloc_id &&
                     !dispatch->bounded_extent.alloc_id &&
                     !dispatch->adaptive_block_dim;
            });
          };
          if (arch != Arch::cuda || inside_control || !flat(branch->condition) ||
              !std::all_of(branch->bodies.begin(), branch->bodies.end(), flat))
            throw std::invalid_argument(
                "Selected CUDA masked branch requires flat kernel regions on CUDA");
        }
        if (host && inside_control && !arch_is_cpu(arch))
          throw std::invalid_argument(
              "Native GPU control cannot contain host selection");
        has_host_selection |= host;
        branch->selection.validate(branch->bodies.size());
        if (!active_controls.insert(branch.get()).second)
          throw std::invalid_argument(
              "Fixed control descriptors contain a cycle");
        // Conditions execute exactly once. Only native GPU control restricts
        // their execution mode; conditional fact joins apply to both modes.
        const bool native_control = !host;
        validate(branch->condition, recording_state, native_control);
        auto joined = recording_state.initialized;
        bool first = branch->selection.has_default;
        for (const auto &body : branch->bodies) {
          auto facts = recording_state.control_branch();
          validate(body, facts, native_control);
          if (first) {
            joined = std::move(facts.initialized);
            first = false;
          } else {
            for (auto it = joined.begin(); it != joined.end();) {
              if (facts.initialized.count(*it))
                ++it;
              else
                it = joined.erase(it);
            }
          }
        }
        // With no default, the untouched incoming state is also a path. Only
        // facts true on every path reach the suffix; opaque facts are
        // unchanged.
        recording_state.initialized = std::move(joined);
        active_controls.erase(branch.get());
        continue;
      }
      if (const auto *value =
              std::get_if<std::shared_ptr<const Parallel>>(&operation)) {
        const auto &parallel = *value;
        if (arch != Arch::cuda || inside_control || !parallel ||
            parallel->branches.size() < 2)
          throw std::invalid_argument(
              "Fixed CUDA parallel requires a flat CUDA region and at least "
              "two branches");
        for (const auto &branch : parallel->branches) {
          if (branch.empty())
            throw std::invalid_argument(
                "Fixed CUDA parallel requires nonempty branches");
          for (const auto &item : branch) {
            const auto *dispatch = std::get_if<Dispatch>(&item);
            if (!dispatch || dispatch->bounded_extent != kDeviceNullPtr ||
                dispatch->indirect_dispatch != kDeviceNullPtr)
              throw std::invalid_argument(
                  "Fixed CUDA parallel requires ordinary kernel branches");
          }
          auto branch_state = recording_state.control_branch();
          validate(branch, branch_state, true);
        }
        continue;
      }
      if (const auto *value =
              std::get_if<std::shared_ptr<const While>>(&operation)) {
        const auto &loop = *value;
        if (!loop || loop->predicate.device != &device ||
            loop->predicate.offset % 4 || loop->body.empty())
          throw std::invalid_argument(
              "Fixed while requires aligned predicate storage and nonempty "
              "bodies");
        if ((arch == Arch::vulkan) != bool(loop->vulkan))
          throw std::invalid_argument(
              "Fixed while requires a matching selected Vulkan recipe on "
              "Vulkan only");
        if (loop->cuda_control) {
          const auto flat = [](const auto &region) {
            return std::all_of(region.begin(), region.end(), [](const auto &op) {
              return std::holds_alternative<Dispatch>(op);
            });
          };
          if (arch != Arch::cuda || inside_control || !flat(loop->condition) ||
              !flat(loop->body))
            throw std::invalid_argument(
                "Selected CUDA control requires flat kernel "
                "regions on CUDA");
          loop->cuda_control->validate(
              loop->condition.size(), loop->condition.size() + loop->body.size(),
              loop->continue_while_nonzero);
          for (const auto &inner : loop->cuda_control->inners)
            if (inner.predicate.device != &device ||
                !inner.predicate.alloc_id || inner.predicate.offset % 4)
              throw std::invalid_argument(
                  "Invalid CUDA nested predicate storage");
        }
        if (loop->vulkan) {
          const auto &control = *loop->vulkan;
          const auto valid = [&](DevicePtr ptr) {
            return ptr.device == &device && ptr.alloc_id && ptr.offset % 4 == 0;
          };
          const auto flat = [](const auto &region) {
            return std::all_of(region.begin(), region.end(),
                               [](const auto &op) {
                                 return std::holds_alternative<Dispatch>(op);
                               });
          };
          if (inside_control || !loop->continue_while_nonzero ||
              !valid(control.counter) ||
              (control.status != kDeviceNullPtr && !valid(control.status)) ||
              !flat(loop->condition) || !flat(loop->body))
            throw std::invalid_argument(
                "Invalid flat Vulkan structured recording recipe");
          if (control.nested) {
            if (!control.recipe.chunks.empty())
              throw std::invalid_argument(
                  "Nested Vulkan control cannot carry flat chunks");
            control.nested->validate(loop->condition.size(),
                                     loop->condition.size() + loop->body.size(),
                                     loop->max_iterations);
            for (const auto &inner : control.nested->inners)
              if (!valid(inner.predicate) || !valid(inner.counter) ||
                  (inner.status && !valid(*inner.status)))
                throw std::invalid_argument(
                    "Invalid Vulkan inner control storage");
          } else {
            control.recipe.validate(loop->max_iterations);
          }
        }
        if (!active_controls.insert(loop.get()).second)
          throw std::invalid_argument(
              "Fixed control descriptors contain a cycle");
        auto condition = recording_state.control_branch();
        validate(loop->condition, condition, true);
        // The initial condition always executes, including a zero budget. Body
        // facts cannot escape a possibly zero-trip loop.
        recording_state.initialized = condition.initialized;
        auto body = condition;
        validate(loop->body, body, true);
        // Monotonic initialization plus stateless operations is loop-invariant;
        // do not rewalk or unroll regions to validate another iteration.
        for (const auto *resource : condition.initialized)
          if (!body.initialized.count(resource))
            throw std::invalid_argument(
                "Loop providers cannot invalidate established initialization "
                "facts");
        active_controls.erase(loop.get());
        continue;
      }
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        if (dispatch->adaptive_block_dim &&
            (arch != Arch::cuda || dispatch->bounded_extent == kDeviceNullPtr ||
             !dispatch->bounded_binding_identity))
          throw std::invalid_argument(
              "Adaptive launch requires a CUDA bounded binding identity");
        if (dispatch->bounded_extent != kDeviceNullPtr &&
            ((!arch_is_cpu(arch) && arch != Arch::cuda) || inside_control ||
             dispatch->indirect_dispatch != kDeviceNullPtr ||
             dispatch->bounded_extent.device != &device ||
             dispatch->bounded_extent.offset % 4 || !dispatch->bounded_capacity ||
             dispatch->bounded_capacity >
                 std::numeric_limits<std::int32_t>::max()))
          throw std::invalid_argument(
              "Exact bounded launch requires a flat CPU/CUDA range and i32 capacity");
        if (dispatch->indirect_dispatch != kDeviceNullPtr &&
            (arch != Arch::vulkan || inside_control ||
             dispatch->indirect_dispatch.device != &device ||
             dispatch->indirect_dispatch.offset % 4))
          throw std::invalid_argument(
              "Indirect dispatch requires a flat Vulkan recording and aligned packet");
        if (!dispatch->kernel || !dispatch->context ||
            dispatch->kernel->ret_size ||
            !dispatch->context->argpack_ptrs.empty())
          throw std::invalid_argument(
              "Fixed execution requires loaded entries without returns or "
              "ArgPacks");
        std::set<std::vector<int>> image_arguments;
        for (const auto &image : dispatch->images) {
          if (!image || !image_arguments.insert(image->argument()).second)
            throw std::invalid_argument("Duplicate or null kernel image owner");
          image->validate(arch, device, *dispatch->kernel, *dispatch->context,
                          recording_state);
          state_->image_bindings.push_back(image);
        }
        for (const auto &[index, param] : dispatch->kernel->nested_parameters) {
          if ((param.ptype == ParameterType::kTexture &&
               !image_arguments.count(index)) ||
              param.ptype == ParameterType::kRWTexture ||
              param.ptype == ParameterType::kTextureCollection)
            throw std::invalid_argument(
                "Fixed textures require generation-qualified resource owners");
        }
      } else if (const auto *copy = std::get_if<BufferCopy>(&operation)) {
        if (copy->source.device != &device ||
            copy->destination.device != &device)
          throw std::invalid_argument(
              "Fixed copy storage belongs to another device");
      } else if (const auto *fill = std::get_if<BufferFill>(&operation)) {
        if (fill->destination.device != &device ||
            fill->destination.offset % 4 || fill->bytes % 4)
          throw std::invalid_argument(
              "Fixed fill requires aligned 32-bit storage");
      } else {
        const auto &native =
            std::get<std::shared_ptr<NativeOperation>>(operation);
        if (!native || native->arch() != arch || &native->device() != &device)
          throw std::invalid_argument(
              "Fixed native operation belongs to another execution device");
        const auto mode = native->execution_mode();
        if (!((arch_is_cpu(arch) && mode == NativeOperationMode::cpu) ||
              (arch == Arch::cuda &&
               (mode == NativeOperationMode::cuda_capture ||
                mode == NativeOperationMode::cuda_ordered ||
                mode == NativeOperationMode::cuda_synchronous)) ||
              (arch == Arch::vulkan &&
               mode == NativeOperationMode::vulkan_inline)))
          throw std::invalid_argument(
              "Fixed native operation mode/backend mismatch");
        if (inside_control && arch == Arch::cuda &&
            mode != NativeOperationMode::cuda_capture)
          throw std::invalid_argument(
              "GPU control cannot contain host-invoked native operations");
        native->validate_recording(recording_state);
        if (owners.insert(native.get()).second) {
          state_->native_operations.push_back(native);
          if (native->workspace_bytes() >
              std::numeric_limits<std::size_t>::max() - state_->workspace_bytes)
            throw std::invalid_argument(
                "Native workspace accounting overflows address space");
          state_->workspace_bytes += native->workspace_bytes();
        }
      }
    }
  };
  validate(operations, recording_state, false);
#ifdef TI_WITH_CUDA
  if (binding_replay) {
    std::vector<std::vector<std::uint64_t>> invariants;
    invariants.reserve(state_->native_operations.size());
    for (const auto &operation : state_->native_operations)
      invariants.push_back(operation->cuda_binding_frame_invariants());
    binding_replay->validate_invariants(std::move(invariants));
  }
#endif
  state_->native_workspace_bytes = state_->workspace_bytes;
  if (has_host_selection && !arch_is_cpu(arch)) {
    if (arch != Arch::cuda && arch != Arch::vulkan)
      throw std::invalid_argument("Host selection requires CUDA or Vulkan");
    state_->host_cuda = arch == Arch::cuda;
    // The full validation above must finish before any backend allocation or
    // capture. Child blocks receive incoming facts, not the final joined facts.
    std::function<std::function<std::int32_t()>(DevicePtr)> reader;
#ifdef TI_WITH_CUDA
    if (arch == Arch::cuda) {
      if (!state_->fault_domain)
        state_->fault_domain = std::make_shared<RuntimeFaultDomain>(arch, 0);
      reader = [](DevicePtr ptr) {
        return [source = cuda_pointer(ptr)] {
          std::int32_t value;
          CUDADriver::get_instance().memcpy_device_to_host(&value, source, 4);
          return value;
        };
      };
    }
#endif
#ifdef TI_WITH_VULKAN
    if (arch == Arch::vulkan) {
      if (!gfx_runtime)
        throw std::invalid_argument("Host Vulkan selection requires a runtime");
      auto staging = device.allocate_memory_unique(
          {4, false, true, false, AllocUsage::None});
      if (staging.second != RhiResult::success)
        throw std::runtime_error("Cannot allocate host selector staging");
      state_->host_staging = std::move(staging.first);
      state_->workspace_bytes += 4;
      reader = [&, staging = state_->host_staging.get()](DevicePtr ptr) {
        auto command = device.get_compute_stream()->new_command_list_unique();
        if (command.second != RhiResult::success)
          throw std::runtime_error("Cannot prepare host selector readback");
        std::shared_ptr<CommandList> commands(std::move(command.first));
        commands->memory_barrier();
        commands->buffer_copy(staging->get_ptr(), ptr, 4);
        commands->memory_barrier();
        return [device = &device, gfx_runtime, staging, commands] {
          (void)gfx_runtime->flush_if_pending();
          device->get_compute_stream()->submit_synced(commands.get());
          void *mapped = nullptr;
          if (device->map(*staging, &mapped) != RhiResult::success)
            throw std::runtime_error("Cannot map host selector staging");
          std::int32_t value;
          std::memcpy(&value, mapped, 4);
          device->unmap(*staging);
          return value;
        };
      };
    }
#endif
    if (!reader)
      throw std::invalid_argument("Host selection backend is not compiled");
    std::function<State::HostSteps(const std::vector<Operation> &,
                                  NativeRecordingState &)>
        prepare;
    prepare = [&](const std::vector<Operation> &region,
                  NativeRecordingState &facts) {
      State::HostSteps steps;
      std::vector<Operation> block;
      const auto flush = [&] {
        if (block.empty())
          return;
        auto prepared = std::make_shared<FixedExecution>(
            arch, device, gfx_runtime, block, &facts, state_->fault_domain);
        // Shared provider workspace was counted once by the outer validation.
        const auto extra = prepared->workspace_bytes() -
                           prepared->state_->native_workspace_bytes;
        if (prepared->argument_bytes() >
                std::numeric_limits<std::size_t>::max() - state_->bytes ||
            extra > std::numeric_limits<std::size_t>::max() -
                        state_->workspace_bytes)
          throw std::invalid_argument("Host control storage size overflow");
        state_->bytes += prepared->argument_bytes();
        state_->workspace_bytes += extra;
        steps.push_back([prepared] { prepared->run(); });
        block.clear();
      };
      for (const auto &operation : region) {
        const auto *value =
            std::get_if<std::shared_ptr<const Branch>>(&operation);
        if (!value || (*value)->mode != BranchExecutionMode::host) {
          block.push_back(operation);
          continue;
        }
        flush();
        const auto &branch = **value;
        struct PreparedBranch {
          std::function<std::int32_t()> read;
          State::HostSteps condition;
          std::vector<State::HostSteps> bodies;
        };
        auto prepared = std::make_shared<PreparedBranch>();
        prepared->read = reader(branch.selector);
        prepared->condition = prepare(branch.condition, facts);
        auto joined = facts.initialized;
        bool first = branch.selection.has_default;
        for (const auto &body : branch.bodies) {
          auto selected = facts.control_branch();
          prepared->bodies.push_back(prepare(body, selected));
          if (first) {
            joined = std::move(selected.initialized);
            first = false;
          } else {
            for (auto it = joined.begin(); it != joined.end();) {
              if (selected.initialized.count(*it))
                ++it;
              else
                it = joined.erase(it);
            }
          }
        }
        facts.initialized = std::move(joined);
        steps.push_back([prepared, selection = branch.selection] {
          State::run_host_steps(prepared->condition);
          const auto index =
              selection.select(prepared->read(), prepared->bodies.size());
          if (index < prepared->bodies.size())
            State::run_host_steps(prepared->bodies[index]);
        });
      }
      flush();
      return steps;
    };
    auto facts = sequence_state ? *sequence_state : NativeRecordingState{};
    state_->host_steps = prepare(operations, facts);
    state_->kind = 5;
    if (sequence_state)
      *sequence_state = std::move(recording_state);
    return;
  }
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    std::function<std::vector<State::CpuStep>(const std::vector<Operation> &)>
        prepare;
    prepare = [&](const std::vector<Operation> &operations) {
      std::vector<State::CpuStep> steps;
      for (const auto &operation : operations) {
        if (const auto *value =
                std::get_if<std::shared_ptr<const Branch>>(&operation)) {
          const auto &branch = **value;
          auto *selector =
              static_cast<std::int32_t *>(cpu_pointer(branch.selector, 4));
          struct PreparedBranch {
            std::vector<State::CpuStep> condition;
            std::vector<std::vector<State::CpuStep>> bodies;
          };
          auto prepared = std::make_shared<PreparedBranch>();
          prepared->condition = prepare(branch.condition);
          for (const auto &body : branch.bodies)
            prepared->bodies.push_back(prepare(body));
          steps.push_back(
              {nullptr, {}, [selector, selection = branch.selection, prepared] {
                 State::run_cpu_steps(prepared->condition);
                 const auto index =
                     selection.select(*selector, prepared->bodies.size());
                 if (index < prepared->bodies.size())
                   State::run_cpu_steps(prepared->bodies[index]);
               }});
          continue;
        }
        if (const auto *value =
                std::get_if<std::shared_ptr<const While>>(&operation)) {
          const auto &loop = **value;
          auto *predicate =
              static_cast<std::uint32_t *>(cpu_pointer(loop.predicate, 4));
          auto condition = std::make_shared<std::vector<State::CpuStep>>(
              prepare(loop.condition));
          auto body =
              std::make_shared<std::vector<State::CpuStep>>(prepare(loop.body));
          steps.push_back(
              {nullptr,
               {},
               [condition, body, predicate, budget = loop.max_iterations,
                nonzero = loop.continue_while_nonzero] {
                 State::run_cpu_steps(*condition);
                 for (std::uint64_t iteration = 0;
                      iteration < budget && ((*predicate != 0) == nonzero);
                      ++iteration) {
                   State::run_cpu_steps(*body);
                   State::run_cpu_steps(*condition);
                 }
               }});
          continue;
        }
        if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
          auto *loaded = dynamic_cast<llvm_aot::KernelImpl *>(dispatch->kernel);
          auto *launcher =
              loaded ? dynamic_cast<cpu::KernelLauncher *>(&loaded->launcher())
                     : nullptr;
          if (!launcher)
            throw std::invalid_argument("Fixed CPU entry is not registered");
          if (steps.empty() || steps.back().launcher != launcher)
            steps.push_back({launcher, {}, {}});
          steps.back().launches.push_back(
              launcher->prepare_launch(loaded->handle(), *dispatch->context));
          state_->bytes += dispatch->context->arg_buffer_size +
                           dispatch->context->arg_buffer_prefix_size();
        } else if (const auto *native =
                       std::get_if<std::shared_ptr<NativeOperation>>(
                           &operation)) {
          steps.push_back(
              {nullptr, {}, [native = *native] { native->run_cpu(); }});
        } else {
          State::CpuStep step;
          if (const auto *copy = std::get_if<BufferCopy>(&operation)) {
            auto bytes = copy->bytes;
            auto *dst = cpu_pointer(copy->destination, bytes);
            auto *src = cpu_pointer(copy->source, bytes);
            check_pointer_copy(dst, src, bytes);
            step.native = [dst, src, bytes] {
              if (bytes)
                std::memcpy(dst, src, bytes);
            };
          } else {
            const auto &fill = std::get<BufferFill>(operation);
            auto *dst = static_cast<std::uint32_t *>(
                cpu_pointer(fill.destination, fill.bytes));
            auto count = fill.bytes / 4;
            auto value = fill.value;
            step.native = [dst, count, value] {
              if (count)
                std::fill_n(dst, count, value);
            };
          }
          steps.push_back(std::move(step));
        }
      }
      return steps;
    };
    state_->cpu_steps = prepare(operations);
    if (sequence_state)
      *sequence_state = std::move(recording_state);
    return;
  }
#endif
#ifdef TI_WITH_CUDA
  if (arch == Arch::cuda) {
    if (!state_->fault_domain &&
        std::any_of(state_->native_operations.begin(), state_->native_operations.end(),
                    [](const auto &op) {
                      return op->execution_mode() == NativeOperationMode::cuda_ordered;
                    }))
      state_->fault_domain = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 0);
    using CudaOperation = cuda::FixedGraph::Operation;
    std::size_t ordered_operations = 0;
    bool synchronous_operations = false;
    auto account = [&](const cuda::FixedGraph &graph) {
      if (graph.argument_bytes() >
              std::numeric_limits<std::size_t>::max() - state_->bytes ||
          graph.control_bytes() >
              std::numeric_limits<std::size_t>::max() - state_->workspace_bytes)
        throw std::invalid_argument("Fixed CUDA storage size overflow");
      state_->bytes += graph.argument_bytes();
      state_->workspace_bytes += graph.control_bytes();
      if (graph.capture_bytes() >
          std::numeric_limits<std::size_t>::max() - state_->workspace_bytes)
        throw std::invalid_argument("Fixed CUDA capture storage size overflow");
      state_->workspace_bytes += graph.capture_bytes();
    };
    auto flush = [&](std::vector<CudaOperation> &native) {
      if (native.empty())
        return;
      auto graph =
          std::make_unique<cuda::FixedGraph>(native, state_->fault_domain);
      account(*graph);
      state_->cuda_steps.emplace_back(std::move(graph));
      native.clear();
    };
    std::function<std::vector<CudaOperation>(const std::vector<Operation> &)>
        prepare;
    prepare = [&](const std::vector<Operation> &operations) {
      std::vector<CudaOperation> native;
      for (const auto &operation : operations) {
        if (const auto *branch =
                std::get_if<std::shared_ptr<const Branch>>(&operation)) {
          auto result = std::make_shared<cuda::FixedGraph::Branch>();
          result->selector = cuda_pointer((*branch)->selector);
          result->selection = (*branch)->selection;
          result->mode = (*branch)->mode;
          result->condition = prepare((*branch)->condition);
          for (const auto &body : (*branch)->bodies)
            result->bodies.push_back(prepare(body));
          native.emplace_back(std::move(result));
        } else if (const auto *parallel =
                       std::get_if<std::shared_ptr<const Parallel>>(
                           &operation)) {
          auto result = std::make_shared<cuda::FixedGraph::Parallel>();
          for (const auto &branch : (*parallel)->branches)
            result->branches.push_back(prepare(branch));
          native.emplace_back(std::move(result));
        } else if (const auto *value =
                       std::get_if<std::shared_ptr<const While>>(&operation)) {
          const auto &loop = **value;
          auto result = std::make_shared<cuda::FixedGraph::While>();
          result->predicate = cuda_pointer(loop.predicate);
          result->max_iterations = loop.max_iterations;
          result->continue_while_nonzero = loop.continue_while_nonzero;
          result->condition = prepare(loop.condition);
          result->body = prepare(loop.body);
          if (loop.cuda_control) {
            result->recipe.emplace();
            result->recipe->encoding = loop.cuda_control->encoding;
            for (const auto &inner : loop.cuda_control->inners)
              result->recipe->inners.push_back(
                  {cuda_pointer(inner.predicate), inner.condition_begin,
                   inner.body_begin, inner.end, inner.max_iterations});
          }
          native.emplace_back(std::move(result));
        } else if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
          auto *loaded = dynamic_cast<llvm_aot::KernelImpl *>(dispatch->kernel);
          auto *launcher =
              loaded ? dynamic_cast<cuda::KernelLauncher *>(&loaded->launcher())
                     : nullptr;
          if (!launcher)
            throw std::invalid_argument("Fixed CUDA entry is not registered");
          native.emplace_back(cuda::FixedGraph::Dispatch{
              launcher, loaded->handle(), dispatch->context, {}, {},
              dispatch->bounded_extent == kDeviceNullPtr
                  ? nullptr
                  : cuda_pointer(dispatch->bounded_extent),
              dispatch->bounded_capacity, dispatch->adaptive_block_dim,
              dispatch->grouped_update, dispatch->bounded_binding_identity});
        } else if (const auto *external =
                       std::get_if<std::shared_ptr<NativeOperation>>(
                           &operation)) {
          const auto mode = (*external)->execution_mode();
          if (mode == NativeOperationMode::cuda_synchronous ||
              mode == NativeOperationMode::cuda_ordered) {
            ordered_operations += mode == NativeOperationMode::cuda_ordered;
            synchronous_operations |= mode == NativeOperationMode::cuda_synchronous;
            flush(native);
            state_->cuda_steps.emplace_back(*external);
            continue;
          }
          native.emplace_back(cuda::FixedGraph::Dispatch{
              nullptr,
              {},
              nullptr,
              [external = *external](void *stream) {
                external->record_cuda(stream);
              },
              [external = *external] {
                return external->take_cuda_capture_resources();
              }});
        } else if (const auto *copy = std::get_if<BufferCopy>(&operation)) {
          auto *dst = cuda_pointer(copy->destination);
          auto *src = cuda_pointer(copy->source);
          auto bytes = copy->bytes;
          check_pointer_copy(dst, src, bytes);
          native.emplace_back(cuda::FixedGraph::Dispatch{
              nullptr, {}, nullptr, [dst, src, bytes](void *stream) {
                if (bytes)
                  CUDADriver::get_instance().memcpy_device_to_device_async(
                      dst, src, bytes, stream);
              }});
        } else {
          const auto &fill = std::get<BufferFill>(operation);
          auto *dst = cuda_pointer(fill.destination);
          auto count = fill.bytes / 4;
          auto value = fill.value;
          native.emplace_back(cuda::FixedGraph::Dispatch{
              nullptr, {}, nullptr, [dst, count, value](void *stream) {
                if (count)
                  CUDADriver::get_instance().memsetd32_async(dst, value, count,
                                                             stream);
              }});
        }
      }
      return native;
    };
    auto native = prepare(operations);
    if (binding_replay) {
      struct FrameOperands {
        std::shared_ptr<void> bindings;
        std::vector<std::shared_ptr<NativeOperation>> operations;
        std::vector<std::shared_ptr<NativeImageKernelBinding>> images;
      };
      auto retained = std::make_shared<FrameOperands>();
      retained->bindings = std::move(binding_operands);
      retained->operations = state_->native_operations;
      retained->images = state_->image_bindings;
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      state_->cuda_frame = cuda::FixedGraph::capture_binding_frame(
          native, std::move(retained), state_->fault_domain);
      account(*state_->cuda_frame);
      binding_replay->prepare(state_->cuda_frame->binding_source(),
                              state_->cuda_frame);
      state_->cuda_binding_replay = std::move(binding_replay);
      state_->kind = 1;
    } else if (state_->cuda_steps.empty()) {
      state_->cuda_graph =
          std::make_unique<cuda::FixedGraph>(native, state_->fault_domain);
      state_->kind = 1;
      account(*state_->cuda_graph);
    } else {
      flush(native);
      state_->kind = synchronous_operations ? 3 : 4;
      if (ordered_operations)
        state_->cuda_ordered = std::make_unique<cuda::OrderedSubmissions>(
            ordered_operations, state_->fault_domain);
    }
    if (sequence_state)
      *sequence_state = std::move(recording_state);
    return;
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    std::vector<gfx::GfxRuntime::GraphRecordingOperation> native;
    const auto bind_dispatch = [&](const Dispatch &dispatch) {
      auto *loaded = dynamic_cast<gfx::KernelImpl *>(dispatch.kernel);
      if (!loaded || (gfx_runtime && gfx_runtime != &loaded->runtime()))
        throw std::invalid_argument(
            "Fixed Vulkan entries must share a runtime");
      gfx_runtime = &loaded->runtime();
      return gfx::GfxRuntime::GraphDispatch{
          loaded->handle(), dispatch.context, dispatch.indirect_dispatch};
    };
    const auto retain =
        [&](std::unique_ptr<gfx::GraphReplayRegistration> graph) {
          const auto info = graph->snapshot_stats();
          state_->bytes += info.known_persistent_argument_bytes -
                           info.known_persistent_control_bytes;
          state_->workspace_bytes += info.known_persistent_control_bytes;
          state_->vulkan_graphs.push_back(std::move(graph));
        };
    const auto flush = [&] {
      if (native.empty())
        return;
      if (!gfx_runtime)
        throw std::invalid_argument(
            "Fixed Vulkan operations require a runtime");
      retain(gfx_runtime->prepare_fixed_graph(native, {}));
      native.clear();
    };
    for (const auto &operation : operations) {
      if (const auto *value =
              std::get_if<std::shared_ptr<const While>>(&operation)) {
        const auto &loop = **value;
        if (!loop.max_iterations) {
          for (const auto &op : loop.condition)
            native.push_back({bind_dispatch(std::get<Dispatch>(op)), {}});
          continue;
        }
        flush();
        std::vector<gfx::GfxRuntime::GraphDispatch> dispatches;
        const auto append = [&](const auto &region) {
          for (const auto &op : region)
            dispatches.push_back(bind_dispatch(std::get<Dispatch>(op)));
        };
        append(loop.condition);
        append(loop.body);
        if (loop.vulkan->nested) {
          gfx::GfxRuntime::GraphNestedStructuredControl control;
          control.outer_predicate = loop.predicate;
          control.outer_counter = loop.vulkan->counter;
          control.outer_status = loop.vulkan->status;
          control.outer_has_status = control.outer_status != kDeviceNullPtr;
          control.outer_condition_dispatch_count = loop.condition.size();
          control.outer_max_iterations =
              static_cast<std::uint32_t>(loop.max_iterations);
          for (const auto &source : loop.vulkan->nested->inners) {
            gfx::GfxRuntime::GraphNestedStructuredInnerControl inner;
            inner.inner_predicate = source.predicate;
            inner.inner_counter = source.counter;
            inner.inner_has_status = bool(source.status);
            inner.inner_status = source.status.value_or(kDeviceNullPtr);
            inner.inner_condition_dispatch_begin = source.condition_begin;
            inner.inner_body_dispatch_begin = source.body_begin;
            inner.inner_dispatch_end = source.end;
            inner.inner_max_iterations = source.max_iterations;
            inner.inner_chunk_size = source.chunk_size;
            control.inner_controls.push_back(inner);
          }
          retain(gfx_runtime->prepare_fixed_nested_structured_graph(
              dispatches, control, {}));
          continue;
        }
        append(loop.condition);
        gfx::GfxRuntime::GraphStructuredControl control;
        control.predicate = loop.predicate;
        control.counter = loop.vulkan->counter;
        control.status = loop.vulkan->status;
        control.has_status = control.status != kDeviceNullPtr;
        control.initial_dispatch_count = loop.condition.size();
        for (const auto &chunk : loop.vulkan->recipe.chunks) {
          control.max_iterations = chunk.iterations;
          control.strategy = chunk.strategy;
          retain(gfx_runtime->prepare_fixed_structured_graph(dispatches,
                                                             control, {}));
          control.execute_initial_dispatches = false;
        }
      } else if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        native.push_back({bind_dispatch(*dispatch), {}});
      } else if (const auto *external =
                     std::get_if<std::shared_ptr<NativeOperation>>(
                         &operation)) {
        gfx::GfxRuntime::GraphRecordingOperation op;
        op.inline_recording = true;
        op.images = (*external)->vulkan_image_bindings();
        op.external = [external = *external](Device *device,
                                             CommandList *commands) {
          commands->memory_barrier();
          external->record_vulkan(device, commands);
          commands->memory_barrier();
        };
        native.push_back(std::move(op));
      } else {
        gfx::GfxRuntime::GraphRecordingOperation op;
        op.inline_recording = true;
        if (const auto *copy = std::get_if<BufferCopy>(&operation)) {
          if (copy->source.offset % 4 || copy->destination.offset % 4 ||
              copy->bytes % 4)
            throw std::invalid_argument(
                "Vulkan copies require 4-byte alignment");
          op.buffers = {{copy->source, copy->destination}};
          op.external = [copy = *copy](Device *, CommandList *commands) {
            if (copy.bytes) {
              commands->memory_barrier();
              commands->buffer_copy(copy.destination, copy.source, copy.bytes);
              commands->memory_barrier();
            }
          };
        } else {
          const auto &fill = std::get<BufferFill>(operation);
          op.buffers = {{fill.destination}};
          op.external = [fill](Device *, CommandList *commands) {
            if (fill.bytes) {
              commands->memory_barrier();
              commands->buffer_fill(fill.destination, fill.bytes, fill.value);
              commands->memory_barrier();
            }
          };
        }
        native.push_back(std::move(op));
      }
    }
    flush();
    state_->vulkan_runtime = gfx_runtime;
    state_->kind = 2;
    if (sequence_state)
      *sequence_state = std::move(recording_state);
    return;
  }
#endif
  throw std::invalid_argument("Fixed execution requires CPU, CUDA or Vulkan");
}

FixedExecution::~FixedExecution() = default;
void FixedExecution::run() {
  if (state_->fault_domain)
    state_->fault_domain->throw_if_submission_disallowed("run fixed execution");
  try {
    run_impl();
  } catch (const BackendRuntimeError &error) {
    if (state_->fault_domain)
      state_->fault_domain->report_backend_error(error, 0);
    throw;
  }
}
void FixedExecution::retire_completed_binding(bool backend_safe) {
#ifdef TI_WITH_CUDA
  if (state_->cuda_binding_replay) {
    auto lock = CUDAContext::get_instance().get_submission_lock_guard();
    state_->cuda_binding_replay->release_completed_frame(
        state_->cuda_frame.get(),
        backend_safe && (!state_->fault_domain ||
                         state_->fault_domain->backend_calls_safe()));
  }
#endif
}

void FixedExecution::run_impl() {
  if (!state_->host_steps.empty()) {
    if (state_->host_failed)
      throw std::runtime_error(
          "Host control execution failed; prepare a new instance");
    try {
#ifdef TI_WITH_CUDA
      if (state_->host_cuda) {
        auto &cuda = CUDAContext::get_instance();
        auto guard = cuda.get_guard();
        auto lock = cuda.get_submission_lock_guard();
        State::run_host_steps(state_->host_steps);
      } else
#endif
      {
        State::run_host_steps(state_->host_steps);
      }
    } catch (...) {
      state_->host_failed = true;
      throw;
    }
    return;
  }
#ifdef TI_WITH_LLVM
  State::run_cpu_steps(state_->cpu_steps);
#endif
#ifdef TI_WITH_CUDA
  if (state_->cuda_binding_replay) {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    state_->cuda_binding_replay->run(state_->cuda_frame->binding_source(),
                                     state_->cuda_frame);
  }
  if (state_->cuda_graph)
    state_->cuda_graph->run();
  if (!state_->cuda_steps.empty()) {
    if (state_->cuda_failed)
      throw std::runtime_error(
          "Host-invoked CUDA execution failed; prepare a new instance");
    auto &cuda = CUDAContext::get_instance();
    auto guard = cuda.get_guard();
    auto lock = cuda.get_submission_lock_guard();
    auto &driver = CUDADriver::get_instance();
    bool idle = false;
    if (state_->cuda_ordered)
      state_->cuda_ordered->begin();
    try {
      for (const auto &step : state_->cuda_steps) {
        if (const auto *graph =
                std::get_if<std::unique_ptr<cuda::FixedGraph>>(&step)) {
          (*graph)->run();
          idle = false;
        } else {
          const auto &operation = std::get<std::shared_ptr<NativeOperation>>(step);
          if (operation->execution_mode() == NativeOperationMode::cuda_ordered) {
            state_->cuda_ordered->run(*operation);
            idle = false;
            continue;
          }
          if (!idle)
            driver.context_synchronize();
          try {
            operation->run_cuda_synchronous();
          } catch (...) {
            // A provider can enqueue work before reporting an error. Complete
            // it before allowing owner retirement; preserve the original error.
            driver.context_synchronize.call();
            throw;
          }
          driver.context_synchronize();
          idle = true;
        }
      }
      if (state_->cuda_ordered)
        state_->cuda_ordered->finish();
    } catch (const BackendRuntimeError &error) {
      state_->cuda_failed = true;
      if (state_->cuda_ordered)
        state_->cuda_ordered->finish_after_failure(&error);
      throw;
    } catch (...) {
      state_->cuda_failed = true;
      if (state_->cuda_ordered)
        state_->cuda_ordered->finish_after_failure();
      throw;
    }
  }
#endif
#ifdef TI_WITH_VULKAN
  if (state_->vulkan_graphs.size() > 1) {
    state_->vulkan_runtime->begin_submission_batch();
    try {
      for (const auto &graph : state_->vulkan_graphs)
        graph->launch_prepared();
    } catch (...) {
      try {
        state_->vulkan_runtime->end_submission_batch();
      } catch (...) {
      }
      throw;
    }
    state_->vulkan_runtime->end_submission_batch();
  } else if (!state_->vulkan_graphs.empty()) {
    state_->vulkan_graphs.front()->launch_prepared();
  }
#endif
}
std::uint32_t FixedExecution::execution_kind() const {
  return state_->kind;
}
std::size_t FixedExecution::argument_bytes() const {
  return state_->bytes;
}
std::size_t FixedExecution::workspace_bytes() const {
  return state_->workspace_bytes;
}
}  // namespace taichi::lang
