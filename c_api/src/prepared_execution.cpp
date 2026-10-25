#include "argument_binding.h"
#include "execution_resources.h"
#include "execution_binding_family.h"
#include "native_resource_registry.h"
#include "execution_domain.h"
#include "execution_domain_binding.h"
#include "vulkan_execution_domain.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <set>
#include <limits>
#include <map>
#include "taichi/aot/execution_plan.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_sort.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_image.h"
#include "taichi/runtime/native_acceleration_structure.h"
#include "taichi/runtime/native_kernel_result.h"
#include "taichi/program/runtime_fault.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/binding_frame_replay.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/cuda/sparse_runtime_view.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/rhi/cuda/cuda_context.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi_gfx_impl.h"
#include "taichi/runtime/fixed_execution_vulkan.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#endif

namespace capi {
using namespace taichi::lang;

// One public preparation owns both initialization and replay recordings. Its
// domain reservation counts the logical instance, independent of phase count.
struct PreparedDomainInstance final : RuntimeClient {
  std::vector<std::shared_ptr<RuntimeClient>> parts;
#ifdef TI_WITH_CUDA
  std::vector<std::shared_ptr<cuda::SparseRuntimeView>> cuda_views;
  struct CudaResult final : RuntimeCompletionObserver {
    std::vector<std::shared_ptr<cuda::SparseRuntimeView::DiagnosticSnapshot>>
        snapshots;
    void observe() const override {
      std::exception_ptr error;
      for (const auto &snapshot : snapshots) {
        try {
          snapshot->observe();
        } catch (...) {
          if (!error)
            error = std::current_exception();
        }
      }
      if (error)
        std::rethrow_exception(error);
    }
  };
  std::vector<std::shared_ptr<CudaResult>> cuda_results;
  std::shared_ptr<RuntimeCompletionObserver> collect_cuda_result(
      void *stream,
      std::size_t slot) {
    const auto &result = cuda_results.at(slot);
    for (std::size_t i = 0; i < cuda_views.size(); ++i)
      cuda_views[i]->enqueue_diagnostics(*result->snapshots.at(i), stream);
    return result;
  }
#endif
#ifdef TI_WITH_VULKAN
  std::shared_ptr<gfx::GfxRuntime::FixedGraphWorkspace> shared_workspace;
#endif
  std::mutex mutex;
  std::exception_ptr retirement_error;
  bool safely_released() {
    std::lock_guard<std::mutex> lock(mutex);
#ifdef TI_WITH_CUDA
    if (!cuda_views.empty())
      return false;
#endif
    return parts.empty() && !retirement_error;
  }
  void retire() override {
    retire_after({});
  }
  void retire_after(const RuntimeCompletion &prefix) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (retirement_error)
      std::rethrow_exception(retirement_error);
    for (const auto &part : parts) {
      try {
        part->retire();
      } catch (...) {
        if (!retirement_error)
          retirement_error = std::current_exception();
      }
    }
    if (retirement_error)
      std::rethrow_exception(retirement_error);
    parts.clear();
#ifdef TI_WITH_CUDA
    try {
      // The domain has proved its prefix complete; all command parts are
      // already retired. Return managers before releasing tree allocation
      // loans.
      for (const auto &view : cuda_views)
        view->retire();
      cuda_views.clear();
      cuda_results.clear();
    } catch (...) {
      retirement_error = std::current_exception();
      throw;
    }
#endif
#ifdef TI_WITH_VULKAN
    if (shared_workspace) {
      shared_workspace->retire_shared_ray_loans(prefix);
      shared_workspace->retire_shared_sparse_loans(prefix);
      shared_workspace->retire_shared_image_loans();
      shared_workspace.reset();
    }
#endif
  }
};

struct PreparedExecution {
  AotModule *owner{nullptr};
  std::shared_ptr<ExecutionBindingFamily> family;
  std::vector<TiMemory> memories;
  std::unordered_set<TiMemory> replay_memories;
  std::vector<TiNativeResource> external_resources;
  std::unordered_set<TiNativeResource> replay_external_resources;
  std::unordered_set<const NativeResource *> replay_resources;
  TiPreparedExecutionInfo info{};
  bool live{false};
  bool binding_retired{false};
  // Independent instances keep this registry entry and its reverse dependency
  // edges after public handle release. Domain capacity bounds resident entries.
  bool released{false};
  bool domain_initialized{false};
  bool domain_instance_admitted{false};
  std::shared_ptr<PreparedExecutionDomain> domain;
  PreparedExecutionInstance domain_registration;
  std::shared_ptr<PreparedDomainInstance> domain_instance;
#ifdef TI_WITH_VULKAN
  std::shared_ptr<gfx::GfxRuntime::FixedGraphWorkspace> vulkan_workspace;
  std::shared_ptr<const vulkan::ExecutionDomain::Commands> vulkan_workspace_initialization;
#endif
  std::shared_ptr<RuntimeCompletionCudaEventPool> bootstrap_events;
  RuntimeCompletion bootstrap;
  bool requires_default_prefix{true};
  std::vector<DeviceAllocation> preparation_allocations;
  std::vector<SubmissionParameterValue> submission_values;
  std::vector<std::size_t> submission_widths;
  // Declared before backend recordings so owned buffers are destroyed last.
  std::shared_ptr<ExecutionResourceStorage> resources;
  std::vector<std::shared_ptr<NativeResource>> native_resources;
  // Direct-kernel preparations expose one latest-result frame. A completion
  // ticket does not snapshot it; each replay overwrites the same binding.
  std::shared_ptr<NativeKernelResultStorage> kernel_result;
  bool result_valid{false};
  uint64_t native_resource_workspace_bytes{0};
  uint64_t native_resource_payload_bytes{0};
  // The backend consumes these cold descriptors before they are released.
  std::unique_ptr<BoundKernelArguments> kernel_arguments;
  std::unique_ptr<BoundGraphArguments> graph_arguments;
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
  std::vector<std::unique_ptr<BoundKernelArguments>> plan_arguments;
  std::vector<std::shared_ptr<ExternalSynchronizationDomain>> replay_access;
  std::vector<std::shared_ptr<ExternalSynchronizationDomain>> initial_access;
  std::unique_ptr<FixedExecution> native;
  std::unique_ptr<FixedExecution> initialization;

  void close_domain() {
    if (domain)
      domain->close();
  }

  void publish_bootstrap(Runtime &runtime) {
    if (!domain)
      return;
    if (!requires_default_prefix) {
#ifdef TI_WITH_CUDA
      if (domain->cuda)
        // Join during initialize(), so prepare does not occupy a submission
        // slot while waiting for an allocation producer. Capacity one remains
        // sufficient for both this join and the following initialization.
        return;
#endif
      std::vector<DeviceAllocation>().swap(preparation_allocations);
      return;
    }
    std::vector<DeviceAllocation>().swap(preparation_allocations);
    const auto &owner = runtime.runtime_fault_domain();
#ifdef TI_WITH_VULKAN
    if (domain->vulkan) {
      auto &gfx = dynamic_cast<::GfxRuntime &>(runtime).get_gfx_runtime();
      bootstrap = RuntimeCompletion::from_stream_semaphore(
          taichi::Arch::vulkan, owner->program_domain(), owner->allocate_submission_sequence(),
          gfx.record_completion_semaphore(), owner);
      return;
    }
#endif
#ifdef TI_WITH_CUDA
    auto &cuda_context = CUDAContext::get_instance();
    auto submission = cuda_context.get_submission_lock_guard();
    auto current = cuda_context.get_guard();
    bootstrap_events =
        std::make_shared<RuntimeCompletionCudaEventPool>(owner, 1);
    bootstrap = RuntimeCompletion::from_cuda_stream(
        owner->snapshot().program_domain, owner->allocate_submission_sequence(),
        nullptr, owner, bootstrap_events);
#endif
  }

  void release_native_resources() {
    kernel_result.reset();
    result_valid = false;
    while (!native_resources.empty())
      native_resources.pop_back();
    native_resource_workspace_bytes = 0;
    native_resource_payload_bytes = 0;
  }
  void refresh_resource_bytes() {
    uint64_t workspace = 0, payload = 0;
    std::unordered_set<const NativeResource *> counted;
    counted.reserve(native_resources.size());
    for (const auto &resource : native_resources) {
      if (!counted.insert(resource.get()).second)
        continue;
      const auto bytes = resource->workspace_bytes();
      if (bytes > std::numeric_limits<uint64_t>::max() - workspace)
        throw MemoryBindingError("Native resource workspace size overflow");
      workspace += bytes;
      const auto contents = resource->device_payload_bytes();
      if (contents > std::numeric_limits<uint64_t>::max() - payload)
        throw MemoryBindingError("Native resource payload size overflow");
      payload += contents;
    }
    native_resource_workspace_bytes = workspace;
    native_resource_payload_bytes = payload;
  }
  uint64_t workspace_bytes() const {
    uint64_t total = native_resource_workspace_bytes;
#ifdef TI_WITH_CUDA
    if (domain_instance)
      for (const auto &view : domain_instance->cuda_views) {
        if (view->bytes() > std::numeric_limits<uint64_t>::max() - total)
          throw MemoryBindingError("Prepared workspace size overflow");
        total += view->bytes();
      }
#endif
    for (const auto bytes :
         {native ? native->workspace_bytes() : 0,
          initialization ? initialization->workspace_bytes() : 0}) {
      if (bytes > std::numeric_limits<uint64_t>::max() - total)
        throw MemoryBindingError("Prepared workspace size overflow");
      total += bytes;
    }
#ifdef TI_WITH_VULKAN
    if (vulkan_workspace) {
      const auto bytes = vulkan_workspace->bytes();
      if (bytes > std::numeric_limits<uint64_t>::max() - total)
        throw MemoryBindingError("Prepared workspace size overflow");
      total += bytes;
    }
#endif
    return total;
  }
  uint64_t array_bytes() const {
    const auto local = resources ? resources->bytes() : 0;
    const auto shared = family && family->arrays->storage
                            ? family->arrays->storage->bytes()
                            : 0;
    if (shared > std::numeric_limits<uint64_t>::max() - local)
      throw MemoryBindingError("Prepared resource size overflow");
    return local + shared;
  }
  ~PreparedExecution() {
    initialization.reset();
    retire_binding();
    native.reset();
    release_native_resources();
  }

  void retire_binding(bool already_waited = false) noexcept {
    if (!family || !native || binding_retired)
      return;
    bool safe = family->runtime->runtime_fault_domain()->backend_calls_safe();
    if (safe && !already_waited) {
      try {
        family->runtime->wait();
      } catch (...) {
        safe = false;
      }
    }
    native->retire_completed_binding(safe);
    binding_retired = true;
  }

  void release_source_bindings() {
    contexts.clear();
    graph_arguments.reset();
    kernel_arguments.reset();
    plan_arguments.clear();
  }

  void bind_discarded_results(
      Runtime &runtime,
      const std::vector<FixedExecution::Operation> &operations,
      bool replay) {
    using Operations = std::vector<FixedExecution::Operation>;
    std::unordered_set<const void *> active;
    std::unordered_map<LaunchContextBuilder *, NativeKernelResultPool *> scopes;
    // Keep scope identities alive until the complete walk is validated.
    std::vector<std::unique_ptr<NativeKernelResultPool>> pools;
    auto make_pool = [&]() -> NativeKernelResultPool & {
      pools.push_back(std::make_unique<NativeKernelResultPool>(runtime.arch,
                                                              runtime.get()));
      return *pools.back();
    };
    std::function<void(const Operations &, NativeKernelResultPool &)> visit;
    visit = [&](const Operations &steps, NativeKernelResultPool &pool) {
      for (const auto &step : steps) {
        if (const auto *dispatch = std::get_if<FixedExecution::Dispatch>(&step)) {
          if (!dispatch->kernel || !dispatch->context)
            throw MemoryBindingError("Prepared dispatch is missing its kernel or context");
          if (!dispatch->kernel->ret_size)
            continue;
          auto [scope, inserted] = scopes.emplace(dispatch->context, &pool);
          if (!inserted && scope->second != &pool)
            throw MemoryBindingError("Concurrent dispatches require distinct argument contexts");
          // Direct kernel preparations already own an observable frame.
          if (!dispatch->context->prepared_result)
            dispatch->context->prepared_result = pool.acquire(*dispatch->kernel);
          continue;
        }
        auto enter = [&](const auto &node) {
          if (!node || !active.insert(node.get()).second)
            throw MemoryBindingError("Invalid prepared control structure");
        };
        if (const auto *branch = std::get_if<
                std::shared_ptr<const FixedExecution::Branch>>(&step)) {
          enter(*branch);
          visit((*branch)->condition, pool);
          for (const auto &body : (*branch)->bodies)
            visit(body, pool);
          active.erase(branch->get());
        } else if (const auto *loop = std::get_if<
                       std::shared_ptr<const FixedExecution::While>>(&step)) {
          enter(*loop);
          visit((*loop)->condition, pool);
          visit((*loop)->body, pool);
          active.erase(loop->get());
        } else if (const auto *parallel = std::get_if<
                       std::shared_ptr<const FixedExecution::Parallel>>(&step)) {
          enter(*parallel);
          for (const auto &body : (*parallel)->branches)
            visit(body, make_pool());
          active.erase(parallel->get());
        }
      }
    };
    visit(operations, make_pool());
    for (const auto &pool : pools)
      for (const auto &result : pool->resources()) {
        native_resources.push_back(result);
        if (replay)
          replay_resources.insert(result.get());
      }
  }

  void prepare_operations(
      Runtime &runtime,
      const std::vector<FixedExecution::Operation> &operations,
      const std::vector<FixedExecution::Operation> &initializers = {},
      const VulkanSubmissionRecipe *submission_recipe = nullptr,
      const OrderedStorageRing *storage_ring = nullptr) {
    gfx::GfxRuntime *gfx_runtime = nullptr;
    std::unordered_set<const NativeResource *> qualified_producers;
#ifdef TI_WITH_VULKAN
    if (auto *gfx = dynamic_cast<::GfxRuntime *>(&runtime))
      gfx_runtime = &gfx->get_gfx_runtime();
#endif
    try {
      // Cold binding precedes capture of domain/family operands. Initialization
      // has a separate lifetime; later family members do not replay it.
      bind_discarded_results(runtime, operations, true);
      if ((!family || !family->prepared) && !initializers.empty())
        bind_discarded_results(runtime, initializers, false);
      FixedExecution::CudaExecutionBinding selected{domain ? domain->cuda : nullptr, {}};
#ifdef TI_WITH_CUDA
      struct CudaClosure {
        bool needs_view{false};
        std::vector<int> trees;
        std::set<cuda::KernelLauncher *> launchers;
      };
      std::map<LlvmRuntimeExecutor *, CudaClosure> cuda_closures;
      std::set<std::pair<cuda::KernelLauncher *, int>> cuda_selected;
      std::function<void(const std::vector<FixedExecution::Operation> &)>
          collect_cuda;
      collect_cuda = [&](const auto &steps) {
        for (const auto &step : steps) {
          if (const auto *dispatch =
                  std::get_if<FixedExecution::Dispatch>(&step)) {
            auto *kernel =
                dynamic_cast<llvm_aot::KernelImpl *>(dispatch->kernel);
            auto *launcher =
                kernel
                    ? dynamic_cast<cuda::KernelLauncher *>(&kernel->launcher())
                    : nullptr;
            if (!launcher)
              throw MemoryBindingError(
                  "CUDA domain requires selected CUDA kernels");
            if (!cuda_selected
                     .emplace(launcher, kernel->handle().get_launch_id())
                     .second)
              continue;
            const auto requirements =
                launcher->execution_domain_requirements(kernel->handle());
            auto &closure = cuda_closures[requirements.executor];
            closure.needs_view |= requirements.runtime_view;
            closure.trees.insert(closure.trees.end(),
                                 requirements.trees.begin(),
                                 requirements.trees.end());
            closure.launchers.insert(launcher);
          } else if (const auto *loop = std::get_if<
                         std::shared_ptr<const FixedExecution::While>>(&step)) {
            if (!*loop)
              throw MemoryBindingError("CUDA domain has an empty loop");
            collect_cuda((*loop)->condition);
            collect_cuda((*loop)->body);
          } else if (const auto *branch = std::get_if<
                         std::shared_ptr<const FixedExecution::Branch>>(
                         &step)) {
            if (!*branch)
              throw MemoryBindingError("CUDA domain has an empty branch");
            collect_cuda((*branch)->condition);
            for (const auto &body : (*branch)->bodies)
              collect_cuda(body);
          } else if (const auto *parallel = std::get_if<
                         std::shared_ptr<const FixedExecution::Parallel>>(
                         &step)) {
            if (!*parallel)
              throw MemoryBindingError(
                  "CUDA domain has an empty parallel region");
            for (const auto &body : (*parallel)->branches)
              collect_cuda(body);
          }
        }
      };
      if (selected.domain) {
        collect_cuda(initializers);
        collect_cuda(operations);
      }
#endif
#ifdef TI_WITH_VULKAN
      FixedExecution::VulkanExecutionBinding selected_vulkan;
      if (domain && domain->vulkan) {
        selected_vulkan.domain = domain->vulkan;
        vulkan_workspace = gfx_runtime->create_fixed_graph_workspace();
        selected_vulkan.workspace = vulkan_workspace;
        for (const auto &resource : native_resources)
          selected_vulkan.private_resources.insert(resource.get());
        std::vector<gfx::GfxRuntime::SharedImage> shared_images;
        std::vector<NativeResourceUse> shared_rays;
        const auto ray_uses = [&](const std::vector<NativeResourceUse> &uses) {
          for (const auto &use : uses)
            if (use.resource && !selected_vulkan.private_resources.count(use.resource.get()))
              shared_rays.push_back(use);
        };
        const auto image_use = [&](const std::shared_ptr<NativeSampledImage> &image,
                                    bool sampled_only) {
          if (!image)
            throw MemoryBindingError("Native image access has no owner");
          const auto handle = image->gfx_managed_owner();
          if (!handle)
            return;
          const auto format = image->parameters().format;
          gfx::GfxRuntime::SharedImage use{handle, image->gfx_sampled_allocation(), image};
          use.depth_aspect = format == BufferFormat::depth16 || format == BufferFormat::depth32f;
          use.access = sampled_only ? gfx::SharedImageUse::read
                                    : gfx::SharedImageUse::exclusive;
          shared_images.push_back(std::move(use));
          qualified_producers.insert(image.get());
        };
        bool nonflat = false;
        const auto collect = [&](const auto &steps) {
          for (const auto &step : steps) {
            const auto *dispatch = std::get_if<FixedExecution::Dispatch>(&step);
            if (!dispatch) {
              if (const auto *native = std::get_if<std::shared_ptr<NativeOperation>>(&step)) {
                const auto requirements = (*native)->vulkan_recording_requirements();
                ray_uses(requirements.stateful_resources);
                if (requirements.image_uses)
                  for (const auto &use : *requirements.image_uses)
                    image_use(use.image, use.sampled_only);
              }
              nonflat |= std::holds_alternative<std::shared_ptr<const FixedExecution::While>>(step) ||
                         std::holds_alternative<std::shared_ptr<const FixedExecution::Branch>>(step) ||
                         std::holds_alternative<std::shared_ptr<const FixedExecution::Parallel>>(step);
              continue;
            }
            for (const auto &binding : dispatch->resources) {
              ray_uses(binding->vulkan_stateful_resources());
              const auto image_binding = std::dynamic_pointer_cast<NativeImageKernelBinding>(binding);
              if (!image_binding)
                continue;
              image_use(image_binding->image(), image_binding->sampled_only());
            }
          }
        };
        collect(initializers);
        collect(operations);
        if (nonflat && (!shared_images.empty() || !shared_rays.empty()))
          throw MemoryBindingError("Shared Vulkan resources currently require flat dispatches");
        // Sparse state belongs to the complete prepared instance, including
        // initialization and every control-flow arm. Borrow the selected
        // physical kernels once, before recording any of those phases.
        std::vector<gfx::GfxRuntime::KernelHandle> sparse_kernels;
        std::unordered_set<int> selected_kernels;
        std::function<void(const std::vector<FixedExecution::Operation> &)>
            collect_sparse;
        collect_sparse = [&](const auto &steps) {
          for (const auto &step : steps) {
            if (const auto *dispatch = std::get_if<FixedExecution::Dispatch>(&step)) {
              const auto *kernel = dynamic_cast<gfx::KernelImpl *>(dispatch->kernel);
              if (!kernel || &kernel->runtime() != gfx_runtime)
                throw MemoryBindingError("Vulkan domain kernels require the same runtime");
              if (selected_kernels.insert(kernel->handle().get_launch_id()).second)
                sparse_kernels.push_back(kernel->handle());
            } else if (const auto *loop = std::get_if<
                           std::shared_ptr<const FixedExecution::While>>(&step)) {
              if (!*loop)
                throw MemoryBindingError("Vulkan domain has an empty loop");
              collect_sparse((*loop)->condition);
              collect_sparse((*loop)->body);
            } else if (const auto *branch = std::get_if<
                           std::shared_ptr<const FixedExecution::Branch>>(&step)) {
              if (!*branch)
                throw MemoryBindingError("Vulkan domain has an empty branch");
              collect_sparse((*branch)->condition);
              for (const auto &body : (*branch)->bodies)
                collect_sparse(body);
            } else if (const auto *parallel = std::get_if<
                           std::shared_ptr<const FixedExecution::Parallel>>(&step)) {
              if (!*parallel)
                throw MemoryBindingError("Vulkan domain has an empty parallel region");
              for (const auto &body : (*parallel)->branches)
                collect_sparse(body);
            }
          }
        };
        collect_sparse(initializers);
        collect_sparse(operations);
        // Reserve after collecting every phase and before recording commands.
        // The Program submission guard is held by the borrowed runtime entry;
        // GFX uses its existing host gate for the reservation/prefix transaction.
        gfx_runtime->prepare_fixed_shared_images(
            vulkan_workspace, shared_images, runtime.runtime_fault_domain());
        gfx_runtime->prepare_fixed_shared_ray_resources(
            vulkan_workspace, shared_rays, runtime.runtime_fault_domain());
        gfx_runtime->prepare_fixed_shared_sparse_resources(
            vulkan_workspace, sparse_kernels, runtime.runtime_fault_domain());
        // These exact loans already publish a complete producer prefix. They
        // do not justify skipping any other public or unknown dependency.
        for (const auto &use : shared_rays)
          qualified_producers.insert(use.resource.get());
        // Only factories invoked for this prepared instance populate this
        // private owner list. Public/borrowed native resources remain outside
        // it and need a separate cross-instance layout lease contract.
        for (const auto &resource : native_resources) {
          const auto image = std::dynamic_pointer_cast<NativeSampledImage>(resource);
          if (!image)
            continue;
          const auto &params = image->parameters();
          gfx_runtime->prepare_fixed_private_image(
              vulkan_workspace, image->gfx_sampled_allocation(),
              bool(params.usage & ImageAllocUsage::Sampled)
                  ? ImageLayout::shader_read : ImageLayout::shader_read_write,
              image);
        }
      }
      const auto *vulkan_binding = selected_vulkan.domain ? &selected_vulkan : nullptr;
#else
      const FixedExecution::VulkanExecutionBinding *vulkan_binding = nullptr;
#endif
      if (domain) {
        struct DomainOperands final : RuntimeCompletionResources {
          std::shared_ptr<ExecutionResourceStorage> arrays;
          std::vector<std::shared_ptr<NativeResource>> objects;
          std::size_t retained_resource_count(
              std::uint32_t) const noexcept override {
            return objects.size() + (arrays ? 1 : 0);
          }
        };
        auto retained = std::make_shared<DomainOperands>();
        retained->arrays = resources;
        retained->objects = native_resources;
        selected.operands = std::move(retained);
        domain_instance = std::make_shared<PreparedDomainInstance>();
        selected.retirement_parts = &domain_instance->parts;
#ifdef TI_WITH_VULKAN
        if (vulkan_binding) {
          domain_instance->shared_workspace = vulkan_workspace;
          selected_vulkan.operands = selected.operands;
          selected_vulkan.retirement_parts = &domain_instance->parts;
        }
#endif
      }
      const auto *execution_binding = selected.domain ? &selected : nullptr;
      NativeRecordingState sequence =
          family ? family->initial_facts : NativeRecordingState{};
      struct FrameOperands {
        std::shared_ptr<ExecutionResourceStorage> views;
        std::shared_ptr<SharedExecutionResources> shared;
        std::vector<std::shared_ptr<NativeResource>> objects;
      };
      std::shared_ptr<FrameOperands> operands;
      if (family) {
        operands = std::make_shared<FrameOperands>();
        operands->views = resources;
        operands->shared = family->arrays;
        for (const auto &object : native_resources)
          if (replay_resources.count(object.get()))
            operands->objects.push_back(object);
      }
      const auto prepare_recordings = [&] {
        if (family && !family->prepared) {
          if (!initializers.empty()) {
            auto initial_operands = std::make_shared<FrameOperands>(*operands);
            initial_operands->objects = native_resources;
            family->initialization_operands = std::move(initial_operands);
            family->initialization = std::make_unique<FixedExecution>(
                runtime.arch, runtime.get(), gfx_runtime, initializers, &sequence,
                runtime.runtime_fault_domain());
          } else {
            family->initialized = true;
          }
          family->initial_facts = sequence;
        } else if (!family && !initializers.empty()) {
          initialization = std::make_unique<FixedExecution>(
              runtime.arch, runtime.get(), gfx_runtime, initializers, &sequence,
              runtime.runtime_fault_domain(), nullptr, nullptr, nullptr, nullptr,
              execution_binding, vulkan_binding);
          if (!initial_access.empty())
            initialization->bind_external_access(std::move(initial_access));
        }
        selected.parameter_count = submission_values.size();
#ifdef TI_WITH_VULKAN
        selected_vulkan.parameter_count = submission_values.size();
#endif
        native = std::make_unique<FixedExecution>(
            runtime.arch, runtime.get(), gfx_runtime, operations, &sequence,
            runtime.runtime_fault_domain(), family ? family->replay : nullptr,
            family ? operands : nullptr, submission_recipe, storage_ring,
            execution_binding, vulkan_binding);
        if (!replay_access.empty())
          native->bind_external_access(std::move(replay_access));
        if (family) {
          family->prepared = true;
          family->arrays->sealed = true;
        }
      };
      if (domain) {
        std::function<std::shared_ptr<RuntimeCompletionObserver>(void *,
                                                                 std::size_t)>
            collect_cuda_result;
#ifdef TI_WITH_CUDA
        if (std::any_of(
                cuda_closures.begin(), cuda_closures.end(),
                [](const auto &entry) { return entry.second.needs_view; })) {
          collect_cuda_result = [owner = domain_instance](void *stream,
                                                          std::size_t slot) {
            return owner->collect_cuda_result(stream, slot);
          };
        }
#endif
        domain_registration = domain->prepare_owned_instance(
            domain_instance,
            [&](const auto &instance) {
              selected.instance = instance.cuda;
#ifdef TI_WITH_CUDA
              for (const auto &[executor, closure] : cuda_closures) {
                if (!closure.needs_view)
                  continue;
                auto view = std::make_shared<cuda::SparseRuntimeView>();
                domain_instance->cuda_views.push_back(view);
                view->initialize(
                    *executor, executor->cuda_runtime_view_trees(closure.trees),
                    runtime.runtime_fault_domain());
                for (auto *launcher : closure.launchers)
                  selected.runtime_views.emplace(launcher, view);
              }
              if (!domain_instance->cuda_views.empty()) {
                for (std::size_t slot = 0;
                     slot < domain->cuda->snapshot().capacity; ++slot) {
                  auto result =
                      std::make_shared<PreparedDomainInstance::CudaResult>();
                  for (const auto &view : domain_instance->cuda_views)
                    result->snapshots.push_back(
                        view->prepare_diagnostic_snapshot());
                  domain_instance->cuda_results.push_back(std::move(result));
                }
              }
#endif
#ifdef TI_WITH_VULKAN
              selected_vulkan.instance = instance.vulkan;
#endif
              prepare_recordings();
#ifdef TI_WITH_VULKAN
              if (vulkan_binding) {
                auto commands = domain->vulkan->new_command_list();
                vulkan_workspace->record_initialization(*commands);
                vulkan_workspace_initialization = domain->vulkan->prepare_commands(
                    {commands.get()}, vulkan_workspace);
              }
#endif
            },
            std::move(collect_cuda_result));
        domain_instance_admitted = true;
      } else
        prepare_recordings();
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
    std::uint64_t dispatch_count = 0;
    std::unordered_set<const NativeResource *> private_resources;
    std::unordered_set<const NativeResource *> visited_resources;
    std::function<void(const NativeResource *)> inspect_resource;
    inspect_resource = [&](const NativeResource *resource) {
      if (!resource || qualified_producers.count(resource) ||
          !visited_resources.insert(resource).second)
        return;
      const auto dependencies = resource->preparation_dependencies();
      if (!dependencies) {
        requires_default_prefix = true;
        return;
      }
      preparation_allocations.insert(preparation_allocations.end(),
                                     dependencies->allocations.begin(),
                                     dependencies->allocations.end());
      for (const auto *dependency : dependencies->resources) {
        if (!dependency || (!private_resources.count(dependency) &&
                            !qualified_producers.count(dependency)))
          requires_default_prefix = true;
        else
          inspect_resource(dependency);
      }
    };
    if (domain) {
      requires_default_prefix = !memories.empty();
      if (resources) {
        const auto allocations = resources->private_preparation_allocations();
        if (allocations)
          preparation_allocations = *allocations;
        else
          requires_default_prefix = true;
      }
      for (const auto &resource : native_resources)
        private_resources.insert(resource.get());
      for (const auto &resource : native_resources)
        inspect_resource(resource.get());
      for (const auto handle : external_resources)
        if (!qualified_producers.count(
                runtime.native_resource_registry().get(handle).get()))
          requires_default_prefix = true;
#ifdef TI_WITH_VULKAN
      if (vulkan_workspace && vulkan_workspace->has_field_dependencies())
        requires_default_prefix = true;
#endif
    }
    std::function<void(const std::vector<FixedExecution::Operation> &)>
        count_dispatches;
    count_dispatches = [&](const std::vector<FixedExecution::Operation>
                               &steps) {
      for (const auto &step : steps) {
        if (const auto *dispatch =
                std::get_if<FixedExecution::Dispatch>(&step)) {
          ++dispatch_count;
#ifdef TI_WITH_CUDA
          if (domain && domain->cuda) {
            const auto *kernel =
                dynamic_cast<llvm_aot::KernelImpl *>(dispatch->kernel);
            auto *launcher =
                kernel
                    ? dynamic_cast<cuda::KernelLauncher *>(&kernel->launcher())
                    : nullptr;
            if (!launcher || launcher->has_field_dependencies(kernel->handle()))
              requires_default_prefix = true;
          }
#endif
        } else if (const auto *operation =
                       std::get_if<std::shared_ptr<NativeOperation>>(&step)) {
          if (domain)
            inspect_resource(operation->get());
        } else if (const auto *branch = std::get_if<
                       std::shared_ptr<const FixedExecution::Branch>>(&step)) {
          count_dispatches((*branch)->condition);
          for (const auto &body : (*branch)->bodies)
            count_dispatches(body);
        } else if (const auto *parallel = std::get_if<
                       std::shared_ptr<const FixedExecution::Parallel>>(
                       &step)) {
          for (const auto &branch : (*parallel)->branches)
            count_dispatches(branch);
        } else if (const auto *loop = std::get_if<
                       std::shared_ptr<const FixedExecution::While>>(&step)) {
          count_dispatches((*loop)->condition);
          count_dispatches((*loop)->body);
        }
      }
    };
    if (domain) {
      count_dispatches(initializers);
      dispatch_count = 0;
    }
    count_dispatches(operations);
    if (requires_default_prefix)
      preparation_allocations.clear();
    if (storage_ring) {
      // The compact template expands to equal frames during cold parsing.
      // One replay selects one frame; memory statistics cover every frame.
      if (dispatch_count % storage_ring->capacity)
        throw MemoryBindingError("Storage ring dispatch counts differ between frames");
      dispatch_count /= storage_ring->capacity;
    }
    if (dispatch_count > std::numeric_limits<std::uint32_t>::max())
      throw MemoryBindingError(
          "Static dispatch count exceeds its C API representation");
    info.dispatch_count = static_cast<std::uint32_t>(dispatch_count);
    info.execution_kind = native->execution_kind();
    info.argument_bytes = native->argument_bytes() +
                          (initialization ? initialization->argument_bytes()
                           : family && family->initialization
                               ? family->initialization->argument_bytes()
                               : 0);
    refresh_resource_bytes();
    release_source_bindings();
    live = true;
  }

  void prepare(
      Runtime &runtime,
      const std::vector<std::pair<aot::Kernel *, LaunchContextBuilder *>>
          &dispatches) {
    std::vector<FixedExecution::Operation> operations;
    for (const auto &[kernel, context] : dispatches)
      operations.emplace_back(FixedExecution::Dispatch{kernel, context});
    prepare_operations(runtime, operations);
  }

  void run() {
    if (!live)
      throw MemoryBindingError("Prepared execution has a retired dependency");
    if (family && family->failed)
      throw MemoryBindingError(
          "Binding family initialization failed; prepare a new family");
    result_valid = false;
    if (domain) {
      if (!domain_initialized)
        throw MemoryBindingError(
            "Initialize the selected execution domain before submitting");
      (void)native->submit();
    } else {
      native->run();
    }
    ++info.launches;
    result_valid = bool(kernel_result);
  }

  void invalidate() {
    live = false;
    initialization.reset();
    retire_binding(true);
    native.reset();
#ifdef TI_WITH_VULKAN
    // In-flight commands also own the workspace. Releasing the public facade
    // cannot release its read reservation before the final domain prefix.
    vulkan_workspace_initialization.reset();
    vulkan_workspace.reset();
#endif
    release_native_resources();
    release_source_bindings();
    resources.reset();
    family.reset();
  }

  bool needs_initialization() const {
    if (domain)
      return !domain_initialized;
    return family ? !family->initialized : bool(initialization);
  }

  void finish_initialization() {
    initialization.reset();
    for (std::size_t i = native_resources.size(); i-- > 0;)
      if (!replay_resources.count(native_resources[i].get()))
        native_resources.erase(native_resources.begin() + i);
    refresh_resource_bytes();
    if (resources)
      resources->retain(replay_memories);
    info.argument_bytes = native->argument_bytes();
  }

  void initialize(Runtime &runtime) {
    if (!live)
      throw MemoryBindingError("Prepared execution has a retired dependency");
    if (domain) {
      if (domain_initialized)
        return;
      try {
#ifdef TI_WITH_CUDA
        if (domain->cuda && !requires_default_prefix &&
            !preparation_allocations.empty()) {
          auto allocations = std::move(preparation_allocations);
          domain->join_allocation_initialization(runtime.get(), allocations)
              .wait();
          domain->cuda->collect_ready(1);
        }
#endif
        // Follow the required producer prefix, not an all-domain wait. Fully
        // described private plans need only their allocation initialization;
        // shared GFX loans retain their exact producer dependencies. Keep both
        // sets of recording operands until the domain retires; dropping an
        // initializer wrapper must not free a still-resident graph's argument
        // storage.
#ifdef TI_WITH_VULKAN
        if (domain->vulkan) {
          auto dependencies = vulkan_workspace->shared_resource_dependencies();
          if (bootstrap.valid())
            dependencies.push_back(bootstrap);
          domain->vulkan->submit_instance(domain_registration.vulkan,
              vulkan_workspace_initialization, dependencies).wait();
          if (initialization)
            initialization->submit().wait();
        } else
#endif
        if (initialization)
          initialization
              ->submit(bootstrap.valid()
                           ? std::vector<RuntimeCompletion>{bootstrap}
                           : std::vector<RuntimeCompletion>{})
              .wait();
        else if (bootstrap.valid())
          bootstrap.wait();
        bootstrap = {};
        domain_initialized = true;
      } catch (...) {
        live = false;
        throw;
      }
      return;
    }
    if (family) {
      family->initialize();
      return;
    }
    if (!initialization)
      return;
    try {
      initialization->run();
      runtime.wait();
      // Commands must retire before any of their input or scratch storage.
      finish_initialization();
    } catch (...) {
      // Initialization may have partially written data. Retain its leases
      // until a synchronized close/owner retirement; never retry implicitly.
      live = false;
      throw;
    }
  }
};

class PreparedExecutionRegistry {
 public:
  explicit PreparedExecutionRegistry(Runtime &runtime) : runtime_(runtime) {
  }
  TiPreparedExecution publish(std::unique_ptr<PreparedExecution> execution) {
    prune_released();
    const auto handle =
        reinterpret_cast<TiPreparedExecution>(Runtime::new_handle_token());
    auto &memories = execution->memories;
    std::sort(memories.begin(), memories.end(), std::less<TiMemory>());
    memories.erase(std::unique(memories.begin(), memories.end()),
                   memories.end());
    auto *entry = execution.get();
    entries_.emplace(handle, std::move(execution));
    try {
      modules_[entry->owner].insert(handle);
      if (entry->family)
        families_[entry->family.get()].insert(handle);
      for (auto memory : memories)
        memories_[memory].insert(handle);
      for (auto resource : entry->external_resources)
        resources_[resource].insert(handle);
    } catch (...) {
      remove_dependencies(handle, *entry);
      entries_.erase(handle);
      throw;
    }
    return handle;
  }
  PreparedExecution &get(TiPreparedExecution handle) {
    const auto found = entries_.find(handle);
    if (found == entries_.end() || found->second->released)
      throw MemoryBindingError("Prepared handle is not owned by this runtime");
    return *found->second;
  }
  uint64_t family_size(const ExecutionBindingFamily *family) const {
    const auto found =
        families_.find(const_cast<ExecutionBindingFamily *>(family));
    return found == families_.end() ? 0 : found->second.size();
  }
  void destroy(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if (entry.domain) {
      if (entry.domain->instance_retired(entry.domain_registration)) {
        // Dependency retirement can release this instance while its queue
        // stays open. Do not retain a dead host entry until whole-domain close.
        entry.retire_binding(true);
        remove_dependencies(handle, entry);
        entries_.erase(handle);
        return;
      }
      // Host-only release. Code/storage edges remain authoritative through
      // explicit domain close, dependency destruction or runtime retirement.
      entry.released = true;
      return;
    }
    if ((entry.native || entry.initialization) &&
        runtime_.runtime_fault_domain()->backend_calls_safe())
      runtime_.wait();
    entry.retire_binding(true);
    remove_dependencies(handle, entry);
    entries_.erase(handle);
  }
  void finish_instance_retirement(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if (!entry.domain || !entry.domain->instance_retired(entry.domain_registration))
      throw MemoryBindingError("Execution instance retirement is not complete");
    remove_dependencies(handle, entry);
    entries_.erase(handle);
  }
  void initialize(TiPreparedExecution handle) {
    auto &entry = get(handle);
    const bool needed = entry.needs_initialization();
    entry.initialize(runtime_);
    if (entry.domain)
      return;
    if (!needed)
      return;
    const auto finish = [&](TiPreparedExecution member) {
      auto &call = get(member);
      if (!call.live)
        return;
      if (call.family)
        call.finish_initialization();
      for (auto memory : call.memories)
        if (!call.replay_memories.count(memory))
          remove_edge(memories_, memory, member);
      call.memories.erase(
          std::remove_if(call.memories.begin(), call.memories.end(),
                         [&](TiMemory memory) {
                           return !call.replay_memories.count(memory);
                         }),
          call.memories.end());
      call.replay_memories.clear();
      call.replay_resources.clear();
      for (auto resource : call.external_resources)
        if (!call.replay_external_resources.count(resource))
          remove_edge(resources_, resource, member);
      call.external_resources.erase(
          std::remove_if(call.external_resources.begin(),
                         call.external_resources.end(), [&](auto resource) {
                           return !call.replay_external_resources.count(resource);
                         }),
          call.external_resources.end());
      call.replay_external_resources.clear();
    };
    if (entry.family) {
      auto &family = *entry.family;
      const auto members = families_.at(&family);
      for (auto member : members)
        finish(member);
      for (auto object = family.objects.begin();
           object != family.objects.end();) {
        if (!family.plan.replay_bindings.count(object->first))
          object = family.objects.erase(object);
        else
          ++object;
      }
      std::unordered_set<TiMemory> retained;
      for (auto array = family.arrays->arrays.begin();
           array != family.arrays->arrays.end();) {
        if (!family.plan.replay_bindings.count(array->first))
          array = family.arrays->arrays.erase(array);
        else {
          retained.insert(array->second.memory);
          ++array;
        }
      }
      if (family.arrays->storage)
        family.arrays->storage->retain(retained);
      family.initialization_memories.clear();
      family.initialization_external_resources.clear();
    } else {
      finish(handle);
    }
  }
  void run(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if (!entry.domain && entry.needs_initialization())
      initialize(handle);
    entry.run();
  }
  RuntimeCompletion submit(TiPreparedExecution handle,
                           const std::vector<RuntimeCompletion> &dependencies,
                           bool sample_gpu = false) {
    auto &entry = get(handle);
    if (!entry.live || !entry.domain || !entry.domain_initialized)
      throw MemoryBindingError(
          "Submit requires an initialized live execution-domain instance");
    auto result =
        entry.native->submit(dependencies, entry.submission_values, sample_gpu);
    ++entry.info.launches;
    return result;
  }
  void preparation_failed(TiPreparedExecution handle) {
    auto &entry = get(handle);
    entry.live = false;
    entry.released = true;
    // A rejected reservation executes no preparation. A failed preparation
    // retires its complete group before returning. Neither case needs to
    // drain or invalidate successful peers. After admission (e.g. bootstrap
    // failure), or when cleanup failed, keep the existing cold close proof.
    if (entry.domain_instance_admitted ||
        (entry.domain_instance && !entry.domain_instance->safely_released()))
      entry.close_domain();
    remove_dependencies(handle, entry);
    entries_.erase(handle);
  }
  void retire_memory(TiMemory memory) {
    // A storage generation owns its consumers, not the whole selected queue.
    // Native instance retirement waits its exact prefix and releases the
    // complete initialization/replay group before invalidating these bindings.
    retire(memories_, memory, true, true);
  }
  void retire_resource(TiNativeResource resource, bool synchronize) {
    retire(resources_, resource, synchronize);
  }
  void retire_module(AotModule *module) {
    retire(modules_, module);
  }
  void clear(bool synchronize) {
    // Close before erasing any entry: domain retirement may release recording
    // references, but the registry still owns the actual C API storage here.
    for (const auto &entry : entries_)
      entry.second->close_domain();
    if (synchronize && !entries_.empty() &&
        runtime_.runtime_fault_domain()->backend_calls_safe())
      runtime_.wait();
    modules_.clear();
    memories_.clear();
    resources_.clear();
    families_.clear();
    for (auto &entry : entries_)
      entry.second->retire_binding(true);
    entries_.clear();
  }

 private:
  void prune_released() {
    for (auto it = entries_.begin(); it != entries_.end();) {
      auto &entry = *it->second;
      if (entry.released && entry.domain && entry.domain->snapshot().closed) {
        entry.close_domain();  // A failed provider retirement is not a proof.
        remove_dependencies(it->first, entry);
        it = entries_.erase(it);
      } else {
        ++it;
      }
    }
  }
  template <typename Map, typename Key>
  void retire(Map &map, Key key, bool synchronize = true,
              bool memory_instances_only = false) {
    auto found = map.find(key);
    if (found == map.end())
      return;
    // Retirement is a cold boundary. Reverse indexes visit only dependent
    // executions and remove all their edges; warm replay never scans owners.
    const std::vector<TiPreparedExecution> handles(found->second.begin(),
                                                   found->second.end());
    bool default_queue = false;
    for (auto handle : handles) {
      auto &entry = *entries_.at(handle);
      if (!entry.domain)
        default_queue = true;
      if (memory_instances_only && entry.domain)
        entry.domain->retire_instance(entry.domain_registration);
      else
        entry.close_domain();
    }
    if (synchronize && default_queue &&
        runtime_.runtime_fault_domain()->backend_calls_safe())
      runtime_.wait();
    for (auto handle : handles) {
      auto &entry = *entries_.at(handle);
      remove_dependencies(handle, entry);
      entry.invalidate();
      if (entry.released)
        entries_.erase(handle);
    }
  }
  template <typename Map, typename Key>
  void remove_edge(Map &map, Key key, TiPreparedExecution handle) {
    auto found = map.find(key);
    if (found != map.end()) {
      found->second.erase(handle);
      if (found->second.empty())
        map.erase(found);
    }
  }
  void remove_dependencies(TiPreparedExecution handle,
                           PreparedExecution &entry) {
    if (entry.family)
      remove_edge(families_, entry.family.get(), handle);
    remove_edge(modules_, entry.owner, handle);
    for (auto memory : entry.memories)
      remove_edge(memories_, memory, handle);
    for (auto resource : entry.external_resources)
      remove_edge(resources_, resource, handle);
    entry.owner = nullptr;
    entry.memories.clear();
    entry.external_resources.clear();
  }
  Runtime &runtime_;
  std::unordered_map<TiPreparedExecution, std::unique_ptr<PreparedExecution>>
      entries_;
  std::unordered_map<AotModule *, std::unordered_set<TiPreparedExecution>>
      modules_;
  std::unordered_map<TiMemory, std::unordered_set<TiPreparedExecution>>
      memories_;
  std::unordered_map<TiNativeResource, std::unordered_set<TiPreparedExecution>>
      resources_;
  std::unordered_map<ExecutionBindingFamily *,
                     std::unordered_set<TiPreparedExecution>>
      families_;
};
}  // namespace capi

capi::PreparedExecutionRegistry &Runtime::prepared_registry() {
  if (!prepared_)
    prepared_ = std::make_shared<capi::PreparedExecutionRegistry>(*this);
  return *prepared_;
}
void Runtime::retire_prepared_memory(TiMemory memory) {
  if (prepared_)
    prepared_->retire_memory(memory);
}
void Runtime::retire_prepared_native_resource(TiNativeResource resource,
                                             bool synchronize) {
  if (prepared_)
    prepared_->retire_resource(resource, synchronize);
}
void Runtime::retire_prepared_module(AotModule *owner) {
  if (prepared_)
    prepared_->retire_module(owner);
}
void Runtime::clear_prepared(bool synchronize) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (prepared_)
    prepared_->clear(synchronize);
}

TiPreparedExecution ti_prepare_kernel_ext(TiRuntime runtime,
                                          TiKernel kernel,
                                          uint32_t arg_count,
                                          const TiArgument *args,
                                          uint32_t binding_count,
                                          const TiNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto arguments = capi::bind_kernel_arguments(rt, kernel, arg_count, args,
                                               binding_count, bindings);
  if (!arguments)
    return TI_NULL_HANDLE;
  auto prepared = std::make_unique<capi::PreparedExecution>();
  prepared->owner = arguments->owner;
  prepared->memories = arguments->memories;
  prepared->external_resources = arguments->native_resources;
  prepared->kernel_arguments = std::move(arguments);
  if (prepared->kernel_arguments->kernel->ret_size) {
    prepared->kernel_result =
        std::make_shared<taichi::lang::NativeKernelResultStorage>(
            rt.arch, rt.get(), *prepared->kernel_arguments->kernel);
    prepared->kernel_arguments->context->prepared_result = prepared->kernel_result;
    prepared->native_resources.push_back(prepared->kernel_result);
    prepared->replay_resources.insert(prepared->kernel_result.get());
  }
  taichi::lang::FixedExecution::Dispatch dispatch{
      prepared->kernel_arguments->kernel,
      prepared->kernel_arguments->context.get()};
  dispatch.resources = prepared->kernel_arguments->kernel_resources;
  prepared->prepare_operations(rt, {std::move(dispatch)});
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiPreparedExecution ti_prepare_compute_graph_ext(
    TiRuntime runtime,
    TiComputeGraph graph,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto arguments = capi::bind_graph_arguments(rt, graph, arg_count, args,
                                              binding_count, bindings);
  if (!arguments)
    return TI_NULL_HANDLE;
  auto prepared = std::make_unique<capi::PreparedExecution>();
  prepared->owner = arguments->owner;
  prepared->memories = arguments->memories;
  prepared->external_resources = arguments->native_resources;
  prepared->graph_arguments = std::move(arguments);
  const auto &bound = *prepared->graph_arguments;
  using namespace taichi::lang;
  if (!bound.dispatch_arguments.empty()) {
    std::vector<FixedExecution::Operation> operations;
    for (const auto &arguments : bound.dispatch_arguments) {
      FixedExecution::Dispatch dispatch{arguments->kernel, arguments->context.get()};
      dispatch.resources = arguments->kernel_resources;
      operations.emplace_back(std::move(dispatch));
    }
    prepared->prepare_operations(rt, std::move(operations));
    return rt.prepared_registry().publish(std::move(prepared));
  }
  std::vector<std::pair<aot::Kernel *, LaunchContextBuilder *>> dispatches;
  for (const auto &dispatch : bound.graph->dispatches) {
    if (!dispatch.compiled_kernel || dispatch.cuda_bounded_dispatch ||
        dispatch.indirect_dispatch_arg)
      throw MemoryBindingError(
          "Prepared AOT graph requires ordinary loaded dispatches");
    auto context =
        std::make_unique<LaunchContextBuilder>(dispatch.compiled_kernel);
    aot::CompiledGraph::init_runtime_context(dispatch,
                                             bound.arguments, *context);
    for (int i = 0; i < dispatch.symbolic_args.size(); ++i) {
      const auto &name = dispatch.symbolic_args[i].name;
      const auto grad = bound.gradients.find(name);
      if (grad != bound.gradients.end()) {
        auto *array =
            reinterpret_cast<const Ndarray *>(bound.arguments.at(name).val);
        context->set_arg_ndarray_with_grad({i}, *array, *grad->second);
      }
    }
    dispatches.emplace_back(dispatch.compiled_kernel, context.get());
    prepared->contexts.push_back(std::move(context));
  }
  prepared->prepare(rt, dispatches);
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

static TiPreparedExecution prepare_execution_plan_impl(
    TiRuntime runtime,
    TiAotModule module,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings,
    std::shared_ptr<capi::ExecutionBindingFamily> family,
    std::shared_ptr<capi::PreparedExecutionDomain> domain = {},
    uint32_t parameter_count = 0,
    const char *const *parameter_names = nullptr) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module);
  TI_CAPI_ARGUMENT_NULL_RV(description);
  if ((arg_count && !args) || (binding_count && !bindings) ||
      (parameter_count && !parameter_names) ||
      description_size > std::numeric_limits<std::size_t>::max())
    throw MemoryBindingError("Invalid execution plan input descriptors");
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  if (domain) {
    if (rt.arch != domain->backend() ||
        !domain->belongs_to(rt.runtime_fault_domain()))
      throw MemoryBindingError(
          "Execution domain belongs to another runtime generation");
    if (!domain->snapshot().accepting)
      throw MemoryBindingError("Execution domain is closed for admission");
  }
  auto &mod = *reinterpret_cast<AotModule *>(module);
  if (&mod.runtime() != &rt)
    throw MemoryBindingError(
        "Execution plan module belongs to another runtime");
  using namespace taichi::lang;
  const bool rebinding = bool(family);
  aot::ExecutionPlan parsed;
  if (family) {
    if (family->runtime != &rt || family->module != &mod || family->failed)
      throw MemoryBindingError("Binding family owner is invalid or failed");
  } else {
    try {
      parsed =
          aot::ExecutionPlan::parse(std::string(description, description_size));
      if (domain && parsed.cuda_binding_frames)
        throw MemoryBindingError(
            "Independent execution domains require independent bindings");
      if (parsed.cuda_binding_frames)
        family = std::make_shared<capi::ExecutionBindingFamily>(
            rt, mod, std::move(parsed));
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
  }
  const auto &plan = family ? family->plan : parsed;
  if (parameter_count && (!domain || family))
    throw MemoryBindingError(
        "Submission parameters require an independent execution domain");
  if (plan.vulkan_submission && rt.arch != taichi::Arch::vulkan)
    throw MemoryBindingError("Selected Vulkan submission requires Vulkan");
  const auto &resource_description =
      rebinding ? family->frame_resources : plan.resources;
  std::unordered_map<std::string, const TiArgument *> arguments;
  for (uint32_t i = 0; i < arg_count; ++i) {
    if (!args[i].name || !*args[i].name ||
        !arguments.emplace(args[i].name, &args[i].argument).second)
      throw MemoryBindingError(
          "Execution plan argument names must be unique and nonempty");
  }
  std::unordered_map<std::string, std::size_t> parameter_slots;
  for (uint32_t i = 0; i < parameter_count; ++i) {
    if (!parameter_names[i] || !*parameter_names[i] ||
        !parameter_slots.emplace(parameter_names[i], i).second)
      throw MemoryBindingError(
          "Submission parameter names must be unique and nonempty");
    const std::string name(parameter_names[i]);
    const auto argument = arguments.find(name);
    if (argument == arguments.end() ||
        argument->second->type != TI_ARGUMENT_TYPE_SCALAR ||
        (argument->second->value.scalar.type != TI_DATA_TYPE_U64 &&
         argument->second->value.scalar.type != TI_DATA_TYPE_I64 &&
         argument->second->value.scalar.type != TI_DATA_TYPE_U32 &&
         argument->second->value.scalar.type != TI_DATA_TYPE_I32) ||
        !plan.replay_bindings.count(name) ||
        plan.initialization_bindings.count(name))
      throw MemoryBindingError(
          "Submission parameters require replay-only i32/u32/i64/u64 "
          "scalars: " +
          name);
  }
  std::unordered_set<std::string> patched_parameters;
  std::unordered_set<std::string> frozen_metadata;
  if (rebinding) {
    for (const auto &[name, value] : arguments)
      if (!family->frame_arguments.count(name))
        throw MemoryBindingError(
            "Binding family accepts only replay argument names: " + name);
    for (const auto &name : family->frame_arguments) {
      if (arguments.count(name))
        continue;
      if (plan.replay_bindings.count(name) &&
          !family->persistent_public_arguments.count(name))
        throw MemoryBindingError("Missing binding family replay argument: " +
                                 name);
      arguments.emplace(name, &family->original_arguments.at(name));
      if (!plan.replay_bindings.count(name))
        frozen_metadata.insert(name);
    }
  } else {
    if (arguments.size() != plan.arguments.size())
      throw MemoryBindingError(
          "Execution plan argument set does not match its slots");
    for (const auto &name : plan.arguments)
      if (!arguments.count(name))
        throw MemoryBindingError("Missing execution plan argument: " + name);
    if (family)
      for (const auto &[name, value] : arguments)
        family->original_arguments.emplace(name, *value);
  }
  std::unordered_map<std::string, const TiNamedNdArrayBinding *> layouts;
  std::unordered_set<std::string> used_bindings;
  for (uint32_t i = 0; i < binding_count; ++i) {
    if (!bindings[i].name || !arguments.count(bindings[i].name) ||
        !layouts.emplace(bindings[i].name, &bindings[i]).second)
      throw MemoryBindingError(
          "Invalid or duplicate execution plan ndarray binding");
  }

  auto prepared = std::make_unique<capi::PreparedExecution>();
  prepared->owner = &mod;
  prepared->family = family;
  prepared->domain = domain;
  prepared->submission_values.resize(parameter_count);
  for (uint32_t i = 0; i < parameter_count; ++i)
    prepared->submission_widths.push_back(
        data_type_size(capi::ndarray_scalar_type(
            arguments.at(parameter_names[i])->value.scalar.type)));
  // Reject unmanaged entry paths before private allocation/provider setup.
  // This is a cold validation; submit uses only the frozen epoch closure.
  for (const auto &[name, argument] : arguments)
    if (!frozen_metadata.count(name) &&
        argument->type == TI_ARGUMENT_TYPE_NDARRAY)
      rt.get_memory(argument->value.ndarray.memory, bool(domain));
  for (const auto &[name, binding] : layouts)
    if (!frozen_metadata.count(name) && binding->gradient)
      rt.get_memory(binding->gradient->memory, bool(domain));
  // A frozen structure source carries its whole named ABI, not merely a member
  // path or record width. Check public sources before resolving private data;
  // private descriptors are checked after metadata resolution, before allocation.
  auto validate_structure = [&](const aot::ExecutionPlan::StructArrayLayout &required,
                                bool private_storage) {
    const auto argument = arguments.find(required.array);
    const auto binding = layouts.find(required.array);
    if (argument == arguments.end() ||
        argument->second->type != TI_ARGUMENT_TYPE_NDARRAY ||
        binding == layouts.end() || !binding->second->layout ||
        binding->second->gradient ||
        binding->second->layout_version != aot::StructLayout::kVersion)
      throw MemoryBindingError("Structure constraint requires a typed ndarray: " +
                               required.array);
    const auto &provided = rt.get_layout(binding->second->layout);
    const auto &expected = rt.get_layout(mod.get_layout_handle(required.layout));
    if (provided.type != expected.type || provided.stride != expected.stride)
      throw MemoryBindingError("Structure layout does not match frozen source: " +
                               required.array);
    const auto &array = argument->second->value.ndarray;
    capi::struct_ndarray_storage_layout(array, provided);
    if (required.exact_shape.size() != array.shape.dim_count ||
        !std::equal(required.exact_shape.begin(), required.exact_shape.end(),
                    array.shape.dims))
      throw MemoryBindingError("Structure shape does not match frozen source: " +
                               required.array);
    if (!private_storage)
      capi::validated_struct_ndarray_bytes(
          rt, array, binding->second->layout, binding->second->layout_version,
          bool(domain));
    used_bindings.insert(required.array);
  };
  std::vector<const aot::ExecutionPlan::StructArrayLayout *> private_structures;
  for (const auto &constraint : plan.constraints) {
    const auto *required =
        std::get_if<aot::ExecutionPlan::StructArrayLayout>(&constraint);
    if (!required || (rebinding && !plan.replay_bindings.count(required->array)))
      continue;
    if (arguments.count(required->array))
      validate_structure(*required, false);
    else
      private_structures.push_back(required);
  }
  // Optional providers are admitted before allocating private plan resources.
  // There is no dependency probing on replay, nor on unrelated operations.
  auto local_preparation =
      family ? nullptr : capi::make_native_preparation_context(rt);
  auto &preparation = family ? *family->preparation : *local_preparation;
  preparation.fault_domain->throw_if_submission_disallowed("prepare AOT plan");
  // Independent plans have a local context, so this callback cannot escape to
  // a binding family or ordinary provider. Both owned arrays and ArgPacks use
  // the same cold transport; factories still see completed construction data.
  // Exact ranges of completed construction uploads, not a list of owned
  // handles: module aliases and uninitialized private arrays prove nothing.
  // This proof is scoped to this prepare. Factories cannot modify inputs.
  std::unordered_map<DeviceAllocationId,
                     std::vector<std::pair<std::uint64_t, std::size_t>>>
      completed_uploads;
  if (domain) {
    preparation.upload_owned_data = [domain, &rt,
                                     &completed_uploads](const auto &uploads) {
      domain->upload_owned_data(rt.get(), uploads);
      for (const auto &upload : uploads)
        completed_uploads[upload.destination.alloc_id].emplace_back(
            upload.destination.offset, upload.bytes);
    };
    preparation.complete_owned_initialization =
        [&, domain](std::shared_ptr<NativeOperation> operation,
                    const std::vector<NativeArrayBinding> &inputs) {
          for (const auto &input : inputs) {
            if (input.pointer.device != &rt.get())
              return false;
            const auto ranges = completed_uploads.find(input.pointer.alloc_id);
            if (ranges == completed_uploads.end())
              return false;
            const bool ready = std::any_of(
                ranges->second.begin(), ranges->second.end(),
                [&](const auto &upload) {
                  const auto start = upload.first;
                  return input.pointer.offset >= start &&
                         input.pointer.offset - start <= upload.second &&
                         input.bytes <=
                             upload.second - (input.pointer.offset - start);
                });
            if (!ready)
              return false;
          }
          return domain->complete_owned_initialization(
              rt.get(), preparation.gfx_runtime, std::move(operation), inputs);
        };
  }
  std::vector<std::shared_ptr<NativeOperation>> resource_initializers;
  struct RestoreVulkanInitialization {
    NativePreparationContext &context;
    decltype(context.defer_vulkan_initialization) previous;
    ~RestoreVulkanInitialization() {
      context.defer_vulkan_initialization = std::move(previous);
    }
  } restore_vulkan_initialization{
      preparation, std::move(preparation.defer_vulkan_initialization)};
  if (domain && rt.arch == taichi::Arch::vulkan) {
    // Domain plans have no shared binding-family context. The local sink and
    // its captured descriptors disappear after cold binding; submitted work
    // retains the resulting operations through the existing prepared owner.
    preparation.defer_vulkan_initialization =
        [&](std::shared_ptr<NativeOperation> operation) {
          resource_initializers.push_back(std::move(operation));
        };
  }
  struct RestoreArgPackResolver {
    NativePreparationContext &context;
    decltype(context.resolve_argpack_layout) previous;
    ~RestoreArgPackResolver() {
      context.resolve_argpack_layout = std::move(previous);
    }
  } restore_argpack_resolver{preparation,
                              std::move(preparation.resolve_argpack_layout)};
  preparation.resolve_argpack_layout = [&rt, &mod](const std::string &entry,
                                                   std::uint32_t argument) {
    const auto handle = mod.get_kernel_handle(entry);
    if (!handle)
      throw MemoryBindingError("ArgPack layout kernel was not exported: " + entry);
    return rt.get_kernel_argpack_layout(handle, argument);
  };
  std::set<std::pair<std::string, std::uint32_t>> probed;
  std::map<const aot::ExecutionPlan::Native *,
           NativeOperationRegistry::DecodedOperation>
      decoded;
  auto preflight = [&](const aot::ExecutionPlan::Native *native,
                       NativeObjectKind kind) {
    try {
      auto request = native_operation_registry().decode(*native, kind);
      if (probed.emplace(request.kind(), request.version()).second &&
          !native_operation_registry()
               .probe(request.kind(), request.version(), rt.arch,
                      preparation.providers, kind)
               .available)
        throw MemoryBindingError(native->provider + "/" + native->operation +
                                 " provider unavailable on this runtime");
      decoded.emplace(native, std::move(request));
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
  };
  std::vector<const aot::ExecutionPlan::Operation *> steps;
  std::function<void(const std::vector<aot::ExecutionPlan::Operation> &)>
      collect_steps;
  collect_steps = [&](const std::vector<aot::ExecutionPlan::Operation>
                          &sequence) {
    for (const auto &operation : sequence) {
      steps.push_back(&operation);
      if (const auto *value =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::Branch>>(
                  &operation)) {
        const auto &branch = **value;
        using Lowering = aot::ExecutionPlan::Branch::Lowering;
        if (domain && (branch.lowering == Lowering::cuda_host ||
                       branch.lowering == Lowering::vulkan_host))
          throw MemoryBindingError(
              "Independent execution domains require device branch selection");
        const bool matches = branch.lowering == Lowering::cpu
                                 ? arch_is_cpu(rt.arch)
                             : branch.lowering == Lowering::vulkan_host
                                 ? rt.arch == taichi::Arch::vulkan
                                 : rt.arch == taichi::Arch::cuda;
        if (!matches)
          throw MemoryBindingError(
              "AOT branch requires its declared backend lowering");
        collect_steps(branch.condition);
        for (const auto &body : branch.bodies)
          collect_steps(body);
      }
      if (const auto *parallel =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::Parallel>>(
                  &operation)) {
        if ((*parallel)->lowering ==
            aot::ExecutionPlan::Parallel::Lowering::vulkan) {
          if (rt.arch != taichi::Arch::vulkan || !domain || !domain->vulkan)
            throw MemoryBindingError(
                "AOT Vulkan parallel requires an independent Vulkan execution "
                "domain");
        } else if (rt.arch != taichi::Arch::cuda) {
          throw MemoryBindingError("AOT CUDA parallel requires a CUDA runtime");
        }
        for (const auto &branch : (*parallel)->branches)
          collect_steps(branch);
      }
      if (const auto *loop =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::While>>(
                  &operation)) {
        if ((*loop)->cuda_host && (rt.arch != taichi::Arch::cuda || domain))
          throw MemoryBindingError(
              "AOT host while requires CUDA without an independent execution domain");
        if ((*loop)->cuda_control && rt.arch != taichi::Arch::cuda)
          throw MemoryBindingError(
              "AOT CUDA control requires a CUDA runtime");
        if ((rt.arch == taichi::Arch::vulkan) != bool((*loop)->vulkan))
          throw MemoryBindingError(
              "AOT while requires a matching selected Vulkan recipe on "
              "Vulkan only");
        collect_steps((*loop)->condition);
        collect_steps((*loop)->body);
      }
    }
  };
  if (!rebinding)
    collect_steps(plan.initialization);
  collect_steps(plan.operations);
#ifdef TI_WITH_CUDA
  if (domain && domain->cuda) {
    // Reuse the cold flat operation inventory. Qualify code before private
    // arrays or native provider objects can be materialized. No hot scans.
    std::unordered_set<std::string> qualified;
    for (const auto *step : steps) {
      const auto *dispatch = std::get_if<aot::ExecutionPlan::Kernel>(step);
      if (!dispatch || !qualified.insert(dispatch->entry).second)
        continue;
      auto handle = mod.get_kernel_handle(dispatch->entry);
      if (!handle)
        throw MemoryBindingError("Execution plan kernel was not exported: " +
                                 dispatch->entry);
      auto *kernel =
          dynamic_cast<llvm_aot::KernelImpl *>(rt.get_kernel(handle).kernel);
      auto *launcher =
          kernel ? dynamic_cast<cuda::KernelLauncher *>(&kernel->launcher())
                 : nullptr;
      if (!launcher)
        throw MemoryBindingError(
            "Execution domain requires a CUDA kernel launcher");
      launcher->execution_domain_requirements(kernel->handle());
    }
  }
#endif
  for (const auto index : resource_description.order)
    if (const auto *native = std::get_if<NativeOperationDescription>(
            &resource_description.declarations[index].value))
      preflight(native, NativeObjectKind::resource);
  for (const auto *step : steps) {
    const auto &operation = *step;
    if (const auto *native =
            std::get_if<aot::ExecutionPlan::Native>(&operation))
      preflight(native, NativeObjectKind::operation);
  }
  if (rebinding) {
    for (const auto &name : family->persistent_public_arguments) {
      if (!family->frame_arguments.count(name))
        continue;
      const auto &old = family->original_arguments.at(name);
      const auto &next = *arguments.at(name);
      if (old.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT &&
          next.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT) {
        if (old.value.native_resource != next.value.native_resource)
          throw MemoryBindingError(
              "Shared native resource changed; prepare a new binding family: " +
              name);
        rt.native_resource_registry().get(next.value.native_resource);
        continue;
      }
      if (old.type != TI_ARGUMENT_TYPE_NDARRAY ||
          next.type != TI_ARGUMENT_TYPE_NDARRAY)
        throw MemoryBindingError(
            "Shared provider resources require unchanged array bindings: " +
            name);
      const auto &a = old.value.ndarray;
      const auto &b = next.value.ndarray;
      const auto size_a = capi::validated_ndarray_bytes(rt, a);
      const auto size_b = capi::validated_ndarray_bytes(rt, b);
      const auto memory_a = rt.get_memory(a.memory);
      const auto memory_b = rt.get_memory(b.memory);
      const auto range_a = resolve_native_storage_range(
          rt.arch, memory_a.allocation.get_ptr(memory_a.byte_offset), size_a);
      const auto range_b = resolve_native_storage_range(
          rt.arch, memory_b.allocation.get_ptr(memory_b.byte_offset), size_b);
      if (a.elem_type != b.elem_type ||
          a.shape.dim_count != b.shape.dim_count ||
          a.elem_shape.dim_count != b.elem_shape.dim_count ||
          !std::equal(a.shape.dims, a.shape.dims + a.shape.dim_count,
                      b.shape.dims) ||
          !std::equal(a.elem_shape.dims,
                      a.elem_shape.dims + a.elem_shape.dim_count,
                      b.elem_shape.dims) ||
          !range_a.proven_same_range(range_b))
        throw MemoryBindingError(
            "Shared provider resource input changed; prepare a new binding "
            "family: " +
            name);
    }
  }
  std::map<std::string, std::shared_ptr<NativeResource>> native_objects;
  for (const auto &[name, arg] : arguments) {
    if (arg->type != TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
      continue;
    auto object = rt.native_resource_registry().get(arg->value.native_resource);
    native_objects.emplace(name, std::move(object));
    prepared->external_resources.push_back(arg->value.native_resource);
    if (family && !rebinding && plan.replay_bindings.count(name))
      family->persistent_public_arguments.insert(name);
  }
  std::map<std::string, const NativeOperationDescription *> allocator_descriptions;
  for (const auto index : resource_description.order) {
    const auto &resource = resource_description.declarations[index];
    if (const auto *native =
            std::get_if<NativeOperationDescription>(&resource.value))
      allocator_descriptions.emplace(resource.name, native);
  }
  auto resolve_allocator = [&](const std::string &name) {
    auto found = native_objects.find(name);
    if (found == native_objects.end()) {
      std::shared_ptr<NativeResource> object;
      if (rebinding) {
        object = family->objects.at(name);
      } else {
        const auto description = allocator_descriptions.find(name);
        if (description == allocator_descriptions.end() ||
            !description->second->bindings.empty())
          throw MemoryBindingError(
              "Array allocator requires an input-free native resource: " + name);
        try {
          object = native_operation_registry().prepare_resource(
              preparation, decoded.at(description->second), {});
        } catch (const std::invalid_argument &error) {
          throw MemoryBindingError(error.what());
        }
      }
      prepared->native_resources.push_back(object);
      found = native_objects.emplace(name, std::move(object)).first;
    }
    auto allocator =
        std::dynamic_pointer_cast<NativeMemoryAllocator>(found->second);
    if (!allocator)
      throw MemoryBindingError("Native resource is not an array allocator: " + name);
    return allocator;
  };
  auto resource_bindings = capi::bind_execution_resources(
      rt, mod, resource_description, arguments, layouts,
      family ? family->arrays.get() : nullptr, frozen_metadata,
      resolve_allocator, bool(domain), [&] {
        for (const auto *required : private_structures)
          validate_structure(*required, true);
      }, preparation.upload_owned_data);
  prepared->resources = std::move(resource_bindings.storage);
  prepared->memories.insert(prepared->memories.end(),
                             resource_bindings.native_memories.begin(),
                             resource_bindings.native_memories.end());
  used_bindings.insert(resource_bindings.layout_sources.begin(),
                        resource_bindings.layout_sources.end());
  struct Storage {
    DevicePtr pointer;
    uint64_t size;
    uint64_t element_bytes;
  };
  auto native_arrays = std::move(resource_bindings.native_arrays);
  auto storage = [&](const std::string &name, uint64_t offset) {
    if (const auto native = native_arrays.find(name);
        native != native_arrays.end() && native->second.strides)
      throw MemoryBindingError("Execution memory operation requires packed storage: " + name);
    const auto found = arguments.find(name);
    if (found == arguments.end())
      throw MemoryBindingError(
          "Execution storage requires an ndarray binding: " + name);
    const auto &arg = *found->second;
    if (arg.type != TI_ARGUMENT_TYPE_NDARRAY)
      throw MemoryBindingError("Execution memory operations require ndarrays");
    const auto &array = arg.value.ndarray;
    uint64_t element_bytes = 0;
    uint64_t bytes;
    if (array.elem_type == TI_DATA_TYPE_UNKNOWN) {
      const auto binding = layouts.find(name);
      if (binding == layouts.end() || binding->second->gradient)
        throw MemoryBindingError("Structure memory operation requires its layout: " + name);
      bytes = capi::validated_struct_ndarray_bytes(
          rt, array, binding->second->layout, binding->second->layout_version,
          bool(domain));
      element_bytes = rt.get_layout(binding->second->layout).stride;
      used_bindings.insert(name);
    } else {
      bytes = capi::validated_ndarray_bytes(rt, array, &element_bytes, bool(domain));
    }
    if (offset > bytes)
      throw MemoryBindingError(
          "Execution memory offset exceeds ndarray extent");
    const auto allocation = rt.get_memory(array.memory, bool(domain));
    prepared->memories.push_back(array.memory);
    return Storage{
        allocation.allocation.get_ptr(allocation.byte_offset + offset),
        bytes - offset, element_bytes};
  };
  auto array_binding =
      [&](const std::string &slot) -> const NativeArrayBinding & {
    auto found = native_arrays.find(slot);
    if (found == native_arrays.end()) {
      storage(slot, 0);
      const auto &array = arguments.at(slot)->value.ndarray;
      found =
          native_arrays
              .emplace(slot, capi::bind_native_array(rt, array, bool(domain)))
              .first;
    }
    return found->second;
  };
  std::map<std::string, NativeStorageRange> storage_ranges;
  auto storage_range =
      [&](const std::string &slot) -> const NativeStorageRange & {
    auto found = storage_ranges.find(slot);
    if (found == storage_ranges.end()) {
      const auto &array = array_binding(slot);
      if (array.strides)
        throw MemoryBindingError("Packed range constraint cannot qualify an affine member: " + slot);
      found = storage_ranges
                  .emplace(slot, resolve_native_storage_range(
                                     rt.arch, array.pointer, array.bytes))
                  .first;
    }
    return found->second;
  };
  // Resolve certificates before creating provider state or recording commands.
  // These maps disappear after prepare; replay carries only frozen bindings.
  for (const auto &constraint : plan.constraints) {
    if (std::holds_alternative<aot::ExecutionPlan::StructArrayLayout>(constraint))
      continue;  // Validated before allocating private resources.
    if (rebinding) {
      bool relevant = true;
      const auto used = [&](const std::string &name) {
        return plan.replay_bindings.count(name) != 0;
      };
      if (const auto *layout =
              std::get_if<aot::ExecutionPlan::ArrayLayout>(&constraint))
        relevant = used(layout->array);
      else if (const auto *pair =
                   std::get_if<aot::ExecutionPlan::Disjoint>(&constraint))
        relevant = used(pair->left) && used(pair->right);
      else if (const auto *pair =
                   std::get_if<aot::ExecutionPlan::ResourceAlias>(&constraint))
        relevant = used(pair->left) && used(pair->right);
      if (!relevant)
        continue;
    }
    try {
      if (const auto *layout =
              std::get_if<aot::ExecutionPlan::ArrayLayout>(&constraint)) {
        validate_native_array_layout(array_binding(layout->array),
                                     layout->required);
      } else if (std::holds_alternative<aot::ExecutionPlan::Disjoint>(
                     constraint)) {
        const auto &pair = std::get<aot::ExecutionPlan::Disjoint>(constraint);
        const auto &a = array_binding(pair.left);
        const auto &b = array_binding(pair.right);
        if (a.strides || b.strides) {
          const auto left = resolve_native_dense_array(rt.arch, a);
          const auto right = resolve_native_dense_array(rt.arch, b);
          if (!left.proven_disjoint(right) &&
              !(pair.allow_same_view && left.proven_same_mapping(right)))
            throw MemoryBindingError("Cannot prove affine bindings disjoint or identical");
          continue;
        }
        const auto &left = storage_range(pair.left);
        const auto &right = storage_range(pair.right);
        bool accepted = left.proven_disjoint(right);
        if (!accepted && pair.allow_same_view &&
            left.proven_same_range(right)) {
          const auto &a = array_binding(pair.left);
          const auto &b = array_binding(pair.right);
          accepted = a.scalar_type == b.scalar_type && a.shape == b.shape &&
                     a.element_shape == b.element_shape;
        }
        if (!accepted)
          throw std::invalid_argument(
              std::string("Cannot prove array bindings disjoint") +
              (pair.allow_same_view ? " or the same view: " : ": ") +
              pair.left + " and " + pair.right);
      } else if (const auto *pair =
                     std::get_if<aot::ExecutionPlan::ResourceAlias>(&constraint)) {
        const auto resolve = [&](const std::string &name,
                                 const std::vector<std::string> &path) {
          const auto found = arguments.find(name);
          if (found == arguments.end() ||
              found->second->type != TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
            throw MemoryBindingError("Resource alias requires public native resources: " + name);
          auto &registry = rt.native_resource_registry();
          return registry.get(registry.resolve(
              found->second->value.native_resource, path));
        };
        if (resolve(pair->left, pair->left_path) !=
            resolve(pair->right, pair->right_path))
          throw MemoryBindingError("Native resource alias mismatch: " + pair->left +
                                   " and " + pair->right);
      } else {
        const auto &group =
            std::get<aot::ExecutionPlan::DisjointWrites>(constraint);
        std::vector<NativeStorageRange> reads;
        std::vector<NativeStorageWrite> writes;
        std::vector<std::optional<std::size_t>> read_indices(
            group.reads.size());
        for (std::size_t i = 0; i < group.reads.size(); ++i) {
          if (rebinding && !plan.replay_bindings.count(group.reads[i]))
            continue;
          read_indices[i] = reads.size();
          reads.push_back(storage_range(group.reads[i]));
        }
        for (const auto &write : group.writes) {
          if (rebinding && !plan.replay_bindings.count(write.array))
            continue;
          auto inplace = write.inplace_read;
          if (inplace && !read_indices[*inplace])
            inplace.reset();
          if (inplace) {
            const auto &a = array_binding(write.array);
            const auto &b = array_binding(group.reads[*inplace]);
            if (a.scalar_type != b.scalar_type || a.shape != b.shape ||
                a.element_shape != b.element_shape)
              inplace.reset();
            else
              inplace = read_indices[*inplace];
          }
          writes.push_back({storage_range(write.array), inplace});
        }
        validate_native_disjoint_writes(reads, writes);
      }
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
  }
  auto native_inputs = [&](const NativeOperationDescription &description) {
    NativeOperationBindings inputs;
    for (const auto &[port, slot] : description.bindings) {
      if (const auto object = native_objects.find(slot);
          object != native_objects.end()) {
        inputs.emplace(port, object->second);
        continue;
      }
      inputs.emplace(port, array_binding(slot));
    }
    return inputs;
  };
  std::map<std::string, std::shared_ptr<const NativeHostBlobResource::Bytes>>
      host_blobs;
  for (const auto index : resource_description.order) {
    const auto &resource = resource_description.declarations[index];
    // Allocator factories were resolved after metadata validation, before
    // arrays. Reuse their owner instead of creating a second private pool.
    if (native_objects.count(resource.name))
      continue;
    if (rebinding &&
        (std::holds_alternative<NativeOperationDescription>(resource.value) ||
         std::holds_alternative<aot::ExecutionResources::HostBlob>(
             resource.value))) {
      auto object = family->objects.at(resource.name);
      prepared->native_resources.push_back(object);
      native_objects.emplace(resource.name, std::move(object));
      continue;
    }
    if (const auto *blob =
            std::get_if<aot::ExecutionResources::HostBlob>(&resource.value)) {
      auto found = host_blobs.find(blob->data);
      if (found == host_blobs.end()) {
        auto bytes = std::make_shared<const NativeHostBlobResource::Bytes>(
            mod.get().load_data(blob->data, blob->byte_size));
        found = host_blobs.emplace(blob->data, std::move(bytes)).first;
      } else if (found->second->size() != blob->byte_size) {
        throw MemoryBindingError(
            "Execution host blob size differs between declarations");
      }
      auto object = std::make_shared<NativeHostBlobResource>(rt.arch, rt.get(),
                                                             found->second);
      prepared->native_resources.push_back(object);
      native_objects.emplace(resource.name, std::move(object));
      continue;
    }
    if (const auto *native =
            std::get_if<NativeOperationDescription>(&resource.value)) {
      try {
        // Provider factories may inspect data before replay. Until a factory
        // declares an epoch-safe preparation contract, external operands are
        // accepted only by operations within the complete prepared recipe.
        for (const auto &[port, slot] : native->bindings) {
          const auto input = arguments.find(slot);
          if (input != arguments.end() &&
              input->second->type == TI_ARGUMENT_TYPE_NDARRAY)
            rt.get_memory(input->second->value.ndarray.memory);
        }
        auto object = native_operation_registry().prepare_resource(
            preparation, decoded.at(native), native_inputs(*native));
        prepared->native_resources.push_back(object);
        native_objects.emplace(resource.name, std::move(object));
      } catch (const std::invalid_argument &error) {
        throw MemoryBindingError(error.what());
      }
    }
  }
  std::vector<FixedExecution::Operation> initializers;
  for (const auto &zero : resource_bindings.zero_initializations) {
    const auto allocation = rt.get_memory(zero.memory);
    initializers.emplace_back(
        FixedExecution::BufferFill{allocation.allocation.get_ptr(0),
                                   static_cast<std::size_t>(zero.bytes), 0});
  }
  std::vector<FixedExecution::Operation> explicit_initializers;
  std::vector<FixedExecution::Operation> operations;
  const auto control_pointer = [&](const std::string &name) {
    const auto &array = array_binding(name);
    if (array.scalar_type != PrimitiveType::i32 || array.bytes != 4 ||
        !array.element_shape.empty() ||
        !std::all_of(array.shape.begin(), array.shape.end(),
                     [](std::uint64_t extent) { return extent == 1; }) ||
        array.base_offset % 4)
      throw MemoryBindingError(
          "Control requires one aligned scalar i32 element: " + name);
    return array.pointer;
  };
  std::function<bool(const std::vector<aot::ExecutionPlan::Operation> &,
                     std::vector<FixedExecution::Operation> &)>
      bind_steps;
  bool preparing_initialization = true;
  bind_steps = [&](const std::vector<aot::ExecutionPlan::Operation> &sequence,
                   std::vector<FixedExecution::Operation> &operations) {
    // One map per ordered region and prepared frame, never in the potentially
    // shared binding-family context. Recursive control/parallel regions have
    // independent storage. Initializers retain their original private owners.
    std::map<SortWorkspaceDescription::Signature,
             std::shared_ptr<NativeSortWorkspace>> sort_workspaces;
    for (const auto &operation : sequence) {
      if (const auto *value =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::Branch>>(
                  &operation)) {
        const auto &branch = **value;
        auto bound = std::make_shared<FixedExecution::Branch>();
        bound->selector = control_pointer(branch.selector);
        bound->selection = branch.selection;
        using Lowering = aot::ExecutionPlan::Branch::Lowering;
        if (branch.lowering == Lowering::cuda_masked)
          bound->mode = BranchExecutionMode::cuda_masked;
        else if (branch.lowering == Lowering::cuda_host ||
                 branch.lowering == Lowering::vulkan_host)
          bound->mode = BranchExecutionMode::host;
        if (!bind_steps(branch.condition, bound->condition))
          return false;
        for (const auto &body : branch.bodies) {
          bound->bodies.emplace_back();
          if (!bind_steps(body, bound->bodies.back()))
            return false;
        }
        operations.emplace_back(std::move(bound));
        continue;
      } else if (const auto *parallel = std::get_if<
                     std::shared_ptr<const aot::ExecutionPlan::Parallel>>(
                     &operation)) {
        auto bound = std::make_shared<FixedExecution::Parallel>();
        for (const auto &branch : (*parallel)->branches) {
          bound->branches.emplace_back();
          if (!bind_steps(branch, bound->branches.back()))
            return false;
        }
        operations.emplace_back(std::move(bound));
        continue;
      } else if (const auto *value = std::get_if<
                     std::shared_ptr<const aot::ExecutionPlan::While>>(
                     &operation)) {
        const auto &loop = **value;
        auto bound = std::make_shared<FixedExecution::While>();
        bound->predicate = control_pointer(loop.predicate);
        bound->max_iterations = loop.max_iterations;
        bound->continue_while_nonzero = loop.continue_while_nonzero;
        bound->host_control = loop.cuda_host;
        if (loop.cuda_control) {
          bound->cuda_control.emplace();
          bound->cuda_control->encoding = loop.cuda_control->encoding;
          for (const auto &source : loop.cuda_control->inners)
            bound->cuda_control->inners.push_back(
                {control_pointer(source.predicate), source.condition_begin,
                 source.body_begin, source.end, source.max_iterations});
        }
        if (loop.vulkan) {
          bound->vulkan.emplace();
          bound->vulkan->recipe = loop.vulkan->recipe;
          bound->vulkan->counter = control_pointer(loop.vulkan->counter);
          std::vector<std::string> controls{loop.predicate,
                                            loop.vulkan->counter};
          if (loop.vulkan->status) {
            controls.push_back(*loop.vulkan->status);
            bound->vulkan->status = control_pointer(*loop.vulkan->status);
          }
          if (loop.vulkan->nested) {
            bound->vulkan->nested.emplace();
            for (const auto &source : loop.vulkan->nested->inners) {
              VulkanNestedControlRecipe<DevicePtr>::Inner inner;
              inner.predicate = control_pointer(source.predicate);
              inner.counter = control_pointer(source.counter);
              controls.push_back(source.predicate);
              controls.push_back(source.counter);
              if (source.status) {
                inner.status = control_pointer(*source.status);
                controls.push_back(*source.status);
              }
              inner.condition_begin = source.condition_begin;
              inner.body_begin = source.body_begin;
              inner.end = source.end;
              inner.max_iterations = source.max_iterations;
              inner.chunk_size = source.chunk_size;
              bound->vulkan->nested->inners.push_back(std::move(inner));
            }
          }
          std::vector<const NativeStorageRange *> ranges;
          ranges.reserve(controls.size());
          for (const auto &name : controls)
            ranges.push_back(&storage_range(name));
          for (std::size_t i = 0; i < controls.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
              if (!ranges[i]->proven_disjoint(*ranges[j]))
                throw MemoryBindingError(
                    "Vulkan loop controls must be disjoint");
        }
        if (!bind_steps(loop.condition, bound->condition) ||
            !bind_steps(loop.body, bound->body))
          return false;
        operations.emplace_back(std::move(bound));
        continue;
      }
      if (const auto *kernel =
              std::get_if<aot::ExecutionPlan::Kernel>(&operation)) {
        auto handle = mod.get_kernel_handle(kernel->entry);
        if (!handle)
          throw MemoryBindingError("Execution plan kernel was not exported: " +
                                   kernel->entry);
        std::vector<TiArgument> kernel_args;
        std::vector<TiNdArrayBinding> kernel_bindings;
        std::map<uint32_t, std::shared_ptr<NativeResource>> kernel_resources;
        const auto &parameters =
            rt.get_kernel(handle).kernel->nested_parameters;
        for (std::size_t i = 0; i < kernel->arguments.size(); ++i) {
          const auto &name = kernel->arguments[i];
          if (const auto object = native_objects.find(name);
              object != native_objects.end()) {
            kernel_resources.emplace(static_cast<uint32_t>(i), object->second);
            // The private owner map supplies this slot. Never synthesize or
            // reinterpret a public native-resource handle.
            kernel_args.emplace_back();
            continue;
          }
          kernel_args.push_back(*arguments.at(name));
          if (auto binding = layouts.find(name); binding != layouts.end()) {
            const auto &value = *binding->second;
            const auto parameter = parameters.find({static_cast<int>(i)});
            // A plan-wide primal may also be read by ordinary helper kernels.
            // Supply its gradient only where the compiled dispatch ABI uses it.
            // Keep malformed/structure bindings for the common validator, and
            // reject gradients unused by every dispatch at the end of prepare.
            if (value.gradient && !value.layout && value.layout_version == 0 &&
                (parameter == parameters.end() ||
                 !parameter->second.needs_grad))
              continue;
            used_bindings.insert(name);
            kernel_bindings.push_back({static_cast<uint32_t>(i),
                                       value.layout_version, value.layout,
                                       value.gradient});
          }
        }
        auto bound = capi::bind_kernel_arguments(
            rt, handle, static_cast<uint32_t>(kernel_args.size()),
            kernel_args.data(), static_cast<uint32_t>(kernel_bindings.size()),
            kernel_bindings.data(),
            kernel->bounded_range && !kernel->bounded_range->cuda,
            kernel_resources, bool(domain));
        if (!bound)
          return false;
        std::vector<SubmissionParameterPatch> patches;
        for (std::size_t i = 0; i < kernel->arguments.size(); ++i) {
          const auto &name = kernel->arguments[i];
          const auto slot = parameter_slots.find(name);
          if (slot == parameter_slots.end())
            continue;
          const auto parameter = parameters.find({static_cast<int>(i)});
          if (parameter == parameters.end() || parameter->second.is_array ||
              parameter->second.get_dtype() !=
                  capi::ndarray_scalar_type(
                      arguments.at(name)->value.scalar.type))
            throw MemoryBindingError(
                "Submission parameter compiled scalar ABI does not match: " +
                name);
          patches.push_back({slot->second,
                             bound->kernel->args_type->get_element_offset(
                                 {static_cast<int>(i)}),
                             prepared->submission_widths.at(slot->second)});
          patched_parameters.insert(name);
          bound->context->mutable_argument_ranges.emplace_back(
              patches.back().argument_offset, patches.back().bytes);
        }
        bound->context->append_dispatch_label(kernel->label);
        DevicePtr indirect_packet{kDeviceNullPtr};
        DevicePtr bounded_extent{kDeviceNullPtr};
        std::uint32_t bounded_capacity = 0;
        std::uint32_t adaptive_block_dim = 0;
        bool grouped_update = false;
        const void *bounded_binding_identity = nullptr;
        if (kernel->bounded_range) {
          const auto &range = *kernel->bounded_range;
          if ((range.cuda && rt.arch != taichi::Arch::cuda) ||
              (!range.cuda && !arch_is_cpu(rt.arch)))
            throw MemoryBindingError("Exact bounded launch backend mismatch");
          const auto &extent = array_binding(range.extent);
          validate_native_array_layout(extent, {{2}, {}, 4, 4, true, "i32"});
          if (extent.scalar_type != PrimitiveType::i32 || extent.bytes != 8)
            throw MemoryBindingError(
                "Exact bounded extent requires a packed i32[2]");
          bounded_extent = extent.pointer;
          bounded_capacity = range.capacity;
          adaptive_block_dim = range.block_dim;
          grouped_update = range.grouped_update;
          // std::map references are stable through cold preparation. Equal
          // storage under two symbolic names remains independently bindable.
          bounded_binding_identity = &extent;
          const auto storage = resolve_native_storage_range(
              rt.arch, extent.pointer, extent.bytes);
          if (storage.memory_offset % 4)
            throw MemoryBindingError("Exact bounded extent must be i32 aligned");
          if (!range.cuda) {
            bound->context->set_cpu_bounded_range(
                reinterpret_cast<void *>(
                    static_cast<std::uintptr_t>(storage.memory_offset)),
                range.capacity);
          }
        }
        if (kernel->indirect_packet) {
          if (rt.arch != taichi::Arch::vulkan)
            throw MemoryBindingError("Vulkan indirect launch requires Vulkan");
          const auto &packet = array_binding(*kernel->indirect_packet);
          if (packet.scalar_type != PrimitiveType::u32 || !packet.element_shape.empty() ||
              packet.shape.size() != 1 || packet.shape[0] < 3 ||
              packet.bytes < 3 * sizeof(std::uint32_t) || packet.pointer.offset % 4)
            throw MemoryBindingError(
                "Indirect launch packet requires at least three packed u32 values");
          indirect_packet = packet.pointer;
        }
        prepared->memories.insert(prepared->memories.end(),
                                  bound->memories.begin(),
                                  bound->memories.end());
        operations.emplace_back(FixedExecution::Dispatch{
            bound->kernel, bound->context.get(), indirect_packet,
            bounded_extent, bounded_capacity, adaptive_block_dim,
            grouped_update, bounded_binding_identity, bound->kernel_resources,
            std::move(patches)});
        prepared->plan_arguments.push_back(std::move(bound));
      } else if (const auto *copy =
                     std::get_if<aot::ExecutionPlan::Copy>(&operation)) {
        const auto src = storage(copy->source, copy->source_offset);
        const auto dst = storage(copy->destination, copy->destination_offset);
        const auto bytes = copy->bytes.value_or(src.size);
        if (bytes > src.size || bytes > dst.size ||
            bytes > std::numeric_limits<std::size_t>::max())
          throw MemoryBindingError("Execution copy exceeds ndarray extent");
        if (bytes && src.pointer.alloc_id == dst.pointer.alloc_id &&
            src.pointer.offset < dst.pointer.offset + bytes &&
            dst.pointer.offset < src.pointer.offset + bytes)
          throw MemoryBindingError("Execution copy slices must not overlap");
        operations.emplace_back(FixedExecution::BufferCopy{
            dst.pointer, src.pointer, static_cast<std::size_t>(bytes)});
      } else if (const auto *native =
                     std::get_if<aot::ExecutionPlan::Native>(&operation)) {
        try {
          const auto &registry = native_operation_registry();
          const auto &request = decoded.at(native);
          const auto inputs = native_inputs(*native);
          if (request.kind() == SortOperation::kind) {
            // Only owner-local workspace policy belongs here. The common
            // schema/binder owns all operation-specific input semantics.
            auto descriptor = registry.bind<SortOperation>(request, inputs);
            if (!preparing_initialization) {
              const auto description = SortWorkspaceDescription::from(descriptor);
              auto &workspace = sort_workspaces[description.signature()];
              if (!workspace) {
                workspace = prepare_sort_workspace(rt.arch, rt.get(), description);
                prepared->native_resources.push_back(workspace);
                prepared->replay_resources.insert(workspace.get());
              }
              descriptor.workspace = workspace;
            }
            operations.emplace_back(registry.prepare(preparation, descriptor));
          } else {
            operations.emplace_back(registry.prepare(preparation, request, inputs));
          }
        } catch (const std::invalid_argument &error) {
          throw MemoryBindingError(error.what());
        }
      } else {
        const auto &fill = std::get<aot::ExecutionPlan::Fill>(operation);
        const auto dst = storage(fill.destination, fill.offset);
        const auto bytes = fill.bytes.value_or(dst.size);
        const auto &array = arguments.at(fill.destination)->value.ndarray;
        if (bytes > dst.size ||
            bytes > std::numeric_limits<std::size_t>::max() ||
            rt.get_memory(array.memory, bool(domain)).base_offset % 4)
          throw MemoryBindingError(
              "Execution fill exceeds ndarray extent or alignment");
        operations.emplace_back(FixedExecution::BufferFill{
            dst.pointer, static_cast<std::size_t>(bytes), fill.value});
      }
    }
    return true;
  };
  if (!rebinding && !bind_steps(plan.initialization, explicit_initializers))
    return TI_NULL_HANDLE;
  if (family && !rebinding) {
    family->objects = native_objects;
    std::vector<TiMemory> dependencies;
    for (const auto &name : plan.initialization_bindings) {
      const auto found = arguments.find(name);
      if (found != arguments.end() &&
          found->second->type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        family->initialization_external_resources.insert(
            found->second->value.native_resource);
      if (found != arguments.end() &&
          found->second->type == TI_ARGUMENT_TYPE_NDARRAY)
        dependencies.push_back(found->second->value.ndarray.memory);
      if (const auto binding = layouts.find(name);
          binding != layouts.end() && binding->second->gradient)
        dependencies.push_back(binding->second->gradient->memory);
    }
    if (prepared->resources)
      prepared->resources->remove_owned_dependencies(dependencies);
    if (family->arrays->storage)
      family->arrays->storage->remove_owned_dependencies(dependencies);
    family->initialization_memories.insert(dependencies.begin(),
                                           dependencies.end());
    for (const auto &name : family->persistent_public_arguments) {
      if (!plan.replay_bindings.count(name))
        continue;
      const auto &argument = *arguments.at(name);
      if (argument.type == TI_ARGUMENT_TYPE_NDARRAY)
        family->persistent_memories.insert(argument.value.ndarray.memory);
      else if (argument.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        family->persistent_external_resources.insert(
            argument.value.native_resource);
    }
  }
  preparing_initialization = false;
  if (!bind_steps(plan.operations, operations))
    return TI_NULL_HANDLE;
  if (patched_parameters.size() != parameter_slots.size())
    throw MemoryBindingError(
        "Submission parameter is not used by a replay kernel");
  if (used_bindings.size() != layouts.size())
    throw MemoryBindingError(
        "Ndarray layout/gradient binding is unused by any kernel");
  if (domain) {
    auto external_access = [&](const std::set<std::string> &names) {
      std::vector<std::shared_ptr<ExternalSynchronizationDomain>> access;
      auto add = [&](TiMemory handle) {
        const auto memory = rt.get_memory(handle, true);
        if (memory.borrowed && memory.borrowed->external_access)
          access.push_back(memory.borrowed->external_access);
      };
      for (const auto &name : names) {
        const auto argument = arguments.find(name);
        if (argument != arguments.end() &&
            argument->second->type == TI_ARGUMENT_TYPE_NDARRAY)
          add(argument->second->value.ndarray.memory);
        const auto binding = layouts.find(name);
        if (binding != layouts.end() && binding->second->gradient)
          add(binding->second->gradient->memory);
      }
      return access;
    };
    prepared->initial_access = external_access(plan.initialization_bindings);
    prepared->replay_access = external_access(plan.replay_bindings);
    if (rt.arch == taichi::Arch::vulkan && !prepared->initial_access.empty())
      throw MemoryBindingError(
          "Vulkan external initialization requires a separately qualified producer epoch");
  }
  {
    for (const auto &name : plan.replay_bindings) {
      if (const auto arg = arguments.find(name); arg != arguments.end() &&
          arg->second->type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        prepared->replay_external_resources.insert(
            arg->second->value.native_resource);
      if (const auto object = native_objects.find(name);
          object != native_objects.end()) {
        prepared->replay_resources.insert(object->second.get());
        continue;
      }
      // Affine views have no separate allocation. Struct parents are retained
      // by the replay closure; Field roots live with this execution's module.
      if (const auto member = native_arrays.find(name);
          member != native_arrays.end() && member->second.strides)
        continue;
      const auto &argument = *arguments.at(name);
      if (argument.type == TI_ARGUMENT_TYPE_NDARRAY) {
        prepared->replay_memories.insert(argument.value.ndarray.memory);
        // A view's public storage ancestor may not appear as a direct kernel
        // argument. Retiring that ancestor must still invalidate this plan.
        prepared->memories.push_back(argument.value.ndarray.memory);
      }
      if (const auto binding = layouts.find(name);
          binding != layouts.end() && binding->second->gradient)
        prepared->replay_memories.insert(binding->second->gradient->memory);
    }
  }
  if (family) {
    if (!family->initialized)
      prepared->external_resources.insert(
          prepared->external_resources.end(),
          family->initialization_external_resources.begin(),
          family->initialization_external_resources.end());
    prepared->external_resources.insert(
        prepared->external_resources.end(),
        family->persistent_external_resources.begin(),
        family->persistent_external_resources.end());
    prepared->replay_external_resources.insert(
        family->persistent_external_resources.begin(),
        family->persistent_external_resources.end());
    if (!family->initialized)
      prepared->memories.insert(prepared->memories.end(),
                                family->initialization_memories.begin(),
                                family->initialization_memories.end());
    prepared->memories.insert(prepared->memories.end(),
                              family->persistent_memories.begin(),
                              family->persistent_memories.end());
    prepared->replay_memories.insert(family->persistent_memories.begin(),
                                     family->persistent_memories.end());
    if (family->arrays->storage)
      family->arrays->storage->remove_owned_dependencies(prepared->memories);
  }
  if (prepared->resources)
    prepared->resources->remove_owned_dependencies(prepared->memories);
  // Operation factories can defer initialization too (for example an SBT).
  // Collect after binding both regions, before any user initializer consumes
  // those owners. No factory initialization is part of steady replay.
  for (auto &operation : resource_initializers)
    initializers.emplace_back(std::move(operation));
  for (auto &operation : explicit_initializers)
    initializers.emplace_back(std::move(operation));
  if (plan.vulkan_submission && operations.size() != plan.operations.size())
    throw MemoryBindingError("Selected Vulkan submission changed replay boundaries");
  if (plan.storage_ring && operations.size() != plan.operations.size())
    throw MemoryBindingError("Ordered storage ring changed replay boundaries");
  if (domain) {
    // Register the complete code/storage closure before a domain may retain
    // any recording, including a partially prepared initializer.
    auto &registry = rt.prepared_registry();
    const auto handle = registry.publish(std::move(prepared));
    try {
      auto &entry = registry.get(handle);
      entry.prepare_operations(
          rt, operations, initializers,
          plan.vulkan_submission ? &*plan.vulkan_submission : nullptr,
          plan.storage_ring ? &*plan.storage_ring : nullptr);
      entry.publish_bootstrap(rt);
    } catch (...) {
      registry.preparation_failed(handle);
      throw;
    }
    return handle;
  }
  prepared->prepare_operations(rt, operations, initializers,
                              plan.vulkan_submission ? &*plan.vulkan_submission : nullptr,
                              plan.storage_ring ? &*plan.storage_ring : nullptr);
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiPreparedExecution ti_prepare_execution_plan_ext(
    TiRuntime runtime,
    TiAotModule module,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  return prepare_execution_plan_impl(runtime, module, description_size,
                                     description, arg_count, args,
                                     binding_count, bindings, {});
}

#ifdef TI_WITH_CUDA
TiPreparedExecution ti_prepare_execution_plan_on_domain_ext(
    TiRuntime runtime,
    TiAotModule module,
    TiCudaExecutionDomain domain,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!domain) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "domain");
    return TI_NULL_HANDLE;
  }
  return prepare_execution_plan_impl(runtime, module, description_size,
                                     description, arg_count, args,
                                     binding_count, bindings, {},
                                     std::make_shared<capi::PreparedExecutionDomain>(
                                         capi::get_execution_domain(domain)));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiPreparedExecution ti_prepare_execution_plan_parameters_on_domain_ext(
    TiRuntime runtime,
    TiAotModule module,
    TiCudaExecutionDomain domain,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings,
    uint32_t parameter_count,
    const char *const *parameter_names) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!domain) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "domain");
    return TI_NULL_HANDLE;
  }
  return prepare_execution_plan_impl(
      runtime, module, description_size, description, arg_count, args,
      binding_count, bindings, {},
      std::make_shared<capi::PreparedExecutionDomain>(capi::get_execution_domain(domain)),
      parameter_count, parameter_names);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
#endif

#ifdef TI_WITH_VULKAN
TiPreparedExecution ti_prepare_execution_plan_on_vulkan_domain_ext(
    TiRuntime runtime, TiAotModule module, TiVulkanExecutionDomain domain,
    uint64_t description_size, const char *description,
    uint32_t arg_count, const TiNamedArgument *args,
    uint32_t binding_count, const TiNamedNdArrayBinding *bindings,
    uint32_t parameter_count, const char *const *parameter_names) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(domain);
  return prepare_execution_plan_impl(
      runtime, module, description_size, description, arg_count, args,
      binding_count, bindings, {},
      std::make_shared<capi::PreparedExecutionDomain>(capi::get_execution_domain(domain)),
      parameter_count, parameter_names);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
#endif

static TiBool try_submit_prepared_impl(TiRuntime runtime,
                                       TiPreparedExecution execution,
                                       uint32_t dependency_count,
                                       const TiCompletion *dependencies,
                                       TiCompletion *completion,
                                       bool ready_only,
                                       uint32_t parameter_count = 0,
                                       const uint64_t *parameters = nullptr,
                                       bool sample_gpu = false) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(completion);
  *completion = TI_NULL_HANDLE;
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(execution);
  if (dependency_count && !dependencies)
    throw MemoryBindingError("Missing completion dependency array");
  std::vector<taichi::lang::RuntimeCompletion> predecessors;
  predecessors.reserve(dependency_count);
  for (uint32_t i = 0; i < dependency_count; ++i)
    predecessors.push_back(capi::get_completion(dependencies[i]));
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto &entry = rt.prepared_registry().get(execution);
  if (entry.submission_values.size() != parameter_count ||
      (parameter_count && !parameters))
    throw MemoryBindingError(
        "Submission parameter count does not match preparation");
  // This host snapshot has no device effects and is never the DMA source.
  // The instance copies it to its pinned storage only after admission.
  for (uint32_t i = 0; i < parameter_count; ++i)
    if (entry.submission_widths[i] == 4 && parameters[i] > UINT32_MAX)
      throw MemoryBindingError(
          "32-bit submission parameter requires zero upper 32 bits");
  if (parameter_count)
    std::copy_n(parameters, parameter_count, entry.submission_values.begin());
  if (ready_only) {
    const auto &entry = rt.prepared_registry().get(execution);
    if (!entry.live || !entry.domain || !entry.domain_initialized)
      throw MemoryBindingError(
          "Ready submit requires an initialized live execution-domain "
          "instance");
    const auto &owner = rt.runtime_fault_domain();
    owner->throw_if_new_submission_disallowed("ready domain submission");
    const auto domain = entry.domain->snapshot();
    if (!domain.accepting || domain.closed)
      throw MemoryBindingError("Execution domain is closing or closed");
    // Validate the entire list before querying any prefix. A pending valid
    // ticket must not hide a foreign or invalid ticket later in the list.
    for (const auto &dependency : predecessors) {
      if (entry.domain->backend() == taichi::Arch::cuda)
        dependency.validate_cuda_dependency(domain.owner, owner);
      else
        dependency.reusable_stream_dependency(entry.domain->backend(), domain.owner, owner);
    }
    for (const auto &dependency : predecessors)
      if (!dependency.done())
        return TI_FALSE;
  }
  capi::CompletionPublication publication;
  try {
    *completion = publication.commit(
        rt.prepared_registry().submit(execution, predecessors, sample_gpu));
  } catch (const std::exception &error) {
#ifdef TI_WITH_CUDA
    if (dynamic_cast<const taichi::lang::cuda::ExecutionDomainBusy *>(&error))
      return TI_FALSE;
#endif
#ifdef TI_WITH_VULKAN
    if (dynamic_cast<const taichi::lang::vulkan::ExecutionDomain::Busy *>(&error))
      return TI_FALSE;
#endif
    throw;
  }
  return TI_TRUE;
  TI_CAPI_TRY_CATCH_END();
  return TI_FALSE;
}

TiBool ti_try_submit_prepared_ext(TiRuntime runtime,
                                  TiPreparedExecution execution,
                                  uint32_t dependency_count,
                                  const TiCompletion *dependencies,
                                  TiCompletion *completion) {
  return try_submit_prepared_impl(runtime, execution, dependency_count,
                                  dependencies, completion, false);
}

TiBool ti_try_submit_prepared_ready_ext(TiRuntime runtime,
                                        TiPreparedExecution execution,
                                        uint32_t dependency_count,
                                        const TiCompletion *dependencies,
                                        TiCompletion *completion) {
  return try_submit_prepared_impl(runtime, execution, dependency_count,
                                  dependencies, completion, true);
}

TiBool ti_try_submit_prepared_parameters_ext(TiRuntime runtime,
                                             TiPreparedExecution execution,
                                             uint32_t dependency_count,
                                             const TiCompletion *dependencies,
                                             TiCompletion *completion,
                                             TiBool ready_only,
                                             uint32_t parameter_count,
                                             const uint64_t *parameters) {
  return try_submit_prepared_impl(
      runtime, execution, dependency_count, dependencies, completion,
      ready_only != TI_FALSE, parameter_count, parameters);
}
TiBool ti_try_submit_prepared_sampled_ext(TiRuntime runtime,
                                          TiPreparedExecution execution,
                                          uint32_t dependency_count,
                                          const TiCompletion *dependencies,
                                          TiCompletion *completion,
                                          TiBool ready_only,
                                          TiBool sample_gpu,
                                          uint32_t parameter_count,
                                          const uint64_t *parameters) {
  return try_submit_prepared_impl(runtime, execution, dependency_count,
                                  dependencies, completion,
                                  ready_only != TI_FALSE, parameter_count,
                                  parameters, sample_gpu != TI_FALSE);
}

TiPreparedExecution ti_prepare_execution_binding_ext(
    TiRuntime runtime,
    TiPreparedExecution source,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(source);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &entry = rt.prepared_registry().get(source);
  if (!entry.live || !entry.family)
    throw MemoryBindingError(
        "Preparing a binding requires a live CUDA binding family");
  return prepare_execution_plan_impl(
      runtime, reinterpret_cast<TiAotModule>(entry.owner), 0, "", arg_count,
      args, binding_count, bindings, entry.family);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_get_prepared_binding_family_info_ext(
    TiRuntime runtime,
    TiPreparedExecution execution,
    TiPreparedBindingFamilyInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  *info = {};
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &entry = rt.prepared_registry().get(execution);
  if (!entry.live)
    throw MemoryBindingError("Prepared execution has a retired dependency");
  if (entry.family) {
    info->identity = entry.family->identity;
    info->bindings = rt.prepared_registry().family_size(entry.family.get());
    info->resource_bytes = entry.family->arrays->storage
                               ? entry.family->arrays->storage->bytes()
                               : 0;
#ifdef TI_WITH_CUDA
    info->executables = entry.family->replay->snapshot().executables;
#endif
  }
  TI_CAPI_TRY_CATCH_END();
}

void Runtime::configure_native_provider(
    std::string provider,
    std::optional<taichi::lang::NativeProviderLocation> location) {
  if (provider.empty() || provider.find('\0') != std::string::npos)
    throw MemoryBindingError("Provider name must be nonempty without NUL");
  if (location) {
    try {
      taichi::lang::validate_native_provider_location(provider, *location);
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
    native_provider_locations_.insert_or_assign(std::move(provider),
                                                std::move(*location));
  } else {
    native_provider_locations_.erase(provider);
  }
}

void Runtime::configure_native_library(std::string name, const char *path) {
  if (name.empty() || name.find('\0') != std::string::npos)
    throw MemoryBindingError(
        "Native library name must be nonempty without NUL");
  try {
    if (path)
      taichi::lang::validate_native_library_location(name, path);
  } catch (const std::invalid_argument &error) {
    throw MemoryBindingError(error.what());
  }
  if (path)
    native_library_locations_.insert_or_assign(std::move(name), path);
  else
    native_library_locations_.erase(name);
}

void ti_configure_native_library_ext(TiRuntime runtime,
                                     const char *name,
                                     const char *path) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(name);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.configure_native_library(name, path);
  TI_CAPI_TRY_CATCH_END();
}

void ti_configure_native_provider_ext(TiRuntime runtime,
                                      const char *provider,
                                      const char *adapter_path,
                                      const char *library_path) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(provider);
  TI_CAPI_INVALID_ARGUMENT(!adapter_path && library_path && *library_path);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  std::optional<taichi::lang::NativeProviderLocation> location;
  if (adapter_path)
    location = {adapter_path, library_path ? library_path : ""};
  rt.configure_native_provider(provider, std::move(location));
  TI_CAPI_TRY_CATCH_END();
}

namespace {
TiFlags native_object_capabilities(TiRuntime runtime,
                                   const char *kind,
                                   uint32_t version,
                                   taichi::lang::NativeObjectKind object_kind) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!runtime || !kind) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "runtime or operation kind");
    return 0;
  }
  if (!*kind || !version) {
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "operation kind or version");
    return 0;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto *capability = taichi::lang::native_operation_registry().find(
      kind, version, rt.arch, object_kind);
  static_assert(static_cast<TiFlags>(taichi::lang::NativeOperationMode::cpu) ==
                TI_NATIVE_OPERATION_CAPABILITY_CPU_BIT_EXT);
  static_assert(
      static_cast<TiFlags>(taichi::lang::NativeOperationMode::cuda_capture) ==
      TI_NATIVE_OPERATION_CAPABILITY_CUDA_CAPTURE_BIT_EXT);
  static_assert(
      static_cast<TiFlags>(taichi::lang::NativeOperationMode::vulkan_inline) ==
      TI_NATIVE_OPERATION_CAPABILITY_VULKAN_INLINE_BIT_EXT);
  static_assert(static_cast<TiFlags>(
                    taichi::lang::NativeOperationMode::cuda_synchronous) ==
                TI_NATIVE_OPERATION_CAPABILITY_CUDA_SYNCHRONOUS_BIT_EXT);
  static_assert(
      static_cast<TiFlags>(taichi::lang::NativeOperationMode::cuda_ordered) ==
      TI_NATIVE_OPERATION_CAPABILITY_CUDA_ORDERED_BIT_EXT);
  return capability ? static_cast<TiFlags>(capability->mode) : 0;
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

void probe_native_object(TiRuntime runtime,
                         const char *kind,
                         uint32_t version,
                         TiNativeOperationAvailabilityExt *availability,
                         taichi::lang::NativeObjectKind object_kind) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(availability);
  *availability = {};
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(kind);
  if (!*kind || !version) {
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "operation kind or version");
    return;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &registry = taichi::lang::native_operation_registry();
  const auto *capability = registry.find(kind, version, rt.arch, object_kind);
  if (capability) {
    taichi::lang::NativeProviderSession providers(
        rt.native_provider_locations(), rt.native_library_locations());
    const auto status =
        registry.probe(kind, version, rt.arch, providers, object_kind);
    availability->capabilities = static_cast<TiFlags>(capability->mode);
    availability->available = status.available ? TI_TRUE : TI_FALSE;
    availability->library_version = status.library_version;
  }
  TI_CAPI_TRY_CATCH_END();
}
}  // namespace

TiFlags ti_get_native_operation_capabilities_ext(TiRuntime runtime,
                                                 const char *kind,
                                                 uint32_t version) {
  return native_object_capabilities(runtime, kind, version,
                                    taichi::lang::NativeObjectKind::operation);
}

TiFlags ti_get_native_resource_capabilities_ext(TiRuntime runtime,
                                                const char *kind,
                                                uint32_t version) {
  return native_object_capabilities(runtime, kind, version,
                                    taichi::lang::NativeObjectKind::resource);
}

void ti_probe_native_operation_ext(
    TiRuntime runtime,
    const char *kind,
    uint32_t version,
    TiNativeOperationAvailabilityExt *availability) {
  probe_native_object(runtime, kind, version, availability,
                      taichi::lang::NativeObjectKind::operation);
}

void ti_probe_native_resource_ext(
    TiRuntime runtime,
    const char *kind,
    uint32_t version,
    TiNativeOperationAvailabilityExt *availability) {
  probe_native_object(runtime, kind, version, availability,
                      taichi::lang::NativeObjectKind::resource);
}

void ti_initialize_prepared_ext(TiRuntime runtime,
                                TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().initialize(execution);
  TI_CAPI_TRY_CATCH_END();
}

void ti_launch_prepared_ext(TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().run(execution);
  TI_CAPI_TRY_CATCH_END();
}
void ti_read_prepared_kernel_result_ext(TiRuntime runtime,
                                        TiPreparedExecution execution,
                                        uint64_t result_size,
                                        void *result_data) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto &entry = rt.prepared_registry().get(execution);
  if (!entry.live || !entry.kernel_result || !entry.result_valid || entry.domain)
    throw MemoryBindingError(
        "Prepared result requires a live direct kernel with a successful submission");
  if (result_size != entry.kernel_result->device_payload_bytes() || !result_data)
    throw MemoryBindingError("Prepared result destination must match compiled size");
  // Explicit observation is the only synchronization point. The submission
  // mutex prevents another replay from overwriting the frame during its copy.
  try {
    rt.wait();
    rt.complete_submission_completions();
    entry.kernel_result->read_completed(result_data, result_size);
  } catch (...) {
    entry.result_valid = false;
    throw;
  }
  TI_CAPI_TRY_CATCH_END();
}
TiSubmissionTicket ti_submit_prepared_ext(TiRuntime runtime,
                                         TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  if (rt.prepared_registry().get(execution).domain)
    throw MemoryBindingError(
        "Selected-domain execution requires ti_try_submit_prepared_ext; "
        "an ordinary queue ticket cannot represent its completion");
  rt.prepare_submission_completion();
  rt.prepared_registry().run(execution);
  return rt.finish_submission_completion();
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
void ti_get_prepared_execution_info_ext(TiRuntime runtime,
                                        TiPreparedExecution execution,
                                        TiPreparedExecutionInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  *info = rt.prepared_registry().get(execution).info;
  TI_CAPI_TRY_CATCH_END();
}
uint64_t ti_get_prepared_execution_workspace_bytes_ext(
    TiRuntime runtime,
    TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!runtime || !execution) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "runtime or execution");
    return 0;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  return rt.prepared_registry().get(execution).workspace_bytes();
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

TiCudaExecutionInstance ti_get_prepared_cuda_instance_ext(
    TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(execution);
#ifdef TI_WITH_CUDA
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto &entry = rt.prepared_registry().get(execution);
  return capi::publish_execution_instance(entry.domain ? entry.domain->cuda : nullptr,
                                          entry.domain_registration.cuda);
#else
  ti_set_last_error(TI_ERROR_NOT_SUPPORTED, "CUDA execution instances unavailable");
#endif
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

#ifdef TI_WITH_VULKAN
TiVulkanExecutionInstance ti_get_prepared_vulkan_instance_ext(
    TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto &entry = rt.prepared_registry().get(execution);
  return capi::publish_execution_instance(entry.domain ? entry.domain->vulkan : nullptr,
                                          entry.domain_registration.vulkan);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
#endif

void ti_destroy_prepared_ext(TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().destroy(execution);
  TI_CAPI_TRY_CATCH_END();
}

void ti_finish_prepared_cuda_instance_retirement_ext(
    TiRuntime runtime, TiPreparedExecution execution) {
  ti_finish_prepared_instance_retirement_ext(runtime, execution);
}

void ti_finish_prepared_instance_retirement_ext(
    TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().finish_instance_retirement(execution);
  TI_CAPI_TRY_CATCH_END();
}

uint64_t ti_get_prepared_execution_resource_bytes_ext(
    TiRuntime runtime,
    TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!runtime || !execution) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "runtime or execution");
    return 0;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  return rt.prepared_registry().get(execution).array_bytes();
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

void ti_get_prepared_execution_memory_info_ext(
    TiRuntime runtime,
    TiPreparedExecution execution,
    TiPreparedExecutionMemoryInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  TI_CAPI_ARGUMENT_NULL(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &prepared = rt.prepared_registry().get(execution);
  *info = {prepared.array_bytes(), prepared.native_resource_payload_bytes,
           prepared.workspace_bytes()};
  TI_CAPI_TRY_CATCH_END();
}
