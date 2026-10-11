#include "taichi/runtime/fixed_execution.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include "taichi/aot/graph_data.h"
#include "taichi/program/launch_context_builder.h"
#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/gfx/aot_graph_data.h"
#endif

namespace taichi::lang {
struct FixedExecution::State {
  std::uint32_t kind{0};
  std::size_t bytes{0};
#ifdef TI_WITH_LLVM
  struct CpuStep {
    cpu::KernelLauncher *launcher{nullptr};
    std::vector<cpu::KernelLauncher::PreparedLaunch> launches;
    std::function<void()> native;
  };
  std::vector<CpuStep> cpu_steps;
#endif
#ifdef TI_WITH_CUDA
  std::unique_ptr<cuda::FixedGraph> cuda_graph;
#endif
#ifdef TI_WITH_VULKAN
  std::unique_ptr<gfx::GraphReplayRegistration> vulkan_graph;
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

FixedExecution::FixedExecution(Arch arch,
                               Device &device,
                               gfx::GfxRuntime *gfx_runtime,
                               const std::vector<Operation> &operations)
    : state_(std::make_unique<State>()) {
  if (operations.empty())
    throw std::invalid_argument("Fixed execution requires a nonempty batch");
  // Validate every descriptor before preparing backend state or commands.
  for (const auto &operation : operations) {
    if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
      if (!dispatch->kernel || !dispatch->context ||
          dispatch->kernel->ret_size ||
          !dispatch->context->argpack_ptrs.empty())
        throw std::invalid_argument(
            "Fixed execution requires loaded entries without returns or "
            "ArgPacks");
      for (const auto &[index, param] : dispatch->kernel->nested_parameters) {
        if (param.ptype == ParameterType::kTexture ||
            param.ptype == ParameterType::kRWTexture)
          throw std::invalid_argument(
              "Fixed textures require generation-qualified resource owners");
      }
    } else if (const auto *copy = std::get_if<BufferCopy>(&operation)) {
      if (copy->source.device != &device || copy->destination.device != &device)
        throw std::invalid_argument(
            "Fixed copy storage belongs to another device");
    } else {
      const auto &fill = std::get<BufferFill>(operation);
      if (fill.destination.device != &device || fill.destination.offset % 4 ||
          fill.bytes % 4)
        throw std::invalid_argument(
            "Fixed fill requires aligned 32-bit storage");
    }
  }
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    for (const auto &operation : operations) {
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        auto *loaded = dynamic_cast<llvm_aot::KernelImpl *>(dispatch->kernel);
        auto *launcher =
            loaded ? dynamic_cast<cpu::KernelLauncher *>(&loaded->launcher())
                   : nullptr;
        if (!launcher)
          throw std::invalid_argument("Fixed CPU entry is not registered");
        auto &steps = state_->cpu_steps;
        if (steps.empty() || steps.back().launcher != launcher)
          steps.push_back({launcher, {}, {}});
        steps.back().launches.push_back(
            launcher->prepare_launch(loaded->handle(), *dispatch->context));
        state_->bytes += dispatch->context->arg_buffer_size;
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
        state_->cpu_steps.push_back(std::move(step));
      }
    }
    return;
  }
#endif
#ifdef TI_WITH_CUDA
  if (arch == Arch::cuda) {
    std::vector<cuda::FixedGraph::Dispatch> native;
    for (const auto &operation : operations) {
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        auto *loaded = dynamic_cast<llvm_aot::KernelImpl *>(dispatch->kernel);
        auto *launcher =
            loaded ? dynamic_cast<cuda::KernelLauncher *>(&loaded->launcher())
                   : nullptr;
        if (!launcher)
          throw std::invalid_argument("Fixed CUDA entry is not registered");
        native.push_back({launcher, loaded->handle(), dispatch->context, {}});
      } else if (const auto *copy = std::get_if<BufferCopy>(&operation)) {
        auto *dst = cuda_pointer(copy->destination);
        auto *src = cuda_pointer(copy->source);
        auto bytes = copy->bytes;
        check_pointer_copy(dst, src, bytes);
        native.push_back(
            {nullptr, {}, nullptr, [dst, src, bytes](void *stream) {
               if (bytes)
                 CUDADriver::get_instance().memcpy_device_to_device_async(
                     dst, src, bytes, stream);
             }});
      } else {
        const auto &fill = std::get<BufferFill>(operation);
        auto *dst = cuda_pointer(fill.destination);
        auto count = fill.bytes / 4;
        auto value = fill.value;
        native.push_back(
            {nullptr, {}, nullptr, [dst, count, value](void *stream) {
               if (count)
                 CUDADriver::get_instance().memsetd32_async(dst, value, count,
                                                            stream);
             }});
      }
    }
    state_->cuda_graph = std::make_unique<cuda::FixedGraph>(native);
    state_->kind = 1;
    state_->bytes = state_->cuda_graph->argument_bytes();
    return;
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    std::vector<gfx::GfxRuntime::GraphRecordingOperation> native;
    for (const auto &operation : operations) {
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        auto *loaded = dynamic_cast<gfx::KernelImpl *>(dispatch->kernel);
        if (!loaded || (gfx_runtime && gfx_runtime != &loaded->runtime()))
          throw std::invalid_argument(
              "Fixed Vulkan entries must share a runtime");
        gfx_runtime = &loaded->runtime();
        native.push_back({{loaded->handle(), dispatch->context}, {}});
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
    if (!gfx_runtime)
      throw std::invalid_argument("Fixed Vulkan operations require a runtime");
    state_->vulkan_graph = gfx_runtime->prepare_fixed_graph(native, {});
    state_->kind = 2;
    state_->bytes =
        state_->vulkan_graph->snapshot_stats().known_persistent_argument_bytes;
    return;
  }
#endif
  throw std::invalid_argument("Fixed execution requires CPU, CUDA or Vulkan");
}

FixedExecution::~FixedExecution() = default;
void FixedExecution::run() {
#ifdef TI_WITH_LLVM
  for (auto &step : state_->cpu_steps) {
    if (step.launcher)
      step.launcher->launch_prepared(step.launches);
    else
      step.native();
  }
#endif
#ifdef TI_WITH_CUDA
  if (state_->cuda_graph)
    state_->cuda_graph->run();
#endif
#ifdef TI_WITH_VULKAN
  if (state_->vulkan_graph)
    state_->vulkan_graph->launch_prepared();
#endif
}
std::uint32_t FixedExecution::execution_kind() const {
  return state_->kind;
}
std::size_t FixedExecution::argument_bytes() const {
  return state_->bytes;
}
}  // namespace taichi::lang
