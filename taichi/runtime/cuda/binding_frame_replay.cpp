#include "taichi/runtime/cuda/binding_frame_replay.h"

#include <deque>
#include <stdexcept>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang::cuda {

struct BindingFrameReplay::State {
  struct Pending {
    RuntimeCompletion completion;
    std::shared_ptr<void> frame;
  };
  std::uint64_t owner_domain;
  std::shared_ptr<RuntimeFaultDomain> fault_domain;
  std::shared_ptr<RuntimeCompletionCudaEventPool> events;
  CUgraphExec executable{nullptr};
  CUgraph active_graph{nullptr};
  std::shared_ptr<void> active;
  std::deque<Pending> pending;
  bool failed{false};
  bool closed{false};
  bool retired_safe{true};

  void retain(const std::shared_ptr<void> &frame) {
    pending.push_back({RuntimeCompletion::from_cuda_stream(
                           owner_domain, 0, nullptr, fault_domain, events),
                       frame});
  }
};

BindingFrameReplay::BindingFrameReplay(
    std::uint64_t owner_domain,
    std::shared_ptr<RuntimeFaultDomain> fault_domain)
    : state_(std::make_unique<State>()) {
  state_->owner_domain = owner_domain;
  state_->fault_domain = std::move(fault_domain);
  state_->events = std::make_shared<RuntimeCompletionCudaEventPool>(
      state_->fault_domain, 0,
      RuntimeCompletionCudaEventPool::Retention::until_clear);
}

BindingFrameReplay::~BindingFrameReplay() {
  retire_backend(true);
  release_frame_leases();
}

void BindingFrameReplay::collect() {
  while (!state_->pending.empty() && state_->pending.front().completion.done())
    state_->pending.pop_front();
}

void BindingFrameReplay::select(void *source_graph,
                                const std::shared_ptr<void> &frame) {
  auto &state = *state_;
  if (state.closed || state.failed)
    throw std::runtime_error("CUDA binding-frame executor is closed or failed");
  if (!source_graph || !frame)
    throw std::invalid_argument("CUDA binding frame needs a graph and owner");
  if (state.fault_domain)
    state.fault_domain->throw_if_submission_disallowed(
        "select CUDA binding frame");
  if (state.active == frame) {
    if (state.active_graph != source_graph)
      throw std::invalid_argument(
          "CUDA binding frame changed its immutable graph");
    return;
  }
  collect();
  auto &driver = CUDADriver::get_instance();
  try {
    if (!state.executable) {
      driver.graph_instantiate_with_flags(&state.executable, source_graph, 0);
    } else {
      state.retain(state.active);
      void *error_node = nullptr;
      std::uint32_t result = 0;
      const auto error = driver.graph_exec_update_v1.call(
          state.executable, source_graph, &error_node, &result);
      if (error != CUDA_SUCCESS || result != 0) {
        // Never replace the selected shared-executable strategy with another
        // executable, recapture, or an ordinary submission on update failure.
        state.failed = true;
        throw BackendRuntimeError(
            Arch::cuda, error, "graph_exec_update_v1",
            fmt::format(
                "CUDA binding-frame update rejected: driver={}, result={}",
                error, result));
      }
    }
  } catch (const BackendRuntimeError &error) {
    if (state.fault_domain)
      state.fault_domain->report_backend_error(error, 0);
    throw;
  }
  state.active = frame;
  state.active_graph = source_graph;
}

void BindingFrameReplay::prepare(void *source_graph,
                                 const std::shared_ptr<void> &frame) {
  select(source_graph, frame);
  state_->retain(frame);
}

void BindingFrameReplay::run(void *source_graph,
                             const std::shared_ptr<void> &frame) {
  select(source_graph, frame);
  try {
    CUDADriver::get_instance().graph_launch(state_->executable, nullptr);
  } catch (const BackendRuntimeError &error) {
    if (state_->fault_domain)
      state_->fault_domain->report_backend_error(error, 0);
    throw;
  }
}

bool BindingFrameReplay::is_active(const void *frame) const {
  return state_->active.get() == frame;
}

bool BindingFrameReplay::failed() const {
  return state_->failed;
}

BindingFrameReplay::Snapshot BindingFrameReplay::snapshot() const {
  return {state_->failed, state_->executable ? 1u : 0u, state_->pending.size(),
          state_->events->snapshot()};
}

bool BindingFrameReplay::retire_backend(bool backend_safe) noexcept {
  auto &state = *state_;
  if (state.closed)
    return backend_safe && state.retired_safe;
  state.closed = true;
  bool safe =
      backend_safe &&
      (!state.fault_domain ||
       (state.fault_domain->backend_calls_safe() &&
        state.fault_domain->state() != RuntimeLifecycleState::kFinalized));
  if (safe) {
    try {
      auto &cuda = CUDAContext::get_instance();
      auto submission = cuda.get_submission_lock_guard();
      auto context = cuda.get_guard();
      auto &driver = CUDADriver::get_instance();
      if (state.executable || !state.pending.empty())
        driver.stream_synchronize(nullptr);
      if (state.executable)
        driver.graph_exec_destroy(state.executable);
    } catch (const BackendRuntimeError &error) {
      safe = false;
      if (state.fault_domain)
        state.fault_domain->report_backend_error(error, 0);
    } catch (...) {
      safe = false;
    }
  }
  state.executable = nullptr;
  state.retired_safe = safe;
  return safe;
}

void BindingFrameReplay::release_frame_leases() noexcept {
  // Never release operands while the executable can still reference them.
  if (!state_->closed)
    return;
  state_->pending.clear();
  state_->active.reset();
  state_->active_graph = nullptr;
  state_->events->clear();
}

}  // namespace taichi::lang::cuda
