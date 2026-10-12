#include "taichi/runtime/aot_runtime_client.h"

#include <atomic>
#include <exception>
#include "taichi/program/program.h"
#include "taichi_core_impl.h"
#ifdef TI_WITH_LLVM
#include "taichi_llvm_impl.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi_vulkan_impl.h"
#endif

namespace taichi::lang {
namespace {
#ifdef TI_WITH_VULKAN
class BorrowedVulkanRuntime final : public ::VulkanRuntime {
 public:
  explicit BorrowedVulkanRuntime(gfx::GfxRuntime &runtime) : runtime_(runtime) {
  }
  Device &get() override {
    return *runtime_.get_ti_device();
  }
  gfx::GfxRuntime &get_gfx_runtime() override {
    return runtime_;
  }

 private:
  gfx::GfxRuntime &runtime_;
};
#endif

std::unique_ptr<::Runtime> borrow_backend(Program &program) {
  const auto arch = program.compile_config().arch;
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch) || arch == Arch::cuda) {
    auto *impl = static_cast<LlvmProgramImpl *>(program.get_program_impl());
    return std::make_unique<capi::LlvmRuntime>(*impl->get_runtime_executor(),
                                               program.result_buffer);
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    auto *runtime = program.get_gfx_runtime_for_borrowing();
    if (!runtime)
      throw std::runtime_error("The Vulkan runtime is not materialized");
    return std::make_unique<BorrowedVulkanRuntime>(*runtime);
  }
#endif
  throw std::invalid_argument(
      "AOT runtime borrowing requires CPU, CUDA or Vulkan");
}
}  // namespace

struct AotRuntimeClient::State {
  State(Program &program, std::unique_ptr<::Runtime> runtime)
      : program(&program),
        runtime(std::move(runtime)),
        arch(program.compile_config().arch),
        device_id(reinterpret_cast<std::uintptr_t>(&this->runtime->get())) {
    if (&this->runtime->get() != program.get_compute_device())
      throw std::logic_error("AOT borrower did not share its Program device");
  }
  std::recursive_mutex mutex;
  Program *program;
  std::unique_ptr<::Runtime> runtime;
  const Arch arch;
  const std::uintptr_t device_id;
  std::atomic<bool> closed{false};
  std::vector<Program::RuntimeResourceSubmissionGuard> scopes;

  void finish(bool synchronize) {
    closed.store(true, std::memory_order_release);
    auto released = std::move(runtime);
    program = nullptr;
    if (released)
      released->retire_resources(synchronize);
  }
};

AotRuntimeClient::AotRuntimeClient(std::unique_ptr<State> state)
    : state_(std::move(state)) {
}
AotRuntimeClient::~AotRuntimeClient() = default;

std::shared_ptr<AotRuntimeClient> AotRuntimeClient::create(Program &program) {
  auto guard = program.acquire_runtime_resource_submission_guard();
  program.ensure_runtime_submission_allowed("borrow AOT runtime");
  auto result = std::shared_ptr<AotRuntimeClient>(new AotRuntimeClient(
      std::make_unique<State>(program, borrow_backend(program))));
  program.retain_runtime_client(result);
  return result;
}

void AotRuntimeClient::enter() {
  auto &state = *state_;
  state.mutex.lock();
  try {
    // Preserve one empty scope for closed wrappers so late close/GC can clear
    // their Python registrations without calling a dead backend.
    Program::RuntimeResourceSubmissionGuard guard;
    if (!state.closed.load(std::memory_order_acquire)) {
      try {
        guard = state.program->acquire_runtime_resource_submission_guard();
        state.program->ensure_runtime_submission_allowed("borrowed AOT call");
        if (!state.program->runtime_submission_allowed()) {
          state.closed.store(true, std::memory_order_release);
          guard.unlock();
        }
      } catch (const TaichiRuntimeError &) {
        // The retained registration transfers cleanup to the owner finalizer.
        state.closed.store(true, std::memory_order_release);
        if (guard.owns_lock())
          guard.unlock();
      }
    }
    state.scopes.emplace_back(std::move(guard));
  } catch (...) {
    state.mutex.unlock();
    throw;
  }
}

void AotRuntimeClient::leave() {
  auto &state = *state_;
  if (state.scopes.empty())
    throw std::logic_error("Unbalanced borrowed AOT scope");
  if (state.program && state.scopes.back().owns_lock())
    state.program->mark_runtime_submission_pending();
  state.scopes.pop_back();
  state.mutex.unlock();
}

void AotRuntimeClient::close() {
  enter();
  auto &state = *state_;
  std::exception_ptr failure;
  if (state.runtime && state.scopes.back().owns_lock()) {
    auto *program = state.program;
    try {
      state.finish(true);
    } catch (...) {
      failure = std::current_exception();
    }
    program->detach_runtime_client(this);
  }
  leave();
  if (failure)
    std::rethrow_exception(failure);
}

void AotRuntimeClient::retire() {
  std::lock_guard<std::recursive_mutex> lock(state_->mutex);
  state_->finish(false);
}

bool AotRuntimeClient::closed() const {
  return state_->closed.load(std::memory_order_acquire);
}
std::uintptr_t AotRuntimeClient::handle() const {
  std::lock_guard<std::recursive_mutex> lock(state_->mutex);
  return reinterpret_cast<std::uintptr_t>(state_->runtime.get());
}
std::uintptr_t AotRuntimeClient::device_identity() const {
  return state_->device_id;
}
Arch AotRuntimeClient::arch() const {
  return state_->arch;
}
}  // namespace taichi::lang
