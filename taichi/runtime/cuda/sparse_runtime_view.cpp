#include "taichi/runtime/cuda/sparse_runtime_view.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/cuda/retired_allocation.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/util/str.h"

namespace taichi::lang::cuda {
namespace {
std::atomic<std::uint64_t> next_view_identity{1};
}

struct SparseRuntimeView::DiagnosticSnapshot::State {
  std::shared_ptr<RuntimeFaultDomain> owner;
  std::uint64_t view_identity{0};
  LlvmSparseRuntimeDiagnostic *host{nullptr};
  // UVA alias of the pinned host storage, not a separate device allocation.
  LlvmSparseRuntimeDiagnostic *device{nullptr};
  ~State() {
    if (host && owner->backend_calls_safe()) {
      try {
        auto guard = CUDAContext::get_instance().get_guard();
        CUDADriver::get_instance().mem_free_host(host);
      } catch (...) {
        // Do not unpin storage after a failed device/owner retirement.
      }
    }
  }
};

struct SparseRuntimeView::State {
  const std::uint64_t identity{next_view_identity.fetch_add(1)};
  std::shared_ptr<RuntimeFaultDomain> owner;
  std::shared_ptr<SparseTreeAdmission::Loan> loan;
  std::vector<LlvmSparseRuntimeViewTree> trees;
  std::vector<void *> allocations;
  std::vector<std::shared_ptr<DiagnosticSnapshot>> snapshots;
  JITModuleCUDA *module{nullptr};
  LLVMRuntime *source{nullptr};
  LLVMRuntime *view{nullptr};
  void *stream{nullptr};
  void *temporary{nullptr};
  void *host{nullptr};
  LlvmSparseRuntimeViewTree *device_trees{nullptr};
  LlvmSparseRuntimeViewStatus *status{nullptr};
  void *release_function{nullptr};
  void *snapshot_function{nullptr};
  std::size_t allocated_bytes{0};
  bool attempted{false};
  bool retired{false};
  bool possibly_bound{false};
  bool ready{false};
  std::mutex retirement_mutex;

  void *allocate(std::size_t bytes) {
    if (!bytes ||
        bytes > std::numeric_limits<std::size_t>::max() - allocated_bytes)
      throw std::invalid_argument("CUDA sparse runtime storage size overflow");
    // Reserve host bookkeeping before allocating device memory.
    allocations.push_back(nullptr);
    auto &driver = CUDADriver::get_instance();
    driver.malloc(&allocations.back(), bytes);
    allocated_bytes += bytes;
    return allocations.back();
  }
  template <typename... Args>
  void call(void *function,
            const char *name,
            void *selected_stream,
            Args... args) {
    auto [pointers, sizes] = module->get_arg_pointers(args...);
    module->launch_function_with_stream(function, name, 1, 1, 0, pointers,
                                        sizes, selected_stream);
  }
  void read(void *device, std::size_t bytes) {
    auto &driver = CUDADriver::get_instance();
    driver.memcpy_device_to_host_async(host, device, bytes, stream);
    wait();
  }
  void wait() {
    auto &driver = CUDADriver::get_instance();
    const auto result = driver.stream_synchronize.call_concurrent(stream);
    if (result != CUDA_SUCCESS) {
      BackendRuntimeError error(
          Arch::cuda, result, "sparse view stream wait",
          driver.stream_synchronize.get_error_message(result));
      owner->report_backend_error(error, 0);
      throw error;
    }
  }
  void check_status(const char *operation) {
    read(status, sizeof(*status));
    const auto value = *static_cast<const LlvmSparseRuntimeViewStatus *>(host);
    if (value.error != LlvmSparseRuntimeViewError::success)
      throw std::invalid_argument(
          fmt::format("CUDA sparse view {} rejected tree {} (status {})",
                      operation, value.tree_id, static_cast<int>(value.error)));
  }
};

SparseRuntimeView::SparseRuntimeView() : state_(std::make_unique<State>()) {
}

SparseRuntimeView::~SparseRuntimeView() {
  try {
    retire();
  } catch (...) {
    // A failed manager return must retain the view AND its tree loans. Domain
    // owners normally quarantine us; this is the final leak-safe fallback.
    state_.release();
  }
}

void SparseRuntimeView::initialize(
    LlvmRuntimeExecutor &executor,
    const std::vector<LlvmSparseRuntimeViewTree> &trees,
    std::shared_ptr<RuntimeFaultDomain> owner) {
  auto &s = *state_;
  if (s.attempted || !owner || owner->backend() != Arch::cuda ||
      trees.empty() ||
      trees.size() > std::size_t(std::numeric_limits<int>::max()) ||
      trees.back().tree_id == std::numeric_limits<int>::max())
    throw std::invalid_argument("Invalid CUDA sparse runtime view preparation");
  owner->throw_if_new_submission_disallowed("prepare CUDA sparse runtime view");
  s.owner = std::move(owner);
  s.attempted = true;
  s.trees = trees;
  s.source = executor.get_llvm_runtime();
  s.module = dynamic_cast<JITModuleCUDA *>(executor.get_runtime_jit_module());
  if (!s.module)
    throw std::invalid_argument(
        "Sparse runtime view requires a CUDA runtime module");
  RuntimeCompletion producer;
  auto &context = CUDAContext::get_instance();
  auto guard = context.get_guard();
  {
    auto submission = context.get_submission_lock_guard();
    s.loan = executor.reserve_cuda_sparse_trees(trees);
    producer = RuntimeCompletion::from_cuda_stream(
        s.owner->snapshot().program_domain,
        s.owner->allocate_submission_sequence(), nullptr, s.owner);
    s.loan->publish_boundary(std::make_shared<RuntimeCompletion>(producer));
  }
  // No global submission gate is held while waiting on this exact producer.
  auto &driver = CUDADriver::get_instance();
  driver.stream_create(&s.stream, CU_STREAM_NON_BLOCKING);
  producer.enqueue_cuda_wait(s.stream, s.owner->snapshot().program_domain,
                             s.owner);
  s.release_function = s.module->lookup_function("runtime_sparse_view_release");
  s.snapshot_function =
      s.module->lookup_function("runtime_sparse_view_snapshot");
  auto *size_function =
      s.module->lookup_function("runtime_sparse_view_get_size");
  auto *initialize_function =
      s.module->lookup_function("runtime_sparse_view_initialize");
  auto *bind_function = s.module->lookup_function("runtime_sparse_view_bind");
  const auto input_bytes = trees.size() * sizeof(trees.front());
  driver.mem_host_alloc(&s.host, std::max(input_bytes, sizeof(std::uint64_t)),
                        0);
  auto *size_result =
      static_cast<std::uint64_t *>(s.allocate(sizeof(std::uint64_t)));
  s.call(size_function, "runtime_sparse_view_get_size", s.stream, size_result);
  s.read(size_result, sizeof(*size_result));
  const auto view_bytes = *static_cast<const std::uint64_t *>(s.host);
  if (!view_bytes || view_bytes % sizeof(std::uint32_t))
    throw std::invalid_argument("Invalid CUDA sparse runtime view size");
  s.view = static_cast<LLVMRuntime *>(s.allocate(view_bytes));
  driver.memsetd32_async(s.view, 0, view_bytes / sizeof(std::uint32_t),
                         s.stream);
  const int capacity = trees.back().tree_id + 1;
  auto *directory = s.allocate(std::size_t(capacity) * sizeof(void *));
  s.temporary = s.allocate(taichi_global_tmp_buffer_size);
  auto *result =
      s.allocate(taichi_result_buffer_entries * sizeof(std::uint64_t));
  s.status =
      static_cast<LlvmSparseRuntimeViewStatus *>(s.allocate(sizeof(*s.status)));
  s.device_trees =
      static_cast<LlvmSparseRuntimeViewTree *>(s.allocate(input_bytes));
  std::memcpy(s.host, trees.data(), input_bytes);
  driver.memcpy_host_to_device_async(s.device_trees, s.host, input_bytes,
                                     s.stream);
  s.call(initialize_function, "runtime_sparse_view_initialize", s.stream,
         s.source, s.view, directory, capacity, s.temporary, result, s.status);
  s.check_status("initialize");
  s.possibly_bound =
      true;  // Driver failure must not free possibly rebound managers.
  s.call(bind_function, "runtime_sparse_view_bind", s.stream, s.view,
         s.device_trees, int(trees.size()), s.status);
  s.read(s.status, sizeof(*s.status));
  if (static_cast<const LlvmSparseRuntimeViewStatus *>(s.host)->error !=
      LlvmSparseRuntimeViewError::success) {
    s.possibly_bound = false;  // Device preflight guarantees zero mutation.
    s.check_status("bind");
  }
  s.ready = true;
}

LLVMRuntime *SparseRuntimeView::bind(LLVMRuntime *source,
                                     const std::vector<int> &trees) const {
  const auto &s = *state_;
  if (!s.ready || source != s.source)
    throw std::invalid_argument(
        "CUDA sparse runtime view source has retired or changed");
  for (int id : trees)
    if (!std::binary_search(
            s.trees.begin(), s.trees.end(), LlvmSparseRuntimeViewTree{id, 0, 0},
            [](const auto &a, const auto &b) { return a.tree_id < b.tree_id; }))
      throw std::invalid_argument(
          "CUDA sparse runtime view omits a selected tree");
  return s.view;
}

void *SparseRuntimeView::temporaries() const {
  if (!state_->ready)
    throw std::invalid_argument("CUDA sparse runtime view is not ready");
  return state_->temporary;
}
std::size_t SparseRuntimeView::bytes() const {
  return state_->allocated_bytes;
}

void SparseRuntimeView::retire_after(const RuntimeCompletion &prefix) {
  if (prefix.valid())
    prefix.wait();
  retire();
}

void SparseRuntimeView::retire() {
  auto &s = *state_;
  std::lock_guard<std::mutex> lock(s.retirement_mutex);
  if (!s.attempted || s.retired)
    return;
  if (!s.owner->backend_calls_safe())
    throw std::runtime_error(
        "Cannot return CUDA sparse managers after a backend fault");
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  if (s.possibly_bound) {
    s.call(s.release_function, "runtime_sparse_view_release", s.stream, s.view,
           s.device_trees, int(s.trees.size()), s.status);
    s.check_status("return");
    s.possibly_bound = false;
  }
  s.ready = false;
  // The private preparation stream also orders failed preparation cleanup.
  if (s.stream)
    s.wait();
  while (!s.allocations.empty()) {
    free_retired_allocation(s.allocations.back(), s.stream);
    s.allocations.pop_back();
  }
  if (s.stream) {
    s.wait();
    driver.stream_destroy(s.stream);
    s.stream = nullptr;
  }
  s.snapshots.clear();
  if (s.host) {
    driver.mem_free_host(s.host);
    s.host = nullptr;
  }
  // Ordinary replay is allowed only after the managers point back to source.
  s.loan.reset();
  s.allocated_bytes = 0;
  s.retired = true;
}

SparseRuntimeView::DiagnosticSnapshot::DiagnosticSnapshot(
    std::unique_ptr<State> state)
    : state_(std::move(state)) {
}
SparseRuntimeView::DiagnosticSnapshot::~DiagnosticSnapshot() = default;

void SparseRuntimeView::DiagnosticSnapshot::observe() const {
  const auto &snapshot = *state_->host;
  if (!snapshot.code)
    return;
  if (snapshot.code != 1)
    throw std::runtime_error("Unknown CUDA sparse runtime diagnostic code");
  const auto end =
      std::find(std::begin(snapshot.message), std::end(snapshot.message), '\0');
  const std::string message(std::begin(snapshot.message), end);
  throw TaichiAssertionError(format_error_message(message, [&](int index) {
    if (index < 0 || index >= taichi_error_message_max_num_arguments)
      throw std::runtime_error("Invalid CUDA sparse diagnostic argument");
    return snapshot.arguments[index];
  }));
}

std::shared_ptr<SparseRuntimeView::DiagnosticSnapshot>
SparseRuntimeView::prepare_diagnostic_snapshot() {
  auto &s = *state_;
  if (!s.ready)
    throw std::invalid_argument("CUDA sparse runtime view is not ready");
  auto data = std::make_unique<DiagnosticSnapshot::State>();
  data->owner = s.owner;
  data->view_identity = s.identity;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  int unified = 0;
  driver.device_get_attribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                              CUDAContext::get_instance().get_device());
  if (!unified)
    throw std::invalid_argument(
        "CUDA sparse diagnostics require unified addressing");
  driver.mem_host_alloc(reinterpret_cast<void **>(&data->host),
                        sizeof(*data->host), 0);
  std::memset(data->host, 0, sizeof(*data->host));
  // As with submission parameters, non-write-combined cuMemHostAlloc memory
  // has the same host/device address under UVA. A DtoH copy here would queue
  // behind a blocked peer's copy engine and defeat independent completion.
  data->device = data->host;
  auto result = std::shared_ptr<DiagnosticSnapshot>(
      new DiagnosticSnapshot(std::move(data)));
  s.snapshots.push_back(result);
  return result;
}

void SparseRuntimeView::enqueue_diagnostics(const DiagnosticSnapshot &snapshot,
                                            void *stream) {
  auto &s = *state_;
  if (!s.ready || snapshot.state_->view_identity != s.identity)
    throw std::invalid_argument(
        "CUDA sparse diagnostic snapshot belongs to another view");
  auto guard = CUDAContext::get_instance().get_guard();
  void *arguments[]{&s.view, &snapshot.state_->device};
  auto &driver = CUDADriver::get_instance();
  driver.launch_kernel(s.snapshot_function, 1, 1, 1, 1, 1, 1, 0, stream,
                       arguments, nullptr);
}

}  // namespace taichi::lang::cuda
