#include "taichi/runtime/cuda/graph_memory_pool.h"

#include <atomic>
#include <limits>
#include <mutex>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"

namespace taichi::lang::cuda {

namespace {
struct PoolHandle {
  void *handle{nullptr};
  std::shared_ptr<BackendFaultReporter> fault_reporter;

  ~PoolHandle() {
    if (handle && (!fault_reporter || fault_reporter->backend_calls_safe())) {
      try {
        auto context = CUDAContext::get_instance().get_guard();
        // CudaDevice's cold retirement owns this handle until all allocation
        // owners and their default-stream frees have finished.
        CUDADriver::get_instance().mem_pool_destroy(handle);
      } catch (...) {
        // A failed context is already reported by the Driver wrapper.
      }
    }
  }
};
}  // namespace

struct GraphMemoryPool::State : CudaDevice::RetainedGraphResource {
  Program *program{nullptr};
  CudaDevice *device{nullptr};
  std::shared_ptr<PoolHandle> pool;
  std::uint64_t retained_bytes{0};
  std::mutex mutex;
  std::atomic<bool> closed{false};

  ~State() override {
    retire_graph_resource();
  }

  void retire_graph_resource() noexcept override {
    if (closed.load()) {
      return;
    }
    auto submission = CUDAContext::get_instance().get_submission_lock_guard();
    std::lock_guard<std::mutex> lock(mutex);
    if (closed.exchange(true)) {
      return;
    }
    // Closing a factory does not destroy the allocator beneath its still-live
    // arrays. Existing allocation records own the handle through their normal
    // retirement. Collection below is nonblocking and cold, never replay.
    pool.reset();
    if (device) {
      try {
        device->collect_graph_memory_pools();
      } catch (...) {
        // Defer reclamation to device teardown after a reported CUDA fault.
      }
    }
    device = nullptr;
  }
};

bool GraphMemoryPool::available() {
  auto &driver = CUDADriver::get_instance();
  return !driver.is_musa() && driver.mem_pool_create.available() &&
         driver.stream_query.available() &&
         driver.mem_pool_destroy.available() &&
         driver.mem_pool_trim_to.available() &&
         driver.mem_pool_get_attribute.available() &&
         driver.mem_alloc_from_pool_async.available() &&
         CUDAContext::get_instance().supports_mem_pool();
}

GraphMemoryPool::GraphMemoryPool(Program &program,
                                 std::uint64_t retained_bytes) {
  auto resource = program.acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(program.compile_config().arch != Arch::cuda || !available(),
              "Graph-owned CUDA memory pools are unavailable");
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto state = std::make_shared<State>();
  state->program = &program;
  state->device = dynamic_cast<CudaDevice *>(program.get_compute_device());
  TI_ERROR_IF(!state->device, "Graph memory pool requires a CUDA device");
  state->retained_bytes = retained_bytes;
  state->pool = std::make_shared<PoolHandle>();
  state->pool->fault_reporter = state->device->backend_fault_reporter();
  auto context = CUDAContext::get_instance().get_guard();
  TaichiCudaMemPoolProps properties{};
  properties.allocation_type = 1;  // CU_MEM_ALLOCATION_TYPE_PINNED
  properties.location_type = 1;    // CU_MEM_LOCATION_TYPE_DEVICE
  properties.location_id =
      static_cast<std::int32_t>(reinterpret_cast<std::intptr_t>(
          CUDAContext::get_instance().get_device()));
  auto &driver = CUDADriver::get_instance();
  driver.mem_pool_create(&state->pool->handle, &properties);
  driver.mem_pool_set_attribute(state->pool->handle,
                                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                &state->retained_bytes);
  state->device->register_graph_memory_pool(state->pool);
  state->device->register_graph_resource(state);
  state_ = std::move(state);
}

GraphMemoryPool::~GraphMemoryPool() {
  state_->retire_graph_resource();
}

Ndarray *GraphMemoryPool::create_ndarray(DataType type,
                                         const std::vector<int> &shape) {
  auto &state = *state_;
  TI_ERROR_IF(state.closed.load(), "Graph memory pool is closed");
  // Validate once, before allocating. No raw pointer import and no repeated
  // storage validation is introduced in ordinary kernel or Graph execution.
  TI_ERROR_IF(!type->is<PrimitiveType>(),
              "Graph memory pool requires a scalar ndarray dtype");
  std::size_t bytes = data_type_size(type);
  TI_ERROR_IF(bytes == 0, "Graph memory pool requires a sized dtype");
  for (int dimension : shape) {
    TI_ERROR_IF(
        dimension <= 0 || bytes > std::numeric_limits<std::size_t>::max() /
                                      static_cast<std::size_t>(dimension),
        "Invalid Graph memory pool ndarray shape");
    bytes *= dimension;
  }
  auto resource = state.program->acquire_runtime_resource_submission_guard();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(state.mutex);
  TI_ERROR_IF(state.closed.load(), "Graph memory pool is closed");
  return state.program->create_ndarray_with_allocator(
      type, shape, [&](std::size_t allocation_bytes) {
        auto allocation = state.device->allocate_memory_from_pool(
            allocation_bytes, state.pool->handle, state.pool);
        state.program->mark_runtime_submission_pending();
        return allocation;
      });
}

void GraphMemoryPool::trim() {
  auto &state = *state_;
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(state.mutex);
  TI_ERROR_IF(state.closed.load(), "Graph memory pool is closed");
  auto context = CUDAContext::get_instance().get_guard();
  CUDADriver::get_instance().mem_pool_trim_to(state.pool->handle, 0);
}

void GraphMemoryPool::close() {
  state_->retire_graph_resource();
}

std::unordered_map<std::string, std::uint64_t> GraphMemoryPool::snapshot() {
  auto &state = *state_;
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.closed.load()) {
    return {{"closed", 1}};
  }
  auto context = CUDAContext::get_instance().get_guard();
  std::unordered_map<std::string, std::uint64_t> result{
      {"closed", 0}, {"release_threshold_bytes", state.retained_bytes}};
  auto &driver = CUDADriver::get_instance();
  // CUmemPool_attribute 5..8; these are this pool's own driver observations,
  // not process VRAM, total Graph storage, or an inferred zero when missing.
  for (const auto &[name, attribute] : {std::pair{"reserved_current_bytes", 5u},
                                        std::pair{"reserved_high_bytes", 6u},
                                        std::pair{"used_current_bytes", 7u},
                                        std::pair{"used_high_bytes", 8u}}) {
    std::uint64_t value = 0;
    driver.mem_pool_get_attribute(state.pool->handle, attribute, &value);
    result.emplace(name, value);
  }
  return result;
}

}  // namespace taichi::lang::cuda
