#define TI_RUNTIME_HOST
#include "cuda_context.h"

#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <stdexcept>

#include "taichi/system/threading.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_profiler.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

namespace {
struct ContextSelection {
  std::recursive_mutex mutex;
  std::atomic<CUDAContext *> context{nullptr};
  bool configuring{false};
  bool failed{false};
  bool cig_context_live{false};
  std::atomic<bool> runtime_failed{false};
  std::exception_ptr first_runtime_failure;
  CudaContextConfig config;
  CudaContextInfo info;
};

ContextSelection &selection() {
  // Same lifetime as the existing process context. The native owner lease is
  // intentionally retained with it, including after Python module teardown.
  static auto *state = new ContextSelection;
  return *state;
}

struct RestoreCurrent {
  CUDADriver &driver;
  void *previous{nullptr};
  explicit RestoreCurrent(CUDADriver &value) : driver(value) {
    driver.context_get_current(&previous);
  }
  ~RestoreCurrent() {
    driver.context_set_current.call(previous);
  }
};

struct CigRetirementFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Cold qualification in the actual newly created context, not a device
// capability inference. Async/pool allocations are deliberately NOT probed in
// this context: some CiG drivers accept allocation, then fault on first access.
// CiG uses retained ordinary device blocks, with event-based retirement.
void qualify_cig_allocations(CUDADriver &driver, CudaContextInfo &info) {
  void *stream = nullptr, *ordinary = nullptr, *managed = nullptr;
  CUgraph graph = nullptr;
  CUgraphExec executable = nullptr;
  bool capturing = false;
  auto cleanup = [&] {
    if (capturing) {
      driver.stream_end_capture.call(stream, &graph);
      capturing = false;
    }
    if (stream)
      driver.stream_synchronize(stream);
    if (executable)
      driver.graph_exec_destroy(executable);
    if (graph)
      driver.graph_destroy(graph);
    if (managed)
      driver.mem_free(managed);
    if (ordinary)
      driver.mem_free(ordinary);
    if (stream)
      driver.stream_destroy(stream);
  };
  try {
    driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    driver.malloc(&ordinary, 4096);
    driver.malloc_managed(&managed, 4096, CU_MEM_ATTACH_GLOBAL);
    driver.stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    capturing = true;
    driver.memsetd32_async(ordinary, 0x1234, 1, stream);
    driver.memsetd32_async(managed, 0x5678, 1, stream);
    driver.stream_end_capture(stream, &graph);
    capturing = false;
    driver.graph_instantiate_with_flags(&executable, graph, 0);
    driver.graph_launch(executable, stream);
    driver.stream_synchronize(stream);
    std::uint32_t value = 0;
    driver.memcpy_device_to_host(&value, ordinary, sizeof(value));
    if (value != 0x1234)
      throw std::runtime_error("CiG captured allocation readback failed");
    driver.memcpy_device_to_host(&value, managed, sizeof(value));
    if (value != 0x5678)
      throw std::runtime_error("CiG managed allocation readback failed");
    info.allocation_checks = 1 | 2 | 16;
  } catch (const std::exception &initial) {
    try {
      cleanup();
    } catch (const std::exception &retirement) {
      throw CigRetirementFailure(std::string("CiG probe failed: ") +
                                 initial.what() + "; retirement failed: " +
                                 retirement.what() + "; context quarantined");
    }
    throw;
  }
  cleanup();
}
}  // namespace

CUDAContext::CUDAContext()
    : profiler_(nullptr), driver_(CUDADriver::get_instance_without_context()) {
  // CUDA initialization
  dev_count_ = 0;
  driver_.init(0);
  driver_.device_get_count(&dev_count_);
  auto &selected = selection();
  auto &configuration = selected.config;
  auto &information = selected.info;
  RestoreCurrent restore(driver_);
  int ordinal = 0;
  if (configuration.mode != CudaContextMode::disabled &&
      !configuration.queue_data.empty()) {
    if (driver_.is_musa() || !driver_.device_get_uuid.available())
      throw std::invalid_argument(
          "Vulkan CiG requires NVIDIA CUDA device identity");
    ordinal = -1;
    for (int candidate = 0; candidate < dev_count_; ++candidate) {
      void *candidate_device = nullptr;
      driver_.device_get(&candidate_device,
                         reinterpret_cast<void *>(candidate));
      CUuuid uuid{};
      driver_.device_get_uuid(&uuid, candidate_device);
      if (std::memcmp(uuid.bytes, configuration.device_uuid.data(), 16) == 0) {
        ordinal = candidate;
        break;
      }
    }
    if (ordinal < 0)
      throw std::invalid_argument(
          "Vulkan CiG UUID has no matching CUDA device");
  }
  driver_.device_get(&device_, reinterpret_cast<void *>(ordinal));
  CUuuid selected_uuid{};
  if (!driver_.is_musa() && driver_.device_get_uuid.available())
    driver_.device_get_uuid(&selected_uuid, device_);
  std::memcpy(information.device_uuid.data(), selected_uuid.bytes, 16);

  char name[128];
  driver_.device_get_name(name, 128, device_);

  TI_TRACE("Using {} device [id={}]: {}",
           cuda::detail::driver_provider_name(driver_.get_provider()), ordinal,
           name);

  if (driver_.is_musa()) {
    int warp_size = 0;
    driver_.device_get_attribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                 device_);
    TI_ERROR_IF(
        !cuda::detail::driver_warp_size_supported(driver_.get_provider(),
                                                  warp_size),
        "The minimal MUSA compatibility path currently requires a 32-lane "
        "warp, but device '{}' reports {} lanes. Wider MUSA warps require "
        "runtime reduction and intrinsic lowering changes before they can run "
        "safely.",
        name, warp_size);
    TI_WARN(
        "Using the experimental MUSA compatibility path through ti.cuda. "
        "Basic PTX kernel execution is enabled; CUDA-specific libraries, "
        "native primitives, Graph paths, and external interop are not "
        "qualified.");
  }

  int cc_major, cc_minor;
  driver_.device_get_attribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_);
  driver_.device_get_attribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_);

  int device_supports_mem_pool = 0;
  if (driver_.is_musa()) {
    TI_TRACE("CUDA asynchronous memory pools are disabled for MUSA.");
  } else if (driver_.get_version_major() > 11 ||
             (driver_.get_version_major() == 11 &&
              driver_.get_version_minor() >= 2)) {
    driver_.device_get_attribute(&device_supports_mem_pool,
                                 CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                                 device_);
  } else {
    TI_WARN(
        "Please consider upgrade your nvidia driver for better device memory "
        "pool"
        "support. Current driver supports CUDA {}.{}, we recommend driver "
        "version"
        "above 470 (CUDA 11.2).",
        driver_.get_version_major(), driver_.get_version_minor());
    device_supports_mem_pool = 0;
  }

  // Select context before touching a default pool or allocating any resources.
  information.driver_entry = !driver_.is_musa() &&
                             driver_.context_create_v4.available() &&
                             driver_.context_destroy.available();
  if (configuration.mode != CudaContextMode::disabled) {
    int capability = 0;
    const auto capability_result = driver_.device_get_attribute.call(
        &capability, 138 /* CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED */,
        device_);
    information.device_capability = capability_result == 0 && capability != 0;
    std::string unavailable;
    if (configuration.queue_data.empty())
      unavailable = "Vulkan external queue binding was not supplied";
    else if (!information.driver_entry ||
             !driver_.context_pop_current.available())
      unavailable = "Driver lacks the Vulkan CiG context ABI";
    else if (!information.device_capability)
      unavailable = "CUDA device does not support Vulkan CiG";
    if (unavailable.empty()) {
      TaichiCudaCigParams cig{2 /* CIG_DATA_TYPE_NV_BLOB */,
                              configuration.queue_data.data()};
      TaichiCudaContextCreateParams params{nullptr, 0, &cig};
      const auto result =
          driver_.context_create_v4.call(&context_, &params, 0, ordinal);
      if (result) {
        unavailable =
            "CiG context creation failed: " + get_cuda_error_message(result);
      } else {
        selected.cig_context_live = true;
        // cuCtxCreate pushes a context. Balance that push, then bind it only
        // for this initialization scope; RestoreCurrent restores the caller.
        void *popped = nullptr;
        driver_.context_pop_current(&popped);
        driver_.context_set_current(context_);
        try {
          qualify_cig_allocations(driver_, information);
          information.vulkan_cig = true;
          vulkan_cig_ = true;
        } catch (const CigRetirementFailure &) {
          throw;
        } catch (const std::exception &error) {
          unavailable = std::string("CiG allocation qualification failed: ") +
                        error.what();
          // A failed destroy prevents fallback and retains the native owner.
          driver_.context_destroy(context_);
          context_ = nullptr;
          selected.cig_context_live = false;
          driver_.context_set_current(restore.previous);
        }
      }
    }
    if (!unavailable.empty()) {
      if (configuration.mode == CudaContextMode::required)
        throw std::runtime_error(unavailable);
      information.fallback_reason = std::move(unavailable);
      configuration.owner.reset();
      configuration.queue_data.clear();
    }
  }
  if (!context_) {
    driver_.primary_context_retain(&context_, ordinal);
    driver_.context_set_current(context_);
  }
  supports_mem_pool_ = cuda::detail::supports_memory_pool(
      driver_.get_version_major(), driver_.get_version_minor(),
      device_supports_mem_pool);
  if (vulkan_cig_)
    supports_mem_pool_ = false;
  if (supports_mem_pool_) {
    void *default_mem_pool;
    driver_.device_get_default_mem_pool(&default_mem_pool, device_);
    // Keep idle cached memory bounded. Runtime statistics expose the pool's
    // reserved/used bytes; a configurable policy belongs at Program startup,
    // not on this process-wide context after allocations may already exist.
    constexpr uint64 kMemPoolReleaseThreshold = 1048576 * 128;
    driver_.mem_pool_set_attribute(default_mem_pool,
                                   CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                                   (void *)&kMemPoolReleaseThreshold);
  }

  TI_TRACE("{} device compute capability: {}.{}",
           cuda::detail::driver_provider_name(driver_.get_provider()), cc_major,
           cc_minor);

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  auto target =
      cuda::detail::resolve_compute_capability_target(cc_major * 10 + cc_minor);
  device_compute_capability_ = target.device_compute_capability;
  codegen_compute_capability_ = target.codegen_compute_capability;
  ptx_version_ = target.ptx_version;

  driver_.device_get_attribute(
      &max_shared_memory_bytes_,
      driver_.is_musa() ? CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
                        : CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      device_);

  mcpu_ = fmt::format("sm_{}", codegen_compute_capability_);
  mattrs_ = fmt::format("+ptx{}", ptx_version_);

  if (target.uses_fallback) {
    TI_WARN(
        "CUDA device capability {} is newer than the bundled LLVM NVPTX "
        "targets. Emitting {} with PTX {} as the highest compatible "
        "fallback.",
        device_compute_capability_, mcpu_, ptx_version_);
  } else {
    TI_TRACE("Emitting CUDA code for {} with PTX {}", mcpu_, ptx_version_);
  }
  information.initialized = true;
  information.generation = 1;
}

std::size_t CUDAContext::get_total_memory() {
  auto guard = get_guard();
  std::size_t ret, _;
  driver_.mem_get_info(&_, &ret);
  return ret;
}

std::size_t CUDAContext::get_free_memory() {
  auto guard = get_guard();
  std::size_t ret, _;
  driver_.mem_get_info(&ret, &_);
  return ret;
}

std::string CUDAContext::get_device_name() {
  constexpr uint32_t kMaxNameStringLength = 128;
  char name[kMaxNameStringLength];
  driver_.device_get_name(name, kMaxNameStringLength /*=128*/, device_);
  std::string str(name);
  return str;
}

void CUDAContext::launch(void *func,
                         const std::string &task_name,
                         std::vector<void *> arg_pointers,
                         std::vector<int> arg_sizes,
                         unsigned grid_dim,
                         unsigned block_dim,
                         std::size_t dynamic_shared_mem_bytes,
                         void *stream) {
  // Direct CUDA primitives enter here without a KernelLauncher transaction.
  // Recursive ownership is intentional when a regular or captured Taichi
  // kernel already owns the wider submission transaction.
  auto submission_lock = get_submission_lock_guard();
  // It is important to keep a handle since in async mode (deleted)
  // a constant folding kernel may happen during a kernel launch
  // then profiler->start and profiler->stop mismatch.
  // TODO: should we keep the handle?

  TI_ERROR_IF(
      dynamic_shared_mem_bytes > 0,
      "CUDA dynamic shared memory is disabled because opt-in allocations can "
      "trigger CUDA_ERROR_ILLEGAL_ADDRESS (IMA). Recompile the kernel with a "
      "SharedArray no larger than 48 KiB.");

  KernelProfilerBase::TaskHandle task_handle;
  // Kernel launch
  if (profiler_) {
    KernelProfilerCUDA *profiler_cuda =
        dynamic_cast<KernelProfilerCUDA *>(profiler_);
    std::string primal_task_name, key;
    bool valid =
        offline_cache::try_demangle_name(task_name, primal_task_name, key);
    profiler_cuda->trace(task_handle, valid ? primal_task_name : task_name,
                         func, grid_dim, block_dim, dynamic_shared_mem_bytes);
  }

  auto context_guard = CUDAContext::get_instance().get_guard();

  // TODO: remove usages of get_current_program here.
  // Make sure there are not too many threads for the device.
  // Note that the CUDA random number generator does not allow more than
  // [saturating_grid_dim * max_block_dim] threads.

  // These asserts are currently remove so that when GGUI calls CUDA kernels,
  // the grid and block dim won't be limited by the limits set by Program. With
  // these limits, GGUI would have to use kernels with grid strided loops, which
  // is harmful to performance. A simple example of rendering a bunny can drop
  // from 2000FPS to 1000FPS because of this. TI_ASSERT(grid_dim <=
  // get_current_program().config.saturating_grid_dim); TI_ASSERT(block_dim <=
  // get_current_program().config.max_block_dim);

  if (grid_dim > 0) {
    auto lock = get_lock_guard();
    driver_.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                          dynamic_shared_mem_bytes, stream, arg_pointers.data(),
                          nullptr);
  }
  if (profiler_)
    profiler_->stop(task_handle);

  if (debug_) {
    // Captured launches describe work; they cannot synchronize that stream.
    // Keep ordinary debug synchronization and let replay/wait report runtime
    // assertions after the recorded graph has actually executed.
    std::uint32_t capture_status = 0;
    if (stream)
      driver_.stream_get_capture_info_v2(stream, &capture_status, nullptr,
                                          nullptr, nullptr, nullptr);
    if (capture_status == 0)
      driver_.stream_synchronize(stream);
  }
}

void CUDAContext::launch_graph(void *executable, void *stream) {
  if (auto *profiler = dynamic_cast<KernelProfilerCUDA *>(profiler_)) {
    profiler->launch_graph(executable, stream);
  } else {
    driver_.graph_launch(executable, stream);
  }
}

CUDAContext::~CUDAContext() {
  // TODO: restore these?
  /*
  CUDADriver::get_instance().cuMemFree(context_buffer);
  for (auto cudaModule: cudaModules)
      CUDADriver::get_instance().cuModuleUnload(cudaModule);
  CUDADriver::get_instance().cuCtxDestroy(context);
  */
}

CUDAContext &CUDAContext::get_instance() {
  auto &selected = selection();
  if (auto *context = selected.context.load(std::memory_order_acquire))
    return *context;
  std::lock_guard<std::recursive_mutex> lock(selected.mutex);
  if (selected.failed)
    throw std::runtime_error(
        "CUDA context initialization failed; use a fresh process");
  if (auto *context = selected.context.load(std::memory_order_relaxed))
    return *context;
  auto *context = new CUDAContext();
  selected.context.store(context, std::memory_order_release);
  return *context;
}

void CUDAContext::configure(CudaContextConfig config) {
  if (config.mode != CudaContextMode::disabled &&
      config.mode != CudaContextMode::required &&
      config.mode != CudaContextMode::automatic)
    throw std::invalid_argument("Unknown CUDA context mode");
  const bool has_data = !config.queue_data.empty();
  if (has_data != bool(config.owner) ||
      config.queue_data.size() > 1024 * 1024 ||
      (config.mode == CudaContextMode::required && !has_data) ||
      (config.mode == CudaContextMode::disabled && has_data))
    throw std::invalid_argument(
        "CiG requires opaque queue bytes and a native owner lease");
  auto &selected = selection();
  std::lock_guard<std::recursive_mutex> lock(selected.mutex);
  if (selected.context.load(std::memory_order_relaxed) || selected.configuring)
    throw std::logic_error(
        "CUDA context selection must precede its first use; hot switching is "
        "unsupported");
  selected.configuring = true;
  selected.config = std::move(config);
  selected.info = {};
  try {
    (void)get_instance();
    selected.configuring = false;
  } catch (...) {
    // Release only after no CiG context exists. An incomplete native rollback
    // quarantines its exact owner; no fallback or same-process retry follows.
    selected.failed = true;
    if (!selected.cig_context_live) {
      selected.config.owner.reset();
      selected.config.queue_data.clear();
    }
    throw;
  }
}

CudaContextInfo CUDAContext::configuration() {
  auto &selected = selection();
  std::lock_guard<std::recursive_mutex> lock(selected.mutex);
  return selected.info;
}

void CUDAContext::record_runtime_failure(
    const BackendRuntimeError &error) noexcept {
  auto &selected = selection();
  auto *context = selected.context.load(std::memory_order_acquire);
  if (!context || !context->is_vulkan_cig())
    return;
  try {
    std::lock_guard<std::recursive_mutex> lock(selected.mutex);
    if (!selected.runtime_failed.load(std::memory_order_relaxed)) {
      selected.first_runtime_failure = std::make_exception_ptr(error);
      selected.runtime_failed.store(true, std::memory_order_release);
    }
  } catch (...) {
    // Preserve failure even if constructing its diagnostic ran out of memory.
    selected.runtime_failed.store(true, std::memory_order_release);
  }
}

void CUDAContext::check_runtime_health() {
  auto &selected = selection();
  if (!selected.runtime_failed.load(std::memory_order_acquire))
    return;
  std::lock_guard<std::recursive_mutex> lock(selected.mutex);
  if (selected.first_runtime_failure)
    std::rethrow_exception(selected.first_runtime_failure);
  throw std::runtime_error("CUDA CiG context has faulted; use a fresh process");
}

}  // namespace taichi::lang
