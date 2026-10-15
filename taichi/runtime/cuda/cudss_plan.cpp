#include "taichi/runtime/cuda/cudss_plan.h"

#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>
#include "taichi/common/dynamic_loader.h"
#include "taichi/cudss/forge_cudss_provider.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {
constexpr int kCudssStatusSuccess = 0;
constexpr int kCudssPhaseAnalysis = 3;
constexpr int kCudssPhaseFactorization = 4;
constexpr int kCudssPhaseRefactorization = 8;
constexpr int kCudssPhaseSolve = 1008;
constexpr int kCudssDataTypeF32 = 0;
constexpr int kCudssDataTypeI32 = 10;
constexpr int kCudssMatrixGeneral = 0;
constexpr int kCudssMatrixSymmetric = 1;
constexpr int kCudssMatrixSpd = 3;
constexpr int kCudssViewFull = 0;
constexpr int kCudssViewLower = 1;
constexpr int kCudssViewUpper = 2;
constexpr int kCudssBaseZero = 0;
constexpr int kCudssLayoutColumnMajor = 0;

void require_cudss_success(std::uint32_t status, const char *operation) {
  TI_ERROR_IF(status != kCudssStatusSuccess,
              "CUDA cuDSS {} failed (status {}).", operation, status);
}

void warn_cudss_failure(std::uint32_t status, const char *operation) {
  if (status != kCudssStatusSuccess) {
    TI_WARN("CUDA cuDSS {} failed during cleanup (status {}).", operation,
            status);
  }
}

void validate_cudss_matrix_contract(int matrix_type, int matrix_view) {
  TI_ERROR_IF(matrix_type != kCudssMatrixGeneral &&
                  matrix_type != kCudssMatrixSymmetric &&
                  matrix_type != kCudssMatrixSpd,
              "CUDA cuDSS matrix_type must be general, symmetric, or spd.");
  TI_ERROR_IF(matrix_view != kCudssViewFull && matrix_view != kCudssViewLower &&
                  matrix_view != kCudssViewUpper,
              "CUDA cuDSS matrix_view must be full, lower, or upper.");
  TI_ERROR_IF(
      matrix_type == kCudssMatrixGeneral && matrix_view != kCudssViewFull,
      "CUDA cuDSS general matrices require the full matrix view.");
}

class CudssProviderRuntime {
 public:
  CudssProviderRuntime(const std::string &adapter_path,
                       const std::string &runtime_library_path) {
    TI_ERROR_IF(adapter_path.empty(),
                "CUDA cuDSS requires a bundled Forge provider adapter.");
    auto &cuda_driver = CUDADriver::get_instance_without_context();
    TI_ERROR_IF(!cuda_driver.nvidia_extensions_available() ||
                    cuda_driver.get_version_major() < 12,
                "CUDA cuDSS requires the NVIDIA CUDA driver API 12.0 or "
                "newer.");
    auto &cublas = CUBLASDriver::get_instance();
    TI_ERROR_IF(!cublas.is_loaded() && !cublas.load_cublas(),
                "CUDA cuDSS requires a compatible user-managed cuBLAS "
                "runtime.");

    loader_ = std::make_unique<DynamicLoader>(adapter_path);
    TI_ERROR_IF(!loader_->loaded(),
                "CUDA cuDSS could not load bundled adapter {}.", adapter_path);
    auto *query_symbol =
        loader_->load_function_optional(TI_FORGE_CUDSS_PROVIDER_QUERY_SYMBOL);
    TI_ERROR_IF(!query_symbol,
                "CUDA cuDSS bundled adapter is missing its query symbol.");
    auto query = reinterpret_cast<TiForgeCudssProviderQueryFn>(query_symbol);
    const auto query_result =
        query(TI_FORGE_CUDSS_PROVIDER_ABI_VERSION, sizeof(api_), &api_);
    TI_ERROR_IF(query_result != TI_FORGE_CUDSS_SUCCESS,
                "CUDA cuDSS bundled adapter rejected Forge provider ABI {} "
                "(result {}).",
                TI_FORGE_CUDSS_PROVIDER_ABI_VERSION,
                static_cast<int>(query_result));
    TI_ERROR_IF(
        api_.struct_size < sizeof(api_) ||
            api_.provider_abi_version != TI_FORGE_CUDSS_PROVIDER_ABI_VERSION ||
            api_.info.struct_size < sizeof(api_.info) ||
            api_.info.provider_abi_version !=
                TI_FORGE_CUDSS_PROVIDER_ABI_VERSION ||
            api_.info.cudss_header_version / 100 != 8,
        "CUDA cuDSS bundled adapter identity is incompatible.");
    constexpr uint64_t required_features =
        TI_FORGE_CUDSS_FEATURE_CSR | TI_FORGE_CUDSS_FEATURE_DENSE_VECTOR |
        TI_FORGE_CUDSS_FEATURE_STAGED_EXECUTION |
        TI_FORGE_CUDSS_FEATURE_VALUE_REBIND |
        TI_FORGE_CUDSS_FEATURE_EXPLICIT_STREAM;
    TI_ERROR_IF((api_.info.features & required_features) != required_features,
                "CUDA cuDSS bundled adapter lacks required features.");
    TI_ERROR_IF(!api_.create_runtime || !api_.destroy_runtime || !api_.create ||
                    !api_.destroy || !api_.set_stream || !api_.config_create ||
                    !api_.config_destroy || !api_.data_create ||
                    !api_.data_destroy || !api_.matrix_create_csr ||
                    !api_.matrix_create_dn || !api_.matrix_destroy ||
                    !api_.matrix_set_values || !api_.matrix_set_csr_pointers ||
                    !api_.execute || !api_.get_last_error,
                "CUDA cuDSS bundled adapter API table is incomplete.");
    TiForgeCudssRuntime candidate_runtime = nullptr;
    TiForgeCudssRuntimeInfo candidate_info{};
    candidate_info.struct_size = sizeof(candidate_info);
    const char *runtime_path =
        runtime_library_path.empty() ? nullptr : runtime_library_path.c_str();
    const auto runtime_result =
        api_.create_runtime(runtime_path, &candidate_runtime, &candidate_info);
    if (runtime_result != TI_FORGE_CUDSS_SUCCESS || !candidate_runtime) {
      if (candidate_runtime) {
        api_.destroy_runtime(candidate_runtime);
      }
      TI_ERROR("CUDA cuDSS vendor runtime initialization failed: {}",
               adapter_error());
    }
    if (candidate_info.version_major != 0 ||
        candidate_info.version_minor != 8) {
      api_.destroy_runtime(candidate_runtime);
      TI_ERROR(
          "CUDA cuDSS adapter loaded an unsupported vendor runtime "
          "version {}.{}.{}.",
          candidate_info.version_major, candidate_info.version_minor,
          candidate_info.version_patch);
    }
    runtime_ = candidate_runtime;
    runtime_info_ = candidate_info;
  }

  CudssProviderRuntime(const CudssProviderRuntime &) = delete;
  CudssProviderRuntime &operator=(const CudssProviderRuntime &) = delete;

  ~CudssProviderRuntime() {
    if (runtime_ && api_.destroy_runtime) {
      const auto result = api_.destroy_runtime(runtime_);
      if (result != TI_FORGE_CUDSS_SUCCESS) {
        TI_WARN("CUDA cuDSS adapter runtime cleanup failed (result {}).",
                static_cast<int>(result));
      }
      runtime_ = nullptr;
    }
  }

  const TiForgeCudssProviderApi &api() const {
    return api_;
  }

  TiForgeCudssRuntime runtime() const {
    return runtime_;
  }

  TiForgeCudssConfigurationApi configuration_api() const {
    auto *symbol = loader_->load_function_optional(
        TI_FORGE_CUDSS_CONFIGURATION_QUERY_SYMBOL);
    TI_ERROR_IF(!symbol,
                "CUDA cuDSS configured recipes require the optional "
                "configuration adapter extension.");
    const auto query =
        reinterpret_cast<TiForgeCudssConfigurationQueryFn>(symbol);
    TiForgeCudssConfigurationApi result{};
    TI_ERROR_IF(
        query(TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION, sizeof(result),
              &result) != TI_FORGE_CUDSS_SUCCESS ||
            result.struct_size < sizeof(result) ||
            result.abi_version != TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION ||
            !result.configure || !result.analysis_memory_estimates,
        "CUDA cuDSS configuration extension is incompatible.");
    return result;
  }

  TiForgeCudssFactorStatisticsApi factor_statistics_api() const {
    auto *symbol = loader_->load_function_optional(
        TI_FORGE_CUDSS_FACTOR_STATISTICS_QUERY_SYMBOL);
    TiForgeCudssFactorStatisticsApi result{};
    if (!symbol) {
      return result;
    }
    const auto query =
        reinterpret_cast<TiForgeCudssFactorStatisticsQueryFn>(symbol);
    // Diagnostics must not become a new execution requirement for old adapters.
    if (query(1, sizeof(result), &result) != TI_FORGE_CUDSS_SUCCESS ||
        result.struct_size != sizeof(result) || result.abi_version != 1 ||
        !result.read_i32) {
      return {};
    }
    return result;
  }

  const TiForgeCudssRuntimeInfo &runtime_info() const {
    return runtime_info_;
  }

  TiForgeCudssAllocatorApi allocator_api() const {
    auto *symbol =
        loader_->load_function_optional(TI_FORGE_CUDSS_ALLOCATOR_QUERY_SYMBOL);
    TI_ERROR_IF(!symbol,
                "CUDA cuDSS Graph allocator extension is unavailable.");
    auto query = reinterpret_cast<TiForgeCudssAllocatorQueryFn>(symbol);
    TiForgeCudssAllocatorApi result{};
    TI_ERROR_IF(query(1, sizeof(result), &result) != TI_FORGE_CUDSS_SUCCESS ||
                    result.struct_size < sizeof(result) ||
                    result.abi_version != 1 || !result.set_allocator,
                "CUDA cuDSS Graph allocator extension is incompatible.");
    return result;
  }

 private:
  std::string adapter_error() const {
    if (!api_.get_last_error) {
      return "adapter error unavailable";
    }
    const auto required = api_.get_last_error(nullptr, 0);
    if (required <= 1) {
      return "adapter call failed without detail";
    }
    std::vector<char> message(required, '\0');
    api_.get_last_error(message.data(), message.size());
    return message.data();
  }

  std::unique_ptr<DynamicLoader> loader_;
  TiForgeCudssProviderApi api_{};
  TiForgeCudssRuntime runtime_{nullptr};
  TiForgeCudssRuntimeInfo runtime_info_{};
};

// Only preparation/capture/retirement calls this allocator. After warming the
// private snapshot it is sealed: captured work cannot create/free persistent
// vendor pointers that would escape a particular CUDA graph's lifetime.
class CudssGraphAllocator {
 public:
  static int allocate(void *context,
                      void **pointer,
                      size_t bytes,
                      void *stream) noexcept {
    auto &owner = *static_cast<CudssGraphAllocator *>(context);
    std::lock_guard<std::mutex> lock(owner.mutex_);
    *pointer = nullptr;
    if (bytes == 0) {
      return 0;
    }
    if (owner.sealed_) {
      ++owner.sealed_allocation_rejections_;
      return 1;
    }
    try {
      CUDADriver::get_instance().malloc_async_impl(pointer, bytes, stream);
      owner.allocations_.emplace(*pointer, bytes);
      owner.live_bytes_ += bytes;
      owner.peak_bytes_ = std::max(owner.peak_bytes_, owner.live_bytes_);
      return 0;
    } catch (...) {
      if (*pointer) {
        try {
          CUDADriver::get_instance().mem_free_async_impl(*pointer, stream);
        } catch (...) {
        }
        *pointer = nullptr;
      }
      return 1;
    }
  }

  static int deallocate(void *context,
                        void *pointer,
                        size_t,
                        void *stream) noexcept {
    auto &owner = *static_cast<CudssGraphAllocator *>(context);
    std::lock_guard<std::mutex> lock(owner.mutex_);
    if (!pointer) {
      return 0;
    }
    if (owner.sealed_) {
      ++owner.sealed_allocation_rejections_;
      return 1;
    }
    const auto found = owner.allocations_.find(pointer);
    if (found == owner.allocations_.end()) {
      return 1;
    }
    try {
      CUDADriver::get_instance().mem_free_async_impl(pointer, stream);
      owner.live_bytes_ -= found->second;
      owner.allocations_.erase(found);
      return 0;
    } catch (...) {
      return 1;
    }
  }

  void seal(bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    sealed_ = value;
  }

  std::array<std::uint64_t, 3> observation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {live_bytes_, peak_bytes_, sealed_allocation_rejections_};
  }

 private:
  mutable std::mutex mutex_;
  bool sealed_{false};
  std::unordered_map<void *, std::size_t> allocations_;
  std::uint64_t live_bytes_{0};
  std::uint64_t peak_bytes_{0};
  std::uint64_t sealed_allocation_rejections_{0};
};

// cuDSS emits small host-to-device parameter copies while recording solves.
// Snapshot each RHS immediately, then upload the packed immutable bytes once
// after capture. Graph nodes read device storage owned by that graph/frame;
// later bindings cannot overwrite it, and replay needs no host transfer.
class CudssCaptureInputs final : public CudaGraphCaptureResources {
 public:
  explicit CudssCaptureInputs(std::shared_ptr<RuntimeFaultDomain> fault_domain)
      : fault_domain_(std::move(fault_domain)) {
  }

  ~CudssCaptureInputs() override {
    release(true);
  }

  void snapshot(void *node, CUDA_MEMCPY3D params) {
    auto &input = inputs_.emplace_back();
    input.node = node;
    input.params = params;
    input.bytes.resize(params.WidthInBytes);
    std::memcpy(
        input.bytes.data(),
        static_cast<const std::uint8_t *>(params.srcHost) + params.srcXInBytes,
        input.bytes.size());
    // The vendor may reuse its host buffer during the very next RHS capture.
    params.srcHost = input.bytes.data();
    params.srcXInBytes = 0;
    CUDADriver::get_instance().graph_memcpy_node_set_params(node, &params);
  }

  void finalize() override {
    if (inputs_.empty())
      return;
    std::vector<std::uint8_t> packed;
    for (auto &input : inputs_) {
      const auto size = packed.size();
      TI_ERROR_IF(size > std::numeric_limits<std::size_t>::max() - 7,
                  "cuDSS capture parameter storage is too large.");
      input.offset = (size + 7) & ~std::size_t(7);
      TI_ERROR_IF(input.bytes.size() >
                      std::numeric_limits<std::size_t>::max() - input.offset,
                  "cuDSS capture parameter storage is too large.");
      packed.resize(input.offset + input.bytes.size());
      std::memcpy(packed.data() + input.offset, input.bytes.data(),
                  input.bytes.size());
    }
    auto &driver = CUDADriver::get_instance();
    driver.malloc(&device_, packed.size());
    bytes_ = packed.size();
    // Synchronous cold upload keeps rollback/lifetime local. It executes no
    // mathematical work, and does not impose synchronization on graph replay.
    driver.memcpy_host_to_device(device_, packed.data(), packed.size());
    for (const auto &input : inputs_) {
      auto params = input.params;
      params.srcMemoryType = static_cast<CUmemorytype>(CU_MEMORYTYPE_DEVICE);
      params.srcHost = nullptr;
      params.srcDevice = static_cast<char *>(device_) + input.offset;
      params.srcXInBytes = 0;
      params.srcPitch = params.WidthInBytes;
      driver.graph_memcpy_node_set_params(input.node, &params);
    }
    inputs_.clear();
  }

  std::uint64_t requested_device_bytes() const override {
    return bytes_;
  }

  void release(bool backend_safe) noexcept override {
    auto *device = std::exchange(device_, nullptr);
    bytes_ = 0;
    if (!device || !backend_safe || !fault_domain_ ||
        !fault_domain_->backend_calls_safe() ||
        fault_domain_->state() == RuntimeLifecycleState::kFinalized)
      return;
    try {
      auto context = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().mem_free(device);
    } catch (...) {
      // Failed contexts must not cause another backend call during unwinding.
    }
  }

 private:
  struct Input {
    void *node{nullptr};
    CUDA_MEMCPY3D params{};
    std::vector<std::uint8_t> bytes;
    std::size_t offset{0};
  };
  std::vector<Input> inputs_;
  void *device_{nullptr};
  std::uint64_t bytes_{0};
  std::shared_ptr<RuntimeFaultDomain> fault_domain_;
};

class CudaCudssPlanImpl final : public CudaCudssPlan {
 public:
  CudaCudssPlanImpl(const CudaCudssMatrixView &matrix,
                    int matrix_type,
                    int matrix_view,
                    const std::string &adapter_path,
                    const std::string &runtime_library_path,
                    std::shared_ptr<RuntimeFaultDomain> fault_domain,
                    const std::vector<int> &configuration,
                    bool graph_owned)
      : rows_(static_cast<std::size_t>(matrix.rows)),
        nonzeros_(static_cast<std::size_t>(matrix.nonzeros)),
        graph_owned_(graph_owned),
        provider_(std::make_unique<CudssProviderRuntime>(adapter_path,
                                                         runtime_library_path)),
        fault_domain_(std::move(fault_domain)) {
    validate_cudss_matrix_contract(matrix_type, matrix_view);
    TI_ERROR_IF(graph_owned && configuration.empty(),
                "CUDA cuDSS Graph plans require a frozen configuration.");
    TI_ERROR_IF(!configuration.empty() &&
                    (configuration.size() != 2 || configuration[0] < 0 ||
                     configuration[0] > 3 || configuration[1] < 0 ||
                     configuration[1] > 1),
                "CUDA cuDSS private configuration must contain a supported "
                "reordering and solve policy.");
    if (!configuration.empty()) {
      configuration_api_ = provider_->configuration_api();
      reordering_ = configuration[0];
      solve_algorithm_ = configuration[1];
    }
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    try {
      require_cudss_success(api.create(runtime, &context_), "handle creation");
      TI_ERROR_IF(!context_, "CUDA cuDSS returned a null handle.");
      require_cudss_success(api.set_stream(runtime, context_, nullptr),
                            "runtime stream binding");
      if (graph_owned_) {
        auto &driver = CUDADriver::get_instance();
        TI_ERROR_IF(
            !driver.malloc_async_impl.available() ||
                !driver.mem_free_async_impl.available() ||
                !CUDAContext::get_instance().supports_mem_pool(),
            "CUDA cuDSS Graph plans require stream-ordered allocation.");
        // cuDSS analysis/factors retain stream-local state. Keep one stream
        // for the whole owner lifetime, including capture and destruction.
        driver.stream_create(&graph_stream_, CU_STREAM_NON_BLOCKING);
        driver.event_create(&graph_fork_event_, CU_EVENT_DISABLE_TIMING);
        driver.event_create(&graph_join_event_, CU_EVENT_DISABLE_TIMING);
        require_cudss_success(api.set_stream(runtime, context_, graph_stream_),
                              "Graph owner stream binding");
        allocator_ = std::make_unique<CudssGraphAllocator>();
        require_cudss_success(
            provider_->allocator_api().set_allocator(
                runtime, context_, allocator_.get(),
                CudssGraphAllocator::allocate, CudssGraphAllocator::deallocate),
            "Graph allocator binding");
        create_graph_snapshot(matrix);
      }
      require_cudss_success(api.config_create(runtime, &config_),
                            "configuration creation");
      if (configuration_api_.configure) {
        require_cudss_success(
            configuration_api_.configure(runtime, config_, reordering_,
                                         solve_algorithm_),
            "frozen configuration round-trip");
      }
      require_cudss_success(api.data_create(runtime, context_, &data_),
                            "solver-data creation");
      const void *row_start =
          graph_owned_ ? seed_buffers_[0] : matrix.row_offsets;
      // cuDSS accepts the canonical three-array CSR form when rowEnd is null.
      // Passing rowOffsets + 1 selects its unsupported four-array CSR form.
      require_cudss_success(
          api.matrix_create_csr(
              runtime, &matrix_, static_cast<std::int64_t>(matrix.rows),
              static_cast<std::int64_t>(matrix.columns),
              static_cast<std::int64_t>(matrix.nonzeros), row_start, nullptr,
              graph_owned_ ? seed_buffers_[1] : matrix.column_indices,
              graph_owned_ ? seed_buffers_[2] : matrix.values,
              kCudssDataTypeI32, kCudssDataTypeI32, kCudssDataTypeF32,
              matrix_type, matrix_view, kCudssBaseZero),
          "CSR descriptor creation");
      if (graph_owned_) {
        prepare_graph_snapshot();
      }
    } catch (...) {
      destroy(true);
      throw;
    }
  }

  ~CudaCudssPlanImpl() override {
    const bool provider_calls_safe =
        fault_domain_ && !fault_domain_->has_fatal_fault();
    if (provider_calls_safe) {
      try {
        auto cuda_submission_guard =
            CUDAContext::get_instance().get_submission_lock_guard();
        auto context_guard = CUDAContext::get_instance().get_guard();
        destroy(true);
        return;
      } catch (...) {
      }
    }
    destroy(false);
  }

  std::shared_ptr<CudssCaptureInputs> capture_inputs_for_recording() const {
    return std::make_shared<CudssCaptureInputs>(fault_domain_);
  }

  void analyze(const CudaCudssMatrixView &matrix) override {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("analyze");
    TI_ERROR_IF(static_cast<std::size_t>(matrix.rows) != rows_ ||
                    matrix.rows != matrix.columns,
                "CUDA cuDSS analyze received a matrix that does not match "
                "the plan shape or dtype.");
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    // Invalidate cached analysis facts before a potentially failing reanalysis.
    // Only configured recipes request these estimates; ordinary plans do not
    // acquire additional vendor queries.
    memory_estimates_status_ = -1;
    memory_estimates_written_ = 0;
    memory_estimates_.fill(-1);
    require_cudss_success(
        api.execute(runtime, context_, kCudssPhaseAnalysis, config_, data_,
                    matrix_, nullptr, nullptr),
        "analysis");
    if (configuration_api_.analysis_memory_estimates) {
      memory_estimates_status_ = configuration_api_.analysis_memory_estimates(
          runtime, context_, data_, memory_estimates_.data(),
          sizeof(memory_estimates_), &memory_estimates_written_);
    }
    analyzed_csr_row_ptr_.resize(rows_ + 1);
    analyzed_csr_col_ind_.resize(matrix.nonzeros);
    CUDADriver::get_instance().memcpy_device_to_host(
        analyzed_csr_row_ptr_.data(), matrix.row_offsets,
        sizeof(int) * analyzed_csr_row_ptr_.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        analyzed_csr_col_ind_.data(), matrix.column_indices,
        sizeof(int) * analyzed_csr_col_ind_.size());
    analyzed_matrix_id_ = matrix.matrix_id;
    analyzed_pattern_version_ = matrix.pattern_version;
    analyzed_shared_pattern_id_ = matrix.shared_pattern_id;
    analyzed_ = true;
    factorized_ = false;
    factorized_from_explicit_values_ = false;
  }

  void factorize(const CudaCudssMatrixView &matrix, bool refactorize) override {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("factorize");
    TI_ERROR_IF(!analyzed_,
                "CUDA cuDSS factorization requires analyze() first.");
    TI_ERROR_IF(refactorize && !factorized_,
                "CUDA cuDSS refactorization requires a prior successful "
                "factorization.");
    validate_analyzed_pattern(matrix);
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    require_cudss_success(api.matrix_set_csr_pointers(
                              runtime, matrix_, matrix.row_offsets, nullptr,
                              matrix.column_indices, matrix.values),
                          "CSR descriptor rebinding");
    factorized_ = false;
    factorized_from_explicit_values_ = false;
    ++factor_invalidations_;
    require_cudss_success(
        api.execute(
            runtime, context_,
            refactorize ? kCudssPhaseRefactorization : kCudssPhaseFactorization,
            config_, data_, matrix_, nullptr, nullptr),
        refactorize ? "refactorization" : "factorization");
    factorized_ = true;
    factorized_matrix_id_ = matrix.matrix_id;
    factorized_pattern_version_ = matrix.pattern_version;
    factorized_numeric_version_ = matrix.numeric_version;
    ++factor_generation_;
  }

  void solve(const CudaCudssMatrixView &matrix,
             void *rhs,
             void *solution) override {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("solve");
    TI_ERROR_IF(!factorized_,
                "CUDA cuDSS solve requires a successful factorization.");
    TI_ERROR_IF(factorized_from_explicit_values_,
                "CUDA cuDSS factors came from explicit Graph values. Use "
                "record_refactor_solve() again or factorize the stored "
                "matrix before a standalone solve.");
    TI_ERROR_IF(
        factorized_matrix_id_ != matrix.matrix_id ||
            factorized_pattern_version_ != matrix.pattern_version ||
            factorized_numeric_version_ != matrix.numeric_version,
        "CUDA cuDSS factorization is stale because the matrix or its "
        "pattern/numeric version changed. Call factorize() again before "
        "solve().");
    bind_dense_vectors(rhs, solution);
    execute_solve();
  }

  std::size_t rows() const noexcept override {
    return rows_;
  }

  std::size_t nonzeros() const noexcept override {
    return nonzeros_;
  }

  void reserve_refactor_solve() override {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("refactorize+solve");
    TI_ERROR_IF(!analyzed_,
                "CUDA cuDSS refactorize+solve requires a prior successful "
                "analysis.");
    TI_ERROR_IF(next_refactor_solve_generation_ ==
                    (std::numeric_limits<std::uint64_t>::max)(),
                "CUDA cuDSS refactorize+solve transaction generation "
                "space exhausted.");
    refactor_solve_inflight_ = true;
    refactor_solve_provider_started_ = false;
    refactor_solve_uses_full_factorization_ = !factorized_;
    active_refactor_solve_generation_ = next_refactor_solve_generation_++;
    factorized_ = false;
    factorized_from_explicit_values_ = false;
    factorized_matrix_id_ = 0;
    factorized_pattern_version_ = 0;
    factorized_numeric_version_ = 0;
    ++factor_invalidations_;
    ++refactor_solve_attempts_;
  }

  void cancel_unsubmitted_refactor_solve() noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (refactor_solve_inflight_ && !refactor_solve_provider_started_) {
      refactor_solve_inflight_ = false;
      refactor_solve_uses_full_factorization_ = false;
      active_refactor_solve_generation_ = 0;
      ++refactor_solve_failures_;
    }
  }

  void execute_reserved_refactor_solve(void *values,
                                       void *rhs,
                                       void *solution) override {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    TI_ERROR_IF(!refactor_solve_inflight_,
                "CUDA cuDSS refactorize+solve has no reserved transaction.");
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    try {
      require_cudss_success(api.matrix_set_values(runtime, matrix_, values),
                            "explicit matrix-values rebinding");
      bind_dense_vectors(rhs, solution);
      refactor_solve_provider_started_ = true;
      const auto factorization_phase = refactor_solve_uses_full_factorization_
                                           ? kCudssPhaseFactorization
                                           : kCudssPhaseRefactorization;
      const auto refactor_status =
          api.execute(runtime, context_, factorization_phase, config_, data_,
                      matrix_, nullptr, nullptr);
      const bool inject_failure =
          debug_fail_next_refactor_solve_after_provider_call_;
      debug_fail_next_refactor_solve_after_provider_call_ = false;
      require_cudss_success(refactor_status, "transactional refactorization");
      TI_ERROR_IF(
          inject_failure,
          "Injected CUDA cuDSS transactional refactorization failure after "
          "the provider call.");
      execute_solve();
      factorized_ = true;
      factorized_from_explicit_values_ = true;
      ++factor_generation_;
      ++refactor_solve_successes_;
    } catch (...) {
      factorized_ = false;
      factorized_from_explicit_values_ = false;
      ++refactor_solve_failures_;
      throw;
    }
  }

  void debug_fail_next_refactor_solve() override {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    require_no_refactor_solve_inflight("failure injection");
    debug_fail_next_refactor_solve_after_provider_call_ = true;
  }

  std::uint64_t submission_retirement_token() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return refactor_solve_inflight_ ? active_refactor_solve_generation_ : 0;
  }

  void on_submission_retired(std::uint64_t token) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (refactor_solve_inflight_ && token != 0 &&
        token == active_refactor_solve_generation_) {
      refactor_solve_inflight_ = false;
      refactor_solve_provider_started_ = false;
      refactor_solve_uses_full_factorization_ = false;
      active_refactor_solve_generation_ = 0;
      ++refactor_solve_retirements_;
    }
  }

  std::unordered_map<std::string, std::uint64_t> statistics() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &runtime_info = provider_->runtime_info();
    return {{"rows", static_cast<std::uint64_t>(rows_)},
            {"provider_abi_version", TI_FORGE_CUDSS_PROVIDER_ABI_VERSION},
            {"provider_version_major", runtime_info.version_major},
            {"provider_version_minor", runtime_info.version_minor},
            {"provider_version_patch", runtime_info.version_patch},
            {"analyzed", analyzed_ ? 1u : 0u},
            {"factorized", factorized_ ? 1u : 0u},
            {"factorized_from_explicit_values",
             factorized_from_explicit_values_ ? 1u : 0u},
            {"factor_generation", factor_generation_},
            {"factor_invalidations", factor_invalidations_},
            {"refactor_solve_inflight", refactor_solve_inflight_ ? 1u : 0u},
            {"refactor_solve_transaction_generation",
             active_refactor_solve_generation_},
            {"refactor_solve_attempts", refactor_solve_attempts_},
            {"refactor_solve_successes", refactor_solve_successes_},
            {"refactor_solve_failures", refactor_solve_failures_},
            {"refactor_solve_retirements", refactor_solve_retirements_},
            {"closed", closed_ ? 1u : 0u}};
  }

  std::unordered_map<std::string, std::int64_t> configuration() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool estimates_valid =
        analyzed_ && memory_estimates_status_ == kCudssStatusSuccess &&
        memory_estimates_written_ >= 6 * sizeof(std::int64_t);
    const auto allocation =
        allocator_ ? allocator_->observation() : std::array<std::uint64_t, 3>{};
    return {
        {"configuration_abi", configuration_api_.abi_version},
        {"graph_owned", graph_owned_ ? 1 : 0},
        {"capture_parameters_device_resident", graph_owned_ ? 1 : 0},
        {"allocator_live_requested_bytes",
         static_cast<std::int64_t>(allocation[0])},
        {"allocator_peak_requested_bytes",
         static_cast<std::int64_t>(allocation[1])},
        {"sealed_allocation_rejections",
         static_cast<std::int64_t>(allocation[2])},
        {"graph_snapshot_bytes",
         static_cast<std::int64_t>(graph_snapshot_bytes_)},
        {"reordering", reordering_},
        {"solve", solve_algorithm_},
        {"factor_statistics_abi", factor_statistics_abi_},
        {"factor_statistics_result", factor_statistics_result_},
        {"factor_statistics_queries", factor_statistics_queries_},
        {"factor_lu_nonzeros", factor_statistics_.lu_nonzeros.value},
        {"factor_lu_nonzeros_status",
         factor_statistics_.lu_nonzeros.vendor_status},
        {"factor_lu_nonzeros_written_bytes",
         factor_statistics_.lu_nonzeros.written_bytes},
        {"factor_superpanels", factor_statistics_.superpanels.value},
        {"factor_superpanels_status",
         factor_statistics_.superpanels.vendor_status},
        {"factor_superpanels_written_bytes",
         factor_statistics_.superpanels.written_bytes},
        {"factor_flops", factor_statistics_.factor_flops.value},
        {"factor_flops_status", factor_statistics_.factor_flops.vendor_status},
        {"factor_flops_written_bytes",
         factor_statistics_.factor_flops.written_bytes},
        {"memory_estimates_status", memory_estimates_status_},
        {"memory_estimates_written_bytes",
         static_cast<std::int64_t>(memory_estimates_written_)},
        {"estimated_device_persistent_bytes",
         estimates_valid ? memory_estimates_[0] : -1},
        {"estimated_device_peak_bytes",
         estimates_valid ? memory_estimates_[1] : -1},
        {"estimated_host_persistent_bytes",
         estimates_valid ? memory_estimates_[2] : -1},
        {"estimated_host_peak_bytes",
         estimates_valid ? memory_estimates_[3] : -1}};
  }

  void claim_graph_phase(int phase) override {
    auto &driver = CUDADriver::get_instance();
    TI_ERROR_IF(
        !driver.stream_get_capture_info_v2.available() ||
            !driver.graph_node_get_type.available() ||
            !driver.graph_memcpy_node_get_params.available() ||
            !driver.graph_memcpy_node_set_params.available(),
        "cuDSS capture requires immutable host-input snapshot support.");
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_ || !graph_owned_ || !factorized_,
                "CUDA cuDSS capture requires a prepared Graph-owned snapshot.");
    TI_ERROR_IF(phase != 0 && phase != kCudssPhaseFactorization &&
                    phase != kCudssPhaseRefactorization,
                "CUDA cuDSS capture numerical phase is unsupported.");
    TI_ERROR_IF(graph_numeric_phase_ != -1 && graph_numeric_phase_ != phase,
                "A cuDSS Graph owner cannot mix fixed factors and numerical "
                "update policies; materialize an independent owner.");
    graph_numeric_phase_ = phase;
  }

  void record_capture(
      int phase,
      void *values,
      const std::vector<std::pair<void *, void *>> &rhs_and_outputs,
      void *stream,
      std::shared_ptr<CudaGraphCaptureResources> &resources) override {
    resources.reset();
    std::shared_ptr<CudssCaptureInputs> capture_inputs;
    TI_ERROR_IF(rhs_and_outputs.empty() || (phase && !values),
                "cuDSS capture requires dense bindings and phase values.");
    std::uint32_t capture_status = 0;
    CUgraph graph = nullptr;
    auto &driver = CUDADriver::get_instance();
    driver.stream_get_capture_info_v2(stream, &capture_status, nullptr, &graph,
                                      nullptr, nullptr);
    if (capture_status == 1) {
      capture_inputs = capture_inputs_for_recording();
      resources = capture_inputs;
    }
    for (std::size_t i = 0; i < rhs_and_outputs.size(); ++i) {
      const auto previous =
          capture_status == 1 ? graph_nodes(graph) : std::vector<void *>{};
      const std::unordered_set<void *> previous_nodes(previous.begin(),
                                                      previous.end());
      record_graph(phase, values, rhs_and_outputs[i].first,
                   rhs_and_outputs[i].second, stream, i == 0);
      if (capture_status != 1)
        continue;
      // Freeze each RHS's vendor staging before recording the next one.
      // One numerical update serves every RHS in this complete region.
      for (auto *node : graph_nodes(graph)) {
        if (previous_nodes.count(node))
          continue;
        std::uint32_t type = 0;
        driver.graph_node_get_type(node, &type);
        if (type != 1 /* memcpy */)
          continue;
        CUDA_MEMCPY3D params{};
        driver.graph_memcpy_node_get_params(node, &params);
        TI_ERROR_IF(params.dstMemoryType != CU_MEMORYTYPE_DEVICE,
                    "cuDSS capture contains a non-device copy destination.");
        if (params.srcMemoryType == CU_MEMORYTYPE_DEVICE)
          continue;
        TI_ERROR_IF(params.srcMemoryType != 1 /* host */ || !params.srcHost ||
                        params.Height != 1 || params.Depth != 1 ||
                        params.srcY || params.srcZ || params.srcLOD,
                    "cuDSS capture host inputs require contiguous 1D storage.");
        capture_inputs->snapshot(node, params);
      }
    }
  }

  void record_graph(int phase,
                    void *values,
                    void *rhs,
                    void *solution,
                    void *stream,
                    bool update_factors = true) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_ || !graph_owned_ || !factorized_ ||
                    graph_numeric_phase_ != phase,
                "CUDA cuDSS captured owner is unavailable.");
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    auto &driver = CUDADriver::get_instance();
    // Capture joins the retained owner stream instead of rebinding warmed
    // cuDSS state to a transient parent stream. These event operations become
    // Graph dependency edges; replay does not call cuDSS or host-synchronize.
    driver.event_record(graph_fork_event_, stream);
    driver.stream_wait_event(graph_stream_, graph_fork_event_, 0);
    try {
      require_cudss_success(
          api.matrix_set_values(runtime, matrix_,
                                phase ? values : seed_buffers_[2]),
          "capture matrix binding");
      bind_dense_vectors(rhs, solution);
      if (phase && update_factors) {
        require_cudss_success(api.execute(runtime, context_, phase, config_,
                                          data_, matrix_, solution_, rhs_),
                              "captured numerical factorization");
      }
      execute_solve();
      driver.event_record(graph_join_event_, graph_stream_);
      driver.stream_wait_event(stream, graph_join_event_, 0);
    } catch (...) {
      factorized_ = false;
      throw;
    }
  }

  void destroy(bool provider_calls_safe) noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    if (!provider_calls_safe || !provider_) {
      return;
    }
    if (allocator_) {
      allocator_->seal(false);
    }
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    if (graph_stream_) {
      warn_cudss_failure(
          CUDADriver::get_instance().stream_synchronize.call(graph_stream_),
          "Graph owner retirement");
    }
    if (solution_) {
      warn_cudss_failure(api.matrix_destroy(runtime, solution_),
                         "solution descriptor destruction");
      solution_ = nullptr;
    }
    if (rhs_) {
      warn_cudss_failure(api.matrix_destroy(runtime, rhs_),
                         "right-hand-side descriptor destruction");
      rhs_ = nullptr;
    }
    if (matrix_) {
      warn_cudss_failure(api.matrix_destroy(runtime, matrix_),
                         "CSR descriptor destruction");
      matrix_ = nullptr;
    }
    if (data_) {
      warn_cudss_failure(api.data_destroy(runtime, context_, data_),
                         "solver-data destruction");
      data_ = nullptr;
    }
    for (auto &buffer : seed_buffers_) {
      if (buffer) {
        warn_cudss_failure(CudssGraphAllocator::deallocate(
                               allocator_.get(), buffer, 0, graph_stream_),
                           "private snapshot retirement");
        buffer = nullptr;
      }
    }
    if (config_) {
      warn_cudss_failure(api.config_destroy(runtime, config_),
                         "configuration destruction");
      config_ = nullptr;
    }
    if (context_) {
      warn_cudss_failure(api.destroy(runtime, context_), "handle destruction");
      context_ = nullptr;
    }
    auto &driver = CUDADriver::get_instance();
    if (graph_fork_event_) {
      driver.event_destroy.call(graph_fork_event_);
      graph_fork_event_ = nullptr;
    }
    if (graph_join_event_) {
      driver.event_destroy.call(graph_join_event_);
      graph_join_event_ = nullptr;
    }
    if (graph_stream_) {
      driver.stream_destroy.call(graph_stream_);
      graph_stream_ = nullptr;
    }
  }

 private:
  static std::vector<void *> graph_nodes(CUgraph graph) {
    auto &driver = CUDADriver::get_instance();
    std::size_t count = 0;
    driver.graph_get_nodes(graph, nullptr, &count);
    std::vector<void *> nodes(count);
    driver.graph_get_nodes(graph, nodes.data(), &count);
    nodes.resize(count);
    return nodes;
  }
  void create_graph_snapshot(const CudaCudssMatrixView &matrix) {
    const std::array<std::size_t, 5> sizes = {
        (rows_ + 1) * sizeof(int), nonzeros_ * sizeof(int),
        nonzeros_ * sizeof(float), rows_ * sizeof(float),
        rows_ * sizeof(float)};
    const std::array<void *, 3> sources = {
        matrix.row_offsets, matrix.column_indices, matrix.values};
    auto &driver = CUDADriver::get_instance();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      require_cudss_success(
          CudssGraphAllocator::allocate(allocator_.get(), &seed_buffers_[i],
                                        sizes[i], nullptr),
          "private snapshot allocation");
      graph_snapshot_bytes_ += sizes[i];
      if (i < sources.size()) {
        driver.memcpy_device_to_device(seed_buffers_[i], sources[i], sizes[i]);
      } else {
        driver.memset(seed_buffers_[i], 0, sizes[i]);
      }
    }
  }

  void prepare_graph_snapshot() {
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    // The private copy was made on the runtime stream. Publish it to the
    // owner's stream at this cold materialization boundary only.
    CUDADriver::get_instance().stream_synchronize(nullptr);
    bind_dense_vectors(seed_buffers_[3], seed_buffers_[4]);
    require_cudss_success(api.execute(runtime, context_, kCudssPhaseAnalysis,
                                      config_, data_, matrix_, solution_, rhs_),
                          "Graph snapshot analysis");
    analyzed_ = true;
    memory_estimates_status_ = configuration_api_.analysis_memory_estimates(
        runtime, context_, data_, memory_estimates_.data(),
        sizeof(memory_estimates_), &memory_estimates_written_);
    require_cudss_success(
        api.execute(runtime, context_, kCudssPhaseFactorization, config_, data_,
                    matrix_, solution_, rhs_),
        "Graph snapshot factorization");
    execute_solve();
    // This is materialization of private data, never Graph prepare() or replay.
    // Finish initial factors before publishing a captured owner.
    CUDADriver::get_instance().stream_synchronize(graph_stream_);
    factorized_ = true;
    const auto diagnostics = provider_->factor_statistics_api();
    factor_statistics_abi_ = diagnostics.abi_version;
    if (diagnostics.read_i32) {
      // DataGet can synchronize. Collect only this completed private snapshot,
      // never on factor/solve replay or on a passive configuration report.
      ++factor_statistics_queries_;
      factor_statistics_result_ =
          diagnostics.read_i32(runtime, context_, data_, &factor_statistics_);
    }
    allocator_->seal(true);
  }

  void require_no_refactor_solve_inflight(const char *operation) const {
    TI_ERROR_IF(refactor_solve_inflight_,
                "CUDA cuDSS {} cannot run while a refactorize+solve "
                "transaction is in flight. Wait for its completion before "
                "reusing this plan.",
                operation);
  }

  void bind_dense_vectors(void *rhs, void *solution) {
    const auto &api = provider_->api();
    auto runtime = provider_->runtime();
    if (!rhs_) {
      try {
        require_cudss_success(
            api.matrix_create_dn(runtime, &rhs_,
                                 static_cast<std::int64_t>(rows_), 1,
                                 static_cast<std::int64_t>(rows_), rhs,
                                 kCudssDataTypeF32, kCudssLayoutColumnMajor),
            "right-hand-side descriptor creation");
        require_cudss_success(
            api.matrix_create_dn(runtime, &solution_,
                                 static_cast<std::int64_t>(rows_), 1,
                                 static_cast<std::int64_t>(rows_), solution,
                                 kCudssDataTypeF32, kCudssLayoutColumnMajor),
            "solution descriptor creation");
      } catch (...) {
        if (solution_) {
          warn_cudss_failure(api.matrix_destroy(runtime, solution_),
                             "solution descriptor rollback");
          solution_ = nullptr;
        }
        if (rhs_) {
          warn_cudss_failure(api.matrix_destroy(runtime, rhs_),
                             "right-hand-side descriptor rollback");
          rhs_ = nullptr;
        }
        throw;
      }
    } else {
      require_cudss_success(api.matrix_set_values(runtime, rhs_, rhs),
                            "right-hand-side rebinding");
      require_cudss_success(api.matrix_set_values(runtime, solution_, solution),
                            "solution rebinding");
    }
  }

  void execute_solve() {
    const auto &api = provider_->api();
    require_cudss_success(
        api.execute(provider_->runtime(), context_, kCudssPhaseSolve, config_,
                    data_, matrix_, solution_, rhs_),
        "solve");
  }

  void validate_analyzed_pattern(const CudaCudssMatrixView &matrix) const {
    TI_ERROR_IF(static_cast<std::size_t>(matrix.rows) != rows_ ||
                    matrix.rows != matrix.columns ||
                    matrix.nonzeros != analyzed_csr_col_ind_.size(),
                "CUDA cuDSS factorize() requires the same sparse pattern "
                "that was passed to analyze(); shape, dtype, or nonzero "
                "count changed.");
    if ((matrix.matrix_id == analyzed_matrix_id_ &&
         matrix.pattern_version == analyzed_pattern_version_) ||
        (analyzed_shared_pattern_id_ != 0 &&
         matrix.shared_pattern_id == analyzed_shared_pattern_id_)) {
      return;
    }
    std::vector<int> row_ptr(analyzed_csr_row_ptr_.size());
    std::vector<int> col_ind(analyzed_csr_col_ind_.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        row_ptr.data(), matrix.row_offsets, sizeof(int) * row_ptr.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        col_ind.data(), matrix.column_indices, sizeof(int) * col_ind.size());
    TI_ERROR_IF(
        row_ptr != analyzed_csr_row_ptr_ || col_ind != analyzed_csr_col_ind_,
        "CUDA cuDSS factorize() requires the same sparse pattern "
        "that was passed to analyze(); a CSR index changed.");
  }

  std::size_t rows_{0};
  TiForgeCudssConfigurationApi configuration_api_{};
  int reordering_{-1};
  int solve_algorithm_{-1};
  std::array<std::int64_t, 16> memory_estimates_{};
  std::int64_t memory_estimates_status_{-1};
  std::size_t memory_estimates_written_{0};
  TiForgeCudssFactorStatistics factor_statistics_{};
  std::int64_t factor_statistics_abi_{0};
  std::int64_t factor_statistics_result_{-1};
  std::int64_t factor_statistics_queries_{0};
  std::size_t nonzeros_{0};
  const bool graph_owned_{false};
  int graph_numeric_phase_{-1};
  void *graph_stream_{nullptr};
  void *graph_fork_event_{nullptr};
  void *graph_join_event_{nullptr};
  std::unique_ptr<CudssGraphAllocator> allocator_;
  std::array<void *, 5> seed_buffers_{};
  std::size_t graph_snapshot_bytes_{0};
  void *context_{nullptr};
  void *config_{nullptr};
  void *data_{nullptr};
  void *matrix_{nullptr};
  void *rhs_{nullptr};
  void *solution_{nullptr};
  bool analyzed_{false};
  bool factorized_{false};
  bool factorized_from_explicit_values_{false};
  bool refactor_solve_inflight_{false};
  bool refactor_solve_provider_started_{false};
  bool refactor_solve_uses_full_factorization_{false};
  bool debug_fail_next_refactor_solve_after_provider_call_{false};
  bool closed_{false};
  std::vector<int> analyzed_csr_row_ptr_;
  std::vector<int> analyzed_csr_col_ind_;
  std::uint64_t analyzed_matrix_id_{0};
  std::uint64_t analyzed_pattern_version_{0};
  std::uint64_t analyzed_shared_pattern_id_{0};
  std::uint64_t factorized_matrix_id_{0};
  std::uint64_t factorized_pattern_version_{0};
  std::uint64_t factorized_numeric_version_{0};
  std::uint64_t factor_generation_{0};
  std::uint64_t factor_invalidations_{0};
  std::uint64_t refactor_solve_attempts_{0};
  std::uint64_t refactor_solve_successes_{0};
  std::uint64_t refactor_solve_failures_{0};
  std::uint64_t refactor_solve_retirements_{0};
  std::uint64_t next_refactor_solve_generation_{1};
  std::uint64_t active_refactor_solve_generation_{0};
  std::unique_ptr<CudssProviderRuntime> provider_;
  std::shared_ptr<RuntimeFaultDomain> fault_domain_;
  mutable std::mutex mutex_;
};

}  // namespace

std::shared_ptr<CudaCudssPlan> make_cuda_cudss_plan(
    const CudaCudssMatrixView &matrix,
    int matrix_type,
    int matrix_view,
    const std::string &adapter_path,
    const std::string &runtime_library_path,
    std::shared_ptr<RuntimeFaultDomain> fault_domain,
    const std::vector<int> &configuration,
    bool graph_owned) {
  TI_ERROR_IF(!matrix.rows || matrix.rows != matrix.columns ||
                  !matrix.nonzeros || !matrix.row_offsets ||
                  !matrix.column_indices || !matrix.values ||
                  (!graph_owned && !matrix.matrix_id) || !fault_domain,
              "cuDSS requires a validated square f32/i32 CSR, live identity, "
              "and fault owner.");
  return std::make_shared<CudaCudssPlanImpl>(
      matrix, matrix_type, matrix_view, adapter_path, runtime_library_path,
      std::move(fault_domain), configuration, graph_owned);
}
}  // namespace taichi::lang
