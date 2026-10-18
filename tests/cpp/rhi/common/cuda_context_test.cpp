#include <gtest/gtest.h>

#include <taichi/common/core.h>
#include <taichi/rhi/cuda/cuda_capability.h>
#include <taichi/rhi/cuda/cuda_context.h>
#include <taichi/rhi/cuda/cuda_device.h>
#include <taichi/rhi/cuda/cusparse_plan.h>
#include <taichi/runtime/cuda/graph_memory_pool.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace taichi::lang {

namespace {

template <typename Function>
struct DriverArguments;
template <typename... Args>
struct DriverArguments<CUDADriverFunction<Args...>> {
  using Tuple = std::tuple<Args...>;
};
template <std::size_t Index, typename Function>
using DriverArgument =
    std::tuple_element_t<Index, typename DriverArguments<Function>::Tuple>;

class ControllableTelemetryMutex {
 public:
  void lock() {
    bool expected = true;
    while (!available_.compare_exchange_weak(
        expected, false, std::memory_order_acquire,
        std::memory_order_relaxed)) {
      expected = true;
      std::this_thread::yield();
    }
  }

  bool try_lock() {
    try_lock_called_.store(true, std::memory_order_release);
    bool expected = true;
    return available_.compare_exchange_strong(
        expected, false, std::memory_order_acquire,
        std::memory_order_relaxed);
  }

  void unlock() {
    available_.store(true, std::memory_order_release);
  }

  void block() {
    try_lock_called_.store(false, std::memory_order_relaxed);
    available_.store(false, std::memory_order_release);
  }

  void wait_for_try_lock() const {
    while (!try_lock_called_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

 private:
  std::atomic<bool> available_{true};
  std::atomic<bool> try_lock_called_{false};
};

}  // namespace

TEST(CUDAVersion, ReportsTheBundledLibdeviceCompatibilityVersion) {
  const auto version = get_cuda_version_string();
  EXPECT_FALSE(version.empty());
  EXPECT_EQ(version.find_first_not_of("0123456789."), std::string::npos);
}

TEST(CUSPARSEDriverABI, DescriptorExtentsMatchTheVendorContract) {
  // cuSPARSE Generic API extents are signed 64-bit even with 32-bit indices.
  // Legacy routines such as Xcoo2csr intentionally retain their int ABI.
  using Csr = decltype(CUSPARSEDriver::cpCreateCsr);
  using Coo = decltype(CUSPARSEDriver::cpCreateCoo);
  using SparseVector = decltype(CUSPARSEDriver::cpCreateSpVec);
  using DenseVector = decltype(CUSPARSEDriver::cpCreateDnVec);
  using CsrGet = decltype(CUSPARSEDriver::cpCsrGet);
  using GetSize = decltype(CUSPARSEDriver::cpGetSize);
  EXPECT_TRUE((std::is_same_v<DriverArgument<1, Csr>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<2, Csr>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<3, Csr>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<1, Coo>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<2, Coo>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<3, Coo>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<1, SparseVector>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<2, SparseVector>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<1, DenseVector>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<1, CsrGet>, std::int64_t *>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<2, CsrGet>, std::int64_t *>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<3, CsrGet>, std::int64_t *>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<1, GetSize>, std::int64_t *>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<2, GetSize>, std::int64_t *>));
  EXPECT_TRUE((std::is_same_v<DriverArgument<3, GetSize>, std::int64_t *>));
}

TEST(CUSPARSEDriverABI, WideExtentsReachTheDynamicCallWithoutAllocation) {
  using Function = decltype(CUSPARSEDriver::cpCreateDnVec);
  // Do not call a deliberately mismatched function pointer on a broken build.
  if constexpr (std::is_same_v<DriverArgument<1, Function>, std::int64_t>) {
    Function function;
    auto callback = +[](cusparseDnVecDescr_t *, std::int64_t size, void *,
                        cudaDataType) -> uint32_t {
      return size == (std::int64_t{1} << 33) + 17 || size == -1 ? 0 : 1;
    };
    std::mutex mutex;
    function.set(reinterpret_cast<void *>(callback));
    function.set_lock(&mutex);
    EXPECT_EQ(function.call(nullptr, (std::int64_t{1} << 33) + 17, nullptr,
                            CUDA_R_32F),
              0);
    EXPECT_EQ(function.call(nullptr, -1, nullptr, CUDA_R_32F), 0);
  } else {
    FAIL() << "cuSPARSE vector extents are narrowed before the vendor call";
  }
}

TEST(CUSPARSECsrPlan, IndependentPlansRetainOnlyTheirOwnExecutionState) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  auto &context = CUDAContext::get_instance();
  auto submission = context.get_submission_lock_guard();
  auto guard = context.get_guard();
  auto &cusparse = CUSPARSEDriver::get_instance();
  if (!cusparse.is_loaded() && !cusparse.load_cusparse())
    GTEST_SKIP();
  auto &driver = CUDADriver::get_instance();
  std::vector<std::unique_ptr<void, void (*)(void *)>> storage;
  auto upload = [&](auto values) {
    void *address = nullptr;
    driver.malloc(&address, sizeof(values));
    storage.emplace_back(
        address,
        +[](void *value) { CUDADriver::get_instance().mem_free(value); });
    driver.memcpy_host_to_device(address, values.data(), sizeof(values));
    return address;
  };
  auto read = [&](void *address, std::size_t size) {
    std::vector<float> values(size);
    driver.memcpy_device_to_host(values.data(), address, size * sizeof(float));
    return values;
  };
  const auto rows = upload(std::array<std::int32_t, 3>{0, 2, 4});
  const auto columns = upload(std::array<std::int32_t, 4>{0, 2, 1, 2});
  const auto values = upload(std::array<float, 4>{2, 3, -1, 4});
  const auto x = upload(std::array<float, 3>{1, 2, 3});
  const auto x2 = upload(std::array<float, 3>{-1, .5f, 2});
  const auto y = upload(std::array<float, 2>{8, -2});
  const auto y2 = upload(std::array<float, 2>{99, 99});
  CudaSparseSpmvDescription description{2, 3, 4, rows, columns, values};
  description.alpha = 2;
  description.beta = .5f;
  auto first = std::make_unique<CudaSparseSpmvPlan>(description);
  description.alpha = 1;
  description.beta = 0;
  CudaSparseSpmvPlan second(description);
  EXPECT_TRUE(first->prepare(x, y, nullptr, true).built);
  EXPECT_TRUE(second.prepare(x2, y2, nullptr, true).built);
  EXPECT_EQ(read(y, 2), (std::vector<float>{8, -2}));
  EXPECT_EQ(read(y2, 2), (std::vector<float>{99, 99}));
  first->record(nullptr);
  second.record(nullptr);
  EXPECT_EQ(read(y, 2), (std::vector<float>{26, 19}));
  EXPECT_EQ(read(y2, 2), (std::vector<float>{4, 7.5f}));

  // The transpose is also the explicit adjoint. Use the generic 64-bit index
  // path with small data, independently of the producer's 32-bit CSR contract.
  description.row_offsets = upload(std::array<std::int64_t, 3>{0, 2, 4});
  description.column_indices = upload(std::array<std::int64_t, 4>{0, 2, 1, 2});
  description.row_type = description.column_type = CUSPARSE_INDEX_64I;
  description.transpose = true;
  CudaSparseSpmvPlan transpose(description);
  const auto q = upload(std::array<float, 2>{5, 7});
  const auto dq = upload(std::array<float, 3>{99, 99, 99});
  transpose.prepare(q, dq, nullptr, true);
  transpose.record(nullptr);
  EXPECT_EQ(read(dq, 3), (std::vector<float>{10, -7, 43}));

  first.reset();
  EXPECT_EQ(read(x, 3), (std::vector<float>{1, 2, 3}));
  std::array<float, 4> changed{1, 3, -2, 4};
  driver.memcpy_host_to_device(values, changed.data(), sizeof(changed));
  const auto reused = second.prepare(x2, y2, nullptr, true);
  EXPECT_FALSE(reused.built);
  EXPECT_EQ(reused.descriptor_creations, 0);
  second.record(nullptr);
  transpose.record(nullptr);
  EXPECT_EQ(read(y2, 2), (std::vector<float>{5, 7}));
  EXPECT_EQ(read(dq, 3), (std::vector<float>{5, -14, 43}));
  const auto rebound = second.prepare(x, y2, nullptr, false);
  EXPECT_FALSE(rebound.built);
  EXPECT_EQ(rebound.descriptor_creations, 1);
  EXPECT_EQ(rebound.descriptor_rebinds, 1);
  second.record(nullptr);
  EXPECT_EQ(read(y2, 2), (std::vector<float>{10, 8}));
  EXPECT_EQ(second.statistics().preprocess_builds,
            second.statistics().preprocessed ? 1 : 0);

  // An empty matrix overwrites old output on every replay, without owning the
  // null index/value buffers. Caller storage outlives all three plans above.
  description = CudaSparseSpmvDescription{
      2, 3, 0, upload(std::array<std::int32_t, 3>{0, 0, 0})};
  CudaSparseSpmvPlan empty(description);
  empty.prepare(x, y2, nullptr, true);
  empty.record(nullptr);
  EXPECT_EQ(read(y2, 2), (std::vector<float>{0, 0}));
}

TEST(CUSPARSEBsrPlan, SharedCoreOwnsOrBorrowsDescriptorsWithoutOwningData) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  auto &context = CUDAContext::get_instance();
  auto submission = context.get_submission_lock_guard();
  auto guard = context.get_guard();
  auto &cusparse = CUSPARSEDriver::get_instance();
  if ((!cusparse.is_loaded() && !cusparse.load_cusparse()) ||
      !cusparse.capabilities().generic_bsr_spmv_available)
    GTEST_SKIP();
  auto &driver = CUDADriver::get_instance();
  std::vector<std::unique_ptr<void, void (*)(void *)>> storage;
  auto upload = [&](auto values) {
    void *address = nullptr;
    driver.malloc(&address, sizeof(values));
    storage.emplace_back(
        address,
        +[](void *value) { CUDADriver::get_instance().mem_free(value); });
    driver.memcpy_host_to_device(address, values.data(), sizeof(values));
    return address;
  };
  auto read = [&](void *address) {
    std::vector<float> values(4);
    driver.memcpy_device_to_host(values.data(), address, 4 * sizeof(float));
    return values;
  };
  const auto rows = upload(std::array<std::int32_t, 3>{0, 2, 3});
  const auto columns = upload(std::array<std::int32_t, 3>{0, 2, 1});
  const auto values =
      upload(std::array<float, 12>{2, 1, 0, 3, -1, 0, 2, 1, 4, -2, 1, 5});
  const auto x = upload(std::array<float, 6>{1, 2, 3, 4, 5, 6});
  const auto x2 = upload(std::array<float, 6>{-1, 0, 2, -2, 1, 3});
  const auto y = upload(std::array<float, 4>{8, -2, 6, 4});
  const auto y2 = upload(std::array<float, 4>{99, 99, 99, 99});
  CudaSparseSpmvDescription description{4, 6, 3, rows, columns, values};
  description.block_size = 2;
  description.algorithm = CUSPARSE_SPMV_BSR_ALG1;
  description.alpha = 2;
  description.beta = .5f;
  auto first = std::make_unique<CudaSparseSpmvPlan>(description);
  EXPECT_TRUE(first->prepare(x, y, nullptr, false).built);
  EXPECT_EQ(read(y), (std::vector<float>{8, -2, 6, 4}));
  first->record(nullptr);
  EXPECT_EQ(read(y), (std::vector<float>{2, 43, 11, 48}));

  cusparseSpMatDescr_t borrowed = nullptr;
  cusparse.cpCreateBsr(&borrowed, 2, 3, 3, 2, 2, rows, columns, values,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F,
                       CUSPARSE_ORDER_ROW);
  auto borrowed_guard = std::unique_ptr<void, void (*)(void *)>(
      borrowed, +[](void *value) {
        CUSPARSEDriver::get_instance().cpDestroySpMat(
            static_cast<cusparseSpMatDescr_t>(value));
      });
  description.alpha = 1;
  description.beta = 0;
  {
    CudaSparseSpmvPlan second(description, borrowed);
    EXPECT_TRUE(second.prepare(x2, y2, nullptr, false).built);
    EXPECT_EQ(read(y2), (std::vector<float>{99, 99, 99, 99}));
    first.reset();
    second.record(nullptr);
    EXPECT_EQ(read(y2), (std::vector<float>{-3, 5, 12, -8}));
    const auto rebound = second.prepare(x, y2, nullptr, false);
    EXPECT_FALSE(rebound.built);
    EXPECT_EQ(rebound.descriptor_creations, 1);
    EXPECT_EQ(rebound.descriptor_rebinds, 1);
    second.record(nullptr);
    EXPECT_EQ(read(y2), (std::vector<float>{-1, 22, 4, 23}));
  }
  // Retirement must leave the caller's BSR descriptor and buffers usable.
  CudaSparseSpmvPlan third(description, borrowed);
  third.prepare(x, y2, nullptr, false);
  std::array<float, 12> changed{1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1};
  driver.memcpy_host_to_device(values, changed.data(), sizeof(changed));
  third.record(nullptr);
  EXPECT_EQ(read(y2), (std::vector<float>{1, 2, 3, 4}));
  description.transpose = true;
  EXPECT_THROW(CudaSparseSpmvPlan{description}, std::invalid_argument);
  description.transpose = false;
  description.rows = 3;
  EXPECT_THROW(CudaSparseSpmvPlan{description}, std::invalid_argument);
}

TEST(CUSPARSETriangularPlan, CapturedValueRefreshUsesReplayValues) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  auto &context = CUDAContext::get_instance();
  auto submission = context.get_submission_lock_guard();
  auto guard = context.get_guard();
  auto &cusparse = CUSPARSEDriver::get_instance();
  if (!cusparse.is_loaded() && !cusparse.load_cusparse())
    GTEST_SKIP();
  const auto caps = cusparse.capabilities();
  if (!caps.spsv_f32_available || !caps.spsm_f32_available ||
      !caps.spsv_value_update_available || !caps.spsm_value_update_available)
    GTEST_SKIP();
  auto &driver = CUDADriver::get_instance();
  std::vector<std::unique_ptr<void, void (*)(void *)>> storage;
  auto upload = [&](auto data) {
    void *address = nullptr;
    driver.malloc(&address, data.size() * sizeof(data[0]));
    storage.emplace_back(
        address,
        +[](void *value) { CUDADriver::get_instance().mem_free(value); });
    driver.memcpy_host_to_device(address, data.data(),
                                 data.size() * sizeof(data[0]));
    return address;
  };
  const auto offsets = upload(std::array<int32_t, 4>{0, 1, 3, 5});
  const auto columns = upload(std::array<int32_t, 5>{0, 0, 1, 1, 2});
  std::array<float, 5> initial{2, 1, 4, -1, 2};
  const auto values = upload(initial), replacement = upload(initial);
  for (int rhs : {1, 2}) {
    SCOPED_TRACE(rhs);
    driver.memcpy_host_to_device(values, initial.data(), sizeof(initial));
    std::vector<float> input(3 * rhs), output(3 * rhs, 99);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < rhs; ++j)
        input[i * rhs + j] = std::array<float, 3>{2, 9, 4}[i] * (j + 1);
    const auto x = upload(input), y = upload(output);
    CudaCsrTriangularPlan plan({3, 5, rhs, offsets, columns, values});
    EXPECT_TRUE(plan.prepare(x, y, nullptr).built);
    driver.stream_synchronize(nullptr);
    driver.memcpy_device_to_host(output.data(), y,
                                 output.size() * sizeof(float));
    EXPECT_EQ(output, std::vector<float>(3 * rhs, 99));
    struct Capture {
      void *stream{nullptr};
      CUgraph graph{nullptr};
      CUgraphExec executable{nullptr};
      bool active{false};
      ~Capture() {
        release();
      }
      void release() {
        auto &driver = CUDADriver::get_instance();
        if (active)
          driver.stream_end_capture.call(stream, &graph);
        if (stream)
          driver.stream_synchronize.call_with_warning(stream);
        if (executable)
          driver.graph_exec_destroy.call_with_warning(executable);
        if (graph)
          driver.graph_destroy.call_with_warning(graph);
        if (stream)
          driver.stream_destroy.call_with_warning(stream);
        stream = nullptr;
        graph = nullptr;
        executable = nullptr;
        active = false;
      }
    } capture;
    driver.stream_create(&capture.stream, CU_STREAM_NON_BLOCKING);
    driver.stream_begin_capture(capture.stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    capture.active = true;
    driver.memcpy_device_to_device_async(values, replacement, sizeof(initial),
                                         capture.stream);
    plan.update_values(static_cast<CUstream>(capture.stream));
    plan.record(static_cast<CUstream>(capture.stream));
    driver.stream_end_capture(capture.stream, &capture.graph);
    capture.active = false;
    driver.graph_instantiate_with_flags(&capture.executable, capture.graph, 0);
    for (float scale : {2.f, .5f, 1.f}) {
      auto changed = initial;
      for (auto &value : changed)
        value *= scale;
      driver.memcpy_host_to_device(replacement, changed.data(),
                                   sizeof(changed));
      driver.graph_launch(capture.executable, capture.stream);
      driver.stream_synchronize(capture.stream);
      driver.memcpy_device_to_host(output.data(), y,
                                   output.size() * sizeof(float));
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < rhs; ++j)
          EXPECT_NEAR(output[i * rhs + j], float((i + 1) * (j + 1)) / scale,
                      1e-6f);
    }
    capture.release();
    auto changed = initial;
    for (auto &value : changed)
      value *= 2;
    driver.memcpy_host_to_device(values, changed.data(), sizeof(changed));
    // Explicit rebinding must not inspect or reuse the retired capture stream.
    plan.update_values(nullptr);
    plan.record(nullptr);
    driver.stream_synchronize(nullptr);
    driver.memcpy_device_to_host(output.data(), y, output.size() * sizeof(float));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < rhs; ++j)
        EXPECT_NEAR(output[i * rhs + j], float((i + 1) * (j + 1)) / 2, 1e-6f);
  }
}

TEST(CUDADriverCompatibility, MapsAndGatesMUSADriverSurface) {
  using cuda::detail::driver_symbol_enabled;
  using cuda::detail::driver_symbol_name;
  using cuda::detail::driver_version_supported;
  using cuda::detail::driver_warp_size_supported;

  EXPECT_EQ(driver_symbol_name(CUDADriverProvider::nvidia_cuda, "cuInit"),
            "cuInit");
  EXPECT_EQ(driver_symbol_name(CUDADriverProvider::musa, "cuInit"),
            "muInit");
  EXPECT_EQ(driver_symbol_name(CUDADriverProvider::musa, "cuMemAlloc_v2"),
            "muMemAlloc_v2");

  EXPECT_TRUE(
      driver_symbol_enabled(CUDADriverProvider::musa, "cuLaunchKernel"));
  EXPECT_FALSE(driver_symbol_enabled(CUDADriverProvider::musa,
                                     "cuGraphInstantiateWithFlags"));
  EXPECT_FALSE(driver_symbol_enabled(CUDADriverProvider::musa,
                                     "cuImportExternalMemory"));

  EXPECT_FALSE(
      driver_version_supported(CUDADriverProvider::nvidia_cuda, 9990));
  EXPECT_TRUE(
      driver_version_supported(CUDADriverProvider::nvidia_cuda, 10000));
  EXPECT_TRUE(driver_version_supported(CUDADriverProvider::musa, 5020));

  EXPECT_TRUE(driver_warp_size_supported(CUDADriverProvider::musa, 32));
  EXPECT_FALSE(driver_warp_size_supported(CUDADriverProvider::musa, 128));
}

TEST(CUDAContext, MemoryPoolSupportRequiresDriverAndDeviceCapability) {
  using cuda::detail::MemoryAllocationRoute;
  using cuda::detail::memory_allocation_route;
  using cuda::detail::supports_memory_pool;

  EXPECT_EQ(memory_allocation_route(supports_memory_pool(11, 1, 1)),
            MemoryAllocationRoute::kSynchronous);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(11, 2, 0)),
            MemoryAllocationRoute::kSynchronous);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(11, 2, 1)),
            MemoryAllocationRoute::kAsyncMemoryPool);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(12, 0, 0)),
            MemoryAllocationRoute::kSynchronous);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(12, 0, 1)),
            MemoryAllocationRoute::kAsyncMemoryPool);
}

TEST(CUDACapability, ResolvesNativeAndCompilerFallbackTargets) {
  using cuda::detail::resolve_compute_capability_target;

  const auto ampere = resolve_compute_capability_target(86);
  EXPECT_EQ(ampere.codegen_compute_capability, 86);
  EXPECT_EQ(ampere.ptx_version, 71);
  EXPECT_FALSE(ampere.uses_fallback);

  const auto blackwell = resolve_compute_capability_target(120);
  EXPECT_EQ(blackwell.codegen_compute_capability, 120);
  EXPECT_EQ(blackwell.ptx_version, 87);
  EXPECT_FALSE(blackwell.uses_fallback);

  const auto future = resolve_compute_capability_target(121);
  EXPECT_EQ(future.codegen_compute_capability, 120);
  EXPECT_EQ(future.ptx_version, 87);
  EXPECT_TRUE(future.uses_fallback);
}

TEST(CUDAContext, SeparatesDeviceCapabilityFromCodegenTarget) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  auto &context = CUDAContext::get_instance();
  const auto target = cuda::detail::resolve_compute_capability_target(
      context.get_compute_capability());
  EXPECT_EQ(context.get_codegen_compute_capability(),
            target.codegen_compute_capability);
  EXPECT_EQ(context.get_mcpu(),
            "sm_" + std::to_string(target.codegen_compute_capability));
  EXPECT_EQ(context.get_mattrs(),
            "+ptx" + std::to_string(target.ptx_version));
}

TEST(CUDADiagnostics, SamplesLockContentionWithoutChangingLockOwnership) {
  CUDASampledLockTelemetry telemetry;
  std::mutex mutex;
  for (uint32_t i = 0; i < CUDASampledLockTelemetry::kSamplingPeriod; ++i) {
    auto lock = telemetry.acquire(mutex);
  }
  EXPECT_EQ(telemetry.snapshot().sampled_acquisitions, 1u);
  EXPECT_EQ(telemetry.snapshot().contended_acquisitions, 0u);

  SampledLockTelemetry<ControllableTelemetryMutex> contended_telemetry;
  ControllableTelemetryMutex controllable_mutex;
  for (uint32_t i = 1;
       i < SampledLockTelemetry<ControllableTelemetryMutex>::kSamplingPeriod;
       ++i) {
    auto lock = contended_telemetry.acquire(controllable_mutex);
  }

  controllable_mutex.block();
  std::thread releaser([&] {
    controllable_mutex.wait_for_try_lock();
    controllable_mutex.unlock();
  });
  {
    auto lock = contended_telemetry.acquire(controllable_mutex);
  }
  releaser.join();

  EXPECT_EQ(contended_telemetry.snapshot().sampled_acquisitions, 1u);
  EXPECT_EQ(contended_telemetry.snapshot().contended_acquisitions, 1u);
}

TEST(CUDADevice, MapLifecycleRejectsInvalidTransitions) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  Device::AllocParams params;
  params.size = sizeof(uint32_t);
  DeviceAllocation allocation;
  ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);
  DeviceAllocationGuard allocation_guard(allocation);

  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  ASSERT_NE(mapped, nullptr);
  *static_cast<uint32_t *>(mapped) = 42;
  EXPECT_EQ(device.map(allocation, &mapped), RhiResult::invalid_usage);
  device.unmap(allocation);

  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  EXPECT_EQ(*static_cast<uint32_t *>(mapped), 42u);
  device.unmap(allocation);

  DeviceAllocation invalid = allocation;
  ++invalid.alloc_id;
  EXPECT_EQ(device.map(invalid, &mapped), RhiResult::invalid_usage);

  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  device.unmap(allocation);
}

TEST(CUDADevice, RejectsStaleWrongDeviceAndOutOfRangeAllocations) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  cuda::CudaDevice other_device;
  Device::AllocParams params;
  params.size = 32;
  DeviceAllocation allocation;
  ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);

  std::array<uint8_t, 8> input{};
  input.fill(42);
  const void *input_ptr = input.data();
  size_t input_size = input.size();
  DevicePtr ptr = allocation.get_ptr(8);
  ASSERT_EQ(device.upload_data(&ptr, &input_ptr, &input_size),
            RhiResult::success);
  DevicePtr out_of_range = allocation.get_ptr(31);
  EXPECT_EQ(device.upload_data(&out_of_range, &input_ptr, &input_size),
            RhiResult::invalid_usage);

  void *mapped = nullptr;
  DeviceAllocation wrong_device = allocation;
  wrong_device.device = &other_device;
  EXPECT_EQ(other_device.map(wrong_device, &mapped), RhiResult::invalid_usage);

  device.dealloc_memory(allocation);
  EXPECT_EQ(device.map(allocation, &mapped), RhiResult::invalid_usage);
  EXPECT_EQ(device.upload_data(&ptr, &input_ptr, &input_size),
            RhiResult::invalid_usage);

  DeviceAllocation replacement;
  ASSERT_EQ(device.allocate_memory(params, &replacement), RhiResult::success);
  EXPECT_NE(replacement.alloc_id, allocation.alloc_id);
  device.dealloc_memory(replacement);
}

TEST(CUDACapability, AsyncTileAdmissionRequiresHardwareAndQualifiedWorkload) {
  using cuda::detail::cuda_async_tile_copy_admitted;
  using cuda::detail::kCudaAsyncTileMinBlsBytes;

  EXPECT_FALSE(cuda_async_tile_copy_admitted(
      75, 70, kCudaAsyncTileMinBlsBytes, 4, true, true));
  EXPECT_FALSE(cuda_async_tile_copy_admitted(
      80, 63, kCudaAsyncTileMinBlsBytes, 4, true, true));
  EXPECT_FALSE(cuda_async_tile_copy_admitted(
      80, 70, kCudaAsyncTileMinBlsBytes - 1, 4, true, true));
  EXPECT_FALSE(cuda_async_tile_copy_admitted(
      80, 70, kCudaAsyncTileMinBlsBytes, 2, true, true));
  EXPECT_FALSE(cuda_async_tile_copy_admitted(
      80, 70, kCudaAsyncTileMinBlsBytes, 4, false, true));
  EXPECT_FALSE(cuda_async_tile_copy_admitted(
      80, 70, kCudaAsyncTileMinBlsBytes, 4, true, false));
  EXPECT_TRUE(cuda_async_tile_copy_admitted(
      80, 70, kCudaAsyncTileMinBlsBytes, 4, true, true));
  EXPECT_TRUE(cuda_async_tile_copy_admitted(
      120, 87, kCudaAsyncTileMinBlsBytes, 8, true, true));
}

TEST(CUDADevice, GraphAllocationLeasePinsARetiredGeneration) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  Device::AllocParams params;
  params.size = 4096;
  DeviceAllocation allocation;
  ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);
  void *allocation_address = device.get_memory_addr(allocation);
  ASSERT_NE(allocation_address, nullptr);

  auto lease = device.acquire_allocation_lease(allocation);
  ASSERT_NE(lease, nullptr);
  device.dealloc_memory(allocation);
  EXPECT_EQ(device.get_memory_addr(allocation), nullptr);

  // The registry handle is retired immediately, but its record and memory
  // remain pinned until the graph lease is gone. A replacement allocation
  // therefore cannot alias the address still captured by the graph.
  DeviceAllocation replacement;
  ASSERT_EQ(device.allocate_memory(params, &replacement), RhiResult::success);
  EXPECT_NE(replacement.alloc_id, allocation.alloc_id);
  EXPECT_NE(device.get_memory_addr(replacement), allocation_address);

  lease.reset();
  device.dealloc_memory(replacement);
  EXPECT_EQ(device.acquire_allocation_lease(allocation), nullptr);
}

TEST(CUDAContext, ToolkitCompatibilityUsesCudaMajorVersion) {
  using cuda::detail::supports_cuda_toolkit_major;

  EXPECT_TRUE(supports_cuda_toolkit_major(11, 11));
  EXPECT_TRUE(supports_cuda_toolkit_major(12, 11));
  EXPECT_TRUE(supports_cuda_toolkit_major(12, 12));
  EXPECT_TRUE(supports_cuda_toolkit_major(13, 12));
  EXPECT_FALSE(supports_cuda_toolkit_major(11, 12));
  EXPECT_FALSE(supports_cuda_toolkit_major(12, 13));
}

TEST(CUDAGraphMemoryPool, DeviceOwnedPoolsPreserveAllocationLeases) {
  if (!CUDADriver::get_instance_without_context().detected() ||
      !cuda::GraphMemoryPool::available()) {
    GTEST_SKIP();
  }
  // No Program, LLVM runtime, ndarray wrapper or Python owner is needed.
  cuda::CudaDevice device;
  cuda::GraphMemoryPool first(device, 0), second(device, 4096);
  const auto a = first.allocate_memory(17 * sizeof(std::uint32_t));
  const auto b = second.allocate_memory(29 * sizeof(std::uint32_t));
  auto &driver = CUDADriver::get_instance();
  driver.stream_synchronize(nullptr);
  EXPECT_EQ(first.snapshot().at("used_current_bytes"), 17 * sizeof(std::uint32_t));
  EXPECT_EQ(second.snapshot().at("used_current_bytes"), 29 * sizeof(std::uint32_t));
  EXPECT_EQ(second.snapshot().at("release_threshold_bytes"), 4096);

  auto lease = device.acquire_allocation_lease(a);
  ASSERT_NE(lease, nullptr);
  auto *address = device.get_memory_addr(a);
  first.close();
  first.close();
  EXPECT_EQ(first.snapshot().at("closed"), 1);
  EXPECT_ANY_THROW(first.allocate_memory(4));
  device.dealloc_memory(a);
  EXPECT_EQ(device.get_memory_addr(a), nullptr);
  // A prepared command may still own the allocation after the public handle
  // and factory retire. Its captured address must remain usable.
  driver.memsetd32(address, 37, 17);
  std::array<std::uint32_t, 17> values{};
  driver.memcpy_device_to_host(values.data(), address, sizeof(values));
  EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                          [](auto value) { return value == 37; }));
  EXPECT_EQ(second.snapshot().at("used_current_bytes"), 29 * sizeof(std::uint32_t));
  lease.reset();
  device.dealloc_memory(b);
  driver.stream_synchronize(nullptr);
  device.collect_graph_memory_pools();
  EXPECT_EQ(second.snapshot().at("used_current_bytes"), 0);
  second.close();
}

TEST(CUDAGraphMemoryPool, DeviceTeardownRetiresStandaloneFactory) {
  if (!CUDADriver::get_instance_without_context().detected() ||
      !cuda::GraphMemoryPool::available()) {
    GTEST_SKIP();
  }
  auto device = std::make_unique<cuda::CudaDevice>();
  cuda::GraphMemoryPool pool(*device, 0);
  const auto allocation = pool.allocate_memory(68);
  ASSERT_NE(device->get_memory_addr(allocation), nullptr);
  device.reset();
  EXPECT_EQ(pool.snapshot().size(), 1);
  EXPECT_EQ(pool.snapshot().at("closed"), 1);
  EXPECT_ANY_THROW(pool.allocate_memory(4));
  pool.close();
}

TEST(CUDADevice, RejectsDeallocationWhileMapped) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  Device::AllocParams params;
  params.size = 32;
  DeviceAllocation allocation;
  ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);

  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  ASSERT_NE(mapped, nullptr);
  std::memset(mapped, 73, params.size);

  device.dealloc_memory(allocation);
  EXPECT_EQ(device.map(allocation, &mapped), RhiResult::invalid_usage);
  device.unmap(allocation);

  std::array<uint8_t, 32> output{};
  void *output_ptr = output.data();
  size_t output_size = output.size();
  DevicePtr ptr = allocation.get_ptr();
  ASSERT_EQ(device.readback_data(&ptr, &output_ptr, &output_size),
            RhiResult::success);
  EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                          [](uint8_t value) { return value == 73; }));

  device.dealloc_memory(allocation);
  EXPECT_EQ(device.map(allocation, &mapped), RhiResult::invalid_usage);
}

TEST(CUDADevice, ConcurrentMapAndDeallocationNeverExposeFreedMemory) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  Device::AllocParams params;
  params.size = 64;

  for (int iteration = 0; iteration < 16; ++iteration) {
    DeviceAllocation allocation;
    ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);
    std::atomic<bool> start{false};

    std::thread mapper([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      void *mapped = nullptr;
      if (device.map(allocation, &mapped) == RhiResult::success) {
        std::memset(mapped, iteration, params.size);
        device.unmap(allocation);
      }
    });
    std::thread releaser([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      device.dealloc_memory(allocation);
    });

    start.store(true, std::memory_order_release);
    mapper.join();
    releaser.join();

    void *probe = nullptr;
    if (device.map(allocation, &probe) == RhiResult::success) {
      device.unmap(allocation);
      device.dealloc_memory(allocation);
    }
    EXPECT_EQ(device.map(allocation, &probe), RhiResult::invalid_usage);
  }
}

TEST(CUDADevice, ConcurrentAllocateCopyReadbackAndRelease) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  constexpr int kThreads = 4;
  constexpr int kIterations = 32;
  constexpr size_t kBytes = 128;
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      Device::AllocParams params;
      params.size = kBytes;
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        DeviceAllocation src;
        DeviceAllocation dst;
        if (device.allocate_memory(params, &src) != RhiResult::success ||
            device.allocate_memory(params, &dst) != RhiResult::success) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }

        std::array<uint8_t, 128> input{};
        input.fill(static_cast<uint8_t>(thread_index + iteration));
        const void *input_ptr = input.data();
        size_t input_size = input.size();
        DevicePtr src_ptr = src.get_ptr();
        DevicePtr dst_ptr = dst.get_ptr();
        if (device.upload_data(&src_ptr, &input_ptr, &input_size) !=
            RhiResult::success) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }

        device.memcpy_internal(dst_ptr, src_ptr, kBytes);
        std::array<uint8_t, 128> output{};
        void *output_ptr = output.data();
        size_t output_size = output.size();
        if (device.readback_data(&dst_ptr, &output_ptr, &output_size) !=
                RhiResult::success ||
            output != input) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        device.dealloc_memory(src);
        device.dealloc_memory(dst);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
}

TEST(CUDADevice, RuntimeAsyncAllocationUsesItsAllocationStream) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  auto &driver = CUDADriver::get_instance();
  const auto before = driver.get_telemetry_snapshot();
  cuda::CudaDevice device;
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 256;
  params.use_memory_pool = true;
  DeviceAllocation allocation = device.allocate_memory_runtime(params);
  ASSERT_EQ(allocation.device, &device);
  device.dealloc_memory(allocation);
  CUDADriver::get_instance().stream_synchronize(nullptr);
  const auto after = driver.get_telemetry_snapshot();
  if (CUDAContext::get_instance().supports_mem_pool()) {
    EXPECT_GE(after.async_allocation_calls, before.async_allocation_calls + 1);
    EXPECT_GE(after.async_free_calls, before.async_free_calls + 1);
  } else {
    EXPECT_GE(after.sync_allocation_fallback_calls,
              before.sync_allocation_fallback_calls + 1);
    EXPECT_GE(after.sync_free_fallback_calls,
              before.sync_free_fallback_calls + 1);
  }
}

TEST(CUDADevice, RuntimeAddressQueryTreatsRetiredAllocationAsNull) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 256;
  params.use_memory_pool = true;
  DeviceAllocation allocation = device.allocate_memory_runtime(params);
  ASSERT_NE(device.get_memory_addr(allocation), nullptr);
  device.dealloc_memory(allocation);
  EXPECT_EQ(device.get_memory_addr(allocation), nullptr);
  CUDADriver::get_instance().stream_synchronize(nullptr);
}

}  // namespace taichi::lang
