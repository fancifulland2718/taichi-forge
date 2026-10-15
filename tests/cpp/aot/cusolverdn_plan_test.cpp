#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <array>
#include <cstdlib>
#include <type_traits>
#include "taichi/external_runtime/forge_cusolverdn_provider.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_provider.h"

namespace taichi::lang {
namespace {
struct SolverOwner {
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCusolverDnExecutionApi api{};
  TiForgeCusolverDnPlan handle{nullptr};
  ~SolverOwner() {
    if (handle)
      EXPECT_EQ(api.destroy_plan(handle), TI_FORGE_RUNTIME_PROVIDER_SUCCESS);
  }
};

class SolverCapture final : public NativeOperation {
 public:
  SolverCapture(Device &device,
                std::shared_ptr<SolverOwner> plan,
                void *input,
                void *factor,
                size_t factor_bytes,
                void *rhs,
                void *output,
                size_t rhs_bytes,
                void *workspace,
                size_t workspace_bytes,
                bool refactor)
      : device_(device),
        plan_(std::move(plan)),
        input_(input),
        factor_(factor),
        factor_bytes_(factor_bytes),
        rhs_(rhs),
        output_(output),
        rhs_bytes_(rhs_bytes),
        workspace_(workspace),
        workspace_bytes_(workspace_bytes),
        refactor_(refactor) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return workspace_bytes_;
  }
  void record_cuda(void *stream) override {
    ++recordings;
    auto &driver = CUDADriver::get_instance();
    TiForgeCusolverDnExecDesc exec{sizeof(exec),
                                   0,
                                   reinterpret_cast<uint64_t>(workspace_),
                                   workspace_bytes_,
                                   reinterpret_cast<uint64_t>(output_),
                                   reinterpret_cast<uint64_t>(stream)};
    if (refactor_) {
      driver.memcpy_device_to_device_async(factor_, input_, factor_bytes_,
                                           stream);
      plan_->runtime->check(plan_->api.execute(plan_->handle, &exec), "factor");
    }
    driver.memcpy_device_to_device_async(output_, rhs_, rhs_bytes_, stream);
    exec.phase = 1;
    plan_->runtime->check(plan_->api.execute(plan_->handle, &exec), "solve");
  }
  int recordings{0};

 private:
  Device &device_;
  std::shared_ptr<SolverOwner> plan_;
  void *input_, *factor_;
  size_t factor_bytes_;
  void *rhs_, *output_;
  size_t rhs_bytes_;
  void *workspace_;
  size_t workspace_bytes_;
  bool refactor_;
};

template <typename Scalar>
void qualify_solver() {
  const auto *library = std::getenv("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH");
  const auto *adapter = std::getenv("TI_CUSOLVERDN_TEST_ADAPTER");
  if (!library || !adapter)
    GTEST_SKIP() << "Set actual cuSOLVER vendor and common adapter paths";
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  std::vector<DeviceAllocationUnique> allocations;
  auto allocate = [&](size_t bytes) {
    if (!bytes)
      return static_cast<void *>(nullptr);
    auto memory = device.allocate_memory_unique({bytes});
    if (memory.second != RhiResult::success)
      throw std::runtime_error("Solver test allocation failed");
    auto *pointer = device.get_memory_addr(*memory.first);
    allocations.push_back(std::move(memory.first));
    return pointer;
  };
  constexpr int n = 17, nrhs = 3;
  std::array<Scalar, n * n> a{};
  std::array<Scalar, n * nrhs> rhs{}, expected{};
  for (int i = 0; i < n; ++i) {
    a[i * n + i] = 4;
    if (i)
      a[i * n + i - 1] = -1;
    for (int j = i + 1; j < n; ++j)
      a[i * n + j] = 999;  // ignored upper triangle
  }
  for (int r = 0; r < nrhs; ++r) {
    for (int i = 0; i < n; ++i)
      expected[r * n + i] = Scalar(r + 1) * Scalar(i + 1) / 8;
    for (int i = 0; i < n; ++i)
      rhs[r * n + i] = 4 * expected[r * n + i] -
                       (i ? expected[r * n + i - 1] : 0) -
                       (i + 1 < n ? expected[r * n + i + 1] : 0);
  }
  auto *input = allocate(sizeof(a)), *factor = allocate(sizeof(a));
  auto *b = allocate(sizeof(rhs)), *x = allocate(sizeof(rhs));
  auto *info = allocate(8);
  driver.memcpy_host_to_device(input, a.data(), sizeof(a));
  driver.memcpy_host_to_device(b, rhs.data(), sizeof(rhs));
  driver.memsetd32(info, 0xffffffffu, 2);
  auto plan = std::make_shared<SolverOwner>();
  {
    NativeProviderSession session({{"cusolverdn", {adapter, library}},
                                   {"unused", {"missing-adapter.dll", ""}}});
    plan->runtime = session.runtime({"cusolverdn",
                                     TI_FORGE_CUSOLVERDN_PROVIDER_QUERY_SYMBOL,
                                     "generic Cholesky C API"});
  }
  std::weak_ptr<NativeProviderRuntime> runtime_lease = plan->runtime;
  plan->api = plan->runtime->execution_api<TiForgeCusolverDnExecutionApi>();
  EXPECT_THROW(plan->runtime->execution_api<TiForgeCusolverDnExecutionApi>(0),
               NativeProviderError);
  TiForgeCusolverDnPlanDesc desc{sizeof(desc),
                                 std::is_same_v<Scalar, double> ? 1u : 0u,
                                 n,
                                 nrhs,
                                 reinterpret_cast<uint64_t>(factor),
                                 reinterpret_cast<uint64_t>(info)};
  TiForgeCusolverDnPlanInfo details{sizeof(details)};
  auto invalid = desc;
  invalid.rows = -1;
  EXPECT_EQ(plan->api.create_plan(plan->runtime->handle(), &invalid,
                                  &plan->handle, &details),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(plan->handle, nullptr);
  plan->runtime->check(plan->api.create_plan(plan->runtime->handle(), &desc,
                                             &plan->handle, &details),
                       "create plan");
  auto *workspace = allocate(details.device_workspace_bytes);
  auto operation = [&](bool refactor) {
    return std::make_shared<SolverCapture>(
        device, plan, input, factor, sizeof(a), b, x, sizeof(rhs), workspace,
        details.device_workspace_bytes, refactor);
  };
  auto first = operation(true), second = operation(false);
  auto full = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr,
      std::vector<FixedExecution::Operation>{first});
  auto solve = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr,
      std::vector<FixedExecution::Operation>{second});
  std::array<int, 2> status{};
  driver.memcpy_device_to_host(status.data(), info, sizeof(status));
  EXPECT_EQ(status, (std::array<int, 2>{-1, -1}));
  plan.reset();
  full->run();
  std::array<Scalar, n * n> preserved{};
  driver.memcpy_device_to_host(preserved.data(), input, sizeof(a));
  EXPECT_EQ(preserved, a);
  // Reusing factors must not refactor a now-invalid input matrix.
  driver.memsetd32(input, 0, sizeof(a) / 4);
  full.reset();
  first.reset();
  for (int repeat = 0; repeat < 2; ++repeat) {
    solve->run();
    std::array<Scalar, n * nrhs> result{};
    driver.memcpy_device_to_host(result.data(), x, sizeof(result));
    for (int i = 0; i < n * nrhs; ++i)
      EXPECT_NEAR(result[i], expected[i],
                  (std::is_same_v<Scalar, double> ? 1e-12 : 2e-5));
  }
  driver.memcpy_device_to_host(status.data(), info, sizeof(status));
  EXPECT_EQ(status, (std::array<int, 2>{0, 0}));
  EXPECT_EQ(second->recordings, 1);
  second.reset();
  ASSERT_FALSE(runtime_lease.expired());
  solve->run();
  solve.reset();  // retire queued work before releasing the last plan/library
  EXPECT_TRUE(runtime_lease.expired());
}

TEST(CusolverDnSharedPlan, F32CapturedFactorsOutliveProducerAndInput) {
  qualify_solver<float>();
}
TEST(CusolverDnSharedPlan, F64CapturedFactorsOutliveProducerAndInput) {
  qualify_solver<double>();
}
}  // namespace
}  // namespace taichi::lang
#endif
