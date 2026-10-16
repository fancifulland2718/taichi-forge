#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "taichi/external_runtime/forge_cublaslt_provider.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_provider.h"

namespace taichi::lang {
namespace {
struct MatmulOwner {
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCublasLtExecutionApi api{};
  TiForgeCublasLtPlan handle{nullptr};
  ~MatmulOwner() {
    if (handle)
      EXPECT_EQ(api.destroy_plan(handle), TI_FORGE_RUNTIME_PROVIDER_SUCCESS);
  }
};

class MatmulCapture final : public NativeOperation {
 public:
  MatmulCapture(Device &device,
                std::shared_ptr<MatmulOwner> plan,
                TiForgeCublasLtExecDesc desc)
      : device_(device), plan_(std::move(plan)), desc_(desc) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return desc_.workspace_bytes;
  }
  void record_cuda(void *stream) override {
    ++recordings;
    desc_.cuda_stream = reinterpret_cast<uint64_t>(stream);
    plan_->runtime->check(plan_->api.execute(plan_->handle, &desc_), "matmul");
  }
  int recordings{0};

 private:
  Device &device_;
  std::shared_ptr<MatmulOwner> plan_;
  TiForgeCublasLtExecDesc desc_;
};

void qualify_matmul(uint32_t layout,
                    uint32_t epilogue,
                    uint32_t batch,
                    bool transpose_a,
                    bool transpose_b) {
  const auto *library = std::getenv("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH");
  const auto *adapter = std::getenv("TI_CUBLASLT_TEST_ADAPTER");
  if (!library || !adapter)
    GTEST_SKIP() << "Set actual cuBLASLt vendor and common adapter paths";
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  std::vector<DeviceAllocationUnique> allocations;
  auto allocate = [&](size_t bytes) -> uint64_t {
    if (!bytes)
      return 0;
    auto memory = device.allocate_memory_unique({bytes});
    if (memory.second != RhiResult::success)
      throw std::runtime_error("Matmul test allocation failed");
    auto address =
        reinterpret_cast<uint64_t>(device.get_memory_addr(*memory.first));
    allocations.push_back(std::move(memory.first));
    return address;
  };
  constexpr int m = 65, n = 49, k = 97;
  std::vector<float> a(batch * m * k), b(batch * k * n),
      expected(batch * m * n);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = float(int(i % 17) - 8) / 16;
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = float(int(i % 13) - 6) / 16;
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = float(int(i % 7) - 3) / 8;
  auto input_a = allocate(a.size() * sizeof(float));
  auto input_b = allocate(b.size() * sizeof(float));
  auto output = allocate(expected.size() * sizeof(float));
  const auto pointer = [](uint64_t address) {
    return reinterpret_cast<void *>(address);
  };
  driver.memcpy_host_to_device(pointer(input_a), a.data(),
                               a.size() * sizeof(float));
  driver.memcpy_host_to_device(pointer(input_b), b.data(),
                               b.size() * sizeof(float));
  driver.memcpy_host_to_device(pointer(output), expected.data(),
                               expected.size() * sizeof(float));
  auto initial = std::make_shared<MatmulOwner>();
  {
    NativeProviderSession session({{"cublaslt", {adapter, library}},
                                   {"unused", {"missing-adapter.dll", ""}}});
    initial->runtime =
        session.runtime({"cublaslt", TI_FORGE_CUBLASLT_PROVIDER_QUERY_SYMBOL,
                         "documented f32 matmul configuration C API"});
  }
  initial->api = initial->runtime->execution_api<TiForgeCublasLtExecutionApi>();
  EXPECT_THROW(initial->runtime->execution_api<TiForgeCublasLtExecutionApi>(0),
               NativeProviderError);
  TiForgeCublasLtPlanDesc desc{sizeof(desc),
                               layout,
                               m,
                               n,
                               k,
                               batch,
                               uint32_t(transpose_a),
                               uint32_t(transpose_b),
                               epilogue,
                               0.75f,
                               0.25f,
                               4 * 1024 * 1024};
  TiForgeCublasLtPlanInfo info{sizeof(info)};
  auto invalid = desc;
  invalid.m = -1;
  EXPECT_EQ(initial->api.create_plan(initial->runtime->handle(), &invalid,
                                     nullptr, &initial->handle, &info),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(initial->handle, nullptr);
  invalid = desc;
  invalid.k = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(initial->api.create_plan(initial->runtime->handle(), &invalid,
                                     nullptr, &initial->handle, &info),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(initial->handle, nullptr);
  initial->runtime->check(
      initial->api.create_plan(initial->runtime->handle(), &desc, nullptr,
                               &initial->handle, &info),
      "select matmul");
  EXPECT_EQ(info.heuristic_calls, 1u);
  EXPECT_EQ(info.restoration_calls, 0u);

  auto restored = std::make_shared<MatmulOwner>();
  restored->runtime = initial->runtime;
  restored->api = initial->api;
  TiForgeCublasLtPlanInfo restored_info{sizeof(restored_info)};
  // An invalid restore publishes no handle and does not poison later creation.
  auto changed = info.algorithm;
  changed.workspace_bytes ^= 1u;
  EXPECT_EQ(
      restored->api.create_plan(restored->runtime->handle(), &desc, &changed,
                                &restored->handle, &restored_info),
      TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(restored->handle, nullptr);
  changed = info.algorithm;
  changed.present_mask &= ~1u;
  EXPECT_EQ(
      restored->api.create_plan(restored->runtime->handle(), &desc, &changed,
                                &restored->handle, &restored_info),
      TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(restored->handle, nullptr);
  restored->runtime->check(
      restored->api.create_plan(restored->runtime->handle(), &desc,
                                &info.algorithm, &restored->handle,
                                &restored_info),
      "restore matmul");
  EXPECT_EQ(restored_info.heuristic_calls, 0u);
  EXPECT_EQ(restored_info.restoration_calls, 1u);
  EXPECT_EQ(restored_info.algorithm.workspace_bytes,
            info.algorithm.workspace_bytes);
  EXPECT_EQ(restored_info.algorithm.present_mask, info.algorithm.present_mask);
  EXPECT_TRUE(std::equal(std::begin(info.algorithm.values),
                         std::end(info.algorithm.values),
                         std::begin(restored_info.algorithm.values)));
  initial.reset();
  const auto workspace = allocate(restored_info.algorithm.workspace_bytes);
  TiForgeCublasLtExecDesc exec{sizeof(exec),
                               0,
                               input_a,
                               input_b,
                               output,
                               workspace,
                               restored_info.algorithm.workspace_bytes,
                               0};
  auto bad = exec;
  bad.output = input_a;
  EXPECT_EQ(restored->api.execute(restored->handle, &bad),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  bad = exec;
  ++bad.a;
  EXPECT_EQ(restored->api.execute(restored->handle, &bad),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  if (exec.workspace_bytes) {
    bad = exec;
    --bad.workspace_bytes;
    EXPECT_EQ(restored->api.execute(restored->handle, &bad),
              TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  }
  auto verify = [&](bool update) {
    if (update) {
      for (uint32_t q = 0; q < batch; ++q)
        for (int row = 0; row < m; ++row)
          for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int inner = 0; inner < k; ++inner) {
              const auto ai =
                  q * m * k + (transpose_a ? inner * m + row : row * k + inner);
              const auto bi =
                  q * k * n + (transpose_b ? col * k + inner : inner * n + col);
              sum += double(a[ai]) * b[bi];
            }
            auto &value = expected[q * m * n + row * n + col];
            value = desc.alpha * sum + desc.beta * value;
            if (epilogue)
              value = std::max(value, 0.0f);
          }
    }
    driver.context_synchronize();
    std::vector<float> result(expected.size());
    driver.memcpy_device_to_host(result.data(), pointer(output),
                                 result.size() * sizeof(float));
    for (size_t i = 0; i < result.size(); ++i)
      ASSERT_NEAR(result[i], expected[i],
                  2e-5f * std::max(1.0f, std::abs(expected[i])))
          << i;
  };
  verify(false);  // Cold selection/restoration must not execute beta feedback.
  restored->runtime->check(restored->api.execute(restored->handle, &exec),
                           "direct matmul");
  verify(true);
  auto operation = std::make_shared<MatmulCapture>(device, restored, exec);
  std::weak_ptr<NativeProviderRuntime> lease = restored->runtime;
  auto graph = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr,
      std::vector<FixedExecution::Operation>{operation});
  verify(false);  // Capturing must not execute beta feedback either.
  restored.reset();
  graph->run();
  verify(true);
  for (auto &value : a)
    value *= -0.5f;
  driver.memcpy_host_to_device(pointer(input_a), a.data(),
                               a.size() * sizeof(float));
  graph->run();
  verify(true);
  EXPECT_EQ(operation->recordings, 1);
  operation.reset();
  ASSERT_FALSE(lease.expired());
  graph->run();
  graph.reset();  // Retire outstanding work before releasing plan and library.
  EXPECT_TRUE(lease.expired());
}

TEST(CublasLtSharedPlan, RowMajorDirectAndCapturedFeedback) {
  qualify_matmul(0, 0, 1, false, false);
}
TEST(CublasLtSharedPlan, BatchedTransposedFeedback) {
  qualify_matmul(0, 0, 3, true, true);
}
TEST(CublasLtSharedPlan, ColumnMajorReluFrozenConfiguration) {
  qualify_matmul(1, 1, 2, false, true);
}
}  // namespace
}  // namespace taichi::lang
#endif
