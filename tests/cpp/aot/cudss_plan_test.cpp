#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <array>
#include <cstdlib>
#include <utility>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/cudss_plan.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
namespace {
class CudssCaptureOperation final : public NativeOperation {
 public:
  CudssCaptureOperation(Device &device,
                        std::shared_ptr<CudaCudssPlan> plan,
                        int phase,
                        void *values,
                        std::vector<std::pair<void *, void *>> bindings)
      : device_(device),
        plan_(std::move(plan)),
        phase_(phase),
        values_(values),
        bindings_(std::move(bindings)) {
    plan_->claim_graph_phase(phase);
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void record_cuda(void *stream) override {
    ++recordings;
    plan_->record_capture(phase_, values_, bindings_, stream, resources_);
  }
  std::shared_ptr<CudaGraphCaptureResources> take_cuda_capture_resources()
      override {
    return std::exchange(resources_, {});
  }
  int recordings{0};

 private:
  Device &device_;
  std::shared_ptr<CudaCudssPlan> plan_;
  int phase_;
  void *values_;
  std::vector<std::pair<void *, void *>> bindings_;
  std::shared_ptr<CudaGraphCaptureResources> resources_;
};

TEST(CudssSharedPlan, CapturesIndependentBindingsWithoutProgramOrCompiler) {
  const auto *library = std::getenv("TI_CUDSS_TEST_LIBRARY");
  const auto *adapter = std::getenv("TI_CUDSS_TEST_ADAPTER");
  if (!library || !adapter)
    GTEST_SKIP() << "Set cuDSS runtime and adapter paths for actual provider "
                    "qualification";
  ASSERT_TRUE(CUDADriver::get_instance_without_context().detected());
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 1);
  std::vector<DeviceAllocationUnique> allocations;
  auto allocate = [&](std::size_t bytes) {
    auto allocation = device.allocate_memory_unique({bytes});
    if (allocation.second != RhiResult::success)
      throw std::runtime_error("Test CUDA allocation failed");
    auto *pointer = device.get_memory_addr(*allocation.first);
    allocations.push_back(std::move(allocation.first));
    return pointer;
  };
  std::array<int, 5> rows{0, 2, 5, 8, 10};
  std::array<int, 10> columns{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  std::array<float, 10> values{4, -1, -1, 4, -1, -1, 4, -1, -1, 3};
  std::array<float, 4> rhs{2, 4, 6, 9}, rhs2{4, 8, 12, 18};
  auto *r = allocate(sizeof(rows));
  auto *c = allocate(sizeof(columns));
  auto *v = allocate(sizeof(values));
  auto *v2 = allocate(sizeof(values));
  auto *b = allocate(sizeof(rhs));
  auto *b2 = allocate(sizeof(rhs2));
  auto *x = allocate(sizeof(rhs));
  auto *x2 = allocate(sizeof(rhs));
  auto *x3 = allocate(sizeof(rhs));
  driver.memcpy_host_to_device(r, rows.data(), sizeof(rows));
  driver.memcpy_host_to_device(c, columns.data(), sizeof(columns));
  driver.memcpy_host_to_device(b, rhs.data(), sizeof(rhs));
  driver.memcpy_host_to_device(b2, rhs2.data(), sizeof(rhs2));
  const CudaCudssMatrixView matrix{4, 4, 10, r, c, v};
  auto check = [&](void *output, float scale) {
    std::array<float, 4> result{};
    driver.memcpy_device_to_host(result.data(), output, sizeof(result));
    for (int i = 0; i < 4; ++i)
      EXPECT_NEAR(result[i], (i + 1) * scale, 2e-5f);
  };
  for (int phase : {0, 4, 8}) {
    SCOPED_TRACE(phase);
    driver.memcpy_host_to_device(v, values.data(), sizeof(values));
    driver.memsetd32(x, 0, 4);
    driver.memsetd32(x2, 0, 4);
    driver.memsetd32(x3, 0, 4);
    // This owner warms and seals private data; no Program, fields or compiler.
    auto plan = make_cuda_cudss_plan(matrix, 3, 0, adapter, library, fault,
                                     {0, 0}, true);
    std::weak_ptr<CudaCudssPlan> lease = plan;
    const auto before = plan->configuration();
    ASSERT_GT(before.at("allocator_live_requested_bytes"), 0);
    auto first = std::make_shared<CudssCaptureOperation>(
        device, plan, phase, phase ? v : nullptr,
        std::vector<std::pair<void *, void *>>{{b, x}, {b2, x2}});
    auto second = std::make_shared<CudssCaptureOperation>(
        device, plan, phase, phase ? v2 : nullptr,
        std::vector<std::pair<void *, void *>>{{b, x3}});
    auto graph1 = std::make_unique<FixedExecution>(
        Arch::cuda, device, nullptr,
        std::vector<FixedExecution::Operation>{first});
    auto graph2 = std::make_unique<FixedExecution>(
        Arch::cuda, device, nullptr,
        std::vector<FixedExecution::Operation>{second});
    EXPECT_EQ(first->recordings, 1);
    EXPECT_EQ(second->recordings, 1);
    EXPECT_GT(graph1->workspace_bytes(), 0);
    EXPECT_GT(graph2->workspace_bytes(), 0);
    check(x, 0);
    check(x2, 0);
    check(x3, 0);
    EXPECT_EQ(plan->configuration().at("sealed_allocation_rejections"), 0);
    EXPECT_THROW(plan->claim_graph_phase(phase ? 0 : 4), std::string);
    auto doubled = values;
    auto quadrupled = values;
    for (auto &value : doubled)
      value *= 2;
    for (auto &value : quadrupled)
      value *= 4;
    if (phase) {
      driver.memcpy_host_to_device(v, doubled.data(), sizeof(doubled));
      driver.memcpy_host_to_device(v2, quadrupled.data(), sizeof(quadrupled));
    } else {
      // The immutable factors must outlive and ignore the original seed values.
      driver.memsetd32(v, 0, 10);
    }
    plan.reset();
    for (int i = 0; i < 3; ++i) {
      graph1->run();
      check(x, phase ? .5f : 1);
      check(x2, phase ? 1 : 2);
      graph2->run();
      check(x3, phase ? .25f : 1);
    }
    EXPECT_EQ(first->recordings, 1);
    EXPECT_EQ(second->recordings, 1);
    first.reset();
    second.reset();
    ASSERT_FALSE(lease.expired());
    graph1->run();
    graph1.reset();
    check(x, phase ? .5f : 1);
    graph2.reset();
    EXPECT_TRUE(lease.expired());
  }
}
}  // namespace
}  // namespace taichi::lang
#endif
