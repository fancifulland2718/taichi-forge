#include "gtest/gtest.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation_registry.h"

#ifdef TI_WITH_CUDA
#include <array>
#include "taichi/rhi/cuda/cuda_device.h"

namespace taichi::lang {
namespace {
struct OrderedDescription {
  static constexpr const char *kind = "test_ordered_copy";
  static constexpr std::uint32_t version = 1;
};

class Transaction final : public CudaProviderCompletionResource {
 public:
  void reserve() {
    if (active)
      throw std::runtime_error("transaction is in flight");
    active = ++generation;
  }
  std::uint64_t submission_retirement_token() const override {
    return active;
  }
  void on_submission_retired(std::uint64_t token) noexcept override {
    if (token && token == active) {
      active = 0;
      ++retired;
    }
  }
  std::uint64_t generation{0}, active{0}, retired{0};
};

class OrderedCopy final : public NativeOperation {
 public:
  OrderedCopy(Device &device,
              std::shared_ptr<Transaction> transaction,
              void *source,
              void *destination,
              bool fail = false)
      : device_(device),
        transaction_(std::move(transaction)),
        source_(source),
        destination_(destination),
        fail_(fail) {
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
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return transaction_;
  }
  void reserve_cuda_ordered() override {
    transaction_->reserve();
  }
  void run_cuda_ordered(void *stream) override {
    ++calls;
    CUDADriver::get_instance().memcpy_device_to_device_async(
        destination_, source_, 64, stream);
    if (fail_)
      throw std::runtime_error("ordered provider failed after enqueue");
  }
  int calls{0};

 private:
  Device &device_;
  std::shared_ptr<Transaction> transaction_;
  void *source_, *destination_;
  bool fail_;
};

TEST(CudaOrderedSubmission, PreservesOrderSharedRetirementAndPartialFailure) {
  ASSERT_TRUE(CUDADriver::get_instance_without_context().detected());
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  std::array<DeviceAllocationUnique, 3> arrays;
  std::array<void *, 3> pointers{};
  for (int i = 0; i < 3; ++i) {
    auto result = device.allocate_memory_unique({64});
    ASSERT_EQ(result.second, RhiResult::success);
    arrays[i] = std::move(result.first);
    pointers[i] = device.get_memory_addr(*arrays[i]);
    driver.memsetd32(pointers[i], 3, 16);
  }
  const auto fill = [&](int index, std::uint32_t value) {
    return FixedExecution::BufferFill{arrays[index]->get_ptr(), 64, value};
  };
  const auto check = [&](int index, std::uint32_t expected) {
    driver.stream_synchronize(nullptr);
    std::array<std::uint32_t, 16> host{};
    driver.memcpy_device_to_host(host.data(), pointers[index], 64);
    for (auto value : host)
      EXPECT_EQ(value, expected);
  };
  auto transaction = std::make_shared<Transaction>();
  auto copy = std::make_shared<OrderedCopy>(device, transaction, pointers[0],
                                            pointers[1]);
  auto first = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr,
      std::vector<FixedExecution::Operation>{
          fill(0, 7), copy,
          FixedExecution::BufferCopy{arrays[2]->get_ptr(), arrays[1]->get_ptr(),
                                     64}});
  auto second = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr,
      std::vector<FixedExecution::Operation>{copy});
  EXPECT_EQ(first->execution_kind(), 4);
  EXPECT_EQ(second->execution_kind(), 4);
  EXPECT_EQ(copy->calls, 0);
  check(0, 3);  // Preparation performs no mathematical work.
  first->run();
  EXPECT_NE(transaction->active, 0);
  EXPECT_EQ(transaction->retired, 0);
  check(2, 7);
  // The first execution stays alive. A different execution sharing its owner
  // must observe the completed event and retire that owner's exact token.
  second->run();
  EXPECT_EQ(transaction->retired, 1);
  second.reset();  // Close immediately after submission.
  EXPECT_EQ(transaction->retired, 2);
  EXPECT_EQ(transaction->active, 0);
  first.reset();
  EXPECT_EQ(transaction->retired, 2);

  auto failing = std::make_shared<OrderedCopy>(device, transaction, pointers[0],
                                               pointers[1], true);
  {
    FixedExecution failed(Arch::cuda, device, nullptr,
                          {fill(0, 11), failing, fill(2, 99)});
    EXPECT_THROW(failed.run(), std::runtime_error);
    check(1, 11);
    check(2, 7);  // A failure does not submit the suffix.
    EXPECT_THROW(failed.run(), std::runtime_error);
    EXPECT_EQ(failing->calls, 1);
  }
  EXPECT_EQ(transaction->retired, 3);
  EXPECT_EQ(transaction->active, 0);

  transaction->reserve();
  const auto existing = transaction->active;
  {
    FixedExecution rejected(Arch::cuda, device, nullptr, {copy});
    EXPECT_THROW(rejected.run(), std::runtime_error);
  }
  EXPECT_EQ(transaction->active,
            existing);  // Do not retire someone else's reservation.
  transaction->on_submission_retired(existing);

  auto loop = std::make_shared<FixedExecution::While>();
  loop->predicate = arrays[0]->get_ptr();
  loop->max_iterations = 1;
  loop->body = {copy};
  EXPECT_THROW(FixedExecution(Arch::cuda, device, nullptr, {loop}),
               std::invalid_argument);

  NativeOperationRegistry registry;
  registry.define<OrderedDescription>(
      [](Device &, const OrderedDescription &) {});
  registry.register_factory<OrderedDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [&](Arch, Device &, const OrderedDescription &) { return copy; });
  ASSERT_NE(registry.find(OrderedDescription::kind, 1, Arch::cuda), nullptr);
  EXPECT_EQ(registry.find(OrderedDescription::kind, 1, Arch::cuda)->mode,
            NativeOperationMode::cuda_ordered);
  EXPECT_EQ(registry.prepare(Arch::cuda, device, OrderedDescription{}).get(),
            copy.get());
}
}  // namespace
}  // namespace taichi::lang
#endif
