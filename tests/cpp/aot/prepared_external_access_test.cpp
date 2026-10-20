#include <gtest/gtest.h>
#include "taichi/rhi/interop/external_access_epoch.h"

namespace taichi::lang {
namespace {
class CountingExternalDomain final : public ExternalSynchronizationDomain {
 public:
  explicit CountingExternalDomain(std::uint64_t id) : id_(id) {
  }
  std::uint64_t identity() const noexcept override {
    return id_;
  }
  bool consumer_ready() const override {
    ++queries;
    if (fail_ready)
      throw std::runtime_error("ready query failed");
    return ready;
  }
  bool has_consumer_ready_query() const noexcept override {
    return true;
  }
  void acquire_for_consumer(const ExternalStreamDomain &stream) override {
    if (fail_acquire)
      throw std::runtime_error("acquire failed");
    EXPECT_FALSE(active);
    active = true;
    selected = stream;
    ++acquires;
  }
  void release_from_consumer(const ExternalStreamDomain &stream) override {
    EXPECT_TRUE(active);
    EXPECT_TRUE(selected.same_stream(stream));
    active = false;
    ++releases;
    if (fail_release)
      throw std::runtime_error("release failed");
  }
  bool fail_acquire{false}, fail_release{false}, active{false};
  bool ready{true}, fail_ready{false};
  mutable unsigned queries{0};
  unsigned acquires{0}, releases{0};
  ExternalStreamDomain selected;

 private:
  std::uint64_t id_;
};

TEST(PreparedExternalAccess, DeduplicatesColdAndReleasesOncePerEpoch) {
  auto first = std::make_shared<CountingExternalDomain>(1);
  auto second = std::make_shared<CountingExternalDomain>(2);
  PreparedExternalAccess plan({first, second, first});
  EXPECT_EQ(plan.size(), 2);
  const auto selected =
      ExternalStreamDomain::cuda(11, 13, reinterpret_cast<void *>(17));
  for (int i = 0; i < 1000; ++i) {
    auto epoch = plan.acquire(selected);
    epoch.release();
    epoch.release();
  }
  EXPECT_EQ(first->acquires, 1000);
  EXPECT_EQ(first->releases, 1000);
  EXPECT_EQ(second->acquires, 1000);
  EXPECT_EQ(second->releases, 1000);
  EXPECT_THROW(PreparedExternalAccess(
                   {first, std::make_shared<CountingExternalDomain>(1)}),
               std::invalid_argument);
  EXPECT_THROW(PreparedExternalAccess({nullptr}), std::invalid_argument);
  EXPECT_THROW(plan.acquire({}), std::invalid_argument);
  EXPECT_EQ(first->acquires, 1000);
}

TEST(PreparedExternalAccess,
     PartialAcquireAndReleaseFailuresPreserveFirstError) {
  auto first = std::make_shared<CountingExternalDomain>(1);
  auto second = std::make_shared<CountingExternalDomain>(2);
  PreparedExternalAccess plan({first, second});
  const auto stream =
      ExternalStreamDomain::cuda(1, 2, reinterpret_cast<void *>(3));
  first->fail_release = true;
  second->fail_acquire = true;
  try {
    auto epoch = plan.acquire(stream);
    FAIL() << "Expected acquisition failure";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(), "acquire failed");
  }
  EXPECT_EQ(first->acquires, 1);
  EXPECT_EQ(first->releases, 1);
  EXPECT_EQ(second->releases, 0);
  second->fail_acquire = false;
  second->fail_release = true;
  {
    auto epoch = plan.acquire(stream);
    EXPECT_THROW(epoch.release(), std::runtime_error);
  }
  EXPECT_EQ(first->releases, 2);
  EXPECT_EQ(second->releases, 1);
  first->fail_release = second->fail_release = false;
  try {
    auto epoch = plan.acquire(stream);
    throw std::runtime_error("body failed");
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(), "body failed");
  }
  EXPECT_EQ(first->releases, 3);
  EXPECT_EQ(second->releases, 2);
}
}  // namespace
}  // namespace taichi::lang

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "taichi/program/program.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/interop/vulkan_cuda_interop.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {
TEST(PreparedExternalAccess,
     UnreadyInputRejectsWholeRecipeWithoutDeviceEffects) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  auto [buffer, status] =
      device.allocate_memory_unique({sizeof(std::uint32_t)});
  ASSERT_EQ(status, RhiResult::success);
  auto domain = program.cuda_execution_domains()->create({1, 0});
  struct Operands final : RuntimeCompletionResources {
    DeviceAllocationUnique allocation;
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      return 1;
    }
  };
  auto operands = std::make_shared<Operands>();
  operands->allocation = std::move(buffer);
  FixedExecution::CudaExecutionBinding binding{domain, operands};
  FixedExecution recipe(
      Arch::cuda, device, nullptr,
      {FixedExecution::BufferFill{operands->allocation->get_ptr(),
                                  sizeof(std::uint32_t), 12345}},
      nullptr, program.runtime_fault_domain(), nullptr, nullptr, nullptr,
      nullptr, &binding);
  auto first = std::make_shared<CountingExternalDomain>(1);
  auto second = std::make_shared<CountingExternalDomain>(2);
  second->ready = false;
  recipe.bind_external_access({first, second, first});
  EXPECT_ANY_THROW(recipe.submit({RuntimeCompletion{}}));
  EXPECT_EQ(first->queries, 0);
  for (int i = 0; i < 1000; ++i)
    EXPECT_THROW(recipe.submit(), cuda::ExecutionDomainBusy);
  EXPECT_EQ(domain->snapshot().pending, 0);
  EXPECT_EQ(first->queries, 1000);
  EXPECT_EQ(second->queries, 1000);
  EXPECT_EQ(first->acquires, 0);
  EXPECT_EQ(second->acquires, 0);
  second->fail_ready = true;
  EXPECT_THROW(recipe.submit(), std::runtime_error);
  EXPECT_EQ(domain->snapshot().pending, 0);
  EXPECT_EQ(first->acquires, 0);
  second->fail_ready = false;
  second->ready = true;
  auto done = recipe.submit();
  done.wait();
  EXPECT_EQ(first->acquires, 1);
  EXPECT_EQ(second->acquires, 1);
  EXPECT_EQ(first->releases, 1);
  EXPECT_EQ(second->releases, 1);
  std::uint32_t actual = 0;
  auto context = CUDAContext::get_instance().get_guard();
  CUDADriver::get_instance().memcpy_device_to_host(
      &actual, device.get_memory_addr(*operands->allocation), sizeof(actual));
  EXPECT_EQ(actual, 12345);
  domain->close();
  operands->allocation.reset();
  program.finalize();
}

TEST(PreparedExternalAccess,
     VulkanInteropReplaysAndRetiresAfterStreamDestruction) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  vulkan::VulkanDeviceCreator creator({});
  auto *vk = static_cast<vulkan::VulkanDevice *>(creator.device());
  Device::AllocParams parameters{};
  parameters.size = 32 * sizeof(std::uint32_t);
  parameters.export_sharing = true;
  parameters.usage = AllocUsage::Storage;
  auto [shared, status] = vk->allocate_memory_unique(parameters);
  ASSERT_EQ(status, RhiResult::success);
  parameters.export_sharing = false;
  auto [output, output_status] = vk->allocate_memory_unique(parameters);
  ASSERT_EQ(output_status, RhiResult::success);
  auto adapter = VulkanCudaExternalAllocation::create(vk, &device, *shared);
  struct Operands final : RuntimeCompletionResources {
    std::shared_ptr<VulkanCudaExternalAllocation> adapter;
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      return 1;
    }
  };
  auto operands = std::make_shared<Operands>();
  operands->adapter = adapter;
  auto domain = program.cuda_execution_domains()->create({2, 0});
  FixedExecution::CudaExecutionBinding binding{domain, operands};
  FixedExecution recipe(
      Arch::cuda, device, nullptr,
      {FixedExecution::BufferFill{adapter->cuda_allocation().get_ptr(),
                                  parameters.size, 0x12345678}},
      nullptr, program.runtime_fault_domain(), nullptr, nullptr, nullptr,
      nullptr, &binding);
  recipe.bind_external_access({adapter, adapter});
  auto *stream = static_cast<vulkan::VulkanStream *>(vk->get_compute_stream());
  auto [producer, producer_status] = stream->new_command_list_unique();
  ASSERT_EQ(producer_status, RhiResult::success);
  producer->buffer_fill(shared->get_ptr(), parameters.size, 0);
  ASSERT_TRUE(adapter->release_vulkan_to_cuda(*stream, producer.get()));
  auto [consumer, consumer_status] = stream->new_command_list_unique();
  ASSERT_EQ(consumer_status, RhiResult::success);
  consumer->buffer_copy(output->get_ptr(), shared->get_ptr(), parameters.size);
  program.synchronize();
  testing::CudaExternalGate default_gate;
  default_gate.enqueue_wait(nullptr);
  // A blocked legacy tail must not delay selected CUDA/Vulkan handoffs.
  {
    ScopedExternalProfilerAnnotation cycles("F3/external/cycles");
    for (int i = 0; i < 5; ++i) {
      RuntimeCompletion done;
      {
        ScopedExternalProfilerAnnotation submit("F3/external/submit");
        done = recipe.submit();
      }
      auto consumed = adapter->cycle_vulkan_to_cuda(*stream, consumer.get());
      ASSERT_TRUE(consumed);
      ASSERT_TRUE(consumed->wait());
      done.wait();
      EXPECT_FALSE(default_gate.timed_out());
    }
  }
  std::array<std::uint32_t, 32> result{};
  auto pointer = output->get_ptr();
  void *destination = result.data();
  auto bytes = sizeof(result);
  EXPECT_EQ(vk->readback_data(&pointer, &destination, &bytes),
            RhiResult::success);
  for (auto value : result)
    EXPECT_EQ(value, 0x12345678u);
  EXPECT_FALSE(default_gate.timed_out());
  {
    ScopedExternalProfilerAnnotation release(
        "F3/external/default_gate_release");
    default_gate.release();
  }
  domain->close();
  EXPECT_TRUE(domain->snapshot().closed);
  // The allocation's last consumer stream no longer exists. Its exact event
  // prefix still proves retirement without dereferencing that old stream.
  {
    ScopedExternalProfilerAnnotation retirement(
        "F3/external/retire_after_stream_close");
    EXPECT_NO_THROW(adapter->close());
  }
  EXPECT_TRUE(adapter->closed());
  shared.reset();
  output.reset();
  program.finalize();
}

TEST(PreparedExternalAccess, CompleteRecipeConsumesGateOnlyAfterAdmission) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  auto [buffer, status] =
      device.allocate_memory_unique({3 * sizeof(std::uint32_t)});
  ASSERT_EQ(status, RhiResult::success);
  auto *address = static_cast<std::uint32_t *>(device.get_memory_addr(*buffer));
  auto domain = program.cuda_execution_domains()->create({1, 0});
  auto peer = program.cuda_execution_domains()->create({1, 0});
  struct Operands final : RuntimeCompletionResources {
    DeviceAllocationUnique allocation;
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      return 1;
    }
  };
  auto operands = std::make_shared<Operands>();
  operands->allocation = std::move(buffer);
  FixedExecution::CudaExecutionBinding binding{domain, operands};
  FixedExecution recipe(
      Arch::cuda, device, nullptr,
      {FixedExecution::BufferFill{operands->allocation->get_ptr(),
                                  sizeof(std::uint32_t), 12345}},
      nullptr, program.runtime_fault_domain(), nullptr, nullptr, nullptr,
      nullptr, &binding);
  testing::CudaExternalGate admission_gate, external_gate;
  class GateDomain final : public ExternalSynchronizationDomain {
   public:
    GateDomain(testing::CudaExternalGate &gate, std::uint32_t *release_marker)
        : gate_(gate), marker_(release_marker) {
    }
    std::uint64_t identity() const noexcept override {
      return 42;
    }
    void acquire_for_consumer(const ExternalStreamDomain &stream) override {
      selected = stream;
      gate_.enqueue_wait(stream.native_stream);
      ++acquires;
    }
    void release_from_consumer(const ExternalStreamDomain &stream) override {
      EXPECT_TRUE(selected.same_stream(stream));
      CUDADriver::get_instance().memsetd32_async(marker_, 67890, 1,
                                                 stream.native_stream);
      ++releases;
    }
    unsigned acquires{0}, releases{0};
    ExternalStreamDomain selected;

   private:
    testing::CudaExternalGate &gate_;
    std::uint32_t *marker_;
  };
  auto access = std::make_shared<GateDomain>(external_gate, address + 1);
  recipe.bind_external_access({access, access});
  EXPECT_ANY_THROW(recipe.submit({RuntimeCompletion{}}));
  EXPECT_EQ(access->acquires, 0);
  auto occupied = domain->submit(
      {}, nullptr, [&](void *stream) { admission_gate.enqueue_wait(stream); });
  EXPECT_THROW(recipe.submit(), cuda::ExecutionDomainBusy);
  EXPECT_EQ(access->acquires, 0);
  EXPECT_EQ(access->releases, 0);
  admission_gate.release();
  occupied.wait();
  auto pending = recipe.submit();
  EXPECT_EQ(access->acquires, 1);
  EXPECT_EQ(access->releases, 1);
  EXPECT_EQ(access->selected.native_stream,
            reinterpret_cast<void *>(domain->snapshot().stream));
  EXPECT_EQ(access->selected.owner_domain, domain->snapshot().owner);
  EXPECT_EQ(access->selected.stream_identity, domain->snapshot().identity);
  EXPECT_THROW(recipe.submit(), cuda::ExecutionDomainBusy);
  EXPECT_EQ(access->acquires, 1);
  peer->submit({}, nullptr,
               [&](void *stream) {
                 CUDADriver::get_instance().memsetd32_async(address + 2, 24680,
                                                            1, stream);
               })
      .wait();
  auto current = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address + 2,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 24680u);
  EXPECT_FALSE(pending.done());
  EXPECT_FALSE(admission_gate.timed_out());
  EXPECT_FALSE(external_gate.timed_out());
  external_gate.release();
  pending.wait();
  std::array<std::uint32_t, 2> result{};
  CUDADriver::get_instance().memcpy_device_to_host(result.data(), address,
                                                   sizeof(result));
  EXPECT_EQ(result[0], 12345u);
  EXPECT_EQ(result[1], 67890u);
  domain->close();
  peer->close();
  operands->allocation.reset();
  program.finalize();
}
}  // namespace
}  // namespace taichi::lang
#endif
