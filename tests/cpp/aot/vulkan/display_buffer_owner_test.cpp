#if defined(TI_WITH_VULKAN)
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/program/storage_view.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_external_access.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/completion_wait_set.h"
#include "taichi/runtime/fixed_execution_vulkan.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"

namespace taichi::lang {
namespace {
using DisplayOwner = vulkan::VulkanDisplayBufferOwner;
using Access = vulkan::VulkanExternalAccess;
using Domain = vulkan::ExecutionDomain;
using Point = std::shared_ptr<vulkan::VulkanTimelinePoint>;

DisplayOwner::Selection selection(const std::shared_ptr<Domain> &domain) {
  const auto info = domain->snapshot();
  return {info.owner, info.identity, info.queue_index};
}

Point point(const RuntimeCompletion &ticket,
            const std::shared_ptr<RuntimeFaultDomain> &fault) {
  return std::dynamic_pointer_cast<vulkan::VulkanTimelinePoint>(
      ticket.reusable_stream_dependency(Arch::vulkan, fault->program_domain(), fault));
}

bool ready(const RuntimeCompletion &ticket) {
  CompletionWaitSet wait({ticket}, std::chrono::milliseconds(1));
  return wait.wait(std::chrono::seconds(3)).status == CompletionWaitSet::Status::ready;
}

// Exactly the retirement adapter the UI will register per physical slot. It
// owns no Program/domain/Commands and never reenters storage retirement.
struct SlotRetirement final : RuntimeClient {
  explicit SlotRetirement(std::shared_ptr<DisplayOwner> value)
      : owner(std::move(value)) {}
  std::shared_ptr<DisplayOwner> owner;
  std::atomic<unsigned> retired{0};
  void retire() override {
    owner->close();
    retired.fetch_add(1, std::memory_order_relaxed);
  }
};

struct Slot {
  std::shared_ptr<DisplayOwner> owner;
  storage::StorageOwnerRef registration;
  std::shared_ptr<SlotRetirement> retirement;
};

Slot make_slot(Program &program, const std::shared_ptr<Domain> &domain) {
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  Slot slot;
  slot.owner = DisplayOwner::create(device, 16, selection(domain));
  slot.retirement = std::make_shared<SlotRetirement>(slot.owner);
  slot.registration = program.register_external_dense_storage(
      slot.owner->ndarray_allocation(), 16,
      [owner = slot.owner] {
        // Registry release must not introduce another GPU wait. Its retained
        // adapter already drained and explicitly released the allocation.
        EXPECT_TRUE(owner->snapshot().closed);
      }, slot.owner);
  program.retain_external_storage_client(slot.retirement, slot.registration);
  return slot;
}

storage::DenseStorageDescriptor view(const Slot &slot, std::int64_t offset = 0) {
  storage::DenseStorageLayoutSpec layout;
  layout.scalar_type = PrimitiveType::u32;
  layout.index_shape = {2};
  layout.index_strides_bytes = {4};
  layout.byte_offset = offset;
  layout.access = storage::StorageAccess::kReadWrite;
  auto built = storage::build_dense_storage_descriptor(
      slot.registration, storage::StorageSourceKind::kExternalDense, layout);
  if (!built)
    throw std::runtime_error("Cannot construct display fixture view");
  return *built.descriptor;
}

struct BorrowerRetirement final : RuntimeClient {
  std::weak_ptr<Domain> domain;
  std::shared_ptr<vulkan::ExecutionInstance> instance;
  void retire() override {
    if (auto selected = domain.lock())
      if (instance && !selected->instance_retired(instance))
        selected->retire_instance(instance);
  }
};

struct WriterOperands final : RuntimeCompletionResources {
  std::vector<Program::DenseStorageBorrow> loans;
  DeviceAllocationUnique readback;
  std::atomic<std::size_t> count{0};
  std::size_t retained_resource_count(std::uint32_t) const noexcept override {
    return count.load(std::memory_order_acquire);
  }
  void retire() {
    readback.reset();
    loans.clear();
    count.store(0, std::memory_order_release);
  }
};

struct WriterParts final : RuntimeClient {
  std::vector<std::shared_ptr<RuntimeClient>> parts;
  std::shared_ptr<WriterOperands> operands;
  void retire() override {
    for (const auto &part : parts)
      part->retire();
    parts.clear();
    operands->retire();
  }
};

struct Writer {
  std::unique_ptr<FixedExecution> call;
  std::shared_ptr<vulkan::ExecutionInstance> instance;
  std::shared_ptr<WriterOperands> operands;
};

// Program's real Dense borrowing plus the same FixedExecution implementation
// used by prepared Graph/AOT. Two finite loans become two body operations but
// one physical owner's epoch. No test-only production creator or executor.
Writer make_writer(Program &program,
                   const std::shared_ptr<Domain> &domain,
                   const std::vector<storage::DenseStorageDescriptor> &views) {
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  Writer result;
  result.operands = std::make_shared<WriterOperands>();
  auto parts = std::make_shared<WriterParts>();
  parts->operands = result.operands;
  std::vector<std::shared_ptr<BorrowerRetirement>> borrowers;
  std::vector<std::shared_ptr<ExternalSynchronizationDomain>> access;
  std::vector<FixedExecution::Operation> operations;
  for (std::size_t i = 0; i < views.size(); ++i) {
    auto borrower = std::make_shared<BorrowerRetirement>();
    borrower->domain = domain;
    auto loan = program.borrow_dense_storage(views[i], borrower);
    operations.push_back(FixedExecution::BufferFill{
        loan.allocation.get_ptr(loan.byte_offset), loan.byte_size,
        static_cast<std::uint32_t>(17 + i)});
    access.push_back(loan.external_access);
    result.operands->loans.push_back(std::move(loan));
    borrowers.push_back(std::move(borrower));
  }
  auto allocation = device.allocate_memory_unique(
      {16, false, true, false, AllocUsage::Storage});
  if (allocation.second != RhiResult::success)
    throw std::runtime_error("Cannot allocate fixture readback");
  result.operands->readback = std::move(allocation.first);
  operations.push_back(FixedExecution::BufferCopy{
      result.operands->readback->get_ptr(),
      result.operands->loans.front().allocation.get_ptr(), 16});
  result.operands->count.store(views.size(), std::memory_order_release);
  FixedExecution::VulkanExecutionBinding binding{domain, result.operands};
  binding.retirement_parts = &parts->parts;
  result.instance = domain->prepare_owned_instance(parts, [&](const auto &instance) {
    binding.instance = instance;
    for (const auto &borrower : borrowers)
      borrower->instance = instance;
    result.call = std::make_unique<FixedExecution>(
        Arch::vulkan, device, program.get_gfx_runtime_for_borrowing(), operations,
        nullptr, program.runtime_fault_domain(), nullptr, nullptr, nullptr,
        nullptr, nullptr, &binding);
    result.call->bind_external_access(access);
  });
  return result;
}

void expect_pixels(vulkan::VulkanDevice &device, const Writer &writer) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*writer.operands->readback, &mapped), RhiResult::success);
  const auto *pixels = static_cast<const std::uint32_t *>(mapped);
  EXPECT_EQ(pixels[0], 17u);
  EXPECT_EQ(pixels[1], 17u);
  EXPECT_EQ(pixels[2], 18u);
  EXPECT_EQ(pixels[3], 18u);
  device.unmap(*writer.operands->readback);
}

bool independent(vulkan::VulkanDevice &device, unsigned count = 2) {
  return device.vk_caps().timeline_semaphore &&
         device.created_compute_queue_count() >= count;
}
}  // namespace

TEST(VulkanDisplayOwner, DenseAliasesUseOneEpochAndExactProducer) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!independent(device))
    GTEST_SKIP() << "Requires timeline semaphores and one independent queue";
  auto domain = program.vulkan_execution_domains()->create({3, 2, true});
  auto slot = make_slot(program, domain);
  auto writer = make_writer(program, domain, {view(slot), view(slot, 8)});
  program.synchronize();  // Cold setup may use the ordinary runtime queue.
  const auto fault = program.runtime_fault_domain();
  const auto submitted = device.queue_submission_snapshot().queue_submit_calls;
  const auto unacquired_sequence = fault->next_submission_sequence();
  EXPECT_THROW(writer.call->submit(), std::invalid_argument);
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, submitted);
  EXPECT_EQ(domain->latest_completion().sequence(), 0u);
  EXPECT_EQ(fault->next_submission_sequence(), unacquired_sequence);
  EXPECT_EQ(slot.owner->snapshot().next_epoch, 1u);
  const auto empty_lease = slot.owner->acquire_write();
  slot.owner->cancel_write(empty_lease);
  const auto lease = slot.owner->acquire_write();
  EXPECT_NE(lease, empty_lease);
  EXPECT_THROW(slot.owner->cancel_write(empty_lease), std::invalid_argument);

  // All preparation/default initialization has completed before this gate.
  test_detail::TimelineGate default_gate(device, 0);
  auto first = writer.call->submit({}, {}, true);
  ASSERT_TRUE(ready(first));
  expect_pixels(device, writer);
  EXPECT_EQ(slot.owner->snapshot().next_epoch, 2u);
  const auto sequence = first.sequence();
  const auto next_sequence = fault->next_submission_sequence();
  EXPECT_THROW(writer.call->submit(), std::invalid_argument);
  EXPECT_EQ(domain->latest_completion().sequence(), sequence);
  EXPECT_EQ(fault->next_submission_sequence(), next_sequence);
  {
    test_detail::TimelineGate late_gate(device, domain->snapshot().queue_index);
    auto later = domain->submit(domain->prepare_commands({}));
    EXPECT_FALSE(later.done());
    EXPECT_THROW(slot.owner->seal_write(lease, point(later, fault)),
                 std::invalid_argument);
    EXPECT_TRUE(slot.owner->snapshot().acquired);
    EXPECT_EQ(slot.owner->seal_write(lease, point(first, fault)), point(first, fault));
    EXPECT_TRUE(first.done());  // Exact producer excludes the later queue tail.
  }
  const auto next = slot.owner->acquire_write();
  auto second = writer.call->submit();
  ASSERT_TRUE(ready(second));
  EXPECT_THROW(slot.owner->seal_write(next, point(first, fault)),
               std::invalid_argument);
  EXPECT_NO_THROW(slot.owner->seal_write(next, point(second, fault)));
  EXPECT_EQ(slot.owner->snapshot().next_epoch, 3u);
  expect_pixels(device, writer);
  domain->close();
  EXPECT_TRUE(first.done());
  EXPECT_EQ(slot.retirement->retired.load(), 0u);
}

TEST(VulkanDisplayOwner, CancelledPendingWriterCannotBeReacquired) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!independent(device, 3))
    GTEST_SKIP() << "Controlled peer progress requires two independent queues";
  auto domain = program.vulkan_execution_domains()->create({1, 2});
  auto peer = program.vulkan_execution_domains()->create({1, 1});
  auto slot = make_slot(program, domain);
  auto other = make_slot(program, domain);
  auto writer = make_writer(program, domain, {view(slot), view(slot, 8)});
  auto next_writer = make_writer(program, domain, {view(other), view(other, 8)});
  // A real FixedExecution cold bind rejects the wrong domain and returns its
  // one-instance preparation budget before a correctly bound peer is prepared.
  EXPECT_THROW(make_writer(program, peer, {view(slot), view(slot, 8)}),
               std::invalid_argument);
  auto peer_slot = make_slot(program, peer);
  auto peer_writer = make_writer(program, peer, {view(peer_slot), view(peer_slot, 8)});
  const auto lease = slot.owner->acquire_write();
  const auto other_lease = other.owner->acquire_write();
  test_detail::TimelineGate gate(device, domain->snapshot().queue_index);
  auto pending = writer.call->submit();
  EXPECT_FALSE(pending.done());
  slot.owner->cancel_write(lease);
  const auto before = device.queue_submission_snapshot().queue_submit_calls;
  const auto next_sequence = program.runtime_fault_domain()->next_submission_sequence();
  EXPECT_THROW(slot.owner->acquire_write(), Domain::Busy);
  EXPECT_THROW(next_writer.call->submit(), Domain::Busy);
  EXPECT_EQ(domain->latest_completion().sequence(), pending.sequence());
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
  EXPECT_EQ(program.runtime_fault_domain()->next_submission_sequence(), next_sequence);
  EXPECT_EQ(other.owner->snapshot().next_epoch, 1u);
  EXPECT_TRUE(other.owner->snapshot().acquired);
  const auto peer_lease = peer_slot.owner->acquire_write();
  auto sentinel = peer_writer.call->submit();
  ASSERT_TRUE(ready(sentinel));
  peer_slot.owner->seal_write(peer_lease, point(sentinel, program.runtime_fault_domain()));
  expect_pixels(device, peer_writer);
  EXPECT_FALSE(pending.done());
  gate.release();
  pending.wait();
  const auto fresh = slot.owner->acquire_write();
  EXPECT_THROW(slot.owner->seal_write(fresh, point(pending, program.runtime_fault_domain())),
               std::invalid_argument);
  slot.owner->cancel_write(fresh);
  auto completed = next_writer.call->submit();
  completed.wait();
  other.owner->seal_write(other_lease, point(completed, program.runtime_fault_domain()));
}

TEST(VulkanDisplayOwner, MultipleOwnersRollbackAndLocalPlanAddsNoBoundaries) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!independent(device))
    GTEST_SKIP();
  vulkan::VulkanExecutionQueue queue(device, 7);
  DisplayOwner::Selection selected{7, 19, queue.queue_index()};
  auto left = DisplayOwner::create(device, 16, selected);
  auto right = DisplayOwner::create(device, 16, selected);
  auto access = std::make_shared<Access>(
      std::vector<Access::Owner>{right, left, left});
  EXPECT_EQ(access->size(), 2u);
  auto list = queue.new_command_list();
  list->buffer_fill(left->ndarray_allocation().get_ptr(), 16, 23);
  list->buffer_fill(right->ndarray_allocation().get_ptr(), 16, 29);
  list->memory_barrier();
  auto commands = queue.prepare_external_access(queue.prepare({list.get()}), access);
  EXPECT_THROW(queue.prepare_external_access(commands, access), std::invalid_argument);
  auto wrong = selected;
  ++wrong.domain;
  EXPECT_THROW(left->validate_binding(device, wrong), std::invalid_argument);
  const auto left_lease = left->acquire_write();
  bool admitted = false;
  EXPECT_THROW(queue.submit(commands, {}, [&](auto, const auto &) { admitted = true; }),
               std::invalid_argument);
  EXPECT_FALSE(admitted);
  EXPECT_EQ(left->snapshot().next_epoch, 1u);
  EXPECT_EQ(right->snapshot().next_epoch, 1u);
  const auto right_lease = right->acquire_write();
  EXPECT_THROW(queue.submit(commands, {}, [](auto, const auto &) {
    throw std::runtime_error("rejected before native enqueue");
  }), std::runtime_error);
  EXPECT_EQ(queue.pending(), 0u);
  EXPECT_EQ(left->snapshot().next_epoch, 1u);
  EXPECT_EQ(right->snapshot().next_epoch, 1u);
  const auto before = device.queue_submission_snapshot().submitted_command_buffers;
  const auto completed = queue.submit(commands);
  completed->wait();
  EXPECT_EQ(device.queue_submission_snapshot().submitted_command_buffers, before + 1);
  EXPECT_EQ(left->seal_write(left_lease, completed), completed);
  EXPECT_EQ(right->seal_write(right_lease, completed), completed);
  left->close();
  right->close();
  EXPECT_THROW(queue.submit(commands), std::runtime_error);
}

TEST(VulkanDisplayOwner, ProgramRetirementRevokesOnlyItsStorageAndClosesLateShells) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!independent(device))
    GTEST_SKIP();
  auto domain = program.vulkan_execution_domains()->create({2, 3});
  auto slot = make_slot(program, domain);
  auto other = make_slot(program, domain);
  auto descriptor = view(slot);
  auto writer = make_writer(program, domain, {descriptor, view(slot, 8)});
  auto peer = make_writer(program, domain, {view(other), view(other, 8)});
  auto lease = slot.owner->acquire_write();
  auto ticket = writer.call->submit();
  slot.owner->seal_write(lease, point(ticket, program.runtime_fault_domain()));
  slot.owner->stop();
  program.retire_external_dense_storage(slot.registration);
  EXPECT_EQ(slot.retirement->retired.load(), 1u);
  EXPECT_TRUE(slot.owner->snapshot().closed);
  EXPECT_TRUE(domain->instance_retired(writer.instance));
  EXPECT_FALSE(program.validate_external_dense_storage_owner(slot.registration));
  EXPECT_FALSE(writer.operands->readback);
  EXPECT_THROW(writer.call->submit(), std::invalid_argument);
  EXPECT_ANY_THROW(program.borrow_dense_storage(
      descriptor, std::make_shared<BorrowerRetirement>()));
  EXPECT_TRUE(domain->snapshot().accepting);
  auto peer_lease = other.owner->acquire_write();
  auto peer_ticket = peer.call->submit();
  auto exact = point(peer_ticket, program.runtime_fault_domain());
  other.owner->seal_write(peer_lease, exact);
  program.finalize();
  EXPECT_EQ(other.retirement->retired.load(), 1u);
  EXPECT_TRUE(other.owner->snapshot().closed);
  EXPECT_FALSE(peer.operands->readback);
  EXPECT_TRUE(domain->instance_retired(peer.instance));
  EXPECT_TRUE(ticket.done());
  EXPECT_TRUE(peer_ticket.done());
  EXPECT_TRUE(exact->is_ready());
  EXPECT_NO_THROW(exact->wait());
  // Queries and final owner destruction now have no device allocation to free.
  EXPECT_NO_THROW(slot.owner->close());
  EXPECT_NO_THROW(other.owner->close());
}

namespace {
VKAPI_ATTR VkResult VKAPI_CALL reject_display_submit(
    VkQueue, std::uint32_t, const VkSubmitInfo *, VkFence) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}
}  // namespace

TEST(VulkanDisplayOwner, FailedEnqueuePoisonsUntilNativeDrain) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!independent(device))
    GTEST_SKIP();
  vulkan::VulkanExecutionQueue queue(device, 1);
  auto owner = DisplayOwner::create(device, 16, {7, 19, queue.queue_index()});
  auto access = std::make_shared<Access>(std::vector<Access::Owner>{owner});
  auto list = queue.new_command_list();
  list->buffer_fill(owner->ndarray_allocation().get_ptr(), 16, 37);
  auto commands = queue.prepare_external_access(queue.prepare({list.get()}), access);
  owner->acquire_write();
  auto &api = const_cast<VolkDeviceTable &>(device.vk_device().functions());
  const auto submit = api.vkQueueSubmit;
  api.vkQueueSubmit = reject_display_submit;
  EXPECT_THROW(queue.submit(commands), BackendRuntimeError);
  api.vkQueueSubmit = submit;
  EXPECT_EQ(owner->snapshot().next_epoch, 1u);
  EXPECT_TRUE(owner->snapshot().poisoned);
  EXPECT_FALSE(owner->snapshot().accepting);
  EXPECT_THROW(owner->close(), std::runtime_error);
  queue.close();
  EXPECT_NO_THROW(owner->close());
  EXPECT_TRUE(owner->snapshot().closed);
  EXPECT_TRUE(owner->snapshot().poisoned);
  EXPECT_THROW(owner->acquire_write(), std::runtime_error);
}
}  // namespace taichi::lang
#endif
