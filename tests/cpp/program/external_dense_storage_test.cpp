#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "taichi/rhi/interop/external_sync.h"

namespace taichi::lang {
namespace {

class TestSynchronizationDomain final : public ExternalSynchronizationDomain {
 public:
  explicit TestSynchronizationDomain(std::uint64_t identity)
      : identity_(identity) {
  }

  std::uint64_t identity() const noexcept override {
    return identity_;
  }

  void acquire_for_consumer(const ExternalStreamDomain &) override {
    acquires_.fetch_add(1, std::memory_order_relaxed);
  }

  void release_from_consumer(const ExternalStreamDomain &) override {
    releases_.fetch_add(1, std::memory_order_relaxed);
  }

  int acquires() const noexcept {
    return acquires_.load(std::memory_order_relaxed);
  }

  int releases() const noexcept {
    return releases_.load(std::memory_order_relaxed);
  }

 private:
  std::uint64_t identity_{0};
  std::atomic<int> acquires_{0};
  std::atomic<int> releases_{0};
};

storage::DenseStorageBuildResult build_external_f32(
    storage::StorageOwnerRef owner,
    std::vector<std::int64_t> shape,
    std::vector<std::int64_t> strides,
    std::int64_t byte_offset = 0) {
  storage::DenseStorageLayoutSpec layout;
  layout.scalar_type = PrimitiveType::f32;
  layout.index_shape = std::move(shape);
  layout.index_strides_bytes = std::move(strides);
  layout.byte_offset = byte_offset;
  layout.access = storage::StorageAccess::kReadWrite;
  return storage::build_dense_storage_descriptor(
      std::move(owner), storage::StorageSourceKind::kExternalDense, layout);
}

#ifdef TI_WITH_LLVM

TEST(ExternalDenseStorageTest, ResolvesAllocationAndRejectsStaleGeneration) {
  std::atomic<int> releases{0};
  Program program(Arch::x64);
  auto *backing = program.create_ndarray(PrimitiveType::f32, {8},
                                         ExternalArrayLayout::kNull, false);
  const DeviceAllocation allocation = backing->get_device_allocation();

  const auto owner = program.register_external_dense_storage(
      allocation, 8 * sizeof(float), [&] { ++releases; });
  auto built = build_external_f32(owner, {8}, {sizeof(float)});
  ASSERT_TRUE(built);

  program.with_resolved_dense_storage_bindings(
      {&*built.descriptor},
      [&](const storage::ResolvedDenseBinding *bindings, std::size_t count) {
        ASSERT_EQ(count, 1u);
        EXPECT_TRUE(bindings[0].valid);
        EXPECT_EQ(bindings[0].allocation, allocation);
        EXPECT_EQ(bindings[0].byte_offset, 0u);
        EXPECT_EQ(bindings[0].byte_size, 8 * sizeof(float));
      });
  EXPECT_TRUE(program.validate_external_dense_storage_owner(owner));
  EXPECT_EQ(program.debug_dense_storage_binding_stats().at("external_bindings"),
            1u);

  program.retire_external_dense_storage(owner);
  EXPECT_EQ(releases.load(), 1);
  EXPECT_FALSE(program.validate_external_dense_storage_owner(owner));
  EXPECT_ANY_THROW(program.with_resolved_dense_storage_bindings(
      {&*built.descriptor}, [](const auto *, std::size_t) {}));

  const auto replacement = program.register_external_dense_storage(
      allocation, 8 * sizeof(float), [&] { ++releases; });
  EXPECT_EQ(replacement.external_slot, owner.external_slot);
  EXPECT_NE(replacement.external_generation, owner.external_generation);
  EXPECT_FALSE(program.validate_external_dense_storage_owner(owner));
  EXPECT_TRUE(program.validate_external_dense_storage_owner(replacement));
  program.retire_external_dense_storage(replacement);
  EXPECT_EQ(releases.load(), 2);

  program.delete_ndarray(backing);
}

TEST(ExternalDenseStorageTest, RejectsDescriptorOutsideRegisteredRange) {
  Program program(Arch::x64);
  auto *backing = program.create_ndarray(PrimitiveType::f32, {8},
                                         ExternalArrayLayout::kNull, false);
  const auto owner = program.register_external_dense_storage(
      backing->get_device_allocation(), 8 * sizeof(float));
  auto outside =
      build_external_f32(owner, {5}, {sizeof(float)}, 4 * sizeof(float));
  ASSERT_TRUE(outside);
  EXPECT_ANY_THROW(program.with_resolved_dense_storage_bindings(
      {&*outside.descriptor}, [](const auto *, std::size_t) {}));

  program.retire_external_dense_storage(owner);
  program.delete_ndarray(backing);
}

TEST(ExternalDenseStorageTest, FailedRegistrationDoesNotAssumeOwnership) {
  std::atomic<int> releases{0};
  Program program(Arch::x64);
  EXPECT_ANY_THROW(program.register_external_dense_storage(
      kDeviceNullAllocation, sizeof(float), [&] { ++releases; }));
  EXPECT_EQ(releases.load(), 0);
}

TEST(ExternalDenseStorageTest, RuntimeArgumentRequiresMatchingSyncDomain) {
  Program program(Arch::x64);
  auto *backing = program.create_ndarray(PrimitiveType::f32, {8},
                                         ExternalArrayLayout::kNull, false);
  auto sync = std::make_shared<TestSynchronizationDomain>(73);
  const auto owner = program.register_external_dense_storage(
      backing->get_device_allocation(), 8 * sizeof(float), {}, sync);
  auto built = build_external_f32(owner, {8}, {sizeof(float)});
  ASSERT_TRUE(built);

  storage::RuntimeStorageRequirement requirement;
  requirement.backend = Arch::x64;
  requirement.consumer = storage::RuntimeStorageConsumer::kExternalInterop;
  requirement.dense.require_ndarray_abi = true;
  requirement.dense.require_unique_mapping = true;
  requirement.dense.require_writable = true;
  requirement.dense.accept_external_owner = true;
  requirement.require_external_sync = true;

  storage::RuntimeStorageArgument argument(*built.descriptor, requirement,
                                           sync->identity());
  ASSERT_TRUE(argument.qualification().capabilities.bindable);
  program.with_resolved_runtime_storage_arguments(
      {&argument},
      [&](const storage::ResolvedDenseBinding *bindings, std::size_t count) {
        ASSERT_EQ(count, 1u);
        EXPECT_TRUE(bindings[0].valid);
        EXPECT_EQ(bindings[0].runtime_signature, argument.stable_signature());
        EXPECT_EQ(bindings[0].synchronization_domain_identity,
                  sync->identity());
        EXPECT_TRUE(bindings[0].capabilities.zero_copy_qualified);
      });

  EXPECT_EQ(sync->acquires(), 1);
  EXPECT_EQ(sync->releases(), 1);
  EXPECT_THROW(
      program.with_resolved_runtime_storage_arguments(
          {&argument, &argument},
          [](const auto *, std::size_t) { throw std::runtime_error("stop"); }),
      std::runtime_error);
  EXPECT_EQ(sync->acquires(), 2);
  EXPECT_EQ(sync->releases(), 2);
  storage::RuntimeStorageArgument mismatched(*built.descriptor, requirement,
                                             sync->identity() + 1);
  EXPECT_ANY_THROW(program.with_resolved_runtime_storage_arguments(
      {&mismatched}, [](const auto *, std::size_t) {}));

  program.retire_external_dense_storage(owner);
  program.delete_ndarray(backing);
}

TEST(ExternalDenseStorageTest, RuntimeArgumentRejectsMissingSyncOwner) {
  Program program(Arch::x64);
  auto *backing = program.create_ndarray(PrimitiveType::f32, {4},
                                         ExternalArrayLayout::kNull, false);
  const auto owner = program.register_external_dense_storage(
      backing->get_device_allocation(), 4 * sizeof(float));
  auto built = build_external_f32(owner, {4}, {sizeof(float)});
  ASSERT_TRUE(built);

  storage::RuntimeStorageRequirement requirement;
  requirement.backend = Arch::x64;
  requirement.consumer = storage::RuntimeStorageConsumer::kExternalInterop;
  requirement.dense.require_ndarray_abi = true;
  requirement.dense.require_unique_mapping = true;
  requirement.dense.require_writable = true;
  requirement.dense.accept_external_owner = true;
  requirement.require_external_sync = true;
  storage::RuntimeStorageArgument argument(*built.descriptor, requirement, 91);
  ASSERT_TRUE(argument.qualification().capabilities.bindable);
  EXPECT_ANY_THROW(program.with_resolved_runtime_storage_arguments(
      {&argument}, [](const auto *, std::size_t) {}));

  program.retire_external_dense_storage(owner);
  program.delete_ndarray(backing);
}

TEST(ExternalDenseStorageTest, GraphScopeRetainsReplayableExternalOwner) {
  Program program(Arch::x64);
  auto *backing = program.create_ndarray(PrimitiveType::f32, {8},
                                         ExternalArrayLayout::kNull, false);
  auto sync = std::make_shared<TestSynchronizationDomain>(101);
  const auto owner = program.register_external_dense_storage(
      backing->get_device_allocation(), 8 * sizeof(float), {}, sync);
  auto built = build_external_f32(owner, {8}, {sizeof(float)});
  ASSERT_TRUE(built);

  storage::RuntimeStorageRequirement requirement;
  requirement.backend = Arch::x64;
  requirement.consumer = storage::RuntimeStorageConsumer::kGraphReplay;
  requirement.mode = storage::RuntimeStorageMode::kReplay;
  requirement.dense.require_ndarray_abi = true;
  requirement.dense.require_unique_mapping = true;
  requirement.dense.require_writable = true;
  requirement.dense.accept_external_owner = true;
  requirement.require_external_sync = true;
  storage::RuntimeStorageArgument argument(*built.descriptor, requirement,
                                           sync->identity());
  ASSERT_TRUE(argument.qualification().capabilities.replayable);

  const storage::RuntimeStorageArgument *arguments[] = {&argument};
  EXPECT_ANY_THROW(
      program.retain_runtime_storage_for_graph_submission(arguments, 1));

  std::future<void> retire;
  {
    auto graph_scope = program.acquire_runtime_resource_graph_scope();
    program.retain_runtime_storage_for_graph_submission(arguments, 1);
    EXPECT_EQ(sync->acquires(), 1);
    EXPECT_EQ(sync->releases(), 0);
    retire = std::async(std::launch::async,
                        [&] { program.retire_external_dense_storage(owner); });
    EXPECT_EQ(retire.wait_for(std::chrono::milliseconds(25)),
              std::future_status::timeout);
  }
  EXPECT_EQ(sync->releases(), 1);
  EXPECT_EQ(retire.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  retire.get();
  program.delete_ndarray(backing);
}

TEST(ExternalDenseStorageTest, GraphScopeAcquiresSharedDomainOnlyOnce) {
  Program program(Arch::x64);
  auto *backing = program.create_ndarray(PrimitiveType::f32, {16},
                                         ExternalArrayLayout::kNull, false);
  auto sync = std::make_shared<TestSynchronizationDomain>(102);
  const auto first_owner = program.register_external_dense_storage(
      backing->get_device_allocation(), 16 * sizeof(float), {}, sync);
  const auto second_owner = program.register_external_dense_storage(
      backing->get_device_allocation(), 16 * sizeof(float), {}, sync);
  auto first = build_external_f32(first_owner, {8}, {sizeof(float)});
  auto second =
      build_external_f32(second_owner, {8}, {sizeof(float)}, 8 * sizeof(float));
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);

  storage::RuntimeStorageRequirement requirement;
  requirement.backend = Arch::x64;
  requirement.consumer = storage::RuntimeStorageConsumer::kGraphReplay;
  requirement.mode = storage::RuntimeStorageMode::kReplay;
  requirement.dense.require_ndarray_abi = true;
  requirement.dense.require_unique_mapping = true;
  requirement.dense.require_writable = true;
  requirement.dense.accept_external_owner = true;
  requirement.require_external_sync = true;
  storage::RuntimeStorageArgument first_argument(*first.descriptor, requirement,
                                                 sync->identity());
  storage::RuntimeStorageArgument second_argument(
      *second.descriptor, requirement, sync->identity());

  {
    auto graph_scope = program.acquire_runtime_resource_graph_scope();
    const storage::RuntimeStorageArgument *arguments[] = {&first_argument,
                                                          &second_argument};
    program.retain_runtime_storage_for_graph_submission(arguments, 2);
    EXPECT_EQ(sync->acquires(), 1);
    EXPECT_EQ(sync->releases(), 0);
  }
  EXPECT_EQ(sync->acquires(), 1);
  EXPECT_EQ(sync->releases(), 1);

  program.retire_external_dense_storage(first_owner);
  program.retire_external_dense_storage(second_owner);
  program.delete_ndarray(backing);
}

TEST(ExternalDenseStorageTest, RetireWaitsForSubmissionTransaction) {
  std::atomic<int> releases{0};
  Program program(Arch::x64);
  const auto owner = program.register_external_dense_storage(
      kDeviceNullAllocation, 0, [&] { ++releases; });
  auto empty = build_external_f32(owner, {0}, {sizeof(float)});
  ASSERT_TRUE(empty);

  std::promise<void> callback_entered;
  std::promise<void> allow_callback_exit;
  auto exit_signal = allow_callback_exit.get_future().share();
  auto submission = std::async(std::launch::async, [&] {
    program.with_resolved_dense_storage_bindings(
        {&*empty.descriptor},
        [&](const storage::ResolvedDenseBinding *bindings, std::size_t count) {
          ASSERT_EQ(count, 1u);
          EXPECT_EQ(bindings[0].byte_size, 0u);
          callback_entered.set_value();
          exit_signal.wait();
        });
  });
  callback_entered.get_future().wait();

  auto retire = std::async(std::launch::async, [&] {
    program.retire_external_dense_storage(owner);
  });
  EXPECT_EQ(retire.wait_for(std::chrono::milliseconds(25)),
            std::future_status::timeout);
  EXPECT_EQ(releases.load(), 0);

  allow_callback_exit.set_value();
  EXPECT_EQ(submission.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  submission.get();
  EXPECT_EQ(retire.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  retire.get();
  EXPECT_EQ(releases.load(), 1);
}

TEST(ExternalDenseStorageTest, FinalizeReleasesLiveOwnerExactlyOnce) {
  std::atomic<int> releases{0};
  Program program(Arch::x64);
  const auto owner = program.register_external_dense_storage(
      kDeviceNullAllocation, 0, [&] { ++releases; });
  EXPECT_TRUE(program.validate_external_dense_storage_owner(owner));

  program.finalize();
  EXPECT_EQ(releases.load(), 1);
  EXPECT_FALSE(program.validate_external_dense_storage_owner(owner));
  EXPECT_EQ(program.debug_external_dense_storage_stats().at("closed"), 1u);
  EXPECT_EQ(program.debug_external_dense_storage_stats().at("released_total"),
            1u);

  program.finalize();
  EXPECT_EQ(releases.load(), 1);
}

#endif  // TI_WITH_LLVM

}  // namespace
}  // namespace taichi::lang
