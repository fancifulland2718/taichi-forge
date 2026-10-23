#include "taichi/runtime/gfx/shared_image_read_admission.h"

#include <gtest/gtest.h>
#include <atomic>
#include <thread>

namespace taichi::lang::gfx {
struct SharedImageReadAdmissionTestAccess {
  template <class Boundary>
  static auto index_size(const SharedImageReadAdmission<Boundary> &admission,
                         RuntimeResourceHandle image) {
    const auto &edges = admission.control_->images.at(image)->recordings;
    return std::make_pair(edges.size(), edges.capacity());
  }
};
namespace {

using Access = SharedImageUse;
struct Boundary {
  std::uint64_t sequence{1};
};
using Admission = SharedImageReadAdmission<Boundary>;

struct TestTexture {
  explicit TestTexture(std::shared_ptr<int> destroyed,
                       std::function<void()> on_destroy = {})
      : destroyed(std::move(destroyed)), on_destroy(std::move(on_destroy)) {
  }
  ~TestTexture() noexcept {
    ++*destroyed;
    if (on_destroy)
      on_destroy();
  }
  std::shared_ptr<int> destroyed;
  std::function<void()> on_destroy;
};
using Registry = RuntimeResourceRegistry<TestTexture>;

struct Image {
  RuntimeResourceHandle handle;
  Registry::Lease view;
};

class SharedImageReadAdmissionTest : public ::testing::Test {
 protected:
  Image create(std::function<void()> on_destroy = {}) {
    auto [inserted, handle] =
        registry.emplace(1, destroyed, std::move(on_destroy));
    if (inserted != Registry::Result::kSuccess)
      throw std::runtime_error("test image insertion failed");
    auto [acquired, lease] = registry.acquire(handle);
    if (acquired != Registry::Result::kSuccess)
      throw std::runtime_error("test image acquire failed");
    admission.register_image(handle);
    return {handle, std::move(lease)};
  }
  Admission::PinnedUse pin(const Image &image, Access access) {
    return {{image.handle, access},
            std::make_shared<Registry::Lease>(image.view.clone())};
  }
  void ordinary(const Image &image, Access access, int &calls) {
    admission.ordinary({{image.handle, access}}, [&] { ++calls; });
  }
  std::shared_ptr<std::recursive_mutex> gate =
      std::make_shared<std::recursive_mutex>();
  std::shared_ptr<int> destroyed = std::make_shared<int>(0);
  Registry registry{401};
  Admission admission{gate};
};

TEST_F(SharedImageReadAdmissionTest, OrdinaryChurnPrunesExpiredReverseEdges) {
  auto image = create();
  for (int i = 0; i < 257; ++i) {
    auto permit = admission.prepare({pin(image, Access::exclusive)});
    const auto [size, capacity] =
        SharedImageReadAdmissionTestAccess::index_size(admission, image.handle);
    EXPECT_EQ(size, 1);
    EXPECT_EQ(capacity, 1);
  }
  std::vector<std::shared_ptr<Admission::Permit>> live;
  std::size_t old_capacity = 0;
  int capacity_changes = 0;
  for (int i = 0; i < 257; ++i) {
    live.push_back(admission.prepare({pin(image, Access::exclusive)}));
    const auto [size, capacity] =
        SharedImageReadAdmissionTestAccess::index_size(admission, image.handle);
    EXPECT_EQ(size, live.size());
    if (capacity != old_capacity) {
      ++capacity_changes;
      old_capacity = capacity;
    }
  }
  EXPECT_LE(capacity_changes,
            10);  // geometric growth, not one allocation per edge
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  for (const auto &permit : live)
    EXPECT_THROW(permit->submit([] {}), std::invalid_argument);
  loan.reset();
  for (const auto &permit : live)
    EXPECT_NO_THROW(permit->submit([] {}));
}

TEST_F(SharedImageReadAdmissionTest,
       PreparedWriterRestoringReadBoundaryIsBlocked) {
  auto image = create();
  // Entry and final sampled layouts do not erase the middle storage write.
  auto writer = admission.prepare({pin(image, Access::sampled_at_boundary),
                                   pin(image, Access::exclusive),
                                   pin(image, Access::sampled_at_boundary)});
  auto sampler = admission.prepare({pin(image, Access::sampled_at_boundary)});
  int writes = 0;
  int reads = 0;
  writer->submit([&] { ++writes; });
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  ASSERT_EQ(loan->needs_boundary().size(), 1);
  EXPECT_FALSE(loan->ready());
  EXPECT_THROW(writer->submit([&] { ++writes; }), std::invalid_argument);
  EXPECT_THROW(sampler->submit([&] { ++reads; }), std::invalid_argument);
  loan->publish_boundary(std::make_shared<Boundary>());
  sampler->submit([&] { ++reads; });
  EXPECT_THROW(writer->submit([&] { ++writes; }), std::invalid_argument);
  loan.reset();
  writer->submit([&] { ++writes; });
  EXPECT_EQ(writes, 2);
  EXPECT_EQ(reads, 1);
}

TEST_F(SharedImageReadAdmissionTest,
       MultipleDomainsHoldIndependentReaderOwners) {
  auto image = create();
  auto writer = admission.prepare({pin(image, Access::exclusive)});
  auto first = admission.reserve({pin(image, Access::sampled_at_boundary)});
  first->publish_boundary(std::make_shared<Boundary>());
  auto second = admission.reserve({pin(image, Access::sampled_at_boundary)});
  EXPECT_TRUE(second->ready());
  EXPECT_TRUE(second->needs_boundary().empty());
  first.reset();
  EXPECT_THROW(writer->submit([] {}), std::invalid_argument);
  second.reset();
  EXPECT_NO_THROW(writer->submit([] {}));
}

TEST_F(SharedImageReadAdmissionTest, FanoutReusesExactlyThePublishedPrefix) {
  auto image = create();
  auto first = admission.reserve({pin(image, Access::sampled_at_boundary)});
  EXPECT_THROW(first->dependencies(), std::invalid_argument);
  EXPECT_THROW(first->publish_boundary({}), std::invalid_argument);
  auto prefix = std::make_shared<Boundary>();
  prefix->sequence = 71;
  first->publish_boundary(prefix);
  auto second = admission.reserve({pin(image, Access::sampled_at_boundary)});
  ASSERT_EQ(first->dependencies().size(), 1);
  ASSERT_EQ(second->dependencies().size(), 1);
  EXPECT_EQ(first->dependencies()[0], prefix);
  EXPECT_EQ(second->dependencies()[0], prefix);
  EXPECT_EQ(second->dependencies()[0]->sequence, 71);
  // Two consumers each receive the timeline prefix. A binary semaphore is
  // never consumed by this table, and publishing is not a host wait.
}

TEST_F(SharedImageReadAdmissionTest,
       MixedExistingAndNewImagesPreserveBothPrefixes) {
  auto a = create();
  auto b = create();
  auto first = admission.reserve({pin(a, Access::sampled_at_boundary)});
  auto a_prefix = std::make_shared<Boundary>();
  first->publish_boundary(a_prefix);
  auto combined = admission.reserve({pin(a, Access::sampled_at_boundary),
                                     pin(b, Access::sampled_at_boundary)});
  ASSERT_EQ(combined->needs_boundary().size(), 1);
  EXPECT_EQ(combined->needs_boundary()[0], b.handle);
  auto b_prefix = std::make_shared<Boundary>();
  combined->publish_boundary(b_prefix);
  auto dependencies = combined->dependencies();
  ASSERT_EQ(dependencies.size(), 2);
  EXPECT_NE(std::find(dependencies.begin(), dependencies.end(), a_prefix),
            dependencies.end());
  EXPECT_NE(std::find(dependencies.begin(), dependencies.end(), b_prefix),
            dependencies.end());
  first.reset();
  EXPECT_TRUE(combined->ready());
}

TEST_F(SharedImageReadAdmissionTest,
       WriterPreparedDuringLoanCannotSubmitEarly) {
  auto image = create();
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  auto writer = admission.prepare({pin(image, Access::exclusive)});
  EXPECT_THROW(writer->submit([] {}), std::invalid_argument);
  loan.reset();
  EXPECT_NO_THROW(writer->submit([] {}));
}

TEST_F(SharedImageReadAdmissionTest,
       UploadAndCanvasLayoutChangesShareOneGuard) {
  auto image = create();
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  int calls = 0;
  // Upload: transfer_dst; Canvas copy-source: transfer_src; offscreen target:
  // attachment. All three are exclusive even if final layout is shader_read.
  for (int path = 0; path < 3; ++path)
    EXPECT_THROW(ordinary(image, Access::exclusive, calls),
                 std::invalid_argument);
  EXPECT_EQ(calls, 0);
  ordinary(image, Access::sampled_at_boundary, calls);
  EXPECT_EQ(calls, 1);
  loan.reset();
  ordinary(image, Access::exclusive, calls);
  EXPECT_EQ(calls, 2);
}

TEST_F(SharedImageReadAdmissionTest,
       MultiImagePreflightDoesNotPartiallyRecord) {
  auto destination = create();
  auto source = create();
  auto loan = admission.reserve({pin(source, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  int transitions = 0;
  EXPECT_THROW(admission.ordinary({{destination.handle, Access::exclusive},
                                   {source.handle, Access::exclusive}},
                                  [&] { ++transitions; }),
               std::invalid_argument);
  EXPECT_EQ(transitions, 0);
  int writes = 0;
  ordinary(destination, Access::exclusive, writes);
  EXPECT_EQ(writes, 1);
}

TEST_F(SharedImageReadAdmissionTest,
       BlockerCountTracksAllImagesWithoutAliasing) {
  auto a = create();
  auto b = create();
  auto writer =
      admission.prepare({pin(a, Access::exclusive), pin(a, Access::exclusive),
                         pin(b, Access::exclusive)});
  auto a_loan = admission.reserve({pin(a, Access::sampled_at_boundary),
                                   pin(a, Access::sampled_at_boundary)});
  a_loan->publish_boundary(std::make_shared<Boundary>());
  auto b_loan = admission.reserve({pin(b, Access::sampled_at_boundary)});
  b_loan->publish_boundary(std::make_shared<Boundary>());
  a_loan.reset();
  EXPECT_THROW(writer->submit([] {}), std::invalid_argument);
  b_loan.reset();
  EXPECT_NO_THROW(writer->submit([] {}));
}

TEST_F(SharedImageReadAdmissionTest,
       PendingReservationCannotBeJoinedOrSampled) {
  auto image = create();
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  EXPECT_THROW(admission.reserve({pin(image, Access::sampled_at_boundary)}),
               std::invalid_argument);
  int calls = 0;
  EXPECT_THROW(ordinary(image, Access::sampled_at_boundary, calls),
               std::invalid_argument);
  loan.reset();
  ordinary(image, Access::exclusive, calls);
  EXPECT_EQ(calls, 1);
}

TEST_F(SharedImageReadAdmissionTest, ReservationFailureIsAtomicAcrossImages) {
  auto a = create();
  auto b = create();
  auto bad = pin(b, Access::sampled_at_boundary);
  ++bad.use.image.generation;
  EXPECT_THROW(admission.reserve({pin(a, Access::sampled_at_boundary), bad}),
               std::invalid_argument);
  EXPECT_THROW(admission.reserve({pin(a, Access::sampled_at_boundary),
                                  pin(b, Access::exclusive)}),
               std::invalid_argument);
  int writes = 0;
  ordinary(a, Access::exclusive, writes);
  ordinary(b, Access::exclusive, writes);
  EXPECT_EQ(writes, 2);
}

TEST_F(SharedImageReadAdmissionTest,
       IdentityIncludesRegistryDomainKindAndGeneration) {
  auto image = create();
  for (int part = 0; part < 4; ++part) {
    auto wrong = pin(image, Access::sampled_at_boundary);
    if (part == 0)
      ++wrong.use.image.domain;
    if (part == 1)
      ++wrong.use.image.kind;
    if (part == 2)
      ++wrong.use.image.index;
    if (part == 3)
      ++wrong.use.image.generation;
    EXPECT_THROW(admission.reserve({wrong}), std::invalid_argument);
  }
  auto no_owner = pin(image, Access::sampled_at_boundary);
  no_owner.owner.reset();
  EXPECT_THROW(admission.reserve({no_owner}), std::invalid_argument);
}

TEST_F(SharedImageReadAdmissionTest, PublicViewRetireKeepsExistingLeaseUsable) {
  auto image = create();
  auto sampler = admission.prepare({pin(image, Access::sampled_at_boundary)});
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  EXPECT_EQ(registry.retire(image.handle), Registry::Result::kSuccess);
  image.view.reset();
  EXPECT_EQ(registry.acquire(image.handle).first,
            Registry::Result::kInvalidHandle);
  EXPECT_EQ(*destroyed, 0);
  EXPECT_NO_THROW(sampler->submit([] {}));
  loan.reset();
  EXPECT_EQ(*destroyed, 0);
  sampler.reset();
  EXPECT_EQ(*destroyed, 1);
}

TEST_F(SharedImageReadAdmissionTest,
       LastSubmittedPrefixOwnsLoanAfterFacadeClose) {
  auto image = create();
  auto writer = admission.prepare({pin(image, Access::exclusive)});
  auto facade = admission.reserve({pin(image, Access::sampled_at_boundary)});
  facade->publish_boundary(std::make_shared<Boundary>());
  // The existing Vulkan instance payload / last-prefix retirement owns these
  // references in production. This model only tests the host lifetime rule.
  auto first_prefix = facade;
  auto last_prefix = facade;
  facade.reset();
  first_prefix.reset();
  EXPECT_THROW(writer->submit([] {}), std::invalid_argument);
  last_prefix.reset();
  EXPECT_NO_THROW(writer->submit([] {}));
}

TEST_F(SharedImageReadAdmissionTest,
       FailedPrepareDoesNotUndoSubmittedLayoutFact) {
  auto image = create();
  enum class Layout { attachment, shader_read, transfer_dst };
  Layout tracked = Layout::attachment;
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  // Simulate a successfully completed ordinary transition followed by a later
  // preparation failure. Only the reservation rolls back; tracker does not.
  tracked = Layout::shader_read;
  loan.reset();
  EXPECT_EQ(tracked, Layout::shader_read);
  admission.ordinary({{image.handle, Access::exclusive}}, [&] {
    EXPECT_EQ(tracked, Layout::shader_read);
    tracked = Layout::transfer_dst;
  });
  EXPECT_EQ(tracked, Layout::transfer_dst);
}

TEST_F(SharedImageReadAdmissionTest,
       StalePermitDoesNotBecomeValidAfterSlotReuse) {
  auto image = create();
  auto old_handle = image.handle;
  auto old = admission.prepare({pin(image, Access::exclusive)});
  admission.remove_image(image.handle);
  EXPECT_THROW(old->submit([] {}), std::invalid_argument);
  registry.retire(image.handle);
  image.view.reset();
  old.reset();
  auto replacement = create();
  ASSERT_EQ(replacement.handle.index, old_handle.index);
  ASSERT_NE(replacement.handle.generation, old_handle.generation);
  EXPECT_THROW(admission.ordinary({{old_handle, Access::exclusive}}, [] {}),
               std::invalid_argument);
  EXPECT_NO_THROW(
      admission.ordinary({{replacement.handle, Access::exclusive}}, [] {}));
}

TEST_F(SharedImageReadAdmissionTest, UntrackCannotBypassReaderReservation) {
  auto image = create();
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  EXPECT_THROW(admission.remove_image(image.handle), std::invalid_argument);
  EXPECT_TRUE(loan->ready());
  loan.reset();
  EXPECT_NO_THROW(admission.remove_image(image.handle));
}

TEST_F(SharedImageReadAdmissionTest,
       ClosedRuntimeRejectsAdmissionsButKeepsOwners) {
  auto image = create();
  auto permit = admission.prepare({pin(image, Access::sampled_at_boundary)});
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  registry.retire(image.handle);
  image.view.reset();
  admission.close();
  EXPECT_FALSE(loan->ready());
  EXPECT_THROW(permit->submit([] {}), std::invalid_argument);
  EXPECT_THROW(loan->publish_boundary(std::make_shared<Boundary>()),
               std::invalid_argument);
  EXPECT_EQ(*destroyed, 0);
  permit.reset();
  EXPECT_EQ(*destroyed, 0);
  loan.reset();
  EXPECT_EQ(*destroyed, 1);
}

TEST_F(SharedImageReadAdmissionTest, AdmissionCheckAndSubmitShareOneGate) {
  auto image = create();
  auto permit = admission.prepare({pin(image, Access::exclusive)});
  bool other_thread_acquired = true;
  permit->submit([&] {
    std::thread other([&] {
      other_thread_acquired = gate->try_lock();
      if (other_thread_acquired)
        gate->unlock();
    });
    other.join();
  });
  EXPECT_FALSE(other_thread_acquired);
}

TEST_F(SharedImageReadAdmissionTest, ReentrantLoanCannotSplitCheckAndSubmit) {
  auto image = create();
  auto permit = admission.prepare({pin(image, Access::exclusive)});
  permit->submit([&] {
    EXPECT_THROW(admission.reserve({pin(image, Access::sampled_at_boundary)}),
                 std::invalid_argument);
  });
  EXPECT_THROW(
      permit->submit([] { throw std::runtime_error("record failed"); }),
      std::runtime_error);
  // Exception unwinds the submission depth as well as the gate.
  EXPECT_NO_THROW(admission.reserve({pin(image, Access::sampled_at_boundary)}));
}

TEST_F(SharedImageReadAdmissionTest,
       LoanOwnerDestructionRunsOutsideAdmissionGate) {
  bool destructor_saw_unlocked_gate = false;
  auto image = create([&] {
    std::thread other([&] {
      destructor_saw_unlocked_gate = gate->try_lock();
      if (destructor_saw_unlocked_gate)
        gate->unlock();
    });
    other.join();
  });
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  registry.retire(image.handle);
  image.view.reset();
  loan.reset();
  EXPECT_TRUE(destructor_saw_unlocked_gate);
}

TEST_F(SharedImageReadAdmissionTest, DeadRecordingEdgesDoNotOwnResourceLeases) {
  auto image = create();
  for (int i = 0; i < 64; ++i)
    admission.prepare({pin(image, Access::exclusive)});
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  loan->publish_boundary(std::make_shared<Boundary>());
  loan.reset();
  registry.retire(image.handle);
  image.view.reset();
  EXPECT_EQ(*destroyed, 1);
}

TEST(SharedImageReadAdmissionStandalone, LoanOutlivesRuntimeFacade) {
  auto gate = std::make_shared<std::recursive_mutex>();
  auto destroyed = std::make_shared<int>(0);
  Registry registry{402};
  auto [inserted, handle] = registry.emplace(1, destroyed);
  ASSERT_EQ(inserted, Registry::Result::kSuccess);
  auto [acquired, view] = registry.acquire(handle);
  ASSERT_EQ(acquired, Registry::Result::kSuccess);
  auto admission = std::make_unique<Admission>(gate);
  admission->register_image(handle);
  auto loan =
      admission->reserve({{{handle, Access::sampled_at_boundary},
                           std::make_shared<Registry::Lease>(view.clone())}});
  loan->publish_boundary(std::make_shared<Boundary>());
  registry.retire(handle);
  view.reset();
  admission.reset();
  EXPECT_FALSE(loan->ready());
  EXPECT_EQ(*destroyed, 0);
  loan.reset();
  EXPECT_EQ(*destroyed, 1);
}

TEST_F(SharedImageReadAdmissionTest, SingleUsePreflightDoesNotBypassLoan) {
  auto image = create();
  EXPECT_NO_THROW(admission.require_use({image.handle, Access::exclusive}));
  auto loan = admission.reserve({pin(image, Access::sampled_at_boundary)});
  EXPECT_THROW(
      admission.require_use({image.handle, Access::sampled_at_boundary}),
      std::invalid_argument);
  loan->publish_boundary(std::make_shared<Boundary>());
  EXPECT_NO_THROW(
      admission.require_use({image.handle, Access::sampled_at_boundary}));
  EXPECT_THROW(admission.require_use({image.handle, Access::exclusive}),
               std::invalid_argument);
  loan.reset();
  EXPECT_NO_THROW(admission.require_use({image.handle, Access::exclusive}));
}

TEST_F(SharedImageReadAdmissionTest,
       PhysicalUntrackAfterOwnerCloseRevokesOldPermit) {
  auto image = create();
  auto permit = admission.prepare({pin(image, Access::exclusive)});
  admission.close();
  EXPECT_NO_THROW(admission.remove_image(image.handle));
  EXPECT_THROW(permit->submit([] {}), std::invalid_argument);
  registry.retire(image.handle);
  image.view.reset();
  EXPECT_EQ(*destroyed, 0);
  permit.reset();
  EXPECT_EQ(*destroyed, 1);
}

TEST_F(SharedImageReadAdmissionTest,
       RuntimeAppendScopeRejectsReentrantReservation) {
  auto image = create();
  {
    auto scope = admission.ordinary_scope();
    EXPECT_NO_THROW(admission.require_use({image.handle, Access::exclusive}));
    EXPECT_THROW(admission.reserve({pin(image, Access::sampled_at_boundary)}),
                 std::invalid_argument);
  }
  EXPECT_NO_THROW(admission.reserve({pin(image, Access::sampled_at_boundary)}));
}

TEST_F(SharedImageReadAdmissionTest,
       PostPrefixFailureRestoresSamePreparedWriter) {
  auto image = create();
  auto writer = admission.prepare(
      {pin(image, Access::sampled_at_boundary), pin(image, Access::exclusive)});
  int calls = 0;
  writer->submit([&] { ++calls; });
  {
    auto failed = admission.reserve({pin(image, Access::sampled_at_boundary)});
    failed->publish_boundary(std::make_shared<Boundary>());
    EXPECT_THROW(writer->submit([&] { ++calls; }), std::invalid_argument);
    // A downstream command/instance preparation failure drops only the loan.
    // The same prepared writer can resume without rebuilding its permission.
  }
  EXPECT_NO_THROW(writer->submit([&] { ++calls; }));
  EXPECT_EQ(calls, 2);
}

}  // namespace
}  // namespace taichi::lang::gfx
