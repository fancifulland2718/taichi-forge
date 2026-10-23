#include "taichi/rhi/interop/native_ready_gate.h"

#include <array>
#include <limits>

#include "gtest/gtest.h"

namespace taichi::lang {
namespace {

struct NativeGate {
  std::uint32_t struct_size;
  std::uint32_t schema_version;
  void *owner;
  std::int32_t (*retain_owner)(void *);
  void (*release_owner)(void *);
  std::int32_t (*query)(void *, std::uint64_t, std::uint32_t *);
};

struct NativeOwner {
  int retain_calls{0}, release_calls{0}, live_leases{0};
  std::int32_t retain_error{0}, query_error{0};
  std::uint32_t ready{0};
  std::size_t query_calls{0};
  std::array<std::uint64_t, 8> epochs{};
};

extern "C" {
std::int32_t native_ready_gate_test_retain(void *opaque) noexcept {
  auto &owner = *static_cast<NativeOwner *>(opaque);
  ++owner.retain_calls;
  if (!owner.retain_error)
    ++owner.live_leases;
  return owner.retain_error;
}

void native_ready_gate_test_release(void *opaque) noexcept {
  auto &owner = *static_cast<NativeOwner *>(opaque);
  ++owner.release_calls;
  --owner.live_leases;
}

std::int32_t native_ready_gate_test_query(void *opaque,
                                         std::uint64_t epoch,
                                         std::uint32_t *ready) noexcept {
  auto &owner = *static_cast<NativeOwner *>(opaque);
  if (owner.query_calls < owner.epochs.size())
    owner.epochs[owner.query_calls] = epoch;
  ++owner.query_calls;
  *ready = owner.ready;
  return owner.query_error;
}
}

NativeGate descriptor(NativeOwner &owner) {
  return {sizeof(NativeGate), 1, &owner, native_ready_gate_test_retain,
          native_ready_gate_test_release, native_ready_gate_test_query};
}

TEST(NativeReadyGate, InvalidDescriptorsRetainAndQueryFailuresDoNotLeak) {
  NativeOwner owner;
  const auto valid = descriptor(owner);
  EXPECT_NO_THROW(validate_native_ready_gate(valid));
  std::array<NativeGate, 7> invalid;
  invalid.fill(valid);
  --invalid[0].struct_size;
  ++invalid[1].struct_size;
  invalid[2].schema_version = 2;
  invalid[3].owner = nullptr;
  invalid[4].retain_owner = nullptr;
  invalid[5].release_owner = nullptr;
  invalid[6].query = nullptr;
  for (std::size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_THROW(validate_native_ready_gate(invalid[i]), std::invalid_argument);
    EXPECT_THROW(retain_native_ready_gate(invalid[i]), std::invalid_argument);
  }
  EXPECT_EQ(owner.retain_calls, 0);
  EXPECT_EQ(owner.release_calls, 0);
  EXPECT_EQ(owner.query_calls, 0u);
  EXPECT_EQ(owner.live_leases, 0);

  owner.retain_error = -7;
  EXPECT_THROW(retain_native_ready_gate(valid), std::invalid_argument);
  EXPECT_EQ(owner.retain_calls, 1);
  EXPECT_EQ(owner.release_calls, 0);  // A failed retain acquired no reference.
  EXPECT_EQ(owner.query_calls, 0u);
  EXPECT_EQ(owner.live_leases, 0);

  owner.retain_error = 0;
  {
    auto ready = retain_native_ready_gate(valid);
    EXPECT_EQ(owner.retain_calls, 2);
    EXPECT_EQ(owner.live_leases, 1);
    owner.ready = 2;
    EXPECT_THROW(ready(1), std::runtime_error);
    owner.ready = 1;
    owner.query_error = -9;
    EXPECT_THROW(ready(1), std::runtime_error);  // Error must override true.
    EXPECT_EQ(owner.release_calls, 0);
    EXPECT_EQ(owner.live_leases, 1);
    owner.query_error = 0;
    owner.ready = 0;
    EXPECT_FALSE(ready(1));
    owner.ready = 1;
    EXPECT_TRUE(ready(1));
    EXPECT_THROW(ready(0), std::invalid_argument);
    EXPECT_EQ(owner.query_calls, 4u);  // Zero is rejected before the callback.
    for (std::size_t i = 0; i < 4; ++i)
      EXPECT_EQ(owner.epochs[i], 1u);
  }
  EXPECT_EQ(owner.release_calls, 1);
  EXPECT_EQ(owner.live_leases, 0);
}

TEST(NativeReadyGate, CopiesShareOneLeaseAndPassEpochsWithoutAdvancing) {
  NativeOwner owner;
  std::function<bool(std::uint64_t)> survivor;
  {
    auto gate = descriptor(owner);
    auto original = retain_native_ready_gate(gate);
    auto copy = original;
    survivor = copy;
    gate = {};  // Callback/owner facts must be frozen by the cold adapter.
    EXPECT_EQ(owner.retain_calls, 1);
    EXPECT_EQ(owner.live_leases, 1);
    EXPECT_EQ(owner.query_calls, 0u);
    EXPECT_FALSE(original(1));
    owner.ready = 1;
    EXPECT_TRUE(copy(1));  // False did not advance this epoch.
    EXPECT_TRUE(survivor(17));
    original = {};
    copy = {};
    EXPECT_EQ(owner.release_calls, 0);
    EXPECT_EQ(owner.live_leases, 1);
  }
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  EXPECT_TRUE(survivor(maximum));
  EXPECT_TRUE(survivor(3));  // The adapter owns no monotonic epoch counter.
  EXPECT_EQ(owner.query_calls, 5u);
  const std::array<std::uint64_t, 5> expected{1, 1, 17, maximum, 3};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(owner.epochs[i], expected[i]);
  EXPECT_EQ(owner.retain_calls, 1);
  EXPECT_EQ(owner.release_calls, 0);
  survivor = {};
  EXPECT_EQ(owner.release_calls, 1);
  EXPECT_EQ(owner.live_leases, 0);
}

}  // namespace
}  // namespace taichi::lang
