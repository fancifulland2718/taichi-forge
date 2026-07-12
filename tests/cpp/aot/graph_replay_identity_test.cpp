#include <cstddef>
#include <new>

#include <gtest/gtest.h>
#include <taichi/aot/graph_data.h>

namespace taichi::lang::aot {

TEST(GraphReplayIdentityTest, TokenChangesWhenHostAddressIsReused) {
  alignas(CompiledGraphReplayIdentity)
      std::byte storage[sizeof(CompiledGraphReplayIdentity)];

  auto *first = ::new (storage) CompiledGraphReplayIdentity();
  const uint64_t first_value = first->value();
  first->~CompiledGraphReplayIdentity();

  auto *second = ::new (storage) CompiledGraphReplayIdentity();
  const uint64_t second_value = second->value();

  EXPECT_EQ(static_cast<const void *>(first),
            static_cast<const void *>(second));
  EXPECT_NE(first_value, 0);
  EXPECT_NE(second_value, 0);
  EXPECT_NE(first_value, second_value);

  second->~CompiledGraphReplayIdentity();
}

}  // namespace taichi::lang::aot
