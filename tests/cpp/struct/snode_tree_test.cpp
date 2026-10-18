#include "gtest/gtest.h"

#include <iterator>
#include <thread>

#include "taichi/runtime/snode_tree_id_registry.h"
#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

TEST(SNodeTree, GetSNodeToRootMapping) {
  constexpr int kSNodeSize = 16;
  SNode root{/*depth=*/0, /*t=*/SNodeType::root};
  const std::vector<Axis> axes = {Axis{0}};
  std::vector<int> all_snode_ids;
  for (int i = 0; i < 3; ++i) {
    auto &ptr_snode = root.pointer(axes, kSNodeSize);
    auto &dense_snode = ptr_snode.dense(axes, kSNodeSize);
    auto &leaf_snode = dense_snode.insert_children(SNodeType::place);
    all_snode_ids.push_back(ptr_snode.id);
    all_snode_ids.push_back(dense_snode.id);
    all_snode_ids.push_back(leaf_snode.id);
  }

  auto map = get_snodes_to_root_id(root);
  EXPECT_EQ(map.size(), 1 + 3 * 3);
  for (int id : all_snode_ids) {
    EXPECT_EQ(map.at(id), root.id);
  }
}

TEST(SNodeTreeIdRegistry, FailedLegacyReservationDoesNotClaimAnyIds) {
  SNodeTreeIdRegistry registry;
  EXPECT_TRUE(registry.try_reserve(3));
  EXPECT_ANY_THROW(registry.reserve(std::vector<int>{1, 3}));
  EXPECT_FALSE(registry.contains(1));
  EXPECT_TRUE(registry.contains(3));
  EXPECT_ANY_THROW(registry.reserve(std::vector<int>{2, 2}));
  EXPECT_ANY_THROW(registry.reserve(std::vector<int>{2, 1}));
  EXPECT_ANY_THROW(registry.reserve(std::vector<int>{-1}));
  EXPECT_EQ(registry.reserve(2), (std::vector<int>{0, 1}));
  registry.release(0);
  EXPECT_EQ(registry.reserve(2), (std::vector<int>{0, 2}));
}

TEST(SNodeTreeIdRegistry, ConcurrentJitAndModuleReservationsDoNotAlias) {
  SNodeTreeIdRegistry registry;
  std::vector<int> module_ids;
  std::vector<int> jit_ids;
  std::thread module([&] { module_ids = registry.reserve(16); });
  std::thread jit([&] {
    for (int id = 0; jit_ids.size() < 16; ++id) {
      if (registry.try_reserve(id)) jit_ids.push_back(id);
    }
  });
  module.join();
  jit.join();
  std::vector<int> overlap;
  std::set_intersection(module_ids.begin(), module_ids.end(), jit_ids.begin(),
                        jit_ids.end(), std::back_inserter(overlap));
  EXPECT_TRUE(overlap.empty());
  registry.release(module_ids);
  registry.release(jit_ids);
  EXPECT_EQ(registry.reserve(1), std::vector<int>{0});
}

}  // namespace taichi::lang
