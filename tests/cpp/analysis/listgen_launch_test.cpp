#include "gtest/gtest.h"

#include <limits>

#include "taichi/analysis/listgen_launch.h"

namespace taichi::lang {
namespace {

struct ListgenLaunchTest : ::testing::Test {
  SNode root, parent, child;

  void SetUp() override {
    root.type = SNodeType::root;
    parent.type = SNodeType::pointer;
    parent.parent = &root;
    child.type = SNodeType::bitmasked;
    child.parent = &parent;
    child.num_cells_per_container = 64;
  }
};

TEST_F(ListgenLaunchTest, RootMapsOutputSlicesToThreads) {
  child.parent = &root;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 1);
  child.num_cells_per_container = 64 * taichi_listgen_max_element_size;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 1);
  ++child.num_cells_per_container;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 2);
  child.num_cells_per_container = std::numeric_limits<int64>::max();
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 4080);
}

TEST_F(ListgenLaunchTest, NonrootMapsParentSlicesToBlocks) {
  parent.num_cells_per_container = 257;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 1);
  parent.num_cells_per_container = taichi_listgen_max_element_size + 1;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 2);
  parent.vk_max_active_hint = 1;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 2);
}

TEST_F(ListgenLaunchTest, NestedContainersMultiplyWithoutOverflow) {
  SNode ancestor;
  ancestor.type = SNodeType::pointer;
  ancestor.parent = &root;
  ancestor.num_cells_per_container = 33;
  parent.parent = &ancestor;
  parent.num_cells_per_container = 1025;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 66);
  ancestor.num_cells_per_container = std::numeric_limits<int64>::max();
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 4080);
}

TEST_F(ListgenLaunchTest, HashMappingsAreNotOrdinaryContainerMappings) {
  child.type = SNodeType::hash;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 0);
  child.type = SNodeType::dense;
  parent.type = SNodeType::hash;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 0);
}

TEST_F(ListgenLaunchTest, DynamicUsesDeclaredCapacity) {
  parent.type = SNodeType::dynamic;
  parent.num_cells_per_container = 1025;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 2);
}

TEST_F(ListgenLaunchTest, InvalidGeometryDoesNotClaimAProof) {
  EXPECT_EQ(llvm_listgen_grid_bound(nullptr, 64, 4080), 0);
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 0, 4080), 0);
  parent.num_cells_per_container = 0;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 0);
  parent.num_cells_per_container = 1;
  parent.parent = nullptr;
  EXPECT_EQ(llvm_listgen_grid_bound(&child, 64, 4080), 0);
}

}  // namespace
}  // namespace taichi::lang
