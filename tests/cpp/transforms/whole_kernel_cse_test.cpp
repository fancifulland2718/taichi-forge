#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {

TEST(WholeKernelCSE, SameAddressPointersKeepTheirPointeeTypes) {
  for (bool tensor_first : {false, true}) {
    SCOPED_TRACE(tensor_first);
    SNode place(1, SNodeType::place);
    place.dt = PrimitiveType::i32;
    auto block = std::make_unique<Block>();
    auto &types = TypeFactory::get_instance();
    const DataType tensor = types.get_tensor_type({4}, PrimitiveType::i32);
    const auto first_type = tensor_first ? tensor : PrimitiveType::i32;
    const auto second_type = tensor_first ? PrimitiveType::i32 : tensor;
    auto pointer = [&](DataType pointee) {
      auto ptr = block->push_back<GlobalPtrStmt>(&place, std::vector<Stmt *>{});
      ptr->ret_type = types.get_pointer_type(pointee);
      return ptr;
    };
    auto first = pointer(first_type);
    auto second = pointer(second_type);
    auto first_duplicate = pointer(first_type);
    auto second_duplicate = pointer(second_type);
    auto first_load = block->push_back<GlobalLoadStmt>(first_duplicate);
    first_load->ret_type = first_type;
    auto second_load = block->push_back<GlobalLoadStmt>(second_duplicate);
    second_load->ret_type = second_type;

    // Alias analysis must still recognize that these addresses overlap.
    EXPECT_TRUE(irpass::analysis::definitely_same_address(first, second));
    EXPECT_TRUE(irpass::whole_kernel_cse(block.get()));
    EXPECT_EQ(first_load->as<GlobalLoadStmt>()->src, first);
    EXPECT_EQ(second_load->as<GlobalLoadStmt>()->src, second);
    EXPECT_EQ(block->size(), 4);
    EXPECT_FALSE(irpass::whole_kernel_cse(block.get()));
  }
}

}  // namespace taichi::lang
