#include "gtest/gtest.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {
namespace {

std::unique_ptr<ConstStmt> make_const_i32(int32_t value) {
  return Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32),
      value));
}

TEST(Block, Erase) {
  Block b;
  b.insert(make_const_i32(1));
  b.insert(make_const_i32(2));
  auto s3 = make_const_i32(3);
  auto s3_ptr = s3.get();
  b.insert(std::move(s3));
  b.erase(/*location=*/1);
  EXPECT_EQ(b.size(), 2);
  EXPECT_EQ(b.locate(s3_ptr), 1);
}

TEST(Block, EraseRange) {
  Block b;
  std::vector<ConstStmt *> stmt_ptrs;
  for (int i = 0; i < 5; ++i) {
    auto s = make_const_i32(i);
    stmt_ptrs.push_back(s.get());
    b.insert(std::move(s));
  }
  EXPECT_EQ(b.size(), 5);
  auto begin = b.find(stmt_ptrs[1]);
  auto end = b.find(stmt_ptrs[4]);
  b.erase_range(begin, end);
  EXPECT_EQ(b.size(), 2);
  EXPECT_EQ(b.locate(stmt_ptrs.front()), 0);
  EXPECT_EQ(b.locate(stmt_ptrs.back()), 1);
}

TEST(RangeFor, CloneWithBody) {
  auto begin = make_const_i32(1);
  auto end = make_const_i32(4);
  auto original_body = std::make_unique<Block>();
  original_body->insert(make_const_i32(9));
  RangeForStmt original(begin.get(), end.get(), std::move(original_body), true,
                        2, 64, true, "arg (0)");
  original.reversed = true;
  original.one_to_one = true;

  auto body = std::make_unique<Block>();
  auto body_ptr = body.get();
  auto copied = original.clone_with_body(std::move(body));
  EXPECT_EQ(copied->begin, begin.get());
  EXPECT_EQ(copied->end, end.get());
  EXPECT_TRUE(copied->reversed);
  EXPECT_TRUE(copied->is_bit_vectorized);
  EXPECT_EQ(copied->num_cpu_threads, 2);
  EXPECT_EQ(copied->block_dim, 64);
  EXPECT_TRUE(copied->strictly_serialized);
  EXPECT_TRUE(copied->one_to_one);
  EXPECT_EQ(copied->range_hint, "arg (0)");
  EXPECT_EQ(copied->body.get(), body_ptr);
  EXPECT_EQ(copied->body->parent_stmt(), copied.get());
  EXPECT_TRUE(copied->body->statements.empty());
  EXPECT_TRUE(copied->body->trash_bin.empty());
  EXPECT_EQ(original.body->size(), 1);

  auto cloned = original.clone();
  auto full = cloned->as<RangeForStmt>();
  EXPECT_EQ(full->body->size(), 1);
  EXPECT_NE(full->body->statements[0].get(), original.body->statements[0].get());
  EXPECT_EQ(full->body->parent_stmt(), full);
  EXPECT_EQ(full->range_hint, "arg (0)");
}

}  // namespace
}  // namespace taichi::lang
