#include "gtest/gtest.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/compile_config.h"

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

TEST(DelayedIRModifier, PreservesInsertionGroupsAndErasedOwnership) {
  Block block;
  auto *anchor = block.insert(make_const_i32(0));
  auto *last = block.insert(make_const_i32(9));
  DelayedIRModifier edits;
  VecStatement first;
  first.push_back(make_const_i32(1));
  first.push_back(make_const_i32(2));
  edits.insert_before(anchor, std::move(first));
  edits.insert_before(anchor, make_const_i32(3));
  VecStatement after;
  after.push_back(make_const_i32(4));
  after.push_back(make_const_i32(5));
  edits.insert_after(anchor, std::move(after));
  edits.insert_after(anchor, make_const_i32(6));
  edits.erase(last);
  edits.erase(anchor);
  EXPECT_TRUE(edits.modify_ir());
  std::vector<int> values;
  for (auto &stmt : block.statements) {
    values.push_back(stmt->as<ConstStmt>()->val.val_int32());
    EXPECT_EQ(stmt->parent, &block);
    EXPECT_FALSE(stmt->erased);
  }
  EXPECT_EQ(values, (std::vector<int>{1, 2, 3, 6, 4, 5}));
  EXPECT_TRUE(anchor->erased);
  EXPECT_TRUE(last->erased);
  EXPECT_EQ(block.trash_bin.size(), 2);
  EXPECT_FALSE(edits.modify_ir());
}

TEST(DelayedIRModifier, PreservesDependenciesOnNewAnchors) {
  for (bool before : {false, true}) {
    Block block;
    auto *anchor = block.insert(make_const_i32(0));
    auto middle = make_const_i32(1);
    auto *middle_ptr = middle.get();
    DelayedIRModifier edits;
    if (before) {
      edits.insert_before(anchor, std::move(middle));
      edits.insert_before(middle_ptr, make_const_i32(2));
    } else {
      edits.insert_after(anchor, std::move(middle));
      edits.insert_after(middle_ptr, make_const_i32(2));
    }
    EXPECT_TRUE(edits.modify_ir());
    std::vector<int> values;
    for (auto &stmt : block.statements) {
      values.push_back(stmt->as<ConstStmt>()->val.val_int32());
      EXPECT_EQ(stmt->parent, &block);
    }
    EXPECT_EQ(values, before ? (std::vector<int>{2, 1, 0})
                             : (std::vector<int>{0, 1, 2}));
  }
}

TEST(DelayedIRModifier, EditsChildrenOfErasedContainers) {
  Block block;
  auto *condition = block.insert(make_const_i32(1));
  auto branch = std::make_unique<IfStmt>(condition);
  branch->true_statements = std::make_unique<Block>();
  auto *child = branch->true_statements->insert(make_const_i32(2));
  auto *body = branch->true_statements.get();
  auto *branch_ptr = block.insert(std::move(branch));
  DelayedIRModifier edits;
  edits.insert_before(child, make_const_i32(3));
  edits.insert_after(child, make_const_i32(4));
  edits.insert_before(condition, make_const_i32(5));
  edits.erase(branch_ptr);
  edits.erase(child);
  edits.modify_ir();
  EXPECT_EQ(block.size(), 2);
  EXPECT_EQ(body->size(), 2);
  EXPECT_EQ(body->statements[0]->as<ConstStmt>()->val.val_int32(), 3);
  EXPECT_EQ(body->statements[1]->as<ConstStmt>()->val.val_int32(), 4);
  EXPECT_TRUE(branch_ptr->erased);
  EXPECT_TRUE(child->erased);
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
