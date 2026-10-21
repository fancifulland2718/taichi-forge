#include "gtest/gtest.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/callable.h"

#include <sstream>

namespace taichi::lang {
namespace {

class ArgumentCallable : public Callable {
 public:
  ArgumentCallable() {
    ir = std::make_unique<Block>(this);
    arch = Arch::x64;
  }
  std::string get_name() const override {
    return "argument_prefix";
  }
};

FrontendForStmt *append_loop(Block *root, int id) {
  auto *loop = root->push_back<FrontendForStmt>(
      Expr(Identifier(id)), Expr(0), Expr(3), Arch::x64, ForLoopConfig{})
      ->as<FrontendForStmt>();
  loop->body = std::make_unique<Block>();
  loop->body->set_parent_stmt(loop);
  return loop;
}

std::string serialized(IRNode *root) {
  std::ostringstream output;
  gen_offline_cache_key(root, &output);
  return output.str();
}

}  // namespace

TEST(ReverseSegments, PreservesParameterPrefixAndReversesOnlyUserLoops) {
  ArgumentCallable callable;
  auto *root = callable.ir->as<Block>();
  auto *allocation = root->push_back<FrontendAllocaStmt>(
      Identifier(0), PrimitiveType::f32);
  auto *initialization = root->push_back<FrontendAssignStmt>(
      Expr(Identifier(0)), Expr(2.0f));
  callable.finish_argument_initialization();
  auto *first = append_loop(root, 1);
  auto *second = append_loop(root, 2);
  auto clone = root->clone();
  EXPECT_EQ(serialized(root), serialized(clone.get()));

  irpass::reverse_segments(root);
  ASSERT_EQ(root->size(), 4);
  EXPECT_EQ(root->statements[0].get(), allocation);
  EXPECT_EQ(root->statements[1].get(), initialization);
  EXPECT_EQ(root->statements[2].get(), second);
  EXPECT_EQ(root->statements[3].get(), first);
  irpass::reverse_segments(clone.get());
  EXPECT_EQ(serialized(root), serialized(clone.get()));
}

TEST(ReverseSegments, ParameterOriginContributesToTheCacheKey) {
  ArgumentCallable callable;
  auto *root = callable.ir->as<Block>();
  root->push_back<FrontendAllocaStmt>(Identifier(0), PrimitiveType::i32);
  const auto unmarked = serialized(root);
  callable.finish_argument_initialization();
  EXPECT_NE(unmarked, serialized(root));
}

TEST(ReverseSegments, UserInitializationStillFailsWithoutDamagingTheIr) {
  ArgumentCallable callable;
  auto *root = callable.ir->as<Block>();
  callable.finish_argument_initialization();
  auto *user = root->push_back<FrontendAllocaStmt>(
      Identifier(0), PrimitiveType::i32);
  auto *loop = append_loop(root, 1);
  EXPECT_ANY_THROW(irpass::reverse_segments(root));
  EXPECT_EQ(root->statements[0].get(), user);
  EXPECT_EQ(root->statements[1].get(), loop);
}

}  // namespace taichi::lang
