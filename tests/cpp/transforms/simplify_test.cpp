#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

// Basic tests within a basic block

TEST(Simplify, IndexedLoadsPreserveStoreBarriers) {
  for (bool advanced : {false, true}) {
    for (bool in_branch : {false, true}) {
      SCOPED_TRACE(advanced);
      SCOPED_TRACE(in_branch);
      auto block = std::make_unique<Block>();
      auto ptr = block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
      auto other = block->push_back<GlobalTemporaryStmt>(4, PrimitiveType::i32);
      auto one = block->push_back<ConstStmt>(TypedConstant(1));
      auto first = block->push_back<GlobalLoadStmt>(ptr);
      auto unrelated = block->push_back<GlobalLoadStmt>(other);
      auto duplicate = block->push_back<GlobalLoadStmt>(ptr);
      auto writes = block.get();
      if (in_branch) {
        auto branch = block->push_back<IfStmt>(one)->as<IfStmt>();
        branch->set_true_statements(std::make_unique<Block>());
        writes = branch->true_statements.get();
      }
      writes->push_back<GlobalStoreStmt>(ptr, one);
      auto after = block->push_back<GlobalLoadStmt>(ptr);
      auto result = block->push_back<ReturnStmt>(
          std::vector<Stmt *>{first, unrelated, duplicate, after});
      CompileConfig config;
      config.advanced_optimization = advanced;
      irpass::type_check(block.get(), config);
      irpass::simplify(block.get(), config);
      EXPECT_EQ(result->operand(0), first);
      EXPECT_EQ(result->operand(1), unrelated);
      EXPECT_EQ(result->operand(2), first);
      EXPECT_EQ(result->operand(3), after);
      EXPECT_NO_THROW(irpass::analysis::verify(block.get()));
    }
  }
}

TEST(Simplify, SimplifyLinearizedWithTrivialInputs) {
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto get_root = block->push_back<GetRootStmt>();
  auto linearized_empty = block->push_back<LinearizeStmt>(std::vector<Stmt *>(),
                                                          std::vector<int>());
  SNode root(0, SNodeType::root);
  root.insert_children(SNodeType::dense);
  auto lookup = block->push_back<SNodeLookupStmt>(&root, get_root,
                                                  linearized_empty, false);
  auto get_child = block->push_back<GetChStmt>(lookup, 0);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto linearized_zero = block->push_back<LinearizeStmt>(
      std::vector<Stmt *>(2, zero), std::vector<int>({8, 4}));
  [[maybe_unused]] auto lookup2 = block->push_back<SNodeLookupStmt>(
      root.ch[0].get(), get_child, linearized_zero, true);

  irpass::type_check(block.get(), test_prog.prog()->compile_config());
  EXPECT_EQ(block->size(), 7);

  irpass::simplify(
      block.get(),
      test_prog.prog()->compile_config());  // should lower linearized
  // EXPECT_EQ(block->size(), 11);  // not required to check size here

  irpass::constant_fold(block.get());
  irpass::alg_simp(block.get(), test_prog.prog()->compile_config());
  irpass::die(block.get());  // should eliminate consts
  irpass::simplify(block.get(), test_prog.prog()->compile_config());
  irpass::whole_kernel_cse(block.get());
  if (test_prog.prog()->compile_config().advanced_optimization) {
    // get root, const 0, lookup, get child, lookup
    EXPECT_EQ(block->size(), 5);
  }
}

}  // namespace taichi::lang
