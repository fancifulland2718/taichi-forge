#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/transforms/simplify.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

// Basic tests within a basic block

TEST(Simplify, FastLocalCSERewritesDependentExpressionsAndBranches) {
  auto block = std::make_unique<Block>();
  auto input = block->push_back<ArgLoadStmt>(
      std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto sum = block->push_back<BinaryOpStmt>(BinaryOpType::add, input, one);
  auto duplicate_one = block->push_back<ConstStmt>(TypedConstant(1));
  auto duplicate_sum =
      block->push_back<BinaryOpStmt>(BinaryOpType::add, input, duplicate_one);
  auto branch = block->push_back<IfStmt>(duplicate_sum)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto nested_return = branch->true_statements->push_back<ReturnStmt>(
      std::vector<Stmt *>{duplicate_sum});
  auto result =
      block->push_back<ReturnStmt>(std::vector<Stmt *>{sum, duplicate_sum});
  CompileConfig config;
  config.advanced_optimization = false;
  config.flatten_if = false;
  irpass::type_check(block.get(), config);
  irpass::full_simplify(block.get(), config, {true, false});
  EXPECT_EQ(result->operand(0), sum);
  EXPECT_EQ(result->operand(1), sum);
  EXPECT_EQ(nested_return->operand(0), sum);
  EXPECT_NO_THROW(irpass::analysis::verify(block.get()));
}

TEST(Simplify, FastLocalCSEPreservesSignedZeroAndTypes) {
  auto block = std::make_unique<Block>();
  auto positive = block->push_back<ConstStmt>(TypedConstant(0.0f));
  auto negative = block->push_back<ConstStmt>(TypedConstant(-0.0f));
  auto duplicate = block->push_back<ConstStmt>(TypedConstant(-0.0f));
  auto integer = block->push_back<ConstStmt>(TypedConstant(0));
  auto wide = block->push_back<ConstStmt>(TypedConstant(0.0));
  auto result = block->push_back<ReturnStmt>(
      std::vector<Stmt *>{positive, negative, duplicate, integer, wide});
  CompileConfig config;
  config.advanced_optimization = false;
  config.constant_folding = false;
  irpass::type_check(block.get(), config);
  irpass::full_simplify(block.get(), config, {true, false});
  EXPECT_EQ(result->operand(0), positive);
  EXPECT_EQ(result->operand(1), negative);
  EXPECT_EQ(result->operand(2), negative);
  EXPECT_EQ(result->operand(3), integer);
  EXPECT_EQ(result->operand(4), wide);
  EXPECT_NO_THROW(irpass::analysis::verify(block.get()));
}

TEST(Simplify, FastLocalCSEPreservesSparseActivationBoundaries) {
  for (bool activate : {false, true}) {
    for (int barrier : {0, 1, 2}) {
      SCOPED_TRACE(activate);
      SCOPED_TRACE(barrier);
      SNode root(0, SNodeType::root);
      auto sparse = &root.insert_children(SNodeType::pointer);
      auto block = std::make_unique<Block>();
      auto get_root = block->push_back<GetRootStmt>(&root);
      auto zero = block->push_back<ConstStmt>(TypedConstant(0));
      auto lookup =
          block->push_back<SNodeLookupStmt>(sparse, get_root, zero, activate);
      Block *effects = block.get();
      if (barrier == 2) {
        auto cond = block->push_back<ArgLoadStmt>(
            std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
        auto branch = block->push_back<IfStmt>(cond)->as<IfStmt>();
        branch->set_true_statements(std::make_unique<Block>());
        effects = branch->true_statements.get();
      }
      if (barrier != 0)
        effects->push_back<SNodeOpStmt>(SNodeOpType::deactivate, sparse, lookup);
      auto duplicate =
          block->push_back<SNodeLookupStmt>(sparse, get_root, zero, activate);
      auto result =
          block->push_back<ReturnStmt>(std::vector<Stmt *>{lookup, duplicate});
      CompileConfig config;
      config.advanced_optimization = false;
      config.flatten_if = false;
      irpass::type_check(block.get(), config);
      irpass::full_simplify(block.get(), config, {true, false});
      EXPECT_EQ(result->operand(0), lookup);
      EXPECT_EQ(result->operand(1),
                activate && barrier == 0 ? lookup : duplicate);
      EXPECT_NO_THROW(irpass::analysis::verify(block.get()));
    }
  }
}

TEST(Simplify, LoadsPreserveAtomicAndCallBarriers) {
  for (bool advanced : {false, true}) {
    for (bool nested : {false, true}) {
      for (int effect = 0; effect < 3; ++effect) {
        SCOPED_TRACE(advanced);
        SCOPED_TRACE(nested);
        SCOPED_TRACE(effect);
        auto block = std::make_unique<Block>();
        auto ptr = block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
        auto one = block->push_back<ConstStmt>(TypedConstant(1));
        auto first = block->push_back<GlobalLoadStmt>(ptr);
        auto *effects = block.get();
        if (nested) {
          auto condition = block->push_back<ArgLoadStmt>(
              std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
          auto branch = block->push_back<IfStmt>(condition)->as<IfStmt>();
          branch->set_true_statements(std::make_unique<Block>());
          effects = branch->true_statements.get();
        }
        if (effect == 0) {
          effects->push_back<AtomicOpStmt>(AtomicOpType::add, ptr, one);
        } else if (effect == 1) {
          effects->push_back<ExternalFuncCallStmt>(
              ExternalFuncCallStmt::ASSEMBLY, nullptr, "opaque", "", "",
              std::vector<Stmt *>{ptr}, std::vector<Stmt *>{});
        } else {
          // An opaque call is represented by a statement with global side effects.
          effects->push_back<InternalFuncStmt>("test_opaque_effect",
                                             std::vector<Stmt *>{});
        }
        auto after = block->push_back<GlobalLoadStmt>(ptr);
        auto result = block->push_back<ReturnStmt>(std::vector<Stmt *>{first, after});
        CompileConfig config;
        config.advanced_optimization = advanced;
        config.flatten_if = false;
        irpass::type_check(block.get(), config);
        irpass::simplify(block.get(), config);
        EXPECT_EQ(result->operand(0), first);
        EXPECT_EQ(result->operand(1), after);
        EXPECT_NO_THROW(irpass::analysis::verify(block.get()));
      }
    }
  }
}

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

TEST(Simplify, CFGPreservesOpaqueEffectsAndEscapedLocalStorage) {
  for (bool local : {false, true}) {
    auto block = std::make_unique<Block>();
    Stmt *ptr = local ? block->push_back<AllocaStmt>(PrimitiveType::i32)
                      : block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
    auto value = block->push_back<ConstStmt>(TypedConstant(7));
    if (local)
      block->push_back<LocalStoreStmt>(ptr, value);
    else
      block->push_back<GlobalStoreStmt>(ptr, value);
    block->push_back<InternalFuncStmt>("opaque_effect", std::vector<Stmt *>{ptr});
    Stmt *after = local ? block->push_back<LocalLoadStmt>(ptr)
                        : block->push_back<GlobalLoadStmt>(ptr);
    auto result = block->push_back<ReturnStmt>(std::vector<Stmt *>{after});
    irpass::type_check(block.get(), CompileConfig{});
    irpass::cfg_optimization(block.get(), false, false, false);
    EXPECT_EQ(result->operand(0), after);
    EXPECT_NO_THROW(irpass::analysis::verify(block.get()));
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
