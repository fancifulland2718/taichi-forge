#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace {

TEST(AlgebraicUseIndex, InsertedPowerChainRemainsTrackedAcrossIterations) {
  auto root = std::make_unique<Block>();
  auto zero = root->push_back<ConstStmt>(TypedConstant(0));
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto assertion = root->push_back<AssertStmt>(
      one, "already true", std::vector<Stmt *>{});
  auto seven = root->push_back<ConstStmt>(TypedConstant(7));
  auto power = root->push_back<BinaryOpStmt>(BinaryOpType::pow, zero, seven);
  auto sum = root->push_back<BinaryOpStmt>(BinaryOpType::add, power, power);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{sum});
  CompileConfig config;
  irpass::type_check(root.get(), config);
  EXPECT_TRUE(irpass::alg_simp(root.get(), config));
  auto answer = result->operand(0)->cast<ConstStmt>();
  ASSERT_NE(answer, nullptr);
  EXPECT_TRUE(answer->val.equal_value(0));
  EXPECT_FALSE(answer->erased);
  EXPECT_TRUE(power->erased);
  EXPECT_TRUE(sum->erased);
  EXPECT_TRUE(assertion->erased);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  EXPECT_FALSE(irpass::alg_simp(root.get(), config));
}

TEST(AlgebraicUseIndex, InPlaceCastChangesAndResultReplacementShareIndex) {
  auto root = std::make_unique<Block>();
  auto address = root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto value = root->push_back<GlobalLoadStmt>(address);
  auto cast = [&](UnaryOpType op, Stmt *input, DataType type) {
    auto *stmt = root->push_back<UnaryOpStmt>(op, input)->as<UnaryOpStmt>();
    stmt->cast_type = type;
    return stmt;
  };
  auto wide = cast(UnaryOpType::cast_value, value, PrimitiveType::f64);
  auto narrow = cast(UnaryOpType::cast_value, wide, PrimitiveType::f32);
  auto bits = cast(UnaryOpType::cast_bits, value, PrimitiveType::f32);
  auto back = cast(UnaryOpType::cast_bits, bits, PrimitiveType::i32);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{narrow, back});
  CompileConfig config;
  config.fast_math = false;
  irpass::type_check(root.get(), config);
  EXPECT_TRUE(irpass::alg_simp(root.get(), config));
  EXPECT_EQ(narrow->operand, value);
  EXPECT_EQ(result->operand(0), narrow);
  EXPECT_EQ(result->operand(1), value);
  EXPECT_TRUE(back->erased);
  irpass::die(root.get());
  EXPECT_TRUE(wide->erased);
  EXPECT_TRUE(bits->erased);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(AlgebraicUseIndex, SubtreeRewritePreservesEnclosingDefinitions) {
  auto root = std::make_unique<Block>();
  auto address = root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto value = root->push_back<GlobalLoadStmt>(address);
  auto zero = root->push_back<ConstStmt>(TypedConstant(0));
  auto branch = root->push_back<IfStmt>(value)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto sum = branch->true_statements->push_back<BinaryOpStmt>(
      BinaryOpType::add, value, zero);
  auto result = branch->true_statements->push_back<ReturnStmt>(
      std::vector<Stmt *>{sum, sum});
  CompileConfig config;
  irpass::type_check(root.get(), config);
  EXPECT_TRUE(irpass::alg_simp(branch->true_statements.get(), config));
  EXPECT_EQ(result->operand(0), value);
  EXPECT_EQ(result->operand(1), value);
  EXPECT_TRUE(sum->erased);
  EXPECT_FALSE(value->erased);
  EXPECT_EQ(branch->cond, value);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(AlgebraicUseIndex, InsertedTensorConstantsRetainTheirOperands) {
  auto root = std::make_unique<Block>();
  auto value = root->push_back<ConstStmt>(TypedConstant(3));
  auto tensor =
      root->push_back<MatrixInitStmt>(std::vector<Stmt *>{value, value});
  tensor->ret_type =
      TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  auto difference =
      root->push_back<BinaryOpStmt>(BinaryOpType::sub, tensor, tensor);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{difference});
  CompileConfig config;
  irpass::type_check(root.get(), config);
  EXPECT_TRUE(irpass::alg_simp(root.get(), config));
  auto answer = result->operand(0)->cast<MatrixInitStmt>();
  ASSERT_NE(answer, nullptr);
  ASSERT_EQ(answer->values.size(), 2);
  for (auto *element : answer->values) {
    ASSERT_TRUE(element->is<ConstStmt>());
    EXPECT_TRUE(element->as<ConstStmt>()->val.equal_value(0));
    EXPECT_FALSE(element->erased);
  }
  irpass::die(root.get());
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

}  // namespace
}  // namespace taichi::lang
