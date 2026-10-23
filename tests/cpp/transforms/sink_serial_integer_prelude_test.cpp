#include "gtest/gtest.h"

#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/transforms/sink_serial_integer_prelude.h"

namespace taichi::lang {
namespace {

struct PreludeFixture {
  SNode count{1, SNodeType::place};
  SNode output{1, SNodeType::place};
  std::unique_ptr<Block> root{std::make_unique<Block>()};
  OffloadedStmt *serial;
  OffloadedStmt *range;
  Stmt *value;
  Stmt *pointer{nullptr};
  Stmt *user;

  explicit PreludeFixture(bool field = true, Arch arch = Arch::cuda) {
    count.dt = PrimitiveType::i32;
    output.dt = PrimitiveType::i32;
    serial =
        root->push_back<OffloadedStmt>(OffloadedTaskType::serial, arch, nullptr)
            ->as<OffloadedStmt>();
    range = root->push_back<OffloadedStmt>(OffloadedTaskType::range_for, arch,
                                           nullptr)
                ->as<OffloadedStmt>();
    range->const_begin = range->const_end = true;
    range->end_value = 37;
    if (field) {
      pointer = field_ptr(serial->body.get(), &count);
      value = serial->body->push_back<GlobalLoadStmt>(pointer);
      value->ret_type = PrimitiveType::i32;
    } else {
      auto *arg = serial->body->push_back<ArgLoadStmt>(
          std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
      auto *limit = serial->body->push_back<ConstStmt>(TypedConstant(37));
      value =
          serial->body->push_back<BinaryOpStmt>(BinaryOpType::min, arg, limit);
      value->ret_type = PrimitiveType::i32;
    }
    user = range->body->push_back<UnaryOpStmt>(UnaryOpType::neg, value);
    user->ret_type = PrimitiveType::i32;
    auto *destination = field_ptr(range->body.get(), &output);
    range->body->push_back<GlobalStoreStmt>(destination, user);
  }

  static Stmt *field_ptr(Block *body, SNode *snode) {
    auto *result =
        body->push_back<GlobalPtrStmt>(snode, std::vector<Stmt *>{}, false);
    result->ret_type = TypeFactory::get_instance().get_pointer_type(snode->dt);
    return result;
  }

  bool run(std::unordered_set<Stmt *> hidden = {},
           AutodiffMode mode = AutodiffMode::kNone) {
    return irpass::sink_serial_integer_preludes(root.get(), mode, hidden);
  }
};

TEST(SerialIntegerPrelude,
     MovesWholeSnapshotAndPreservesIdentityAcrossBackends) {
  for (auto arch : {Arch::x64, Arch::cuda, Arch::vulkan}) {
    PreludeFixture f(true, arch);
    if (arch == Arch::x64) {
      EXPECT_FALSE(f.run());
      EXPECT_EQ(f.root->size(), 2);
      PreludeFixture byvalue(false, arch);
      EXPECT_TRUE(byvalue.run());
      EXPECT_EQ(byvalue.root->size(), 1);
      continue;
    }
    EXPECT_TRUE(f.run());
    ASSERT_EQ(f.root->size(), 1);
    EXPECT_EQ(f.root->back(), f.range);
    EXPECT_TRUE(f.serial->erased);
    EXPECT_EQ(f.serial->body->size(), 0);
    EXPECT_EQ(f.range->body->statements[0].get(), f.pointer);
    EXPECT_EQ(f.range->body->statements[1].get(), f.value);
    EXPECT_EQ(f.value->parent, f.range->body.get());
    EXPECT_EQ(f.user->operand(0), f.value);
    EXPECT_FALSE(f.run());
  }
}

TEST(SerialIntegerPrelude,
     NestedIntegerBranchBodyKeepsSnapshotReadOnly) {
  // A read-only count guards nested integer interval checks.
  // The output tree is distinct from every source tree.
  PreludeFixture f;
  SNode lower{1, SNodeType::place};
  SNode upper{1, SNodeType::place};
  lower.dt = upper.dt = PrimitiveType::u64;
  lower.num_active_indices = upper.num_active_indices = 1;
  f.output.num_active_indices = 1;
  f.range->body->statements.clear();
  auto *index = f.range->body->push_back<LoopIndexStmt>(f.range, 0);
  index->ret_type = PrimitiveType::i32;
  auto *condition = f.range->body->push_back<BinaryOpStmt>(BinaryOpType::cmp_lt,
                                                           index, f.value);
  condition->ret_type = PrimitiveType::u1;
  auto *active = f.range->body->push_back<IfStmt>(condition)->as<IfStmt>();
  active->set_true_statements(std::make_unique<Block>());
  auto *body = active->true_statements.get();
  auto load_key = [&](SNode *snode) {
    auto *ptr = body->push_back<GlobalPtrStmt>(
        snode, std::vector<Stmt *>{index}, false);
    ptr->ret_type = TypeFactory::get_instance().get_pointer_type(snode->dt);
    auto *load = body->push_back<GlobalLoadStmt>(ptr);
    load->ret_type = PrimitiveType::u64;
    return load;
  };
  auto *low = load_key(&lower);
  auto *high = load_key(&upper);
  auto *status = body->push_back<AllocaStmt>(PrimitiveType::i32);
  auto *zero = body->push_back<ConstStmt>(TypedConstant(0));
  body->push_back<LocalStoreStmt>(status, zero);
  auto *sentinel = body->push_back<ConstStmt>(TypedConstant(uint64_t(-1)));
  auto *valid =
      body->push_back<BinaryOpStmt>(BinaryOpType::cmp_ne, low, sentinel);
  valid->ret_type = PrimitiveType::u1;
  auto *valid_if = body->push_back<IfStmt>(valid)->as<IfStmt>();
  valid_if->set_true_statements(std::make_unique<Block>());
  auto *valid_body = valid_if->true_statements.get();
  auto *one = valid_body->push_back<ConstStmt>(TypedConstant(1));
  valid_body->push_back<LocalStoreStmt>(status, one);
  auto *nonempty =
      valid_body->push_back<BinaryOpStmt>(BinaryOpType::cmp_lt, low, high);
  nonempty->ret_type = PrimitiveType::u1;
  auto *nonempty_if = valid_body->push_back<IfStmt>(nonempty)->as<IfStmt>();
  nonempty_if->set_true_statements(std::make_unique<Block>());
  auto *two =
      nonempty_if->true_statements->push_back<ConstStmt>(TypedConstant(2));
  nonempty_if->true_statements->push_back<LocalStoreStmt>(status, two);
  auto *selected = body->push_back<LocalLoadStmt>(status);
  selected->ret_type = PrimitiveType::i32;
  auto *destination = body->push_back<GlobalPtrStmt>(
      &f.output, std::vector<Stmt *>{index}, true);
  destination->ret_type =
      TypeFactory::get_instance().get_pointer_type(PrimitiveType::i32);
  body->push_back<GlobalStoreStmt>(destination, selected);
  EXPECT_TRUE(f.run());
  EXPECT_EQ(f.root->size(), 1);
  EXPECT_EQ(condition->operand(1), f.value);
}

TEST(SerialIntegerPrelude, ByValueIntegerDagAndFloatLiteralAreAllowed) {
  PreludeFixture f(false);
  auto *literal = f.serial->body->push_back<ConstStmt>(TypedConstant(1.25f));
  auto *other = f.root
                    ->push_back<OffloadedStmt>(OffloadedTaskType::serial,
                                               Arch::x64, nullptr)
                    ->as<OffloadedStmt>();
  other->body->push_back<UnaryOpStmt>(UnaryOpType::neg, literal);
  EXPECT_TRUE(f.run());
  EXPECT_EQ(literal->parent, f.range->body.get());
  EXPECT_EQ(f.root->size(), 2);
}

TEST(SerialIntegerPrelude, TopLevelShapeIsAnImmutableLeaf) {
  PreludeFixture f(false);
  auto *shape = f.serial->body->push_back<ExternalTensorShapeAlongAxisStmt>(
      0, std::vector<int>{1});
  shape->ret_type = PrimitiveType::i32;
  f.range->body->push_back<UnaryOpStmt>(UnaryOpType::neg, shape);
  EXPECT_TRUE(f.run());
}

TEST(SerialIntegerPrelude, ZeroTripSnapshotAndIntegerDagNeedNoExtraTask) {
  for (bool field : {false, true}) {
    PreludeFixture f(field);
    f.range->end_value = 0;
    EXPECT_TRUE(f.run());
    EXPECT_EQ(f.range->begin_value, 0);
    EXPECT_EQ(f.range->end_value, 0);
  }
}

TEST(SerialIntegerPrelude, PreservesAutodiffAndNonconstantRanges) {
  for (auto mode : {AutodiffMode::kForward, AutodiffMode::kReverse,
                    AutodiffMode::kCheckAutodiffValid}) {
    PreludeFixture f;
    EXPECT_FALSE(f.run({}, mode));
    EXPECT_EQ(f.root->size(), 2);
  }
  for (bool begin : {false, true}) {
    PreludeFixture f;
    (begin ? f.range->const_begin : f.range->const_end) = false;
    EXPECT_FALSE(f.run());
  }
}

TEST(SerialIntegerPrelude, PinsBothImplicitRangeMapsAndEndStmt) {
  PreludeFixture f;
  EXPECT_FALSE(f.run({f.value}));
  auto *later = f.root
                    ->push_back<OffloadedStmt>(OffloadedTaskType::range_for,
                                               Arch::x64, nullptr)
                    ->as<OffloadedStmt>();
  later->end_stmt = f.value;
  EXPECT_FALSE(f.run());
}

TEST(SerialIntegerPrelude, RejectsOtherTaskAndAuxiliaryBlockConsumers) {
  PreludeFixture f;
  auto *other = f.root
                    ->push_back<OffloadedStmt>(OffloadedTaskType::serial,
                                               Arch::x64, nullptr)
                    ->as<OffloadedStmt>();
  other->tls_prologue = std::make_unique<Block>();
  other->tls_prologue->set_parent_stmt(other);
  other->tls_prologue->push_back<UnaryOpStmt>(UnaryOpType::neg, f.value);
  EXPECT_FALSE(f.run());
  other->tls_prologue.reset();
  f.range->bls_epilogue = std::make_unique<Block>();
  f.range->bls_epilogue->set_parent_stmt(f.range);
  EXPECT_FALSE(f.run());
}

TEST(SerialIntegerPrelude, KeepsControlStoresAndAllocationsInPrefix) {
  for (int effect = 0; effect < 3; ++effect) {
    PreludeFixture f;
    if (effect == 0) {
      f.serial->body->push_back<GlobalStoreStmt>(f.pointer, f.value);
    } else if (effect == 1) {
      f.serial->body->push_back<AllocaStmt>(PrimitiveType::i32);
    } else {
      f.serial->body->push_back<IfStmt>(f.value);
    }
    EXPECT_FALSE(f.run());
    EXPECT_FALSE(f.serial->erased);
    EXPECT_EQ(f.value->parent, f.serial->body.get());
  }
}

TEST(SerialIntegerPrelude, RejectsDivisionShiftAndArgpackPrefix) {
  for (auto op :
       {BinaryOpType::div, BinaryOpType::mod, BinaryOpType::bit_shl}) {
    PreludeFixture f(false);
    f.value->as<BinaryOpStmt>()->op_type = op;
    f.range->end_value = 0;
    EXPECT_FALSE(f.run());
  }
  PreludeFixture f(false);
  f.serial->body->statements[0]->as<ArgLoadStmt>()->arg_depth = 1;
  EXPECT_FALSE(f.run());
}

TEST(SerialIntegerPrelude, RejectsSnapshotWritesIncludingNestedAtomic) {
  for (bool atomic : {false, true}) {
    PreludeFixture f;
    auto *branch = f.range->body->push_back<IfStmt>(f.value)->as<IfStmt>();
    branch->set_true_statements(std::make_unique<Block>());
    auto *same_leaf =
        PreludeFixture::field_ptr(branch->true_statements.get(), &f.count);
    if (atomic) {
      branch->true_statements->push_back<AtomicOpStmt>(AtomicOpType::add,
                                                       same_leaf, f.value);
    } else {
      branch->true_statements->push_back<GlobalStoreStmt>(same_leaf, f.value);
    }
    EXPECT_FALSE(f.run());
  }
}

TEST(SerialIntegerPrelude, AllowsIndependentFieldAtomicAndNestedControl) {
  PreludeFixture f;
  auto *branch = f.range->body->push_back<IfStmt>(f.value)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto *other_leaf =
      PreludeFixture::field_ptr(branch->true_statements.get(), &f.output);
  branch->true_statements->push_back<AtomicOpStmt>(AtomicOpType::add,
                                                   other_leaf, f.value);
  EXPECT_TRUE(f.run());
}

TEST(SerialIntegerPrelude, RejectsExternalAliasAndUnknownPointerEvenForRead) {
  for (bool argument : {false, true}) {
    PreludeFixture f;
    Stmt *unknown;
    if (argument) {
      unknown = f.range->body->push_back<ArgLoadStmt>(
          std::vector<int>{0}, PrimitiveType::i32, true, true, 0);
    } else {
      unknown = f.range->body->push_back<ExternalTensorBasePtrStmt>(
          std::vector<int>{0}, false);
    }
    unknown->ret_type =
        TypeFactory::get_instance().get_pointer_type(PrimitiveType::i32);
    auto *read = f.range->body->push_back<GlobalLoadStmt>(unknown);
    read->ret_type = PrimitiveType::i32;
    EXPECT_FALSE(f.run());
  }
}

TEST(SerialIntegerPrelude, RejectsNestedFieldMatrixPointerAndOpaqueConsumer) {
  PreludeFixture f;
  auto *zero = f.range->body->push_back<ConstStmt>(TypedConstant(0));
  auto *first = f.range->body->push_back<MatrixPtrStmt>(f.pointer, zero);
  // A nested tensor pointer must retain a tensor pointee until its next index.
  first->ret_type = TypeFactory::get_instance().get_pointer_type(
      TypeFactory::create_tensor_type({1}, PrimitiveType::i32));
  auto *nested = f.range->body->push_back<MatrixPtrStmt>(first, zero);
  nested->ret_type = f.pointer->ret_type;
  f.range->body->push_back<GlobalStoreStmt>(nested, f.value);
  EXPECT_FALSE(f.run());

  PreludeFixture opaque;
  opaque.range->body->push_back<PrintStmt>(opaque.value);
  EXPECT_FALSE(opaque.run());
}

TEST(SerialIntegerPrelude, RejectsSparsePackedIndexedAndActivatingSnapshot) {
  for (int kind = 0; kind < 5; ++kind) {
    PreludeFixture f;
    if (kind == 0)
      f.count.is_path_all_dense = false;
    else if (kind == 1)
      f.count.is_bit_level = true;
    else if (kind == 2)
      f.count.num_active_indices = 1;
    else if (kind == 3)
      f.pointer->as<GlobalPtrStmt>()->activate = true;
    else
      f.count.dt = PrimitiveType::f32;
    EXPECT_FALSE(f.run());
  }
}

TEST(SerialIntegerPrelude, RejectsSparseDestinationAndSharedStorage) {
  PreludeFixture sparse;
  sparse.output.is_path_all_dense = false;
  EXPECT_FALSE(sparse.run());
  PreludeFixture shared;
  auto *allocation = shared.range->body->push_back<AllocaStmt>(
      std::vector<int>{4}, PrimitiveType::i32, true);
  EXPECT_TRUE(allocation->as<AllocaStmt>()->is_shared);
  EXPECT_FALSE(shared.run());
}

TEST(SerialIntegerPrelude, RejectsForwardReferencesAndUnusedNontrivialWork) {
  PreludeFixture f(false);
  auto *arg = f.serial->body->statements[0].get();
  f.value->set_operand(0, f.value);
  EXPECT_FALSE(f.run());
  f.value->set_operand(0, arg);
  f.serial->body->push_back<ArgLoadStmt>(std::vector<int>{1},
                                         PrimitiveType::i32, false, true, 0);
  EXPECT_FALSE(f.run());
}

}  // namespace
}  // namespace taichi::lang
