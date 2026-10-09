#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace {

using Facts = std::unordered_set<Stmt *>;

// Independent hash-set fixed point used as an oracle for the compact solver.
// Compare every node, including joins and backedges, in both directions.
void compare_reference(ControlFlowGraph &cfg, bool forward) {
  std::unordered_map<CFGNode *, Facts> inputs, outputs;
  for (const auto &node : cfg.nodes)
    outputs[node.get()] = forward ? node->reach_gen : node->live_gen;
  bool changed;
  std::size_t iterations = 0;
  do {
    changed = false;
    // The deliberately simple sweep can move a fact only one edge per round
    // in the backward analysis. Allow a full path through a larger fixture.
    ASSERT_LE(++iterations, cfg.size() + 1);
    for (const auto &owned : cfg.nodes) {
      auto node = owned.get();
      Facts input;
      for (auto neighbor : forward ? node->prev : node->next)
        input.insert(outputs[neighbor].begin(), outputs[neighbor].end());
      auto result = forward ? node->reach_gen : node->live_gen;
      for (auto stmt : input) {
        bool killed = CFGNode::contain_variable(node->live_kill, stmt);
        if (forward) {
          auto destinations = irpass::analysis::get_store_destination(stmt);
          killed =
              destinations.empty() ? node->reach_kill_variable(stmt) : true;
          for (auto dest : destinations)
            killed &= node->reach_kill_variable(dest);
        }
        if (!killed)
          result.insert(stmt);
      }
      changed |= result != outputs[node];
      inputs[node] = std::move(input);
      outputs[node] = std::move(result);
    }
  } while (changed);
  for (const auto &node : cfg.nodes) {
    const auto &input = forward ? node->reach_in : node->live_out;
    const auto &output = forward ? node->reach_out : node->live_in;
    EXPECT_EQ(Facts(input.begin(), input.end()), inputs[node.get()]);
    EXPECT_EQ(Facts(output.begin(), output.end()), outputs[node.get()]);
  }
}

TEST(CFGDataflow, MatchesReferenceAcrossBranchesLoopsAndAliases) {
  for (bool lowered : {false, true}) {
    SCOPED_TRACE(lowered);
    auto root = std::make_unique<Block>();
    auto zero = root->push_back<ConstStmt>(TypedConstant(0));
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto four = root->push_back<ConstStmt>(TypedConstant(4));
    auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
    auto tensor = root->push_back<AllocaStmt>(
        TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32));
    auto element = root->push_back<MatrixPtrStmt>(tensor, zero);
    auto alias = root->push_back<MatrixPtrStmt>(tensor, zero);
    auto other = root->push_back<MatrixPtrStmt>(tensor, one);
    auto global = root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
    auto global_alias =
        root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
    root->push_back<LocalStoreStmt>(local, zero);
    root->push_back<LocalStoreStmt>(element, one);
    root->push_back<GlobalStoreStmt>(global, one);
    auto loop = root->push_back<RangeForStmt>(
                        zero, four, std::make_unique<Block>(), 1, 1, 1, false)
                    ->as<RangeForStmt>();
    auto index = loop->body->push_back<LoopIndexStmt>(loop, 0);
    auto branch = loop->body->push_back<IfStmt>(index)->as<IfStmt>();
    branch->set_true_statements(std::make_unique<Block>());
    branch->set_false_statements(std::make_unique<Block>());
    branch->true_statements->push_back<LocalStoreStmt>(local, index);
    branch->true_statements->push_back<LocalStoreStmt>(alias, zero);
    branch->true_statements->push_back<GlobalStoreStmt>(global_alias, zero);
    branch->false_statements->push_back<LocalStoreStmt>(other, one);
    branch->false_statements->push_back<LocalLoadStmt>(local);
    loop->body->push_back<LocalLoadStmt>(element);
    root->push_back<LocalLoadStmt>(local);
    root->push_back<LocalLoadStmt>(alias);
    root->push_back<GlobalLoadStmt>(global);
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    cfg->reaching_definition_analysis(lowered);
    compare_reference(*cfg, true);
    cfg->live_variable_analysis(lowered, std::nullopt);
    compare_reference(*cfg, false);
    // Rerunning an analysis must replace its universe and facts cleanly.
    cfg->reaching_definition_analysis(lowered);
    compare_reference(*cfg, true);
  }
}

TEST(CFGDataflow, EmptyGraphAndWordBoundaries) {
  auto root = std::make_unique<Block>();
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->reaching_definition_analysis(false);
  cfg->live_variable_analysis(false, std::nullopt);
  compare_reference(*cfg, true);
  compare_reference(*cfg, false);

  auto universe = std::make_shared<CFGStmtSet::Universe>();
  std::vector<Stmt *> statements;
  for (int i = 0; i < 130; ++i) {
    auto stmt = root->push_back<ConstStmt>(TypedConstant(i));
    statements.push_back(stmt);
    universe->insert(stmt);
  }
  CFGStmtSet input, gen, kill, output;
  for (auto set : {&input, &gen, &kill, &output})
    set->reset(universe);
  for (int i : {0, 63, 64, 65, 127, 129})
    input.insert(statements[i]);
  kill.insert(statements[64]);
  kill.insert(statements[129]);
  gen.insert(statements[1]);
  gen.insert(statements[64]);
  EXPECT_TRUE(output.transfer(input, gen, kill));
  EXPECT_FALSE(output.transfer(input, gen, kill));
  EXPECT_EQ(Facts(output.begin(), output.end()),
            Facts({statements[0], statements[1], statements[63], statements[64],
                   statements[65], statements[127]}));
  EXPECT_FALSE(output.contains(statements[129]));
  EXPECT_EQ(output.size(), 6);
  EXPECT_EQ(output.storage_bytes(), 3 * sizeof(uint64_t));
}

TEST(CFGDataflow, MultiDestinationDefinitionsRequireEveryDestinationKilled) {
  for (bool lowered : {false, true}) {
    SCOPED_TRACE(lowered);
    auto root = std::make_unique<Block>();
    auto value = root->push_back<ConstStmt>(TypedConstant(1));
    auto first = root->push_back<AllocaStmt>(PrimitiveType::i32);
    auto second = root->push_back<AllocaStmt>(PrimitiveType::i32);
    ControlFlowGraph cfg;
    auto start = cfg.push_back();
    int begin = root->size();
    // Duplicate destinations must not change the all-destinations rule.
    auto call = root->push_back<ExternalFuncCallStmt>(
        ExternalFuncCallStmt::SHARED_OBJECT, nullptr, "", "", "",
        std::vector<Stmt *>{}, std::vector<Stmt *>{first, second, second});
    auto definition =
        cfg.push_back(root.get(), begin, root->size(), false, nullptr);
    begin = root->size();
    root->push_back<LocalStoreStmt>(first, value);
    auto partial =
        cfg.push_back(root.get(), begin, root->size(), false, definition);
    begin = root->size();
    root->push_back<LocalStoreStmt>(first, value);
    root->push_back<LocalStoreStmt>(second, value);
    auto complete =
        cfg.push_back(root.get(), begin, root->size(), false, partial);
    begin = root->size();
    root->push_back<LocalLoadStmt>(first);
    root->push_back<LocalLoadStmt>(second);
    auto use = cfg.push_back(root.get(), begin, root->size(), false, complete);
    auto end = cfg.push_back();
    cfg.final_node = cfg.size() - 1;
    CFGNode::add_edge(start, definition);
    CFGNode::add_edge(definition, partial);
    CFGNode::add_edge(partial, complete);
    CFGNode::add_edge(complete, use);
    CFGNode::add_edge(use, end);
    cfg.reaching_definition_analysis(lowered);
    compare_reference(cfg, true);
    EXPECT_TRUE(partial->reach_out.contains(call));
    EXPECT_FALSE(complete->reach_out.contains(call));
    cfg.live_variable_analysis(lowered, std::nullopt);
    compare_reference(cfg, false);
  }
}

TEST(CFGDataflow, ConditionalStoresAcrossMultipleWords) {
  auto root = std::make_unique<Block>();
  auto value = root->push_back<ConstStmt>(TypedConstant(1));
  std::vector<Stmt *> variables;
  for (int i = 0; i < 130; ++i) {
    auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
    variables.push_back(local);
    auto branch = root->push_back<IfStmt>(value)->as<IfStmt>();
    branch->set_true_statements(std::make_unique<Block>());
    branch->true_statements->push_back<LocalStoreStmt>(local, value);
  }
  for (auto local : variables)
    root->push_back<LocalLoadStmt>(local);
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  cfg->reaching_definition_analysis(false);
  compare_reference(*cfg, true);
  cfg->live_variable_analysis(false, std::nullopt);
  compare_reference(*cfg, false);
}

TEST(CFGForwarding, ScalarDefinitionsSurviveRewritesAndErasure) {
  for (bool autodiff : {false, true}) {
    auto root = std::make_unique<Block>();
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
    auto other = root->push_back<AllocaStmt>(PrimitiveType::i32);
    auto zero_load = root->push_back<LocalLoadStmt>(local);
    root->push_back<LocalStoreStmt>(local, one);
    auto first = root->push_back<LocalLoadStmt>(local);
    root->push_back<LocalStoreStmt>(other, first);
    root->push_back<LocalStoreStmt>(local, first);  // identical, erased
    auto second = root->push_back<LocalLoadStmt>(other);
    auto sum = root->push_back<BinaryOpStmt>(BinaryOpType::add, first, second);
    root->push_back<LocalStoreStmt>(local, sum);
    auto last = root->push_back<LocalLoadStmt>(local);
    auto result = root->push_back<ReturnStmt>(
        std::vector<Stmt *>{zero_load, first, second, last});
    irpass::type_check(root.get(), CompileConfig{});
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    EXPECT_TRUE(cfg->store_to_load_forwarding(true, autodiff));
    EXPECT_TRUE(result->operand(0)->as<ConstStmt>()->val.equal_value(0));
    EXPECT_EQ(result->operand(1), one);
    EXPECT_EQ(result->operand(2), one);
    EXPECT_EQ(result->operand(3), sum);
    EXPECT_EQ(sum->operand(0), one);
    EXPECT_EQ(sum->operand(1), one);
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  }
}

TEST(CFGForwarding, ScalarJoinsRespectValuesAndVisibility) {
  for (bool same : {false, true}) {
    for (bool visible : {false, true}) {
      auto root = std::make_unique<Block>();
      auto cond = root->push_back<ArgLoadStmt>(
          std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
      auto one = root->push_back<ConstStmt>(TypedConstant(1));
      auto two = root->push_back<ConstStmt>(TypedConstant(2));
      auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
      auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
      branch->set_true_statements(std::make_unique<Block>());
      branch->set_false_statements(std::make_unique<Block>());
      auto lhs = visible ? one : branch->true_statements->push_back<ConstStmt>(
                                    TypedConstant(1));
      auto rhs = visible ? (same ? one : two)
                         : branch->false_statements->push_back<ConstStmt>(
                               TypedConstant(same ? 1 : 2));
      branch->true_statements->push_back<LocalStoreStmt>(local, lhs);
      branch->false_statements->push_back<LocalStoreStmt>(local, rhs);
      auto load = root->push_back<LocalLoadStmt>(local);
      auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{load});
      irpass::type_check(root.get(), CompileConfig{});
      auto cfg = irpass::analysis::build_cfg(root.get());
      cfg->simplify_graph();
      cfg->store_to_load_forwarding(true, false);
      EXPECT_EQ(result->operand(0), same && visible ? one : load);
      EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
    }
  }
}

TEST(CFGForwarding, UnknownScalarDefinitionsShadowKnownStores) {
  for (bool external : {false, true}) {
    auto root = std::make_unique<Block>();
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
    root->push_back<LocalStoreStmt>(local, one);
    if (external) {
      root->push_back<ExternalFuncCallStmt>(
          ExternalFuncCallStmt::SHARED_OBJECT, nullptr, "", "", "",
          std::vector<Stmt *>{}, std::vector<Stmt *>{local});
    } else {
      root->push_back<AtomicOpStmt>(AtomicOpType::add, local, one);
    }
    auto unknown = root->push_back<LocalLoadStmt>(local);
    root->push_back<LocalStoreStmt>(local, one);
    auto known = root->push_back<LocalLoadStmt>(local);
    auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{unknown, known});
    irpass::type_check(root.get(), CompileConfig{});
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    cfg->store_to_load_forwarding(true, false);
    EXPECT_EQ(result->operand(0), unknown);
    EXPECT_EQ(result->operand(1), one);
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  }
}

TEST(CFGForwarding, LoopCarriedScalarIsNotReplacedByItsInitialValue) {
  auto root = std::make_unique<Block>();
  auto zero = root->push_back<ConstStmt>(TypedConstant(0));
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto four = root->push_back<ConstStmt>(TypedConstant(4));
  auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto loop = root->push_back<RangeForStmt>(
                      zero, four, std::make_unique<Block>(), 1, 1, 1, false)
                  ->as<RangeForStmt>();
  auto load = loop->body->push_back<LocalLoadStmt>(local);
  auto sum = loop->body->push_back<BinaryOpStmt>(BinaryOpType::add, load, one);
  loop->body->push_back<LocalStoreStmt>(local, sum);
  auto final = root->push_back<LocalLoadStmt>(local);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{final});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  cfg->store_to_load_forwarding(true, false);
  EXPECT_EQ(sum->operand(0), load);
  EXPECT_EQ(result->operand(0), final);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGForwarding, TensorWritesKeepComponentAliasChecks) {
  for (bool dynamic : {false, true}) {
    auto root = std::make_unique<Block>();
    auto zero = root->push_back<ConstStmt>(TypedConstant(0));
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto two = root->push_back<ConstStmt>(TypedConstant(2));
    auto offset = dynamic ? root->push_back<ArgLoadStmt>(
                                std::vector<int>{0}, PrimitiveType::i32, false,
                                true, 0)
                          : one;
    auto type = TypeFactory::get_instance().get_tensor_type(
        {2}, PrimitiveType::i32);
    auto tensor = root->push_back<AllocaStmt>(type);
    auto values = root->push_back<MatrixInitStmt>(
        std::vector<Stmt *>{one, two});
    values->ret_type = type;
    root->push_back<LocalStoreStmt>(tensor, values);
    auto first = root->push_back<MatrixPtrStmt>(tensor, zero);
    auto other = root->push_back<MatrixPtrStmt>(tensor, offset);
    auto before = root->push_back<LocalLoadStmt>(first);
    root->push_back<LocalStoreStmt>(other, two);
    auto after = root->push_back<LocalLoadStmt>(first);
    auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{before, after});
    irpass::type_check(root.get(), CompileConfig{});
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    cfg->store_to_load_forwarding(true, false);
    EXPECT_EQ(result->operand(0), one);
    EXPECT_EQ(result->operand(1), dynamic ? after : one);
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  }
}

TEST(CFGForwarding, GlobalAliasesAndEntryFactsSurviveWriteFiltering) {
  auto root = std::make_unique<Block>();
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto two = root->push_back<ConstStmt>(TypedConstant(2));
  auto ptr = root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto alias = root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto entry = root->push_back<GlobalLoadStmt>(ptr);
  root->push_back<GlobalStoreStmt>(ptr, one);
  auto before = root->push_back<GlobalLoadStmt>(alias);
  auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
  root->push_back<LocalStoreStmt>(local, two);
  auto local_load = root->push_back<LocalLoadStmt>(local);
  root->push_back<GlobalStoreStmt>(alias, local_load);
  auto after = root->push_back<GlobalLoadStmt>(ptr);
  auto result = root->push_back<ReturnStmt>(
      std::vector<Stmt *>{entry, before, after});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  cfg->store_to_load_forwarding(false, false);
  EXPECT_EQ(result->operand(0), entry);
  EXPECT_EQ(result->operand(1), one);
  EXPECT_EQ(result->operand(2), two);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

}  // namespace
}  // namespace taichi::lang
