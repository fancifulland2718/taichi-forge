#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

#include <limits>
#include <unordered_map>

namespace taichi::lang {

namespace irpass {

namespace {

// A block's net change and maximum prefix change, for every stack it touches.
// Compose summaries once per structured region rather than solving a separate
// graph problem for every stack. Branch joins are conservative upper bounds.
struct StackEffect {
  int64 delta{0};
  int64 peak{0};
  bool known{true};
};

using StackEffects = std::unordered_map<AdStackAllocaStmt *, StackEffect>;

int64 checked_add(int64 a, int64 b) {
  TI_ASSERT_INFO(
      !((b > 0 && a > std::numeric_limits<int64>::max() - b) ||
        (b < 0 && a < std::numeric_limits<int64>::min() - b)),
      "Autodiff stack size exceeds addressable storage");
  return a + b;
}

int64 checked_multiply(int64 value, int64 count) {
  TI_ASSERT(count >= 0);
  if (count == 0)
    return 0;
  TI_ASSERT_INFO(value <= std::numeric_limits<int64>::max() / count &&
                     value >= std::numeric_limits<int64>::min() / count,
                 "Autodiff stack size exceeds addressable storage");
  return value * count;
}

class StructuredStackSize : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void append(AdStackAllocaStmt *stack, const StackEffect &next) {
    auto &current = effects_[stack];
    if (!current.known || !next.known) {
      current.known = false;
      return;
    }
    current.peak =
        std::max(current.peak, checked_add(current.delta, next.peak));
    current.delta = checked_add(current.delta, next.delta);
  }

  void append(const StackEffects &next) {
    for (auto &[stack, effect] : next)
      append(stack, effect);
  }

  StackEffects summarize(Block *block) {
    if (!block)
      return {};
    auto enclosing = std::move(effects_);
    effects_.clear();
    // This analysis does not mutate the statement list.
    for (auto &stmt : block->statements)
      stmt->accept(this);
    for (auto it = effects_.begin(); it != effects_.end();) {
      if (it->first->parent == block) {
        // A stack defined inside a loop is initialized on each iteration, so
        // its own capacity must not be multiplied by the enclosing trip count.
        capacities_[it->first] = it->second;
        it = effects_.erase(it);
      } else {
        ++it;
      }
    }
    auto result = std::move(effects_);
    effects_ = std::move(enclosing);
    return result;
  }

  void repeat(StackEffects effects, int64 count) {
    for (auto &[stack, effect] : effects) {
      if (count == 0) {
        effect = {};
      } else if (effect.known) {
        if (count < 0) {
          // Unknown iteration count: balanced or shrinking loops have a finite
          // peak, but positive growth still requires the existing fallback.
          effect.known = effect.delta <= 0;
          effect.delta = 0;  // zero iterations are possible
        } else {
          effect.peak = checked_add(
              effect.peak, checked_multiply(std::max<int64>(0, effect.delta),
                                            count - 1));
          effect.delta = checked_multiply(effect.delta, count);
        }
      }
      append(stack, effect);
    }
  }

  void visit(Block *block) override {
    append(summarize(block));
  }

  void visit(AdStackAllocaStmt *stmt) override {
    if (stmt->max_size == 0)
      effects_[stmt] = {};
  }

  void visit(AdStackPushStmt *stmt) override {
    auto stack = stmt->stack->as<AdStackAllocaStmt>();
    if (stack->max_size == 0)
      append(stack, {1, 1, true});
  }

  void visit(AdStackPopStmt *stmt) override {
    auto stack = stmt->stack->as<AdStackAllocaStmt>();
    if (stack->max_size == 0)
      append(stack, {-1, 0, true});
  }

  void visit(IfStmt *stmt) override {
    auto left = summarize(stmt->true_statements.get());
    auto right = summarize(stmt->false_statements.get());
    for (auto &[stack, effect] : left) {
      auto other = right.find(stack);
      StackEffect alternative =
          other == right.end() ? StackEffect{} : other->second;
      append(stack, {std::max(effect.delta, alternative.delta),
                     std::max(effect.peak, alternative.peak),
                     effect.known && alternative.known});
    }
    for (auto &[stack, effect] : right) {
      if (left.find(stack) == left.end())
        append(stack, {std::max<int64>(0, effect.delta), effect.peak, effect.known});
    }
  }

  void visit(RangeForStmt *stmt) override {
    int64 count = -1;
    auto begin = stmt->begin->cast<ConstStmt>();
    auto end = stmt->end->cast<ConstStmt>();
    if (begin && end && begin->ret_type == PrimitiveType::i32 &&
        end->ret_type == PrimitiveType::i32) {
      count = std::max<int64>(0, int64(end->val.val_int32()) -
                                   int64(begin->val.val_int32()));
    }
    repeat(summarize(stmt->body.get()), count);
  }

  void visit(WhileStmt *stmt) override {
    repeat(summarize(stmt->body.get()), -1);
  }

  void visit(StructForStmt *stmt) override {
    repeat(summarize(stmt->body.get()), -1);
  }

  void visit(MeshForStmt *stmt) override {
    repeat(summarize(stmt->body.get()), -1);
  }

  // Early exits can skip pops. Do not apply structured summaries unless their
  // paths have been modeled; the CFG analysis remains responsible for them.
  void visit(ContinueStmt *) override {
    has_early_exit_ = true;
  }
  void visit(WhileControlStmt *) override {
    has_early_exit_ = true;
  }
  void visit(ReturnStmt *) override {
    has_early_exit_ = true;
  }

  static void run(IRNode *root) {
    StructuredStackSize pass;
    root->accept(&pass);
    if (pass.has_early_exit_)
      return;
    for (auto &[stack, effect] : pass.capacities_) {
      if (effect.known && pass.effects_.find(stack) == pass.effects_.end()) {
        // Capacity zero denotes adaptive, even for an unused stack.
        auto capacity = std::size_t(std::max<int64>(1, effect.peak));
        TI_ASSERT_INFO(
            capacity <= (std::numeric_limits<std::size_t>::max() -
                         AdStackLayout::header_size) / stack->entry_size_in_bytes(),
            "Autodiff stack size exceeds addressable storage");
        stack->max_size = capacity;
      }
    }
  }

 private:
  StackEffects effects_;
  StackEffects capacities_;
  bool has_early_exit_{false};
};

}  // namespace

bool determine_ad_stack_size(IRNode *root, const CompileConfig &config) {
  auto stacks = irpass::analysis::gather_statements(root, [&](Stmt *s) {
        if (auto ad_stack = s->cast<AdStackAllocaStmt>()) {
          return ad_stack->max_size == 0;  // adaptive
        }
        return false;
      });
  if (stacks.empty()) {
    return false;  // no AD-stacks with adaptive size
  }
  StructuredStackSize::run(root);
  if (std::all_of(stacks.begin(), stacks.end(), [](Stmt *stmt) {
        return stmt->as<AdStackAllocaStmt>()->max_size != 0;
      })) {
    return true;
  }
  auto cfg = analysis::build_cfg(root);
  cfg->simplify_graph();
  cfg->determine_ad_stack_size(config.default_ad_stack_size);
  return true;
}

}  // namespace irpass

}  // namespace taichi::lang
