#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/stmt_use_index.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <functional>
#include <typeindex>

namespace taichi::lang {

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 private:
  std::unordered_set<int> visited_;
  // each scope corresponds to an unordered_set
  std::vector<std::unordered_map<std::size_t, std::unordered_set<Stmt *>>>
      visible_stmts_;
  DelayedIRModifier modifier_;
  std::unordered_map<Block *, std::unordered_set<Stmt *>> erased_;
  size_t sparse_lookup_epoch_{0};
  size_t sparse_read_epoch_{0};

  // Reverse def-use map for the current pass iteration.
  // Rebuilt once per inner iteration; used for O(direct-users) MarkUndone.
  StmtUseIndex uses_;

 public:
  using BasicStmtVisitor::visit;

  WholeKernelCSE() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  bool is_done(Stmt *stmt) {
    return visited_.find(stmt->instance_id) != visited_.end();
  }

  void set_done(Stmt *stmt) {
    visited_.insert(stmt->instance_id);
  }

  void invalidate_sparse_lookups() {
    ++sparse_lookup_epoch_;
    ++sparse_read_epoch_;
  }

  void preprocess_container_stmt(Stmt *) override {
    // A child may deactivate a node. Scalar value numbering remains valid,
    // but physical sparse addresses must be resolved again afterwards.
    invalidate_sparse_lookups();
  }

  // Rewrite only direct users, including container operands and statements
  // queued for branch hoisting. Transfer the index immediately: another
  // replacement in this traversal must see the updated def-use graph.
  void replace_uses(Stmt *replaced, Stmt *replacement) {
    uses_.replace_uses(replaced, replacement,
                       [&](Stmt *user) { visited_.erase(user->instance_id); });
  }

  static std::size_t operand_hash(const Stmt *stmt) {
    std::size_t hash_code{0};
    auto hash_type =
        std::hash<std::type_index>{}(std::type_index(typeid(*stmt))) ^
        std::hash<const Type *>{}(stmt->ret_type);
    if (stmt->is<GlobalPtrStmt>() || stmt->is<LoopUniqueStmt>()) {
      // special cases in common_statement_eliminable()
      return hash_type;
    }
    auto op = stmt->get_operands();
    for (auto &x : op) {
      if (x == nullptr)
        continue;
      // Hash the addresses of the operand pointers.
      hash_code = (hash_code * 33) ^ std::hash<Stmt *>{}(x);
    }
    return hash_type ^ hash_code;
  }

  static bool common_statement_eliminable(Stmt *this_stmt, Stmt *prev_stmt) {
    // Is this_stmt eliminable given that prev_stmt appears before it and has
    // the same type with it?
    if (this_stmt->type() != prev_stmt->type())
      return false;
    // Address equality is not enough for substitution: scalar and tensor
    // GlobalPtrStmts may name the same first element, but matrix lowering
    // interprets their offsets differently (bytes versus components).
    if (this_stmt->ret_type != prev_stmt->ret_type)
      return false;
    if (this_stmt->is<GlobalPtrStmt>()) {
      auto this_ptr = this_stmt->as<GlobalPtrStmt>();
      auto prev_ptr = prev_stmt->as<GlobalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr) &&
             (this_ptr->activate == prev_ptr->activate || prev_ptr->activate);
    }
    if (this_stmt->is<ExternalPtrStmt>()) {
      auto this_ptr = this_stmt->as<ExternalPtrStmt>();
      auto prev_ptr = prev_stmt->as<ExternalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr);
    }
    if (this_stmt->is<LoopUniqueStmt>()) {
      auto this_loop_unique = this_stmt->as<LoopUniqueStmt>();
      auto prev_loop_unique = prev_stmt->as<LoopUniqueStmt>();
      if (irpass::analysis::same_value(this_loop_unique->input,
                                       prev_loop_unique->input)) {
        // Merge the "covers" information into prev_loop_unique.
        // Notice that this_loop_unique->covers is corrupted here.
        prev_loop_unique->covers.insert(this_loop_unique->covers.begin(),
                                        this_loop_unique->covers.end());
        return true;
      }
      return false;
    }
    return irpass::analysis::same_statements(this_stmt, prev_stmt);
  }

  void visit(Stmt *stmt) override {
    if (auto *lookup = stmt->cast<SNodeLookupStmt>()) {
      // A non-activating lookup may name ambient memory before a later
      // activation. Activating lookups are idempotent until a lifetime/call
      // barrier; reads may also be reused until another activation occurs.
      if (lookup->activate && lookup->snode->need_activation())
        ++sparse_read_epoch_;
    } else if (stmt->has_global_side_effect() && !stmt->is<GlobalPtrStmt>() &&
               !stmt->is<MatrixOfGlobalPtrStmt>() &&
               !stmt->is<GlobalStoreStmt>() && !stmt->is<AtomicOpStmt>() &&
               !stmt->is<LocalStoreStmt>()) {
      invalidate_sparse_lookups();
    }
    if (!stmt->common_statement_eliminable())
      return;
    // container_statement does not need to be CSE-ed
    if (stmt->is_container_statement())
      return;
    // Generic visitor for all CSE-able statements.
    std::size_t hash_value = operand_hash(stmt);
    if (auto *lookup = stmt->cast<SNodeLookupStmt>();
        lookup && lookup->snode->need_activation()) {
      hash_value ^=
          lookup->activate ? sparse_lookup_epoch_ : sparse_read_epoch_;
    }
    if (is_done(stmt)) {
      visible_stmts_.back()[hash_value].insert(stmt);
      return;
    }
    for (auto &scope : visible_stmts_) {
      for (auto &prev_stmt : scope[hash_value]) {
        if (common_statement_eliminable(stmt, prev_stmt)) {
          replace_uses(stmt, prev_stmt);
          erased_[stmt->parent].insert(stmt);
          return;
        }
      }
    }
    visible_stmts_.back()[hash_value].insert(stmt);
    set_done(stmt);
  }

  void visit(Block *stmt_list) override {
    visible_stmts_.emplace_back();
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    visible_stmts_.pop_back();
  }

  void visit(IfStmt *if_stmt) override {
    preprocess_container_stmt(if_stmt);
    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->set_true_statements(nullptr);
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->set_false_statements(nullptr);
      }
    }

    // Move common statements at the beginning or the end of both branches
    // outside.
    // P9.B (2026-04-27): Loop the head/tail hoist so that K consecutive
    // matching prefix/suffix statements are extracted in a single visit.
    // Previously only 1 stmt at each end was hoisted per outer fixed-point
    // iteration, so a K-stmt shared prefix needed K outer iterations (each
    // rebuilding the def-use map). Physics-simulation pair-wise dispatch
    // commonly shares 3-10 stmt setup (relative position / mat3 transform /
    // contact frame) before diverging — looping cuts CSE outer iterations
    // proportionally on those kernels.
    if (if_stmt->true_statements && if_stmt->false_statements) {
      auto &true_clause = if_stmt->true_statements;
      auto &false_clause = if_stmt->false_statements;
      // Extend prefix hoist: keep extracting matching head stmts as long as
      // both clauses still have entries and the leading pair compares equal.
      while (!true_clause->statements.empty() &&
             !false_clause->statements.empty() &&
             irpass::analysis::same_statements(
                 true_clause->statements[0].get(),
                 false_clause->statements[0].get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract(0);
        replace_uses(false_clause->statements[0].get(), common_stmt.get());
        modifier_.insert_before(if_stmt, std::move(common_stmt));
        uses_.remove_user(false_clause->statements[0].get());
        false_clause->erase(0);
      }
      // Extend suffix hoist: same idea on the tail. Note insert_after stacks
      // hoisted stmts in reverse extraction order, but the iteration itself
      // walks the suffix from outermost-tail inward, so the relative order of
      // hoisted stmts in the parent block is preserved.
      while (!true_clause->statements.empty() &&
             !false_clause->statements.empty() &&
             irpass::analysis::same_statements(
                 true_clause->statements.back().get(),
                 false_clause->statements.back().get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract((int)true_clause->size() - 1);
        replace_uses(false_clause->statements.back().get(), common_stmt.get());
        modifier_.insert_after(if_stmt, std::move(common_stmt));
        uses_.remove_user(false_clause->statements.back().get());
        false_clause->erase((int)false_clause->size() - 1);
      }
    }

    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
  }

  static bool run(IRNode *node) {
    WholeKernelCSE eliminator;
    bool modified = false;
    while (true) {
      // Rebuild the reverse def-use map once per inner iteration (O(N)).
      // Both invalidation and replacement use direct users. Batch ordinary
      // CSE erasures once per block to avoid repeated vector compaction.
      eliminator.uses_.rebuild(node);
      node->accept(&eliminator);
      bool iteration_modified = eliminator.modifier_.modify_ir();
      for (auto &[block, stmts] : eliminator.erased_) {
        for (auto *stmt : stmts)
          eliminator.uses_.remove_user(stmt);
        block->erase(std::move(stmts));
        iteration_modified = true;
      }
      eliminator.erased_.clear();
      if (iteration_modified)
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace irpass {
bool whole_kernel_cse(IRNode *root) {
  TI_AUTO_PROF;
  // P9.B-2 (2026-04-27): per-OffloadedStmt segmentation.
  //
  // When the root is the post-lower_offload kernel block — a Block whose
  // children are all OffloadedStmts — CSE is provably independent across
  // offloads (each offload generates a separate launch; their stmts cannot
  // be operands of each other). Running the pass per-offload turns each
  // fixed-point iteration's use-index rebuild from O(whole-IR) into
  // O(per-offload), and stops a tiny modification in offload K from
  // re-triggering an inner iteration that re-walks all other offloads.
  //
  // For pre-offload IR, ti.func bodies, or any other shape we keep the
  // original whole-root behavior — no functional change.
  if (auto *block = dynamic_cast<Block *>(root)) {
    if (!block->statements.empty()) {
      bool all_offloaded = true;
      for (auto &s : block->statements) {
        if (!s->is<OffloadedStmt>()) {
          all_offloaded = false;
          break;
        }
      }
      if (all_offloaded) {
        bool modified = false;
        for (auto &s : block->statements) {
          if (WholeKernelCSE::run(s.get()))
            modified = true;
        }
        return modified;
      }
    }
  }
  return WholeKernelCSE::run(root);
}
}  // namespace irpass

}  // namespace taichi::lang
