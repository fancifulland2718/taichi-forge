#include "taichi/analysis/gather_snode_tree_dependencies.h"

#include <algorithm>
#include <unordered_set>

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace irpass::analysis {
namespace {

class SNodeTreeDependencyCollector : public BasicStmtVisitor {
 public:
  void visit(Block *block) override {
    for (SNode *snode : block->stop_gradients) {
      record(snode);
    }
    BasicStmtVisitor::visit(block);
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    record(stmt->snode);
  }

  void visit(FrontendForStmt *stmt) override {
    record(stmt->snode);
    BasicStmtVisitor::visit(stmt);
  }

  void visit(GlobalPtrStmt *stmt) override {
    record(stmt->snode);
  }

  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    for (SNode *snode : stmt->snodes) {
      record(snode);
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    record(stmt->snode);
  }

  void visit(StructForStmt *stmt) override {
    record(stmt->snode);
    BasicStmtVisitor::visit(stmt);
  }

  void visit(GetRootStmt *stmt) override {
    record(stmt->root());
  }

  void visit(SNodeLookupStmt *stmt) override {
    record(stmt->snode);
  }

  void visit(GetChStmt *stmt) override {
    record(stmt->input_snode);
    record(stmt->output_snode);
  }

  void visit(OffloadedStmt *stmt) override {
    record(stmt->snode);
    BasicStmtVisitor::visit(stmt);
  }

  void visit(ClearListStmt *stmt) override {
    record(stmt->snode);
  }

  std::vector<int> result() const {
    std::vector<int> ids(tree_ids_.begin(), tree_ids_.end());
    std::sort(ids.begin(), ids.end());
    return ids;
  }

 private:
  void record(const SNode *snode) {
    if (snode != nullptr) {
      tree_ids_.insert(snode->get_snode_tree_id());
    }
  }

  std::unordered_set<int> tree_ids_;
};

}  // namespace

std::vector<int> gather_snode_tree_dependencies(const Kernel &kernel) {
  TI_ASSERT(kernel.ir != nullptr);
  return gather_snode_tree_dependencies(*kernel.ir);
}

std::vector<int> gather_snode_tree_dependencies(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  ir.accept(&collector);
  return collector.result();
}

}  // namespace irpass::analysis
}  // namespace taichi::lang
