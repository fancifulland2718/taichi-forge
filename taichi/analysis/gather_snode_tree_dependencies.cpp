#include "taichi/analysis/gather_snode_tree_dependencies.h"

#include <algorithm>
#include <unordered_set>

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"

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
    if (stmt->op_type == SNodeOpType::activate) {
      record_hash_activation_path(stmt->snode);
    }
  }

  void visit(FrontendForStmt *stmt) override {
    record(stmt->snode);
    BasicStmtVisitor::visit(stmt);
  }

  void visit(GlobalPtrStmt *stmt) override {
    record(stmt->snode);
    if (stmt->activate) {
      record_hash_activation_path(stmt->snode);
    }
  }

  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    for (SNode *snode : stmt->snodes) {
      record(snode);
      if (stmt->activate) {
        record_hash_activation_path(snode);
      }
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    record(stmt->snode);
    if (stmt->op_type == SNodeOpType::activate) {
      record_hash_activation_path(stmt->snode);
    }
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
    if (stmt->activate) {
      record_hash_activation_path(stmt->snode);
    }
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

  SNodeRelocationStructure relocation_structures() const {
    return relocation_structures_;
  }

  bool may_trigger_hash_overflow() const {
    return may_trigger_hash_overflow_;
  }

 private:
  void record_hash_activation_path(const SNode *snode) {
    for (const SNode *node = snode; node != nullptr; node = node->parent) {
      if (node->type == SNodeType::hash) {
        may_trigger_hash_overflow_ = true;
        return;
      }
    }
  }

  void record(const SNode *snode) {
    if (snode != nullptr) {
      tree_ids_.insert(snode->get_snode_tree_id());
      for (const SNode *node = snode; node != nullptr; node = node->parent) {
        switch (node->type) {
          case SNodeType::pointer:
            relocation_structures_ |= SNodeRelocationStructure::pointer;
            break;
          case SNodeType::bitmasked:
            relocation_structures_ |= SNodeRelocationStructure::bitmasked;
            break;
          case SNodeType::dynamic:
            relocation_structures_ |= SNodeRelocationStructure::dynamic;
            break;
          case SNodeType::hash:
            relocation_structures_ |= SNodeRelocationStructure::hash;
            break;
          default:
            break;
        }
      }
    }
  }

  std::unordered_set<int> tree_ids_;
  SNodeRelocationStructure relocation_structures_{
      SNodeRelocationStructure::none};
  bool may_trigger_hash_overflow_{false};
};

bool tree_contains_non_dense_snode(const SNode &node) {
  if (!node.is_path_all_dense) {
    return true;
  }
  return std::any_of(
      node.ch.begin(), node.ch.end(), [](const std::unique_ptr<SNode> &child) {
        return tree_contains_non_dense_snode(*child);
      });
}

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

SNodeRelocationStructure gather_snode_relocation_structures(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  ir.accept(&collector);
  return collector.relocation_structures();
}

bool may_trigger_hash_overflow(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  ir.accept(&collector);
  return collector.may_trigger_hash_overflow();
}

bool has_non_dense_snode_tree_dependency(
    Program &program,
    const std::vector<SNodeTreeDependency> &dependencies) {
  for (const auto &dependency : dependencies) {
    const SNode *root = program.get_snode_root(dependency.tree_id);
    if (root != nullptr && tree_contains_non_dense_snode(*root)) {
      return true;
    }
  }
  return false;
}

}  // namespace irpass::analysis
}  // namespace taichi::lang
