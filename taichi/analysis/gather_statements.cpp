#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang {

class StmtSearcher : public BasicStmtVisitor {
 private:
  std::function<bool(Stmt *)> test_;
  std::vector<Stmt *> results_;

 public:
  using BasicStmtVisitor::visit;

  explicit StmtSearcher(std::function<bool(Stmt *)> test) : test_(test) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (test_(stmt))
      results_.push_back(stmt);
  }

  static std::vector<Stmt *> run(IRNode *root,
                                 const std::function<bool(Stmt *)> &test) {
    StmtSearcher searcher(test);
    root->accept(&searcher);
    return std::move(searcher.results_);
  }
};

namespace irpass::analysis {
bool any_statement(IRNode *root, const std::function<bool(Stmt *)> &test) {
  class AnyStmtSearcher : public BasicStmtVisitor {
   public:
    const std::function<bool(Stmt *)> &test;
    bool found{false};

    explicit AnyStmtSearcher(const std::function<bool(Stmt *)> &predicate)
        : test(predicate) {
      allow_undefined_visitor = true;
      invoke_default_visitor = true;
    }

    using BasicStmtVisitor::visit;

    void visit(Stmt *stmt) override {
      if (!found) {
        found = test(stmt);
      }
    }

    void visit(Block *block) override {
      for (auto &stmt : block->statements) {
        if (found) {
          break;
        }
        stmt->accept(this);
      }
    }
  } searcher(test);
  root->accept(&searcher);
  return searcher.found;
}

std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test) {
  return StmtSearcher::run(root, test);
}
}  // namespace irpass::analysis

}  // namespace taichi::lang
