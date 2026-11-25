#include "taichi/transforms/loop_invariant_detector.h"

namespace taichi::lang {

class LoopInvariantCodeMotion : public LoopInvariantDetector {
 public:
  using LoopInvariantDetector::visit;

  DelayedIRModifier modifier;

  explicit LoopInvariantCodeMotion(const CompileConfig &config)
      : LoopInvariantDetector(config) {
  }

  void visit(BinaryOpStmt *stmt) override {
    if (is_loop_invariant(stmt, stmt->parent)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(current_loop_stmt(), std::move(replacement));
      modifier.erase(stmt);
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    if (is_loop_invariant(stmt, stmt->parent)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(current_loop_stmt(), std::move(replacement));
      modifier.erase(stmt);
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (config.cache_loop_invariant_global_vars &&
        is_loop_invariant(stmt, stmt->parent)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(current_loop_stmt(), std::move(replacement));
      modifier.erase(stmt);
    }
  }

  void visit(ExternalPtrStmt *stmt) override {
    if (config.cache_loop_invariant_global_vars &&
        is_loop_invariant(stmt, stmt->parent)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(current_loop_stmt(), std::move(replacement));
      modifier.erase(stmt);
    }
  }

  void visit(ArgLoadStmt *stmt) override {
    // A by-value scalar is immutable for this invocation. Exposing it before
    // the inner loop also lets LICM lift expressions that depend on it.
    const bool full_cpu_scalar =
        arch_is_cpu(config.arch) && config.compile_tier == "full" &&
        config.advanced_optimization && stmt->arg_depth == 0 &&
        stmt->arg_id.size() == 1 && stmt->create_load && !stmt->is_ptr &&
        stmt->ret_type->is<PrimitiveType>();
    if ((config.cache_loop_invariant_global_vars || full_cpu_scalar) &&
        is_loop_invariant(stmt, stmt->parent)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(current_loop_stmt(), std::move(replacement));
      modifier.erase(stmt);
    }
  }

  static bool run(IRNode *node, const CompileConfig &config) {
    bool modified = false;

    while (true) {
      LoopInvariantCodeMotion eliminator(config);
      node->accept(&eliminator);
      if (eliminator.modifier.modify_ir())
        modified = true;
      else
        break;
    };

    return modified;
  }
};

namespace irpass {
bool loop_invariant_code_motion(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  return LoopInvariantCodeMotion::run(root, config);
}
}  // namespace irpass

}  // namespace taichi::lang
