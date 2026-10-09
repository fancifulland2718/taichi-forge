#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {

namespace irpass {
bool cfg_optimization(
    IRNode *root,
    bool after_lower_access,
    bool autodiff_enabled,
    bool real_matrix_enabled,
    const std::optional<ControlFlowGraph::LiveVarAnalysisConfig>
        &lva_config_opt) {
  TI_AUTO_PROF;
  auto cfg = analysis::build_cfg(root);
  // The CFG memory facts describe explicit addresses. Opaque calls and sparse
  // lifetime changes can affect global memory without naming those addresses.
  // Keep local optimization, but do not forward/eliminate global accesses
  // using an incomplete memory-effect model.
  bool local_only = after_lower_access;
  bool escaped_locals = false;
  for (const auto &node : cfg->nodes) {
    for (int i = node->begin_location; i < node->end_location; ++i) {
      auto *stmt = node->block->statements[i].get();
      const bool opaque_call = stmt->is<InternalFuncStmt>() ||
                               stmt->is<ExternalFuncCallStmt>() ||
                               stmt->is<FuncCallStmt>();
      if (opaque_call) {
        local_only = true;
        for (auto *operand : stmt->get_operands()) {
          while (operand && operand->is<MatrixPtrStmt>())
            operand = operand->as<MatrixPtrStmt>()->origin;
          if (operand && operand->is<AllocaStmt>())
            escaped_locals = true;
        }
      }
      if (auto *op = stmt->cast<SNodeOpStmt>();
          op && (op->op_type == SNodeOpType::deactivate ||
                 op->op_type == SNodeOpType::append ||
                 op->op_type == SNodeOpType::allocate))
        local_only = true;
    }
  }
  bool result_modified = false;
  if (!real_matrix_enabled && !escaped_locals) {
    cfg->simplify_graph();

    if (cfg->store_to_load_forwarding(local_only, autodiff_enabled)) {
      result_modified = true;
    }
    if (cfg->dead_store_elimination(local_only, lva_config_opt)) {
      result_modified = true;
    }
  }
  // die() is the canonical side-effect-aware SSA dead-instruction pass and
  // already removes unused allocas after CFG forwarding/store elimination.
  // A second CFG-liveness DCE is intentionally not maintained without a
  // profiled missed pattern; duplicating the pass would add compile latency and
  // another alias/AD correctness surface.
  die(root);
  return result_modified;
}
}  // namespace irpass

}  // namespace taichi::lang
