#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
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
  bool result_modified = false;
  if (!real_matrix_enabled) {
    cfg->simplify_graph();

    if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
      result_modified = true;
    }
    if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
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
