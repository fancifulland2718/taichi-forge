#pragma once

namespace taichi::lang {

class CompileConfig;
class IRNode;
class Kernel;

namespace irpass {

// Materialize private Forge Graph one- or two-dimensional shared-stage plans.
// The pass is deliberately fail-closed and changes only tasks whose complete
// offload-plan entry explicitly requests the strategy.
void make_external_shared_staged(IRNode *ir,
                                 const CompileConfig &config,
                                 const Kernel *kernel);

}  // namespace irpass
}  // namespace taichi::lang
