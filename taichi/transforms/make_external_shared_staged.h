#pragma once

namespace taichi::lang {

class CompileConfig;
class IRNode;
class Kernel;

namespace irpass {

// Materialize the private Forge Graph shared_staged_1d execution-plan recipe.
// The pass is deliberately fail-closed and changes only tasks whose complete
// offload-plan entry explicitly requests the strategy.
void make_external_shared_staged(IRNode *ir,
                                 const CompileConfig &config,
                                 const Kernel *kernel);

}  // namespace irpass
}  // namespace taichi::lang
