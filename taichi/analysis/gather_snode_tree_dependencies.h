#pragma once

#include <vector>

#include "taichi/codegen/snode_relocation.h"
#include "taichi/codegen/runtime_state.h"
#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

class Kernel;
class IRNode;
class Program;

namespace irpass::analysis {

// The compiler needs all these facts at the same lowering boundary. Collect
// them in one traversal, including transitive real-function bodies.
struct KernelRuntimeDependencies {
  std::vector<int> tree_ids;
  SNodeRelocationStructure structures{SNodeRelocationStructure::none};
  bool may_trigger_hash_overflow{false};
  std::uint32_t runtime_state{runtime_state_known};
};
KernelRuntimeDependencies gather_kernel_runtime_dependencies(IRNode &ir);

// Returns sorted, unique SNodeTree ids referenced by the specialized kernel
// IR. Both frontend and lowered statement forms are covered so graph
// dependency collection is independent of the compilation tier/cache path.
std::vector<int> gather_snode_tree_dependencies(const Kernel &kernel);
std::vector<int> gather_snode_tree_dependencies(IRNode &ir);
// Definition lifetime includes references that optimization may eliminate.
std::vector<int> gather_frontend_snode_tree_dependencies(IRNode &ir);
std::vector<int> gather_source_snode_tree_dependencies(IRNode &ir);

// Returns true when the lowered IR directly references an SNode whose path
// contains activating state. Unlike the tree-granular helper below, this is
// used by codegen to classify state embedded in one executable.
SNodeRelocationStructure gather_snode_relocation_structures(IRNode &ir);

// Returns true when the lowered IR can execute a hash insertion or allocate a
// compact hash child payload. These are the only hash operations that can set
// the LLVM runtime error slot through table/pool overflow. The result is
// persisted with CompiledKernelData so launch does not rediscover it.
bool may_trigger_hash_overflow(IRNode &ir);

// Returns true when any listed SNodeTree contains an activating (sparse or
// dynamic) node. This deliberately qualifies the complete tree because
// compiled Graph dependency metadata and lifecycle ownership are
// tree-granular.
bool has_non_dense_snode_tree_dependency(
    Program &program,
    const std::vector<SNodeTreeDependency> &dependencies);

}  // namespace irpass::analysis
}  // namespace taichi::lang
