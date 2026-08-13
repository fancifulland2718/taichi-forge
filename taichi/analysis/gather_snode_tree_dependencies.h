#pragma once

#include <vector>

#include "taichi/codegen/snode_relocation.h"
#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

class Kernel;
class IRNode;
class Program;

namespace irpass::analysis {

// Returns sorted, unique SNodeTree ids referenced by the specialized kernel
// IR. Both frontend and lowered statement forms are covered so graph
// dependency collection is independent of the compilation tier/cache path.
std::vector<int> gather_snode_tree_dependencies(const Kernel &kernel);
std::vector<int> gather_snode_tree_dependencies(IRNode &ir);

// Returns true when the lowered IR directly references an SNode whose path
// contains activating state. Unlike the tree-granular helper below, this is
// used by codegen to classify state embedded in one executable.
SNodeRelocationStructure gather_snode_relocation_structures(IRNode &ir);

// Returns true when any listed SNodeTree contains an activating (sparse or
// dynamic) node. This deliberately qualifies the complete tree because
// compiled Graph dependency metadata and lifecycle ownership are
// tree-granular.
bool has_non_dense_snode_tree_dependency(
    Program &program,
    const std::vector<SNodeTreeDependency> &dependencies);

}  // namespace irpass::analysis
}  // namespace taichi::lang
