#pragma once

#include <vector>

namespace taichi::lang {

class Kernel;
class IRNode;

namespace irpass::analysis {

// Returns sorted, unique SNodeTree ids referenced by the specialized kernel
// IR. Both frontend and lowered statement forms are covered so graph
// dependency collection is independent of the compilation tier/cache path.
std::vector<int> gather_snode_tree_dependencies(const Kernel &kernel);
std::vector<int> gather_snode_tree_dependencies(IRNode &ir);

}  // namespace irpass::analysis
}  // namespace taichi::lang
