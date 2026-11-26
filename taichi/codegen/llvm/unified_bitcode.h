#pragma once

#include <functional>
#include <string>
#include <map>
#include <vector>

#include "taichi/common/platform_macros.h"

namespace taichi::lang {
class Kernel;
class Callable;
struct GraphKernelMetadata;
namespace LLVM {

std::string unified_leaf_identity(Callable &callable, bool simplify);

// Lower a private copy before native task scheduling and resource layout.
// The returned bitcode contains numerical instructions and symbolic ports.
TI_DLL_EXPORT std::string emit_unified_bitcode(
    Kernel &kernel,
    GraphKernelMetadata &effects,
    std::map<std::string, std::string> &helpers,
    std::map<std::string, std::string> &identities,
    const std::vector<int> &tree_ids,
    const std::map<int, int> &node_ordinals,
    bool simplify,
    std::vector<std::string> *imports = nullptr,
    const std::function<bool(const std::string &)> &restore_unit = {});

}  // namespace LLVM
}  // namespace taichi::lang
