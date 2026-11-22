#pragma once

#include <map>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "taichi/common/platform_macros.h"

namespace taichi::lang {
class Kernel;
class Program;
class SNode;
struct DeviceCapabilityConfig;
namespace LLVM {
using UnifiedUnitLoader = std::function<std::string(const std::string &)>;

// Describe the source and ABI for a cache lookup after the module has verified
// the root and its helper closure. A miss must compile the numerical definition.
TI_DLL_EXPORT std::unique_ptr<Kernel> declare_unified_kernel(
    Program &program, const std::string &name, const std::string &root_digest,
    const std::string &signature,
    const std::map<std::vector<int>, int> &constant_int_args,
    const DeviceCapabilityConfig &device_caps,
    const std::vector<std::vector<SNode *>> &storage);

// Rebuild a pre-offload kernel from the selected numerical LLVM unit. The
// target Program supplies its own task scheduler, ABI layout and compiler.
TI_DLL_EXPORT std::unique_ptr<Kernel> read_unified_kernel(
    Program &program,
    const std::string &name,
    const std::string &bitcode,
    const std::string &signature,
    const std::map<std::vector<int>, int> &constant_int_args,
    const DeviceCapabilityConfig &device_caps,
    const UnifiedUnitLoader &load_helper,
    const std::vector<std::vector<SNode *>> &storage,
    const std::vector<int> &absent_inputs = {},
    bool *needs_bound_device = nullptr);

}  // namespace LLVM
}  // namespace taichi::lang
