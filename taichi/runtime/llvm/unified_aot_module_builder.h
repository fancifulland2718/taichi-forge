#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "taichi/common/core.h"

namespace taichi::lang {
class AotModuleBuilder;
class SNode;
struct DeviceCapabilityConfig;
struct CompileConfig;
namespace LLVM {
TI_DLL_EXPORT int declare_unified_storage(AotModuleBuilder &builder, const SNode &root,
    const std::vector<std::tuple<const SNode *, int, std::string>> &parameters,
    const std::string &binding_key);
void set_unified_build_directory(AotModuleBuilder &builder,
                                 const std::string &directory, bool cache_captures);
std::string unified_build_object(AotModuleBuilder &builder, const std::string &name);
bool restore_unified_build_object(AotModuleBuilder &builder, const std::string &name,
                                  const std::string &record);
std::unique_ptr<AotModuleBuilder> make_unified_aot_module_builder(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps);
}  // namespace LLVM
}  // namespace taichi::lang
