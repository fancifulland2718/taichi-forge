#pragma once

#include <memory>
#include <string>

namespace taichi::lang {
class AotModuleBuilder;
struct DeviceCapabilityConfig;
struct CompileConfig;
namespace LLVM {
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
