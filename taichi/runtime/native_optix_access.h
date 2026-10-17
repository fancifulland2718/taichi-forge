#pragma once

#include "taichi/runtime/native_resource.h"
#include "taichi/optix/forge_optix_program.h"

namespace taichi::lang {
class RuntimeFaultDomain;
struct NativeProviderDefinition;

// Private, owner-bearing access shared by fixed queries and user programs.
// These handles are never persisted or exposed as unowned consumer arguments.
class NativeOptixContext : public NativeResource {
 public:
  virtual const TiForgeOptixProviderApi &provider_api() const = 0;
  virtual TiForgeOptixContext context_handle() const = 0;
  virtual std::shared_ptr<RuntimeFaultDomain> fault_domain() const = 0;
};

class NativeOptixScene : public NativeResource {
 public:
  virtual std::shared_ptr<NativeOptixContext> context() const = 0;
  // A triangle scene or IAS; a bare GAS is not a public program scene.
  virtual TiForgeOptixProgramScene program_scene() const = 0;
};

namespace native_optix {
const NativeProviderDefinition &provider_definition();
void check(const TiForgeOptixProviderApi &api,
           TiForgeOptixResult result,
           const char *operation);
bool calls_safe(const std::shared_ptr<RuntimeFaultDomain> &fault);
}  // namespace native_optix
}  // namespace taichi::lang
