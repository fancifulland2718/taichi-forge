#include "taichi/runtime/native_optix_access.h"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/native_provider.h"
#include "taichi/optix/forge_optix_runtime_provider.h"

namespace taichi::lang::native_optix {
const NativeProviderDefinition &provider_definition() {
  static const NativeProviderDefinition definition{
      "optix", TI_FORGE_OPTIX_RUNTIME_PROVIDER_QUERY_SYMBOL,
      TI_FORGE_OPTIX_RUNTIME_VERSION_FAMILY};
  return definition;
}

void check(const TiForgeOptixProviderApi &api,
           TiForgeOptixResult result,
           const char *operation) {
  if (result == TI_FORGE_OPTIX_SUCCESS)
    return;
  const auto bytes = api.get_last_error(nullptr, 0);
  std::vector<char> message(std::max(size_t(1), bytes), 0);
  api.get_last_error(message.data(), message.size());
  message.back() = 0;
  throw std::runtime_error(std::string("OptiX ") + operation + " (" +
                           std::to_string(int(result)) +
                           "): " + message.data());
}

bool calls_safe(const std::shared_ptr<RuntimeFaultDomain> &fault) {
  return fault && fault->backend_calls_safe() &&
         fault->state() != RuntimeLifecycleState::kFinalized;
}
}  // namespace taichi::lang::native_optix
