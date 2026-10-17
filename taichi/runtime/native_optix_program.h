#pragma once

#include "taichi/runtime/native_optix_access.h"

namespace taichi::lang {
// Immutable pipeline owner shared by independently bound launch packets.
class NativeOptixProgram : public NativeResource {
 public:
  virtual std::shared_ptr<NativeOptixContext> context() const = 0;
  virtual const TiForgeOptixProgramApi &program_api() const = 0;
  virtual TiForgeOptixProgram program_handle() const = 0;
  virtual const std::string &contract() const = 0;
  virtual uint32_t parameter_bytes() const = 0;
  virtual uint32_t group_kind(uint32_t index) const = 0;
};
}  // namespace taichi::lang
