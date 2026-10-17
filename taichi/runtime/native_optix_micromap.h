#pragma once

#include "taichi/runtime/native_host_blob.h"
#include "taichi/optix/forge_optix_provider.h"

namespace taichi::lang {

// Immutable baked host input. Each GAS imports and owns its device OMM;
// this object never contains a driver allocation or a context handle.
class NativeOptixMicromap final : public NativeResource {
 public:
  NativeOptixMicromap(Device &device,
                      uint32_t triangles,
                      std::shared_ptr<NativeHostBlobResource> data,
                      const NativeHostBlobResource &descriptors,
                      const NativeHostBlobResource *indices);
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  bool retains_input(const std::string &port) const override {
    return port == "data";
  }
  TiForgeOptixMicromapDesc description() const;
  const std::string &fingerprint() const {
    return fingerprint_;
  }

 private:
  Device &device_;
  uint32_t triangles_;
  std::shared_ptr<NativeHostBlobResource> data_;
  std::vector<TiForgeOptixMicromapEntry> descriptors_;
  std::vector<int32_t> indices_;
  std::string fingerprint_;
};

}  // namespace taichi::lang
