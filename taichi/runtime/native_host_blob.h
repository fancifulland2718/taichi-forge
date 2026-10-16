#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include "taichi/runtime/native_resource.h"

namespace taichi::lang {

// Immutable module data consumed by a provider on the host during preparation.
// It has no device allocation and must never be bound as a kernel ndarray.
// Resources with the same content key share bytes within a prepared owner.
class NativeHostBlobResource final : public NativeResource {
 public:
  using Bytes = std::vector<std::uint8_t>;
  NativeHostBlobResource(Arch arch,
                        Device &device,
                        std::shared_ptr<const Bytes> bytes)
      : arch_(arch), device_(device), bytes_(std::move(bytes)) {
    if (!bytes_)
      throw std::invalid_argument("Native host blob requires immutable bytes");
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  const Bytes &bytes() const {
    return *bytes_;
  }

 private:
  Arch arch_;
  Device &device_;
  std::shared_ptr<const Bytes> bytes_;
};

}  // namespace taichi::lang
