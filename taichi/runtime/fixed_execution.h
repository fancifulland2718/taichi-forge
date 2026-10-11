#pragma once

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {
class LaunchContextBuilder;
namespace aot {
class Kernel;
}
namespace gfx {
class GfxRuntime;
}

// Cold bindings shared by loaded AOT entries and native runtime adapters. The
// caller owns registered code, storage and submission/retirement serialization.
// Contexts and operation descriptors are borrowed only during preparation.
class FixedExecution {
 public:
  struct Dispatch {
    aot::Kernel *kernel;
    LaunchContextBuilder *context;
  };
  struct BufferCopy {
    DevicePtr destination;
    DevicePtr source;
    std::size_t bytes;
  };
  struct BufferFill {
    DevicePtr destination;
    std::size_t bytes;
    std::uint32_t value;
  };
  using Operation = std::variant<Dispatch, BufferCopy, BufferFill>;

  FixedExecution(Arch arch,
                 Device &device,
                 gfx::GfxRuntime *gfx_runtime,
                 const std::vector<Operation> &operations);
  ~FixedExecution();
  FixedExecution(const FixedExecution &) = delete;
  FixedExecution &operator=(const FixedExecution &) = delete;
  void run();
  std::uint32_t execution_kind() const;
  std::size_t argument_bytes() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang
