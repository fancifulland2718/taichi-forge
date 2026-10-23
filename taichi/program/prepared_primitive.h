#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "taichi/aot/graph_data.h"
#include "taichi/program/ndarray.h"
#include "taichi/runtime/runtime_client.h"

namespace taichi::lang {

class PreparedNativeStorage;
class NativeOperation;
class Program;
namespace gfx {
class ExternalGraphCommand;
}

// Explicit, immutable recording generation. The source prepared plan still
// uses its lazy Program arena. Recorded native commands own separate scratch,
// consumed by the existing CUDA/GFX materializers and retired before Device.
class PreparedPrimitiveRecording final
    : public RuntimeClient,
      public std::enable_shared_from_this<PreparedPrimitiveRecording> {
 public:
  void retire() override;
  Program *program() const;
  std::size_t workspace_bytes() const;
  const std::vector<aot::Arg> &arguments() const;
  std::vector<SNodeTreeDependency> tree_dependencies() const;
  void validate(Program &program,
                const std::unordered_map<std::string, aot::IValue> &args) const;
  void record_cuda(void *stream);
  void record_vulkan(Device *device, CommandList *commands);
  std::vector<DeviceAllocation> buffers() const;
  std::shared_ptr<aot::CudaGraphCaptureCommand> cuda_command();
  std::shared_ptr<gfx::ExternalGraphCommand> vulkan_command();

 private:
  friend class Program;
  mutable std::mutex mutex_;
  Program *program_{nullptr};
  std::shared_ptr<PreparedNativeStorage> storage_;
  std::shared_ptr<NativeOperation> operation_;
  std::vector<aot::Arg> arguments_;
  std::vector<std::size_t> binding_indices_;
};

// Fixed storage bindings, not a second device recording.
// The Program arena owns GPU radix scratch/pipelines. CPU uses the shared
// native sort operation's retained workspace. Borrowed ndarray wrappers never
// free the underlying allocation.
class PreparedPrimitiveSort {
 private:
  friend class Program;
  std::shared_ptr<PreparedNativeStorage> storage_;
  // The existing AOT CPU operation retains its own stable-sort workspace.
  // Populate it on first execution, under the Program submission guard.
  mutable std::shared_ptr<NativeOperation> cpu_operation_;
  std::array<std::unique_ptr<Ndarray>, 2> views_;
  std::array<std::uintptr_t, 2> cuda_pointers_{};
  int count_{0};
  int key_type_{0};
  int value_type_{0};
  int nan_policy_{0};
  std::size_t key_bytes_{0};
  std::size_t value_bytes_{0};
};

// Columns share one predicate/prefix and a device count. All bindings are
// resolved together; no other submission may interleave the column scatters.
class PreparedPrimitiveCompact {
 private:
  friend class Program;
  std::shared_ptr<PreparedNativeStorage> storage_;
  std::vector<std::uintptr_t> cuda_pointers_;
  std::vector<std::size_t> column_bytes_;
  int count_{0};
};

}  // namespace taichi::lang
