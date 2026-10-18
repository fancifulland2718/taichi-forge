#include "taichi/runtime/native_memory_allocator.h"

#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/graph_memory_pool.h"
#endif

namespace taichi::lang {
namespace {
struct MemoryPoolDescription {
  static constexpr const char *kind = "memory_pool";
  static constexpr std::uint32_t version = 1;
  std::uint64_t retained_bytes{0};
};

#ifdef TI_WITH_CUDA
class CudaMemoryPool final : public NativeMemoryAllocator {
 public:
  CudaMemoryPool(cuda::CudaDevice &device, std::uint64_t retained_bytes)
      : device_(device), pool_(device, retained_bytes) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    // Arrays own/account their allocations; driver-private reserved pages are
    // neither additional logical payload nor an estimate of residency.
    return 0;
  }
  DeviceAllocation allocate_memory(std::size_t bytes, AllocUsage) override {
    // CUDA allocations are untyped; usage is a physical Vulkan constraint.
    return pool_.allocate_memory(bytes);
  }

 private:
  cuda::CudaDevice &device_;
  cuda::GraphMemoryPool pool_;
};
#endif
}  // namespace

void register_memory_allocator_resources(NativeOperationRegistry &registry) {
  registry.define<MemoryPoolDescription>(
      [](Device &, const MemoryPoolDescription &) {},
      NativeObjectKind::resource);
  registry.register_schema<MemoryPoolDescription>(
      "memory", "pool", {}, [](const liong::json::JsonValue &parameters) {
        operation_json::fields(parameters, {"retained_bytes"});
        MemoryPoolDescription value{
            operation_json::optional_integer(parameters, "retained_bytes")
                .value_or(0)};
        return [value](const NativeOperationBindings &) { return value; };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<MemoryPoolDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const MemoryPoolDescription &value) {
        auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
        if (!cuda)
          throw std::invalid_argument("Memory pool requires a CUDA device");
        return std::make_shared<CudaMemoryPool>(*cuda, value.retained_bytes);
      },
      [] {
        return NativeOperationAvailability{cuda::GraphMemoryPool::available(),
                                            0};
      });
#endif
}
}  // namespace taichi::lang
