#include "taichi/runtime/native_scan.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/inclusive_scan.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
const std::array<DataType, 6> &scan_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

void validate(Device &device, const ScanOperation &op) {
  const auto &types = scan_types();
  if (op.value_type < 0 || op.value_type >= types.size() ||
      op.values.pointer.device != &device)
    throw std::invalid_argument("Scan requires a supported same-device array");
  const std::uint64_t width = op.value_type < 3 ? 4 : 8;
  validate_native_array_layout(
      op.values, {{op.size}, {}, width, width, false,
                  types[op.value_type]->to_string()});
  if (op.values.shape[0] > std::numeric_limits<std::size_t>::max() / width ||
      op.values.bytes != op.values.shape[0] * width)
    throw std::invalid_argument("Scan requires exact packed storage extents");
}

class ScanBase : public NativeOperation {
 public:
  ScanBase(Arch arch, Device &device, const ScanOperation &op)
      : arch_(arch), device_(device), size_(op.size), type_(op.value_type) {
    validate_native_shader_storage(arch, op.values.pointer);
    resolve_native_storage_range(arch, op.values.pointer, op.values.bytes);
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return bytes_;
  }

 protected:
  Arch arch_;
  Device &device_;
  std::size_t size_;
  int type_;
  std::size_t bytes_{0};
  void *values_{nullptr};
};

#ifdef TI_WITH_LLVM
class CpuScan final : public ScanBase {
 public:
  CpuScan(Arch arch, Device &device, const ScanOperation &op)
      : ScanBase(arch, device, op) {
    auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device);
    if (!cpu || cpu->map_range_for_cpu_native(op.values.pointer,
                                             op.values.bytes, &values_) !=
                    RhiResult::success)
      throw std::invalid_argument("Scan CPU storage range is invalid");
  }
  void run_cpu() override {
    cpu::inclusive_scan(values_, size_, type_);
  }
};
#endif

#ifdef TI_WITH_CUDA
class CudaScan final : public ScanBase {
 public:
  CudaScan(Device &device, const ScanOperation &op)
      : ScanBase(Arch::cuda, device, op) {
    // The shared CUDA primitive takes int, not an application-level size cap.
    if (size_ > std::size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Scan exceeds the CUDA primitive count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda)
      throw std::invalid_argument("Scan requires a CUDA device");
    values_ = static_cast<char *>(cuda->get_memory_addr(op.values.pointer)) +
              op.values.pointer.offset;
    if (size_ <= 1)
      return;
    bytes_ = cuda::driver_scan_workspace_bytes(
        static_cast<int>(size_), static_cast<cuda::CudaTransformValueType>(type_));
    if (bytes_) {
      Device::AllocParams params{};
      params.size = bytes_;
      auto memory = device.allocate_memory_unique(params);
      if (memory.second != RhiResult::success)
        throw std::runtime_error("Failed to allocate fixed scan workspace");
      workspace_ = std::move(memory.first);
      scratch_ = cuda->get_memory_addr(workspace_->get_ptr());
    }
    cuda::driver_prepare_scan();
  }
  void record_cuda(void *stream) override {
    if (size_ > 1)
      cuda::driver_inclusive_scan_with_workspace(
          values_, static_cast<int>(size_),
          static_cast<cuda::CudaTransformValueType>(type_), scratch_, stream);
  }

 private:
  DeviceAllocationUnique workspace_;
  void *scratch_{nullptr};
};
#endif
}  // namespace

void register_scan_operation(NativeOperationRegistry &registry) {
  registry.define<ScanOperation>(validate);
  registry.register_schema<ScanOperation>(
      "builtin", "scan", {"values"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<ScanOperation> {
        operation_json::fields(parameters, {"size", "dtype"});
        const auto size = operation_json::integer(parameters["size"]);
        if (size > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument("Scan size exceeds host address space");
        const auto dtype = operation_json::name(parameters["dtype"]);
        const auto &types = scan_types();
        int type = 0;
        while (type < types.size() && types[type]->to_string() != dtype)
          ++type;
        if (type == types.size())
          throw std::invalid_argument("Scan requires a supported numeric type");
        return [size, type](const NativeOperationBindings &bindings) {
          return ScanOperation{static_cast<std::size_t>(size), type,
                               bindings.at("values").array()};
        };
      });
#ifdef TI_WITH_LLVM
  registry.register_factory<ScanOperation>(
      host_arch(), NativeOperationMode::cpu,
      [](Arch arch, Device &device, const ScanOperation &op) {
        return std::make_shared<CpuScan>(arch, device, op);
      });
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<ScanOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const ScanOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaScan>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<ScanOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const ScanOperation &op) {
        return gfx::prepare_vulkan_scan_operation(device, op);
      });
#endif
}
}  // namespace taichi::lang
