#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {

// Prepared backend commands own their scratch and recording state. The
// enclosing execution retains input/output leases and retires submitted work
// before releasing this object.
class NativeOperation {
 public:
  virtual ~NativeOperation() = default;
  virtual Arch arch() const = 0;
  virtual Device &device() const = 0;
  virtual std::size_t workspace_bytes() const = 0;
  virtual void run_cpu() {
    throw std::invalid_argument("Native operation has no CPU implementation");
  }
  virtual void record_cuda(void *stream) {
    throw std::invalid_argument("Native operation has no CUDA recording");
  }
  virtual void record_vulkan(Device *device, CommandList *commands) {
    throw std::invalid_argument("Native operation has no Vulkan recording");
  }
};

struct CompactOperation {
  static constexpr const char *kind = "compact";
  static constexpr std::uint32_t version = 1;
  struct Column {
    DevicePtr values;
    DevicePtr output;
    std::size_t item_bytes;
  };
  std::vector<Column> columns;
  DevicePtr flags;
  DevicePtr count;
  std::size_t size;
};

enum class NativeSortKeyType { u32, i32, f32, u64, i64, f64 };

struct SortOperation {
  static constexpr const char *kind = "sort";
  static constexpr std::uint32_t version = 1;
  DevicePtr keys;
  DevicePtr values;  // A null device denotes keys-only sorting.
  std::size_t size;
  NativeSortKeyType key_type;
  std::size_t value_bytes{0};
  std::size_t value_scalar_bytes{0};
};

// Compact row-major interleaved complex f32 values. The inverse is
// unnormalized. Each prepared object owns its provider plan/workspace; exact
// in-place input and output are allowed, partial overlap is not.
struct FftC2COperation {
  static constexpr const char *kind = "fft_c2c_f32";
  static constexpr std::uint32_t version = 1;
  DevicePtr input;
  DevicePtr output;
  std::vector<std::size_t> dimensions;
  std::size_t batch_count{1};
  bool inverse{false};
};

std::size_t fft_c2c_element_count(const FftC2COperation &operation);
std::shared_ptr<NativeOperation> prepare_fft_operation(
    Arch arch,
    Device &device,
    const FftC2COperation &operation);

// Ascending, stable, in-place sort. Floating NaNs sort last and equal keys
// preserve their input order. Payloads are copied as raw records.
std::shared_ptr<NativeOperation> prepare_sort_operation(
    Arch arch,
    Device &device,
    const SortOperation &operation);

// Stable selection by a nonzero i32 predicate. Count is written on every
// replay, including empty input. Preparation submits no mathematical work.
std::shared_ptr<NativeOperation> prepare_compact_operation(
    Arch arch,
    Device &device,
    const CompactOperation &operation);

namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_compact_operation(
    Device &device,
    const CompactOperation &operation);
std::shared_ptr<NativeOperation> prepare_vulkan_sort_operation(
    Device &device,
    const SortOperation &operation);
}

}  // namespace taichi::lang
