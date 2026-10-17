#pragma once

#include <array>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "taichi/rhi/device.h"
#include "taichi/runtime/native_resource.h"
#include "taichi/runtime/cuda/capture_resources.h"
#include "taichi/runtime/cuda/provider_completion_resource.h"

namespace taichi::lang {

// The composition contract of an operation, independent of its provider.
enum class NativeOperationMode : std::uint32_t {
  cpu = 1,
  cuda_capture = 2,
  vulkan_inline = 4,
  cuda_synchronous = 8,
  cuda_ordered = 16,
};

// Cold facts for one unconditional recording sequence, never replay state.
// Structured control must intersect facts at joins instead of treating a
// recorded conditional write as unconditional initialization.
struct NativeRecordingState {
  std::unordered_set<const NativeResource *> initialized;
  // Provider-owned, copyable cold facts (for example, a solver's valid guess).
  // Copies across unconditional phases; never consulted by warm replay.
  template <typename Facts>
  Facts &provider_state(const NativeResource *resource) {
    if (conditional_)
      throw std::invalid_argument(
          "Opaque provider facts require a structured-control merge contract");
    auto &state = provider_states[resource];
    if (!state.has_value())
      state = Facts{};
    return std::any_cast<Facts &>(state);
  }

  // Initialized resources are monotonic facts. A zero-trip body may validate
  // against a copy but cannot publish its newly initialized resources outside.
  // Provider-specific mutable facts need an explicit join/repetition contract.
  NativeRecordingState control_branch() const {
    auto copy = *this;
    copy.conditional_ = true;
    return copy;
  }

 private:
  bool conditional_{false};
  std::unordered_map<const NativeResource *, std::any> provider_states;
};

// Prepared backend commands own their scratch and recording state. The
// enclosing execution retains input/output leases and retires submitted work
// before releasing this object.
class NativeOperation : public NativeResource {
 public:
  ~NativeOperation() override = default;
  virtual NativeOperationMode execution_mode() const {
    if (arch_is_cpu(arch()))
      return NativeOperationMode::cpu;
    if (arch() == Arch::cuda)
      return NativeOperationMode::cuda_capture;
    if (arch() == Arch::vulkan)
      return NativeOperationMode::vulkan_inline;
    throw std::invalid_argument("Unsupported native operation backend");
  }
  virtual void validate_recording(NativeRecordingState &) const {
  }
  virtual void run_cpu() {
    throw std::invalid_argument("Native operation has no CPU implementation");
  }
  virtual void record_cuda(void *stream) {
    throw std::invalid_argument("Native operation has no CUDA recording");
  }
  // Cold identity of operands that CUDA Graph update cannot change (for
  // example, a multidimensional memcpy source). Values, not contents. Shared
  // binding frames compare this before capture, so rejection cannot poison the
  // current executable. Empty means no additional provider restriction.
  virtual std::vector<std::uint64_t> cuda_binding_frame_invariants() const {
    return {};
  }
  // Transfer capture-local storage immediately after each recording, including
  // a failed recording. Repeated captures must produce independent storage.
  virtual std::shared_ptr<CudaGraphCaptureResources> take_cuda_capture_resources() {
    return {};
  }
  // Host invocation under the CUDA owner lock, between whole-context fences.
  // May enqueue work on private streams in the owner's context. Never called
  // while recording; callbacks into Python are not part of this contract.
  virtual void run_cuda_synchronous() {
    throw std::invalid_argument(
        "Native operation has no synchronous CUDA implementation");
  }
  // Host invocation on the supplied CUDA stream, ordered with surrounding
  // graphs. Private streams must join it before returning, including failure.
  // Reservation enqueues nothing and must leave the owner unchanged on error.
  virtual std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource() const {
    return {};
  }
  virtual void reserve_cuda_ordered() {
  }
  virtual void run_cuda_ordered(void *stream) {
    throw std::invalid_argument("Native operation has no ordered CUDA implementation");
  }
  virtual void record_vulkan(Device *device, CommandList *commands) {
    throw std::invalid_argument("Native operation has no Vulkan recording");
  }
  // Required layouts while recording this command. The enclosing GFX recorder
  // owns entry/exit transitions and restores its replay layout invariant.
  virtual std::vector<std::pair<DeviceAllocation, ImageLayout>>
  vulkan_image_bindings() const {
    return {};
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

// Compact row-major C = alpha * op(A) * op(B) + beta * C. Dimensions describe
// op(A) and op(B); transposition changes the physical input shapes. Output
// cannot overlap either input. Scalars and transposes are frozen at prepare.
struct GemmF32Operation {
  static constexpr const char *kind = "gemm_f32";
  static constexpr std::uint32_t version = 1;
  DevicePtr a;
  DevicePtr b;
  DevicePtr output;
  std::size_t rows;
  std::size_t columns;
  std::size_t inner;
  float alpha{1};
  float beta{0};
  bool transpose_a{false};
  bool transpose_b{false};
  std::shared_ptr<NativeResource> context;
};

std::array<std::size_t, 3> gemm_f32_element_counts(const GemmF32Operation &op);
std::shared_ptr<NativeOperation> prepare_gemm_f32_operation(
    Arch arch,
    Device &device,
    const GemmF32Operation &operation);

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
