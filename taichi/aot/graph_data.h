#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include "taichi/ir/type.h"
#include "taichi/program/callable.h"
#include "taichi/aot/module_data.h"
#include "taichi/program/compile_config.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

template <typename T, typename G>
T taichi_union_cast_with_different_sizes(G g);

namespace taichi::lang {
class AotModuleBuilder;
class Ndarray;
class Texture;
class Matrix;
class Kernel;
class CompiledKernelData;

namespace aot {

// Currently only scalar, matrix and ndarray are supported.
enum class ArgKind {
  kScalar,
  kMatrix,
  kNdarray,
  kTexture,
  kRWTexture,
  kUnknown
};

/**
 * Symbolic argument used in building `Dispatch` nodes in the `Graph`.
 */
struct Arg {
  ArgKind tag;
  std::string name;
  // Ndarray: element_dtype = dtype + element_shape
  // Texture: element_shape carries [width, height, depth] info
  //          dtype_id carries channel_format info
  PrimitiveTypeID dtype_id;
  size_t field_dim;
  std::vector<int> element_shape;

  // For texture
  size_t num_channels;  // TODO: maybe rename field_dim and merge?

  // For serialization & deserialization
  explicit Arg()
      : tag(ArgKind::kUnknown),
        name(""),
        dtype_id(PrimitiveTypeID::unknown),
        field_dim(0),
        element_shape({}),
        num_channels(0) {
  }

  explicit Arg(ArgKind tag,
               const std::string &name,

               PrimitiveTypeID dtype_id,
               size_t field_dim,
               const std::vector<int> &element_shape)
      : tag(tag),
        name(name),
        dtype_id(dtype_id),
        field_dim(field_dim),
        element_shape(element_shape),
        num_channels(0) {
  }

  // Python/C++ interface that's user facing.
  explicit Arg(ArgKind tag,
               const std::string &name,
               const DataType &dtype,
               size_t dim = 0,
               const std::vector<int> &element_shape = {})
      : tag(tag), name(name), element_shape(element_shape) {
    field_dim = 0;
    num_channels = 0;
    if (tag == ArgKind::kTexture || tag == ArgKind::kRWTexture) {
      num_channels = dim;
    } else {
      field_dim = dim;
    }
    dtype_id = dtype->as<PrimitiveType>()->type;
  }

  DataType dtype() const {
    return PrimitiveType::get(dtype_id);
  }

  bool operator==(const Arg &other) const {
    return tag == other.tag && name == other.name &&
           field_dim == other.field_dim && dtype_id == other.dtype_id &&
           element_shape == other.element_shape &&
           num_channels == other.num_channels;
  }

  bool operator!=(const Arg &other) const {
    return !(*this == other);
  }

  TI_IO_DEF(name, dtype_id, field_dim, tag, element_shape, num_channels);
};

/**
 * Runtime value used in graph execution.
 */
struct TI_DLL_EXPORT IValue {
 public:
  uint64 val;
  ArgKind tag;

  static IValue create(const Ndarray &ndarray) {
    return IValue(reinterpret_cast<intptr_t>(&ndarray), ArgKind::kNdarray);
  }

  static IValue create(const Texture &tex) {
    return IValue(reinterpret_cast<intptr_t>(&tex), ArgKind::kTexture);
  }

  static IValue create(const Matrix &matrix) {
    return IValue(reinterpret_cast<intptr_t>(&matrix), ArgKind::kMatrix);
  }

  template <typename T,
            typename = std::enable_if_t<!std::is_same<T, Ndarray>::value, void>>
  static IValue create(T v) {
    return IValue(taichi_union_cast_with_different_sizes<uint64>(v),
                  ArgKind::kScalar);
  }

 private:
  IValue(uint64 val, ArgKind tag) : val(val), tag(tag) {
  }
};

class TI_DLL_EXPORT Kernel : public CallableBase {
 public:
  // Rule of 5 to make MSVC happy
  Kernel() = default;
  virtual ~Kernel() = default;
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) = default;
  Kernel &operator=(Kernel &&) = default;

  /**
   * @brief Launches the kernel to the device
   *
   * This does not manage the device to host synchronization.
   *
   * @param ctx Host context
   */
  virtual void launch(LaunchContextBuilder &ctx) = 0;
};

struct CompiledDispatch {
  std::string kernel_name;
  std::vector<Arg> symbolic_args;
  Kernel *compiled_kernel{nullptr};
  taichi::lang::Kernel *ti_kernel{nullptr};

  TI_IO_DEF(kernel_name, symbolic_args);
};

struct CompiledGraphJITCachedKernel {
  std::string kernel_key;
  const CompiledKernelData *compiled_kernel_data{nullptr};
  int llvm_launch_id{-1};
};

struct CompiledGraphRuntimeArgPlan {
  ArgKind tag{ArgKind::kUnknown};
  std::string name;
  std::vector<int> arg_id;
  PrimitiveTypeID dtype_id{PrimitiveTypeID::unknown};
  size_t field_dim{0};
  int type_size{0};
  int arg_buffer_offset{-1};
  std::vector<int> element_shape;
  std::vector<int> ndarray_data_ptr_key;
  std::vector<int> ndarray_grad_ptr_key;
  std::vector<int> ndarray_shape_offsets;
};

struct CompiledGraphDispatchRuntimePlan {
  bool cpu_fast_path{false};
  std::vector<CompiledGraphRuntimeArgPlan> args;
};

struct CompiledGraphCudaState;

struct CompiledGraphCudaStateDeleter {
  void operator()(CompiledGraphCudaState *state) const noexcept;
};

struct CompiledGraphVulkanState;

struct CompiledGraphVulkanStateDeleter {
  void operator()(CompiledGraphVulkanState *state) const noexcept;
};

enum class CompiledGraphBackend : uint8_t {
  none,
  cuda,
  vulkan,
};

enum class CompiledGraphExecutionPath : uint8_t {
  none,
  ordinary_fallback,
  cuda_capture,
  cuda_exact_replay,
  cuda_patched_replay,
  vulkan_record,
  vulkan_replay,
};

enum class CompiledGraphFallbackReason : uint8_t {
  none,
  debug_mode,
  insufficient_dispatches,
  unsupported_arguments,
  resource_unavailable,
  structural_unsupported,
  transient_driver_failure,
  retry_backoff,
  runtime_mode,
  replay_slot_saturated,
};

// Internal diagnostics snapshot. Counters are maintained with integer writes
// on the serialized graph-cache path; labels are constructed only when a
// caller explicitly requests the debug snapshot.
struct CompiledGraphStats {
  CompiledGraphBackend backend{CompiledGraphBackend::none};
  CompiledGraphExecutionPath last_path{CompiledGraphExecutionPath::none};
  CompiledGraphFallbackReason last_fallback_reason{
      CompiledGraphFallbackReason::none};
  uint64_t attempts{0};
  uint64_t ordinary_fallbacks{0};
  uint64_t capture_attempts{0};
  uint64_t captures{0};
  uint64_t exact_replays{0};
  uint64_t patched_replays{0};
  uint64_t recaptures{0};
  uint64_t records{0};
  uint64_t replays{0};
  uint64_t structural_fallbacks{0};
  uint64_t transient_failures{0};
  uint64_t retry_backoff_fallbacks{0};
  uint64_t replay_slot_saturation_fallbacks{0};
  uint64_t capture_exceptions{0};
  uint64_t known_persistent_argument_bytes{0};
  uint64_t last_driver_error{0};
  uint32_t retry_backoff_remaining{0};
  uint32_t consecutive_transient_failures{0};
};

// A transient capture error must not permanently disable an otherwise valid
// graph. Retry periodically with bounded exponential backoff; structural
// incompatibility remains disabled because repeating it cannot recover.
class CompiledGraphCaptureRetryState {
 public:
  static constexpr uint32_t kMaxBackoffInvocations = 32;

  bool should_attempt() noexcept {
    if (structurally_disabled_) {
      return false;
    }
    if (retry_backoff_remaining_ > 0) {
      --retry_backoff_remaining_;
      return false;
    }
    return true;
  }

  void record_success() noexcept {
    consecutive_transient_failures_ = 0;
    retry_backoff_remaining_ = 0;
  }

  void record_structural_failure() noexcept {
    structurally_disabled_ = true;
  }

  void record_transient_failure() noexcept {
    consecutive_transient_failures_ =
        std::min<uint32_t>(consecutive_transient_failures_ + 1, 32);
    const uint32_t shift =
        std::min<uint32_t>(consecutive_transient_failures_ - 1, 5);
    retry_backoff_remaining_ = uint32_t{1} << shift;
  }

  bool structurally_disabled() const noexcept {
    return structurally_disabled_;
  }

  uint32_t consecutive_transient_failures() const noexcept {
    return consecutive_transient_failures_;
  }

  uint32_t retry_backoff_remaining() const noexcept {
    return retry_backoff_remaining_;
  }

 private:
  bool structurally_disabled_{false};
  uint32_t consecutive_transient_failures_{0};
  uint32_t retry_backoff_remaining_{0};
};

class CompiledGraphReplayIdentity {
 public:
  CompiledGraphReplayIdentity()
      : value_(next_value_.fetch_add(1, std::memory_order_relaxed)) {
    // Zero is reserved as an invalid/unregistered token. Reaching it requires
    // exhausting the complete 64-bit process lifetime, at which point
    // continuing would reintroduce identity reuse.
    TI_ASSERT(value_ != 0);
  }
  CompiledGraphReplayIdentity(const CompiledGraphReplayIdentity &) = delete;
  CompiledGraphReplayIdentity &operator=(
      const CompiledGraphReplayIdentity &) = delete;
  CompiledGraphReplayIdentity(CompiledGraphReplayIdentity &&) = delete;
  CompiledGraphReplayIdentity &operator=(
      CompiledGraphReplayIdentity &&) = delete;

  uint64_t value() const noexcept {
    return value_;
  }

 private:
  inline static std::atomic<uint64_t> next_value_{1};
  uint64_t value_{0};
};

struct CompiledGraphJITCache {
  CompiledGraphJITCache() = default;
  ~CompiledGraphJITCache();
  CompiledGraphJITCache(const CompiledGraphJITCache &) = delete;
  CompiledGraphJITCache &operator=(const CompiledGraphJITCache &) = delete;

  // Release all state tied to the current Program/Device while that runtime is
  // still alive. This is intentionally separate from the destructor: Python
  // GC may destroy a graph after ti.reset() has already finalized its Program.
  void clear_runtime_state();
  CompiledGraphStats debug_graph_stats();

  std::vector<CompiledGraphJITCachedKernel> kernels;
  std::vector<CompiledGraphDispatchRuntimePlan> runtime_arg_plans;
  std::unique_ptr<CompiledGraphCudaState, CompiledGraphCudaStateDeleter>
      cuda_graph_state;
  std::unique_ptr<CompiledGraphVulkanState, CompiledGraphVulkanStateDeleter>
      vulkan_graph_state;
  // Detailed counters are opt-in. Failure recovery keeps its own bounded
  // backoff state, so ordinary graph replay does not pay for diagnostics until
  // an internal caller requests _graph_stats.
  bool graph_diagnostics_enabled{false};
  // GfxRuntime may retain replay state after this cache is destroyed. A
  // monotonic token, rather than the reusable cache-object address, prevents a
  // later cache at the same host address from inheriting that state.
  CompiledGraphReplayIdentity graph_replay_identity;
  uint64_t graph_replay_token() const noexcept {
    return graph_replay_identity.value();
  }
  // One replay mutates the cached kernels, argument plans and optional graph
  // capture state as a transaction. Do not expose partially updated state to
  // another caller sharing this cache.
  std::mutex run_mutex;
};

struct TI_DLL_EXPORT CompiledGraph {
  std::vector<CompiledDispatch> dispatches;
  std::unordered_map<std::string, aot::Arg> args;

  CompiledGraph() = default;
  explicit CompiledGraph(std::vector<CompiledDispatch> compiled_dispatches)
      : dispatches(std::move(compiled_dispatches)) {
  }
  CompiledGraph(std::vector<CompiledDispatch> compiled_dispatches,
                std::unordered_map<std::string, aot::Arg> graph_args)
      : dispatches(std::move(compiled_dispatches)),
        args(std::move(graph_args)) {
  }
  CompiledGraph(const CompiledGraph &) = default;
  CompiledGraph &operator=(const CompiledGraph &) = default;
  CompiledGraph(CompiledGraph &&) = default;
  CompiledGraph &operator=(CompiledGraph &&) = default;

  void run(const std::unordered_map<std::string, IValue> &args) const;
  void jit_run(const CompileConfig &compile_config,
               const std::unordered_map<std::string, IValue> &args) const;
  void jit_run_cached(const CompileConfig &compile_config,
                      const std::unordered_map<std::string, IValue> &args,
                      CompiledGraphJITCache &cache) const;

  TI_IO_DEF(dispatches);

  // Internal helper shared by graph replay backends.
  static void init_runtime_context(
      const std::vector<Arg> &paramter_list,
      const std::unordered_map<std::string, IValue> &args,
      LaunchContextBuilder &ctx);
};

}  // namespace aot
}  // namespace taichi::lang
