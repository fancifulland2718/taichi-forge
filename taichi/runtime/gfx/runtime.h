#pragma once
#include "taichi/util/lang_util.h"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "taichi/rhi/device.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/program/compile_config.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/kernel_launcher.h"
#if defined(TI_WITH_VULKAN_POINTER)
#include "taichi/runtime/gfx/snode_allocator.h"
#endif

namespace taichi::lang {
class RuntimeStatistics;
namespace gfx {

using namespace taichi::lang::spirv;

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

using high_res_clock = std::chrono::high_resolution_clock;

// TODO: In the future this isn't necessarily a pointer, since DeviceAllocation
// is already a pretty cheap handle>
using InputBuffersMap =
    std::unordered_map<BufferInfo, DeviceAllocation *, BufferInfoHasher>;

class SNodeTreeManager;
class GfxRuntime;
class GraphReplayRegistry;

TI_DLL_EXPORT uint64_t get_graph_replay_slot_saturation_fallbacks();

enum class GraphReplayLastPath : uint8_t {
  none,
  fallback,
  record,
  replay,
};

enum class GraphReplayFallbackReason : uint8_t {
  none,
  runtime_mode,
  insufficient_tasks,
  structural_unsupported,
  slot_saturated,
};

struct GraphReplayStats {
  uint64_t attempts{0};
  uint64_t recorded{0};
  uint64_t replayed{0};
  uint64_t fallbacks{0};
  uint64_t structural_fallbacks{0};
  uint64_t runtime_mode_fallbacks{0};
  uint64_t slot_saturation_fallbacks{0};
  uint64_t known_persistent_argument_bytes{0};
  GraphReplayLastPath last_path{GraphReplayLastPath::none};
  GraphReplayFallbackReason last_fallback_reason{
      GraphReplayFallbackReason::none};
};

class TI_DLL_EXPORT GraphReplayRegistration {
 public:
  ~GraphReplayRegistration();
  GraphReplayRegistration(const GraphReplayRegistration &) = delete;
  GraphReplayRegistration &operator=(const GraphReplayRegistration &) = delete;
  GraphReplayRegistration(GraphReplayRegistration &&) = delete;
  GraphReplayRegistration &operator=(GraphReplayRegistration &&) = delete;
  uint64_t replay_key() const noexcept {
    return replay_key_;
  }
  GraphReplayStats debug_stats() const;

 private:
  friend class GfxRuntime;
  GraphReplayRegistration(std::shared_ptr<GraphReplayRegistry> registry,
                          uint64_t replay_key);

  std::shared_ptr<GraphReplayRegistry> registry_;
  uint64_t replay_key_{0};
};

class CompiledTaichiKernel {
 public:
  struct RuntimeArrayArg {
    std::vector<int> indices;
    std::vector<int> data_ptr_indices;
    std::vector<int> grad_ptr_indices;
    uint32_t access{0};
  };

  enum class BufferBindingKind : uint8_t {
    Skip,
    ExtArrRw,
    Args,
    ArgPack,
    RetsRw,
    StaticRw,
    StaticLookupRw,
    ChunkedRwArray,
  };

  struct BufferBindingPlan {
    BufferBindingKind kind{BufferBindingKind::Skip};
    BufferInfo buffer;
    int binding{-1};
    uint32_t chunk_count{0};
    DeviceAllocation *static_alloc{nullptr};
    const std::vector<DeviceAllocation> *chunk_array{nullptr};
  };

  struct Params {
    const TaichiKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<std::vector<uint32_t>> spirv_bins;
    std::size_t num_snode_trees{0};

    Device *device{nullptr};
    std::vector<DeviceAllocation *> root_buffers;
    DeviceAllocation *global_tmps_buffer{nullptr};
    DeviceAllocation *hash_overflow_buffer{nullptr};
    DeviceAllocation *listgen_buffer{nullptr};
#if defined(TI_WITH_VULKAN_POINTER)
    // B-3.b (2026-05): 路线 B 阶段开启 vulkan_pointer_independent_pool 后，
    // 每个走独立池的 pointer SNode 对应一块独立 DeviceAllocation；在这里打
    // 包传入，CompiledTaichiKernel 构造时按 sid 注册到 input_buffers_ 以供
    // descriptor set fallback 路径绑定。key = SNode id。默认 (OFF) 为空。
    std::vector<std::pair<int, DeviceAllocation *>> node_allocator_pool_buffers;
    // C-2.5 (2026-05): chunked allocator 的全部 chunk DeviceAllocation 列表。
    // key = SNode id；value = {chunk[0], chunk[1], ..., chunk[N-1]}。
    // 只在 max_chunks > 1 时填充；单 chunk allocator（含 Bump）走上面的
    // node_allocator_pool_buffers 单 buffer 路径。CompiledTaichiKernel 在
    // dispatch 时按 BufferBind.chunk_count>0 调 rw_buffer_array(N)。
    std::vector<std::pair<int, std::vector<DeviceAllocation>>>
        node_allocator_chunk_arrays;
#endif

    PipelineCache *backend_cache{nullptr};
  };

  explicit CompiledTaichiKernel(const Params &ti_params);

  const TaichiKernelAttributes &ti_kernel_attribs() const;

  size_t num_pipelines() const;

  size_t get_args_buffer_size() const;
  size_t get_ret_buffer_size() const;

  Pipeline *get_pipeline(int i);
  ShaderResourceSet *get_cached_resource_set(int i);
  const std::vector<RuntimeArrayArg> &runtime_array_args() const {
    return runtime_array_args_;
  }
  const std::vector<std::vector<int>> &runtime_argpack_args() const {
    return runtime_argpack_args_;
  }
  const std::vector<BufferBindingPlan> &buffer_binding_plan(int i) const {
    return buffer_binding_plans_[i];
  }

  void set_listgen_buffer(DeviceAllocation *listgen_buffer) {
    input_buffers_[BufferInfo(BufferType::ListGen)] = listgen_buffer;
  }

  DeviceAllocation *get_buffer_bind(const BufferInfo &bind) {
    return input_buffers_[bind];
  }

  // C-2.5 (2026-05): retrieve all chunk DeviceAllocations for a chunked
  // NodeAllocatorPool binding. Returns nullptr if `bind` is not a chunked
  // pool. Key = sid stored in bind.root_id.
  const std::vector<DeviceAllocation> *get_chunk_array(
      const BufferInfo &bind) const {
    auto it = chunk_arrays_.find(bind);
    if (it == chunk_arrays_.end()) {
      return nullptr;
    }
    return &it->second;
  }

 private:
  TaichiKernelAttributes ti_kernel_attribs_;
  std::vector<TaskAttributes> tasks_attribs_;

  Device *device_;

  InputBuffersMap input_buffers_;
  // C-2.5 (2026-05): per-binding chunk DeviceAllocation lists for chunked
  // NodeAllocatorPool bindings. Empty when no SNode uses chunked allocator.
  std::unordered_map<BufferInfo, std::vector<DeviceAllocation>, BufferInfoHasher>
      chunk_arrays_;

  size_t args_buffer_size_{0};
  size_t ret_buffer_size_{0};
  std::vector<RuntimeArrayArg> runtime_array_args_;
  std::vector<std::vector<int>> runtime_argpack_args_;
  std::vector<std::vector<BufferBindingPlan>> buffer_binding_plans_;
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
  std::vector<std::unique_ptr<ShaderResourceSet>> cached_resource_sets_;
};

class TI_DLL_EXPORT GfxRuntime {
 public:
  struct Params {
    Device *device{nullptr};
    KernelProfilerBase *profiler{nullptr};
    // VS-1: Vulkan listgen scratch sizing. Defaults keep legacy 32 MiB.
    bool listgen_dynamic_size{false};
    int listgen_buffer_MB{0};
    // VS-2: opt-in descriptor cache plus deferred shader-buffer barriers.
    bool dispatch_cache{false};
    // VS-3: opt-in host-side current-list skip for Vulkan listgen tasks.
    bool listgen_reuse{false};
    // G-4: opt-in per-SNode adaptive downgrade for listgen reuse.
    bool listgen_reuse_adaptive{false};
    // G-1: opt-in ctx args/ret buffer ring with conservative fence-safe
    // recycling. Default OFF preserves legacy allocation behavior.
    bool ctx_buffer_ring{false};
    int ctx_buffer_ring_size{8};
    // G-2: opt-in command-list lazy submit. Default OFF preserves legacy
    // timeout behavior; debug mode can force launch-boundary submits when ON.
    bool cmdlist_lazy_submit{false};
    int cmdlist_max_dispatches{8};
    bool debug{false};
  };

  explicit GfxRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~GfxRuntime();

  using KernelHandle = KernelLauncher::Handle;

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<std::vector<uint32_t>> task_spirv_source_codes;
    std::size_t num_snode_trees{0};
    std::vector<int> snode_tree_ids;
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  // Called only after an explicit Program-side synchronize. Releases compiled
  // pipelines that statically bind the destroyed root allocation. Existing
  // replay registrations survive and can record again for unrelated graphs.
  void retire_snode_tree_kernels(int tree_id);
  std::size_t debug_registered_kernel_count();

  void launch_kernel(KernelHandle handle, LaunchContextBuilder &host_ctx);

  struct GraphDispatch {
    KernelHandle handle;
    LaunchContextBuilder *host_ctx{nullptr};
  };

  struct GraphReplayExecutable {
    using AllocationMap =
        std::unordered_map<std::vector<int>,
                           DeviceAllocation,
                           hashing::Hasher<std::vector<int>>>;

    struct PreparedDispatch {
      CompiledTaichiKernel *kernel{nullptr};
      LaunchContextBuilder *host_ctx{nullptr};
      DeviceAllocation *args_buffer{nullptr};
      const AllocationMap *any_arrays{nullptr};
    };

    struct CachedPreparedDispatch {
      AllocationMap any_arrays;
    };

    struct Slot {
      std::vector<std::unique_ptr<DeviceAllocationGuard>> args_buffers;
      std::vector<size_t> args_buffer_sizes;
      std::vector<std::unique_ptr<ShaderResourceSet>> resource_sets;
      std::unique_ptr<CommandList> cmdlist;
      StreamSemaphore completion;
      std::vector<uint64_t> key;
      bool recorded{false};
    };

    // O4.3: keep this fixed. Bounded elastic growth removed rare fallbacks
    // without a repeatable throughput gain and caused a multi-GiB Vulkan
    // driver-memory high-water mark during graph churn.
    static constexpr size_t kReplaySlots = 8;

    Device *device{nullptr};
    std::vector<CachedPreparedDispatch> cached_prepared;
    std::vector<uint64_t> cached_prepare_key;
    std::vector<Slot> slots;
    size_t next_slot{0};

    void bind_device(Device *new_device);
    bool refresh_prepared_cache(const std::vector<uint64_t> &key,
                                std::vector<PreparedDispatch> &prepared);
    Slot *acquire_ready_slot();
    bool ready_for_retirement() const;
    uint64_t known_persistent_argument_bytes() const;
    void reset();
  };

  struct GraphReplayState {
    GraphReplayExecutable executable;
    uint64_t attempts{0};
    uint64_t recorded{0};
    uint64_t replayed{0};
    uint64_t fallbacks{0};
    uint64_t structural_fallbacks{0};
    uint64_t runtime_mode_fallbacks{0};
    uint64_t slot_saturation_fallbacks{0};
    GraphReplayLastPath last_path{GraphReplayLastPath::none};
    GraphReplayFallbackReason last_fallback_reason{
        GraphReplayFallbackReason::none};
    bool diagnostics_enabled{false};
    bool retirement_requested{false};

    void reset();
  };

  bool try_launch_graph(const std::vector<GraphDispatch> &dispatches,
                        uint64_t replay_key,
                        RuntimeStatistics *statistics);
  std::unique_ptr<GraphReplayRegistration> register_graph_replay(
      uint64_t replay_token);
  bool owns_graph_replay_registration(
      const GraphReplayRegistration &registration) const;

  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size);
  void copy_image(DeviceAllocation dst,
                  DeviceAllocation src,
                  const ImageCopyParams &params);

  DeviceAllocation create_image(const ImageParams &params);
  void track_image(DeviceAllocation image, ImageLayout layout);
  void untrack_image(DeviceAllocation image);
  void transition_image(DeviceAllocation image, ImageLayout layout);

  void synchronize();

  StreamSemaphore flush();
  StreamSemaphore flush_if_pending();
  bool has_pending_command_list() const {
    std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
    return current_cmdlist_ != nullptr;
  }

  Device *get_ti_device() const;

  PipelineCache *get_backend_cache() const;

  void add_root_buffer(size_t root_buffer_size);

  // Install a root buffer at the generation-safe Program SNodeTree id. The
  // indexed overload permits an explicitly destroyed slot to be reused
  // without changing the root binding seen by compiled kernels.
  void add_root_buffer(int root_id, size_t root_buffer_size);

  void remove_root_buffer(int root_id);

  void update_listgen_buffer_for_snode_tree(
      const CompiledSNodeStructs &compiled_structs);

  DeviceAllocation *get_root_buffer(int id) const;

  size_t get_root_buffer_size(int id) const;

  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs);

  bool used_in_kernel(DeviceAllocationId id) {
    std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
    return ndarrays_in_use_.count(id) > 0 || argpacks_in_use_.count(id) > 0;
  }

  static std::pair<const lang::StructType *, size_t>
  get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                   const std::string &layout);

  static std::tuple<const lang::StructType *, size_t, size_t>
  get_struct_type_with_data_layout_impl(const lang::StructType *old_ty,
                                        const std::string &layout);

 private:
  friend class taichi::lang::gfx::SNodeTreeManager;
  friend class GraphReplayRegistry;

  // GfxRuntime owns a single mutable command-recording state (command list,
  // barriers, descriptor caches, temporary buffers, image layouts, and graph
  // replay tables). Public host API calls may arrive from different Python
  // threads, so they must not mutate that state concurrently. Recursive
  // locking is intentional: public operations compose other public operations
  // (for example copy_image -> transition_image and synchronize -> flush).
  mutable std::recursive_mutex host_api_mutex_;

  void ensure_current_cmdlist();
  void submit_current_cmdlist_if_timeout();

  void init_nonroot_buffers();
  void ensure_listgen_buffer_bytes(size_t requested_bytes,
                                   const char *reason);
  void ensure_listgen_capacity_entries(size_t requested_entries,
                                       const char *reason);
  void ensure_listgen_capacity_for_kernel(const CompiledTaichiKernel &kernel);
  void insert_pending_dispatch_barriers();
  void add_pending_dispatch_barrier(DeviceAllocation alloc);
  void clear_pending_dispatch_barriers();
  bool task_uses_listgen_buffer(const TaskAttributes &attribs) const;
  int64 get_sparse_list_version(int snode_id) const;
  struct SparseListState;
  void record_sparse_list_reuse_sample(SparseListState &state,
                                       bool would_skip) const;
  bool sparse_list_task_is_current(const TaskAttributes &attribs);
  void mark_sparse_list_task_launched(const TaskAttributes &attribs);
  void invalidate_sparse_list_cache(int sparse_mutation_snode_id);
  void clear_sparse_list_cache_resident();
  void register_hash_overflow_checks(
      int root_id,
      const CompiledSNodeStructs &compiled_structs);
  void check_hash_overflow_counters();
  void synchronize_impl(bool check_hash_overflow);
  void retire_graph_replay(uint64_t replay_token);
  GraphReplayStats debug_graph_replay_stats(uint64_t replay_token);
  void collect_ready_graph_replays();

  struct HashOverflowWatch {
    int root_id{-1};
    int snode_id{-1};
    size_t overflow_byte_offset{0};
    size_t active_byte_offset{0};
    size_t tombstone_byte_offset{static_cast<size_t>(-1)};
  };

  Device *device_{nullptr};
  KernelProfilerBase *profiler_;

  std::unique_ptr<PipelineCache> backend_cache_{nullptr};

  std::vector<std::unique_ptr<DeviceAllocationGuard>> root_buffers_;
#if defined(TI_WITH_VULKAN_POINTER)
  // 路线 B B-1（2026-04-30）：每棵 SNode tree 上每个 pointer SNode 对应
  // 一个 DeviceNodeAllocator。outer key = root_id（与 root_buffers_ 同步），
  // 内层 map key = SNode id。当前阶段池仍寄居 root_buffer 子区间，allocator
  // 只是「字节等价的间接层」；B-3 阶段池将迁出 root_buffer。
  // 注：用 unordered_map 作外层是为了规避 MSVC 上 vector 增长时对不含
  // noexcept move 的 map 的硬性 copy 需求。
  std::unordered_map<int,
                     std::unordered_map<int, std::unique_ptr<DeviceNodeAllocator>>>
      node_allocators_;
#endif
  std::unique_ptr<DeviceAllocationGuard> global_tmps_buffer_;
  std::unique_ptr<DeviceAllocationGuard> hash_overflow_buffer_;
  std::unique_ptr<DeviceAllocationGuard> listgen_buffer_;
  bool listgen_dynamic_size_{false};
  bool listgen_explicit_size_{false};
  size_t listgen_initial_buffer_size_{0};
  size_t listgen_buffer_size_{0};
  size_t listgen_capacity_entries_{0};
  bool listgen_buffer_used_{false};
  bool dispatch_cache_{false};
  bool listgen_reuse_{false};
  struct SparseListState {
    int64 dirty_epoch{0};
    int64 clean_epoch{-1};
    int64 global_dirty_seen{-1};
    int64 version{0};
    int64 clean_parent_version{-1};
    int parent_snode_id{-1};
    std::uint64_t adaptive_window_bits{0};
    int adaptive_window_size{0};
    int adaptive_hit_count{0};
    bool adaptive_disabled{false};
  };
  bool listgen_reuse_adaptive_{false};
  std::unordered_map<int, SparseListState> sparse_list_states_;
  std::unordered_map<int, std::unordered_set<int>> child_lists_by_parent_;
  int64 sparse_list_global_dirty_epoch_{0};
  int resident_sparse_list_snode_id_{-1};
  std::vector<HashOverflowWatch> hash_overflow_watches_;
  bool hash_overflow_error_reported_{false};
  std::unordered_map<uint64_t, GraphReplayState> graph_replay_states_;
  std::shared_ptr<GraphReplayRegistry> graph_replay_registry_;
  uint64_t next_graph_replay_registration_id_{1};
  bool pending_dispatch_global_barrier_{false};
  std::vector<DeviceAllocation> pending_dispatch_barrier_buffers_;
  std::unordered_set<DeviceAllocationId> pending_dispatch_barrier_buffer_ids_;

  std::vector<std::unique_ptr<DeviceAllocationGuard>> ctx_buffers_;

  // G-1: ctx args/ret buffer ring. Three-stage to be GPU-safe:
  //   pending_pool_   : in-flight on current_cmdlist_ (recording)
  //   submitted_pool_ : submitted to stream, GPU may still be using; entries
  //                     carry an optional completion token when backend exposes
  //                     one (Vulkan submit fence)
  //   free_pool_      : safe to reuse (filled after token completion or sync)
  // Each entry is keyed by (size, usage). Pool cap drops oldest free entries.
  struct PooledBuffer {
    std::unique_ptr<DeviceAllocationGuard> guard;
    size_t size{0};
    AllocUsage usage{AllocUsage::None};
    StreamSemaphore completion;
  };
  size_t buffer_pool_capacity_{64};
  bool ctx_buffer_ring_enabled_{false};
  size_t ctx_buffer_ring_size_{8};
  std::vector<PooledBuffer> pending_pool_;
  std::vector<PooledBuffer> submitted_pool_;
  std::vector<PooledBuffer> free_pool_;

  // Try to take a buffer from free_pool_ matching (size, usage). Returns
  // nullptr if pool disabled or no match.
  std::unique_ptr<DeviceAllocationGuard> try_take_pooled_buffer(
      size_t size,
      AllocUsage usage);
  std::unique_ptr<DeviceAllocationGuard> acquire_ctx_buffer(size_t size,
                                                            AllocUsage usage);
  size_t count_pooled_buffers(size_t size, AllocUsage usage) const;
  bool ctx_buffer_pool_enabled() const;
  // Move pending_pool_ -> submitted_pool_ (called on flush), tagging all
  // entries with the submission completion token returned by Stream::submit().
  void flush_pending_pool_to_submitted(StreamSemaphore completion);
  // Move already completed submitted entries to free_pool_ without blocking.
  size_t recycle_completed_pools_to_free();
  // Wait only for the oldest submitted buffer of the requested size/usage if
  // the backend exposes a completion token. Returns false if unsupported.
  bool wait_for_oldest_submitted_buffer(size_t size, AllocUsage usage);
  // Move submitted_pool_ + pending_pool_ -> free_pool_ (called after wait_idle).
  void recycle_pools_to_free();

  std::unique_ptr<CommandList> current_cmdlist_{nullptr};
  high_res_clock::time_point current_cmdlist_pending_since_;
  size_t current_cmdlist_dispatch_count_{0};
  bool cmdlist_lazy_submit_enabled_{false};
  size_t cmdlist_lazy_submit_min_dispatches_{8};
  bool debug_mode_{false};

  std::unordered_map<int, std::unique_ptr<CompiledTaichiKernel>> ti_kernels_;
  std::unordered_map<int, std::vector<int>> ti_kernel_snode_tree_ids_;
  int next_ti_kernel_id_{0};

  std::unordered_map<DeviceAllocation *, size_t> root_buffers_size_map_;
  std::unordered_map<DeviceAllocationId, ImageLayout> last_image_layouts_;
  // [Note] Why do we need to track ndarrays that are in use?
  // Since we separate cmdlist is async, taichi needs a way to know whether
  // ndarrays are still used by pending kernels to be executed. So we use
  // ndarray_in_use_ to track this so that we can free memory allocated for
  // ndarray whenever it's safe to do so.
  std::unordered_set<DeviceAllocationId> ndarrays_in_use_;
  std::unordered_set<DeviceAllocationId> argpacks_in_use_;
};

GfxRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    Arch arch,
    const DeviceCapabilityConfig &caps,
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    const CompileConfig &compile_config);

}  // namespace gfx
}  // namespace taichi::lang
