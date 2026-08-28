#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace cuda {

constexpr std::size_t kMaxOrdinaryLaunchArgRingSize = 8;

struct RetainedLaunchBufferTelemetrySnapshot {
  std::uint64_t current_bytes{0};
  std::uint64_t peak_bytes{0};
  std::uint64_t allocation_calls{0};
  std::uint64_t release_calls{0};
};

struct GridResidencyTelemetrySnapshot {
  std::uint64_t resolution_calls{0};
  std::uint64_t resolution_failures{0};
  std::uint64_t last_requested_waves{0};
  std::uint64_t last_baseline_grid{0};
  std::uint64_t last_resolved_grid{0};
  std::uint64_t last_active_blocks_per_multiprocessor{0};
  std::uint64_t last_multiprocessor_count{0};
};

struct ArtifactQualificationTelemetrySnapshot {
  std::uint64_t qualification_calls{0};
  std::uint64_t registration_materializations{0};
  std::uint64_t function_attribute_queries{0};
  std::uint64_t occupancy_queries{0};
};

RetainedLaunchBufferTelemetrySnapshot
get_retained_launch_buffer_telemetry_snapshot();
GridResidencyTelemetrySnapshot get_grid_residency_telemetry_snapshot();
ArtifactQualificationTelemetrySnapshot
get_artifact_qualification_telemetry_snapshot();

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct RetainedDeviceBuffer {
    RetainedDeviceBuffer() = default;
    RetainedDeviceBuffer(const RetainedDeviceBuffer &) = delete;
    RetainedDeviceBuffer &operator=(const RetainedDeviceBuffer &) = delete;
    ~RetainedDeviceBuffer();

    void *reserve(std::size_t required_bytes) const;
    void release() const;

    mutable void *ptr{nullptr};
    mutable std::size_t capacity{0};
  };

  struct SparseListState {
    int64 dirty_epoch{0};
    int64 clean_epoch{-1};
    int64 clean_parent_version{-1};
    int64 version{0};
    std::uint64_t adaptive_window_bits{0};
    int adaptive_window_size{0};
    int adaptive_hit_count{0};
    bool adaptive_disabled{false};
  };

  struct Context {
    JITModule *jit_module{nullptr};
    std::vector<int> snode_tree_ids;
    std::vector<std::pair<std::vector<int>, Callable::Parameter>> parameters;
    std::vector<OffloadedTask> offloaded_tasks;
    mutable std::once_flag root_binding_once;
    mutable void *root_binding{nullptr};
    mutable std::shared_ptr<void> root_binding_owner;
    mutable std::array<RetainedDeviceBuffer,
                       kMaxOrdinaryLaunchArgRingSize>
        ordinary_arg_buffers;
    mutable std::size_t ordinary_arg_buffer_cursor{0};
    mutable RetainedDeviceBuffer ordinary_result_buffer;
    mutable std::array<std::once_flag, 3> grid_residency_once;
    mutable std::array<std::vector<OffloadedTask>, 3> grid_residency_tasks;
    bool uses_root_binding{false};
  };

 public:
  explicit KernelLauncher(LLVM::KernelLauncher::Config config);

  struct ArtifactQualification {
    std::string entry_point;
    std::uintptr_t function_identity{0};
    int max_threads_per_block{0};
    int static_shared_memory_bytes{0};
    int constant_memory_bytes{0};
    int local_memory_bytes_per_thread{0};
    int registers_per_thread{0};
    int ptx_version{0};
    int binary_version{0};
    int cache_mode_ca{0};
    int max_dynamic_shared_bytes{0};
    int preferred_shared_carveout{0};
    int block_dim{0};
    int dynamic_shared_bytes{0};
    int active_blocks_per_multiprocessor{0};
    int multiprocessor_count{0};
  };

  struct GraphLaunchPacket {
    Handle handle;
    RuntimeContext context;
    void *device_arg_buffer{nullptr};
    std::size_t arg_buffer_size{0};
    std::size_t device_arg_buffer_size{0};
    std::size_t arg_buffer_prefix_size{0};
    std::uintptr_t bounded_extent{0};
    std::uint32_t bounded_capacity{0};
    bool bounded_range{false};
    std::string dispatch_label;
  };

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;
  // Register a private variant whose offloaded task entries uniformly test a
  // device gate before executing. The source kernel and ordinary registration
  // remain untouched, so only internal masked CUDA Graph replay pays the
  // branch cost.
  Handle register_llvm_kernel_graph_gated(
      const LLVM::CompiledKernelData &compiled);
  void retire_snode_tree(int tree_id) override;
  std::size_t debug_registered_kernel_count() override;
  std::vector<ArtifactQualification> qualify_llvm_kernel_artifacts(
      const LLVM::CompiledKernelData &compiled);
  void debug_reset_sparse_listgen_statistics() override;
  SparseSNodeTreeListgenStatistics debug_sparse_listgen_statistics(
      const std::vector<int> &snode_ids) override;
  bool prepare_cuda_graph_launch(Handle handle,
                                 LaunchContextBuilder &ctx,
                                 GraphLaunchPacket &packet,
                                 void *stream);
  bool prepare_cuda_graph_bounded_range(Handle handle,
                                        LaunchContextBuilder &ctx,
                                        GraphLaunchPacket &packet,
                                        void *extent,
                                        std::uint32_t capacity,
                                        void *stream);
  bool prepare_cuda_graph_gated_launch(Handle handle,
                                       LaunchContextBuilder &ctx,
                                       GraphLaunchPacket &packet,
                                       void *gate,
                                       std::uint32_t expected,
                                       void *stream);
  bool update_cuda_graph_launch(const GraphLaunchPacket &packet,
                                LaunchContextBuilder &ctx,
                                std::vector<uint8_t> &host_arg_buffer,
                                void *stream);
  bool update_cuda_graph_bounded_range(
      GraphLaunchPacket &packet,
      void *extent,
      std::uint32_t capacity,
      std::vector<uint8_t> &host_binding,
      void *stream);
  void capture_cuda_graph_launch(const GraphLaunchPacket &packet,
                                 void *stream);
  bool capture_cuda_graph_bounded_launch(const GraphLaunchPacket &packet,
                                         void *stream,
                                         void **device_node,
                                         std::uint32_t *driver_error);
  bool capture_cuda_graph_updatable_launch(const GraphLaunchPacket &packet,
                                           void *stream,
                                           std::vector<void *> *device_nodes,
                                           std::uint32_t *driver_error);

 private:
  bool prepare_cuda_graph_context(Handle handle,
                                  LaunchContextBuilder &ctx,
                                  RuntimeContext &context);
  bool on_cuda_device(void *ptr);
  int64 get_sparse_list_version(int snode_id) const;
  void record_sparse_list_reuse_sample(SparseListState &state,
                                       bool would_skip) const;
  bool sparse_list_task_is_current(const OffloadedTask &task);
  void mark_sparse_list_task_launched(const OffloadedTask &task);
  void invalidate_sparse_list_cache(int sparse_mutation_snode_id);
  void configure_root_binding(const LLVM::CompiledKernelData &compiled,
                              Context &context);
  void ensure_root_binding(const Context &context);
  const std::vector<OffloadedTask> &resolve_grid_residency_tasks(
      const Context &context,
      std::int32_t waves);

  bool listgen_reuse_adaptive_{false};
  // Sparse-list reuse metadata describes one CUDA runtime, not one launch.
  // Dense kernels never take this lock. Sparse list generation/mutation holds
  // it only while checking state and enqueueing the corresponding task so
  // async host callers preserve submission order without a global launch lock.
  std::mutex sparse_list_mutex_;
  std::unordered_map<int, SparseListState> sparse_list_states_;
  bool sparse_listgen_telemetry_enabled_{false};
  std::unordered_map<int, SparseListgenNodeStatistics>
      sparse_listgen_telemetry_;
  std::unordered_map<int, std::shared_ptr<const Context>> contexts_;
  const bool retain_ordinary_launch_buffers_;
  const std::size_t ordinary_launch_arg_ring_size_;
};

}  // namespace cuda
}  // namespace taichi::lang
