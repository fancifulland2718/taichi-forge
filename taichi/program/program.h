// Program  - Taichi program execution context

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <atomic>
#include <deque>
#include <stack>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <taichi/program/runtime_resource_registry.h>
#include <taichi/program/runtime_completion.h>
#include <taichi/program/runtime_fault.h>
#include <taichi/program/runtime_trace.h>
#include <taichi/program/primitive_workspace.h>

#define TI_RUNTIME_HOST
#include "taichi/aot/module_builder.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/type_factory.h"
#include "taichi/ir/snode.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/argpack.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/callable.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/snode_rw_accessors_bank.h"
#include "taichi/program/context.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/system/threading.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/ir/mesh.h"

namespace taichi::lang {

struct VulkanSparseAssemblyDispatchInfo {
  std::size_t radix_sort_workspace_bytes{0};
  std::size_t scan_workspace_bytes{0};
  bool workspace_growth_synchronized{false};
};

struct CudaSparseAssemblyDispatchInfo {
  std::size_t radix_sort_workspace_bytes{0};
  std::size_t scan_workspace_bytes{0};
  bool workspace_growth_synchronized{false};
};

class Program;
class ExternalSynchronizationDomain;
class ExternalAccessEpoch;
struct ExternalStreamDomain;
namespace storage {
class DenseStorageDescriptor;
class RuntimeStorageArgument;
struct ResolvedDenseBinding;
struct StorageOwnerRef;
}  // namespace storage
namespace runtime_completion_detail {
Program *&active_runtime_submission_program() noexcept;
}  // namespace runtime_completion_detail

class StructCompiler;

/**
 * Note [Backend-specific ProgramImpl]
 * We're working in progress to keep Program class minimal and move all backend
 * specific logic to their corresponding backend ProgramImpls.

 * If you are thinking about exposing/adding attributes/methods to Program
 class,
 * please first think about if it's general for all backends:
 * - If so, please consider adding it to ProgramImpl class first.
 * - Otherwise please add it to a backend-specific ProgramImpl, e.g.
 * LlvmProgramImpl, MetalProgramImpl..
 */

class TI_DLL_EXPORT Program {
 public:
  using Kernel = taichi::lang::Kernel;

  class SNodeTreeLifecycleReadGuard {
   public:
    SNodeTreeLifecycleReadGuard(const SNodeTreeLifecycleReadGuard &) = delete;
    SNodeTreeLifecycleReadGuard &operator=(
        const SNodeTreeLifecycleReadGuard &) = delete;
    SNodeTreeLifecycleReadGuard(SNodeTreeLifecycleReadGuard &&other) noexcept;
    SNodeTreeLifecycleReadGuard &operator=(
        SNodeTreeLifecycleReadGuard &&other) = delete;
    ~SNodeTreeLifecycleReadGuard();

    std::uint64_t epoch() const {
      return epoch_;
    }

   private:
    friend class Program;
    explicit SNodeTreeLifecycleReadGuard(Program *program);

    Program *program_{nullptr};
    Program *previous_program_{nullptr};
    std::shared_lock<std::shared_mutex> lock_;
    std::uint64_t epoch_{0};
  };

  // Graph owns one resource submission transaction around all dispatches.
  // This scope records that fact thread-locally so nested Program launches can
  // trust the Graph's already retained runtime arguments instead of repeating
  // registry lookups for every dispatch.
  class RuntimeResourceGraphScope {
   public:
    RuntimeResourceGraphScope(const RuntimeResourceGraphScope &) = delete;
    RuntimeResourceGraphScope &operator=(const RuntimeResourceGraphScope &) =
        delete;
    RuntimeResourceGraphScope(RuntimeResourceGraphScope &&other) noexcept;
    RuntimeResourceGraphScope &operator=(RuntimeResourceGraphScope &&) = delete;
    ~RuntimeResourceGraphScope();
    void finish_external_access_epoch();

   private:
    friend class Program;
    explicit RuntimeResourceGraphScope(Program *program);

    Program *program_{nullptr};
    Program *previous_program_{nullptr};
    std::unique_lock<std::recursive_mutex> lock_;
    RuntimeResourceGraphScope *previous_scope_{nullptr};
    std::unique_ptr<ExternalAccessEpoch> external_access_epoch_;
  };

  // Shared for ordinary submissions, exclusive for completion recording and
  // Program synchronization. Nested Graph dispatches reuse the outer scope
  // through a thread-local marker, avoiding one shared_mutex operation per
  // segment while preserving a linearizable completion boundary.
  class RuntimeSubmissionScope {
   public:
    RuntimeSubmissionScope(const RuntimeSubmissionScope &) = delete;
    RuntimeSubmissionScope &operator=(const RuntimeSubmissionScope &) = delete;
    RuntimeSubmissionScope(RuntimeSubmissionScope &&other) noexcept;
    RuntimeSubmissionScope &operator=(RuntimeSubmissionScope &&) = delete;
    ~RuntimeSubmissionScope();

   private:
    friend class Program;
    explicit RuntimeSubmissionScope(Program *program);

    Program *program_{nullptr};
    Program *previous_program_{nullptr};
    bool owns_reader_{false};
  };

  // Private F2 bridge used by Graph.submit() plumbing. It keeps one reader
  // boundary around every Python mixed-Graph segment and records exactly one
  // completion after the outer boundary is closed. The object must be
  // finished or destroyed on the thread that created it because the nested
  // submission marker is thread-local.
  class RuntimeSubmissionTransaction {
   public:
    RuntimeSubmissionTransaction(const RuntimeSubmissionTransaction &) =
        delete;
    RuntimeSubmissionTransaction &operator=(
        const RuntimeSubmissionTransaction &) = delete;
    ~RuntimeSubmissionTransaction();

    void mark_submission() noexcept;
    RuntimeCompletion finish();

   private:
    friend class Program;
    explicit RuntimeSubmissionTransaction(Program *program);

    Program *program_{nullptr};
    std::optional<RuntimeSubmissionScope> submission_scope_;
    bool finished_{false};
  };

  uint64 *result_buffer{nullptr};  // Note that this result_buffer is used
                                   // only for runtime JIT functions (e.g.
                                   // `runtime_memory_allocate_aligned`)

  std::vector<std::unique_ptr<Kernel>> kernels;

  std::unique_ptr<KernelProfilerBase> profiler{nullptr};

  // Note: for now we let all Programs share a single TypeFactory for smooth
  // migration. In the future each program should have its own copy.
  static TypeFactory &get_type_factory();

  Program() : Program(default_compile_config.arch) {
  }

  explicit Program(Arch arch);

  ~Program();

  const CompileConfig &compile_config() const {
    return compile_config_;
  }

  struct KernelProfilerQueryResult {
    int counter{0};
    double min{0.0};
    double max{0.0};
    double avg{0.0};
  };

  KernelProfilerQueryResult query_kernel_profile_info(const std::string &name) {
    KernelProfilerQueryResult query_result;
    profiler->query(name, query_result.counter, query_result.min,
                    query_result.max, query_result.avg);
    return query_result;
  }

  void clear_kernel_profile_info() {
    profiler->clear();
  }

  void profiler_start(const std::string &name) {
    profiler->start(name);
  }

  void profiler_stop() {
    profiler->stop();
  }

  KernelProfilerBase *get_profiler() {
    return profiler.get();
  }

  void synchronize();

  TI_FORCE_INLINE void ensure_runtime_submission_allowed(
      const char *operation) const {
    if (!runtime_fault_domain_->submission_allowed()) {
      runtime_fault_domain_->throw_if_submission_disallowed(operation);
    }
  }
  bool runtime_has_fatal_fault() const noexcept {
    return runtime_fault_domain_->has_fatal_fault();
  }
  RuntimeFaultSnapshot runtime_fault_snapshot() const {
    return runtime_fault_domain_->snapshot();
  }
  RuntimeStatisticsSnapshot runtime_statistics_snapshot();
  PrimitiveWorkspaceSnapshot primitive_workspace_snapshot() const noexcept {
    return primitive_workspace_arena_.snapshot();
  }
  PrimitiveWorkspaceArena &primitive_workspace_arena() noexcept {
    return primitive_workspace_arena_;
  }
  const PrimitiveWorkspaceArena &primitive_workspace_arena() const noexcept {
    return primitive_workspace_arena_;
  }
  void set_primitive_workspace_budget(std::uint64_t bytes) noexcept {
    primitive_workspace_arena_.set_budget_bytes(bytes);
  }
  void clear_primitive_workspaces();
  void clear_primitive_workspaces_for(PrimitiveWorkspaceBackend backend,
                                      PrimitiveWorkspaceFamily family);
  RuntimeStatistics &runtime_statistics() noexcept {
    return runtime_fault_domain_->statistics();
  }
  RuntimeTraceRecorder &runtime_trace() noexcept {
    return runtime_trace_;
  }
  const RuntimeTraceRecorder &runtime_trace() const noexcept {
    return runtime_trace_;
  }
  void report_backend_runtime_error(
      const BackendRuntimeError &error,
      std::uint64_t submission_sequence = 0) noexcept {
    runtime_fault_domain_->report_backend_error(error, submission_sequence);
  }
  void debug_inject_runtime_fault(std::int64_t backend_code,
                                  const std::string &operation,
                                  const std::string &message);

  // F2 internal completion API. Existing kernel calls, Graph.run(), native
  // primitives and ti.sync() retain their public return values.
  RuntimeCompletion record_runtime_completion();
  std::unique_ptr<RuntimeSubmissionTransaction>
  begin_runtime_submission_transaction();
  TI_FORCE_INLINE void mark_runtime_submission_pending() noexcept {
    runtime_submission_pending_.store(true, std::memory_order_relaxed);
  }
  TI_FORCE_INLINE void mark_runtime_submission(
      RuntimeSubmissionKind kind = RuntimeSubmissionKind::kKernel) noexcept {
    // The reader gate supplies ordering once completion tracking is enabled.
    // The dirty publication and schema-v1 telemetry are independent relaxed
    // operations; neither imposes cross-counter event ordering.
    mark_runtime_submission_pending();
    runtime_fault_domain_->statistics().record_submission(kind);
    if (runtime_trace_.enabled()) {
      runtime_trace_.record_instant(runtime_trace_kind(kind));
    }
  }
  TI_FORCE_INLINE void record_runtime_submission_stat(
      RuntimeSubmissionKind kind) noexcept {
    runtime_fault_domain_->statistics().record_submission(kind);
    if (runtime_trace_.enabled()) {
      runtime_trace_.record_instant(runtime_trace_kind(kind));
    }
  }
  TI_FORCE_INLINE void record_runtime_submission_failure() noexcept {
    runtime_fault_domain_->statistics().record_submission_failure();
  }
  std::unordered_map<std::string, std::uint64_t>
  debug_runtime_completion_stats() const;

  StreamSemaphore flush();
  StreamSemaphore flush_if_pending();
  bool has_pending_gfx_command_list() const;

  /**
   * Materializes the runtime.
   */
  void materialize_runtime();

  int get_snode_tree_size();

  std::vector<int> get_active_snode_tree_ids() const;

  SparseSNodeTreeStatistics debug_sparse_snode_tree_statistics(int tree_id);

  void debug_reset_sparse_listgen_statistics();

  SNodeTreeLifecycleReadGuard acquire_snode_tree_lifecycle_read_guard();

  std::vector<SNodeTreeDependency> snapshot_snode_tree_dependencies(
      const std::vector<int> &tree_ids) const;

  void validate_snode_tree_dependencies(
      const std::vector<SNodeTreeDependency> &dependencies) const;

  std::uint64_t snode_tree_mutation_epoch() const {
    return snode_tree_mutation_epoch_.load(std::memory_order_acquire);
  }

  Kernel &kernel(const std::function<void(Kernel *)> &body,
                 const std::string &name = "",
                 AutodiffMode autodiff_mode = AutodiffMode::kNone) {
    // Expr::set_allow_store(true);
    auto func = std::make_unique<Kernel>(*this, body, name, autodiff_mode);
    // Expr::set_allow_store(false);
    kernels.emplace_back(std::move(func));
    return *kernels.back();
  }

  Function *create_function(const FunctionKey &func_key);

  const CompiledKernelData &compile_kernel(const CompileConfig &compile_config,
                                           const DeviceCapabilityConfig &caps,
                                           const Kernel &kernel_def);

  const CompiledKernelData *find_cached_kernel(
      const CompileConfig &compile_config,
      const std::string &kernel_key,
      const Kernel &kernel_def);

  // P5.b — parallel batch compilation. Compiles every kernel in `kernels`
  // through the shared KernelCompilationManager, dispatching to
  // `compile_config.num_compile_threads` worker threads. Kernel order is
  // irrelevant: each Kernel is already self-contained C++-level IR, so no
  // inter-kernel dependency exists at this layer. SNode tree lifetime must
  // be stable across this call (do not call destroy_snode_tree concurrently).
  void compile_kernels(const CompileConfig &compile_config,
                       const std::vector<const Kernel *> &kernels);

  // V7 (2026-04-26) — detector used by KernelCodeGen::compile_kernel_to_module
  // to know whether the calling thread is currently acting as a
  // compile_kernels outer worker. When true, the LLVM codegen path skips its
  // own inner compilation_workers pool to avoid double-pool oversubscription
  // (see compile_doc/P5_\u5e76\u884c\u7f16\u8bd1.md and \u4f18\u5316\u603b\u89c4\u5212.md \u00a73.4).
  // Only set when compile_config.compile_dag_scheduler is true.
  static bool in_compile_kernels_worker();

  void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                     LaunchContextBuilder &ctx);

  void launch_registered_kernel(
      const CompiledKernelData &compiled_kernel_data,
      KernelLaunchHandle handle,
      LaunchContextBuilder &ctx);

  // Python ordinary-kernel fast path. Keep cache lookup/compilation and launch
  // in one SNodeTree lifecycle read transaction so explicit tree destruction
  // cannot retire CompiledKernelData or a backend handle in the call gap.
  void compile_and_launch_kernel(const CompileConfig &compile_config,
                                 const DeviceCapabilityConfig &caps,
                                 const Kernel &kernel_def,
                                 LaunchContextBuilder &ctx);

  void check_runtime_error_after_kernel_launch(
      const CompiledKernelData &compiled_kernel_data);

  KernelLauncher &get_kernel_launcher() {
    return program_impl_->get_kernel_launcher();
  }

  DeviceCapabilityConfig get_device_caps() {
    return program_impl_->get_device_caps();
  }

  Kernel &get_snode_reader(SNode *snode);

  Kernel &get_snode_writer(SNode *snode);

  uint64 fetch_result_uint64(int i);

  template <typename T>
  T fetch_result(int i) {
    return taichi_union_cast_with_different_sizes<T>(fetch_result_uint64(i));
  }

  Arch get_host_arch() {
    return host_arch();
  }

  float64 get_total_compilation_time() {
    return total_compilation_time_;
  }

  void finalize();

  static int get_kernel_id() {
    static int id = 0;
    TI_ASSERT(id < 100000);
    return id++;
  }

  static int default_block_dim(const CompileConfig &config);

  // Note this method is specific to LlvmProgramImpl, but we keep it here since
  // it's exposed to python.
  void print_memory_profiler_info();

  // Returns zero if the SNode is statically allocated
  std::size_t get_snode_num_dynamically_allocated(SNode *snode);

  void reset_hash_snode_probe_stats();

  std::vector<int64> get_hash_snode_probe_stats();

  inline SNodeFieldMap *get_snode_to_fields() {
    return &snode_to_fields_;
  }

  inline SNodeRwAccessorsBank &get_snode_rw_accessors_bank() {
    return snode_rw_accessors_bank_;
  }

  /**
   * Destroys a new SNode tree.
   *
   * @param snode_tree The pointer to SNode tree.
   */
  void destroy_snode_tree(SNodeTree *snode_tree);

  /**
   * Adds a new SNode tree.
   *
   * @param root The root of the new SNode tree.
   * @param compile_only Only generates the compiled type
   * @return The pointer to SNode tree.
   *
   * FIXME: compile_only is mostly a hack to make AOT & cross-compilation work.
   * E.g. users who would like to AOT to a specific target backend can do so,
   * even if their platform doesn't support that backend. Unfortunately, the
   * current implementation would leave the backend in a mostly broken state. We
   * need a cleaner design to support both AOT and JIT modes.
   */
  SNodeTree *add_snode_tree(std::unique_ptr<SNode> root, bool compile_only);

  /**
   * Allocates a SNode tree id for a new SNode tree
   *
   * @return The SNode tree id allocated
   *
   * Returns and consumes a free SNode tree id if there is any,
   * Otherwise returns the size of `snode_trees_`
   */
  int allocate_snode_tree_id();

  /**
   * Gets the root of a SNode tree.
   *
   * @param tree_id Index of the SNode tree
   * @return Root of the tree
   */
  SNode *get_snode_root(int tree_id);

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      Arch arch,
      const std::vector<std::string> &caps);

  size_t get_field_in_tree_offset(int tree_id, const SNode *child) {
    return program_impl_->get_field_in_tree_offset(tree_id, child);
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) {
    return program_impl_->get_snode_tree_device_ptr(tree_id);
  }

  DevicePtr get_dense_field_device_ptr(SNode *snode);

  std::size_t get_dense_field_stride(SNode *snode, std::size_t value_size);

  Device *get_compute_device() {
    return program_impl_->get_compute_device();
  }

  Device *get_graphics_device() {
    return program_impl_->get_graphics_device();
  }

  // Internal CPU-native primitive scheduler. The returned object is owned by
  // this Program's LLVM runtime and is never a process-global worker pool.
  ThreadPool *get_cpu_thread_pool();

  // Process-unique identity for this Program lifetime. Unlike the Program
  // address, this token cannot alias a later Program after reset/destruction.
  std::uint64_t runtime_program_generation() const noexcept {
    return runtime_completion_domain_;
  }

  // Internal host-API transaction guard used by native primitives. It does
  // not wait for GPU completion; it only prevents reset/retire from crossing
  // the interval in which a native entry point validates and submits work.
  using RuntimeResourceSubmissionGuard =
      std::unique_lock<std::recursive_mutex>;
  using NdarrayResourceRegistry = RuntimeResourceRegistry<Ndarray>;
  using NdarrayResourceHandle = NdarrayResourceRegistry::Handle;
  using NdarrayResourceLease = NdarrayResourceRegistry::Lease;
  using TextureResourceRegistry = RuntimeResourceRegistry<Texture>;
  using TextureResourceHandle = TextureResourceRegistry::Handle;
  using TextureResourceLease = TextureResourceRegistry::Lease;
  RuntimeResourceSubmissionGuard acquire_runtime_resource_submission_guard() {
    ensure_runtime_submission_allowed("runtime resource submission");
    return RuntimeResourceSubmissionGuard(runtime_resource_submission_mutex_);
  }
  RuntimeResourceGraphScope acquire_runtime_resource_graph_scope() {
    return RuntimeResourceGraphScope(this);
  }
  RuntimeSubmissionScope acquire_runtime_submission_scope() {
    return RuntimeSubmissionScope(this);
  }

  RuntimeResourceHandle capture_argpack_resource_handle(
      const ArgPack *view) const;
  // Internal Graph/native bridge. Callers hold the returned submission guard
  // across backend work, retain runtime arguments once per transaction, and
  // resolve per-dispatch DeviceAllocation placeholders without changing the
  // public Graph API.
  void retain_ndarrays_for_external_submission(
      const std::vector<const Ndarray *> &views);
  void retain_ndarrays_for_external_submission(const Ndarray *const *views,
                                                std::size_t count);
  void retain_runtime_storage_for_graph_submission(
      const storage::RuntimeStorageArgument *const *arguments,
      std::size_t count);
  storage::ResolvedDenseBinding
  resolve_runtime_storage_argument_under_graph_guard(
      const storage::RuntimeStorageArgument &argument);
  void validate_ndarrays_for_external_submission(
      const std::vector<const Ndarray *> &views);
  void validate_ndarrays_for_external_submission(const Ndarray *const *views,
                                                  std::size_t count);
  void resolve_ndarray_launch_context(LaunchContextBuilder &ctx);
  void resolve_ndarray_launch_context_under_guard(LaunchContextBuilder &ctx);
  void resolve_runtime_storage_launch_context_under_guard(
      LaunchContextBuilder &ctx);
  void retain_textures_for_external_submission(
      const std::vector<const Texture *> &views);
  void retain_textures_for_external_submission(const Texture *const *views,
                                                std::size_t count);
  void validate_textures_for_external_submission(
      const std::vector<const Texture *> &views);
  void validate_textures_for_external_submission(const Texture *const *views,
                                                  std::size_t count);
  void resolve_texture_launch_context(LaunchContextBuilder &ctx);
  void resolve_texture_launch_context_under_guard(LaunchContextBuilder &ctx);
  NdarrayResourceLease acquire_ndarray_external_lease(
      RuntimeResourceHandle handle);
  TextureResourceLease acquire_texture_external_lease(const Texture *view);

  // TODO: do we still need result_buffer?
  DeviceAllocation allocate_memory_on_device(std::size_t alloc_size,
                                             uint64 *result_buffer) {
    return program_impl_->allocate_memory_on_device(alloc_size, result_buffer);
  }
  DeviceAllocation allocate_memory_on_device(std::size_t alloc_size,
                                             uint64 *result_buffer,
                                             AllocUsage usage) {
    return program_impl_->allocate_memory_on_device(alloc_size, result_buffer,
                                                   usage);
  }
  DeviceAllocation allocate_texture(const ImageParams &params) {
    return program_impl_->allocate_texture(params);
  }

  Ndarray *create_ndarray(
      const DataType type,
      const std::vector<int> &shape,
      ExternalArrayLayout layout = ExternalArrayLayout::kNull,
      bool zero_fill = false,
      const DebugInfo &dbg_info = DebugInfo());

  ArgPack *create_argpack(const DataType dt);

  std::string get_kernel_return_data_layout() {
    return program_impl_->get_kernel_return_data_layout();
  };

  std::string get_kernel_argument_data_layout() {
    return program_impl_->get_kernel_argument_data_layout();
  };

  std::pair<const StructType *, size_t> get_struct_type_with_data_layout(
      const StructType *old_ty,
      const std::string &layout);

  std::pair<const ArgPackType *, size_t> get_argpack_type_with_data_layout(
      const ArgPackType *old_ty,
      const std::string &layout);

  void delete_ndarray(Ndarray *ndarray);

  void delete_argpack(ArgPack *argpack);

  void delete_texture(Texture *texture);

  std::unordered_map<std::string, std::uint64_t>
  debug_argpack_resource_stats() const;
  std::unordered_map<std::string, std::uint64_t>
  debug_argpack_resource_identity(const ArgPack *view) const;

  std::unordered_map<std::string, std::uint64_t>
  debug_ndarray_resource_stats() const;
  std::unordered_map<std::string, std::uint64_t>
  debug_ndarray_resource_identity(const Ndarray *view) const;

  std::unordered_map<std::string, std::uint64_t>
  debug_texture_resource_stats() const;
  std::unordered_map<std::string, std::uint64_t>
  debug_texture_resource_identity(const Texture *view) const;
  std::unordered_map<std::string, std::uint64_t>
  debug_dense_field_staging_stats();
  std::unordered_map<std::string, std::uint64_t>
  debug_dense_storage_binding_stats() const;

  using ExternalDenseStorageRelease = std::function<void()>;
  // Registers a range that is already addressable by this Program's compute
  // Device. This is a lifecycle/binding primitive, not an OS-handle or raw
  // CUDA/Vulkan memory importer. On success, release is invoked exactly once
  // after retirement/finalization and all submission leases have drained. It
  // may run on a completion or teardown thread and should be nonblocking.
  // Failed registration does not transfer ownership or invoke release.
  storage::StorageOwnerRef register_external_dense_storage(
      DeviceAllocation allocation,
      std::uint64_t allocation_bytes,
      ExternalDenseStorageRelease release = {},
      std::shared_ptr<ExternalSynchronizationDomain> synchronization_domain =
          {});
  void retire_external_dense_storage(const storage::StorageOwnerRef &owner);
  bool validate_external_dense_storage_owner(
      const storage::StorageOwnerRef &owner) noexcept;
  std::unordered_map<std::string, std::uint64_t>
  debug_external_dense_storage_stats() const;
  std::weak_ptr<void> weak_lifetime_token() const noexcept {
    return lifetime_token_;
  }

  using DenseStorageBindingCallback = std::function<void(
      const storage::ResolvedDenseBinding *, std::size_t)>;
  void with_resolved_dense_storage_bindings(
      const std::vector<const storage::DenseStorageDescriptor *> &descriptors,
      const DenseStorageBindingCallback &callback);
  void with_resolved_runtime_storage_arguments(
      const std::vector<const storage::RuntimeStorageArgument *> &arguments,
      const DenseStorageBindingCallback &callback);
  intptr_t get_dense_storage_data_ptr_as_int(
      const storage::ResolvedDenseBinding &binding);

  Texture *create_texture(BufferFormat buffer_format,
                          const std::vector<int> &shape);

  intptr_t get_ndarray_data_ptr_as_int(const Ndarray *ndarray);

  void fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val);

  void copy_ndarray_fast(Ndarray *dst, Ndarray *src);

  void copy_ndarray_from_host(Ndarray *dst,
                              const void *src,
                              std::size_t bytes);

  void copy_ndarray_to_host(Ndarray *src, void *dst, std::size_t bytes);
  void copy_ndarrays_to_host(const Ndarray *const *srcs,
                             void *const *dsts,
                             const std::size_t *bytes,
                             std::size_t count);


  bool cuda_device_transform_available() const;

  bool cuda_toolkit_transform_available() const;

  std::size_t cuda_device_transform_affine_ndarray(Ndarray *src,
                                                   Ndarray *dst,
                                                   int value_type,
                                                   double scale,
                                                   double bias);
  std::size_t cuda_device_transform_affine_member_ndarray(Ndarray *src,
                                                          Ndarray *dst,
                                                          int value_type,
                                                          std::size_t offset,
                                                          std::size_t stride,
                                                          double scale,
                                                          double bias);
  std::size_t cuda_device_transform_affine_strided_ndarray(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride,
      double scale,
      double bias);
  std::size_t cuda_device_transform_affine_packed_strided_ndarray(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      int lane_count,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride,
      double scale,
      double bias);

  std::size_t cuda_device_transform_affine_dense_field(SNode *src,
                                                       SNode *dst,
                                                       int value_type,
                                                       std::size_t n,
                                                       double scale,
                                                       double bias);

  std::size_t cuda_device_zero_dense_field(SNode *dst,
                                           int value_type,
                                           std::size_t n);

  bool cuda_device_add_merge_available() const;

  std::size_t cuda_device_add_merge_ndarray(Ndarray *src,
                                            Ndarray *dst,
                                            int value_type);

  std::size_t cuda_device_add_scaled_ndarray(Ndarray *src,
                                             Ndarray *dst,
                                             int value_type,
                                             double scale);

  std::size_t cuda_device_add_scalar_ndarray_to_ndarray(Ndarray *src,
                                                        Ndarray *dst,
                                                        int value_type,
                                                        double scale);

  std::size_t cuda_device_add_merge_strided_ndarray(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride);

  std::size_t cuda_device_add_merge_dense_field(Ndarray *src,
                                                SNode *dst,
                                                int value_type,
                                                std::size_t n);

  std::size_t cuda_device_add_scaled_dense_field(SNode *src,
                                                 SNode *dst,
                                                 int value_type,
                                                 std::size_t n,
                                                 double scale);

  std::size_t cuda_device_add_scalar_field_to_dense_field(SNode *src,
                                                          SNode *dst,
                                                          int value_type,
                                                          std::size_t n);

  bool cuda_device_indexed_copy_available() const;

  bool cuda_device_indexed_copy_payload_available(std::size_t item_bytes) const;

  std::size_t cuda_device_gather_ndarray(Ndarray *src,
                                         Ndarray *indices,
                                         Ndarray *dst);

  std::size_t cuda_device_gather_strided_ndarray(Ndarray *src,
                                                 Ndarray *indices,
                                                 Ndarray *dst,
                                                 std::size_t item_bytes,
                                                 std::size_t src_offset,
                                                 std::size_t src_stride,
                                                 std::size_t dst_offset,
                                                 std::size_t dst_stride);

  std::size_t cuda_device_gather_dense_field(SNode *src,
                                             Ndarray *indices,
                                             SNode *dst,
                                             int value_type,
                                             std::size_t src_n,
                                             std::size_t dst_n);

  std::size_t cuda_device_gather_dense_field_packed(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n,
                                                    int lane_count);

  std::size_t cuda_device_gather_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t cuda_device_gather_dense_field_indices_field(SNode *src,
                                                           SNode *indices,
                                                           SNode *dst,
                                                           int value_type,
                                                           std::size_t src_n,
                                                           std::size_t indices_n,
                                                           std::size_t dst_n);

  std::size_t cuda_device_gather_add_ndarray(Ndarray *src,
                                             Ndarray *indices,
                                             Ndarray *dst,
                                             int value_type);

  std::size_t cuda_device_gather_add_dense_field(SNode *src,
                                                 Ndarray *indices,
                                                 SNode *dst,
                                                 int value_type,
                                                 std::size_t src_n,
                                                 std::size_t dst_n);

  std::size_t cuda_device_gather_add_dense_field_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n);

  std::size_t cuda_device_scatter_ndarray(Ndarray *src,
                                          Ndarray *indices,
                                          Ndarray *dst);

  std::size_t cuda_device_scatter_strided_ndarray(Ndarray *src,
                                                  Ndarray *indices,
                                                  Ndarray *dst,
                                                  std::size_t item_bytes,
                                                  std::size_t src_offset,
                                                  std::size_t src_stride,
                                                  std::size_t dst_offset,
                                                  std::size_t dst_stride);

  std::size_t cuda_device_scatter_dense_field(SNode *src,
                                              Ndarray *indices,
                                              SNode *dst,
                                              int value_type,
                                              std::size_t src_n,
                                              std::size_t dst_n);

  std::size_t cuda_device_scatter_dense_field_packed(SNode *src,
                                                     Ndarray *indices,
                                                     SNode *dst,
                                                     int value_type,
                                                     std::size_t src_n,
                                                     std::size_t dst_n,
                                                     int lane_count);

  std::size_t cuda_device_scatter_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t cuda_device_scatter_dense_field_indices_field(SNode *src,
                                                            SNode *indices,
                                                            SNode *dst,
                                                            int value_type,
                                                            std::size_t src_n,
                                                            std::size_t indices_n,
                                                            std::size_t dst_n);

  bool cuda_device_scatter_add_available() const;

  std::size_t cuda_device_scatter_add_ndarray(Ndarray *src,
                                              Ndarray *indices,
                                              Ndarray *dst,
                                              int value_type);

  std::size_t cuda_device_scatter_add_member_ndarray(Ndarray *src,
                                                     Ndarray *indices,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     std::size_t offset,
                                                     std::size_t stride);

  std::size_t cuda_device_scatter_add_strided_ndarray(
      Ndarray *src,
      Ndarray *indices,
      Ndarray *dst,
      int value_type,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride);

  std::size_t cuda_device_scatter_add_dense_field(SNode *src,
                                                  Ndarray *indices,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t src_n,
                                                  std::size_t dst_n);

  std::size_t cuda_device_scatter_add_dense_field_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n);

  bool cuda_device_bucket_builder_available() const;

  std::size_t cuda_device_bucket_builder_i32_ndarray(Ndarray *keys,
                                                     Ndarray *values,
                                                     Ndarray *offsets,
                                                     Ndarray *output,
                                                     Ndarray *cursor);

  std::size_t cuda_device_bucket_builder_ndarray(Ndarray *keys,
                                                 Ndarray *values,
                                                 Ndarray *offsets,
                                                 Ndarray *output,
                                                 Ndarray *cursor,
                                                 int value_type);

  std::size_t cuda_device_bucket_builder_dense_field(SNode *keys,
                                                     SNode *values,
                                                     SNode *offsets,
                                                     SNode *output,
                                                     Ndarray *cursor,
                                                     int value_type,
                                                     std::size_t n,
                                                     std::size_t num_bins);

  bool cuda_device_grouped_reduce_available() const;

  std::size_t cuda_device_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                        Ndarray *values,
                                                        Ndarray *output,
                                                        int value_type,
                                                        int op);

  std::size_t cuda_device_grouped_reduce_atomic_dense_field(
      SNode *keys,
      SNode *values,
      SNode *output,
      int value_type,
      std::size_t n,
      std::size_t num_groups,
      int op);

  std::size_t cuda_device_grouped_reduce_atomic_member_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t offset,
      std::size_t stride,
      int op);

  std::size_t cuda_device_grouped_reduce_atomic_strided_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  std::size_t cuda_device_grouped_reduce_atomic_strided_keys_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t keys_offset,
      std::size_t keys_stride,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  std::size_t cuda_device_grouped_reduce_i32_atomic_ndarray(Ndarray *keys,
                                                            Ndarray *values,
                                                            Ndarray *output,
                                                            int op);

  std::size_t cuda_device_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                     Ndarray *values,
                                                     Ndarray *output,
                                                     Ndarray *offsets,
                                                     Ndarray *scratch,
                                                     Ndarray *cursor,
                                                     int op);

  std::size_t cuda_device_grouped_reduce_ndarray(Ndarray *keys,
                                                 Ndarray *values,
                                                 Ndarray *output,
                                                 Ndarray *offsets,
                                                 Ndarray *scratch,
                                                 Ndarray *cursor,
                                                 int value_type,
                                                 int op);

  std::size_t cuda_device_grouped_reduce_segmented_strided_keys_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      Ndarray *offsets,
      Ndarray *scratch,
      Ndarray *cursor,
      int value_type,
      std::size_t keys_offset,
      std::size_t keys_stride,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  bool cuda_sparse_assembly_available() const;

  CudaSparseAssemblyDispatchInfo cuda_sparse_assemble_csr(
      Ndarray *packed_triplets,
      Ndarray *triplet_rows,
      Ndarray *triplet_columns,
      Ndarray *triplet_values,
      Ndarray *sorted_keys,
      Ndarray *sorted_values,
      Ndarray *segment_ids,
      Ndarray *unique_keys,
      Ndarray *segment_offsets,
      Ndarray *unique_values,
      Ndarray *row_offsets,
      Ndarray *column_indices,
      Ndarray *active_count,
      Ndarray *control,
      std::size_t capacity,
      std::size_t rows,
      std::size_t cols);

  bool cuda_device_radix_sort_available() const;

  std::size_t cuda_device_radix_sort_ndarray(Ndarray *keys,
                                             Ndarray *values,
                                             int key_type,
                                             int value_type,
                                             int nan_policy);

  std::size_t cuda_device_radix_sort_dense_field(SNode *keys,
                                                 SNode *values,
                                                 int key_type,
                                                 int value_type,
                                                 std::size_t n,
                                                 int nan_policy);

  void cuda_device_radix_sort_clear_workspace();

  std::size_t cuda_device_radix_sort_workspace_bytes() const;

  bool cuda_cub_radix_sort_available() const;

  std::size_t cuda_cub_radix_sort_ndarray(Ndarray *keys,
                                          Ndarray *values,
                                          int key_type,
                                          int value_type,
                                          int mode,
                                          int nan_policy);

  std::size_t cuda_cub_radix_sort_dense_field(SNode *keys,
                                              SNode *values,
                                              int key_type,
                                              int value_type,
                                              std::size_t n,
                                              int mode,
                                              int nan_policy);

  void cuda_cub_radix_sort_clear_workspace();

  std::size_t cuda_cub_radix_sort_workspace_bytes() const;

  bool cpu_stable_sort_available() const;

  std::size_t cpu_stable_sort_ndarray(Ndarray *keys,
                                      Ndarray *values,
                                      int key_type,
                                      int value_type,
                                      bool descending,
                                      int nan_policy);

  std::size_t cpu_stable_sort_dense_field(SNode *keys,
                                          SNode *values,
                                          int key_type,
                                          int value_type,
                                          std::size_t n,
                                          bool descending,
                                          int nan_policy);

  bool cuda_device_scan_available() const;

  std::size_t cuda_device_inclusive_scan_ndarray(Ndarray *data,
                                                  int value_type);

  std::size_t cuda_device_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                          int value_type);

  std::size_t cuda_device_inclusive_scan_member_ndarray(
      Ndarray *data,
      int value_type,
      std::size_t offset,
      std::size_t stride);

  std::size_t cuda_device_inclusive_reverse_scan_member_ndarray(
      Ndarray *data,
      int value_type,
      std::size_t offset,
      std::size_t stride);

  std::size_t cuda_device_inclusive_scan_dense_field(SNode *data,
                                                      int value_type,
                                                      std::size_t n);

  std::size_t cuda_device_inclusive_reverse_scan_dense_field(
      SNode *data,
      int value_type,
      std::size_t n);

  std::size_t cuda_device_inclusive_scan_dense_field_packed(
      SNode *data,
      int value_type,
      std::size_t n,
      int lane_count);

  std::size_t cuda_device_inclusive_reverse_scan_dense_field_packed(
      SNode *data,
      int value_type,
      std::size_t n,
      int lane_count);

  void cuda_device_scan_clear_workspace();

  std::size_t cuda_device_scan_workspace_bytes() const;

  bool cuda_cub_scan_available() const;

  std::size_t cuda_cub_inclusive_scan_ndarray(Ndarray *data, int value_type);

  std::size_t cuda_cub_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                      int value_type);

  std::size_t cuda_cub_inclusive_scan_member_ndarray(Ndarray *data,
                                                     int value_type,
                                                     std::size_t offset,
                                                     std::size_t stride);

  std::size_t cuda_cub_inclusive_reverse_scan_member_ndarray(
      Ndarray *data,
      int value_type,
      std::size_t offset,
      std::size_t stride);

  std::size_t cuda_cub_inclusive_scan_dense_field(SNode *data,
                                                  int value_type,
                                                  std::size_t n);

  std::size_t cuda_cub_inclusive_reverse_scan_dense_field(SNode *data,
                                                          int value_type,
                                                          std::size_t n);

  std::size_t cuda_cub_inclusive_scan_dense_field_packed(SNode *data,
                                                         int value_type,
                                                         std::size_t n,
                                                         int lane_count);

  std::size_t cuda_cub_inclusive_reverse_scan_dense_field_packed(
      SNode *data,
      int value_type,
      std::size_t n,
      int lane_count);

  void cuda_cub_scan_clear_workspace();

  std::size_t cuda_cub_scan_workspace_bytes() const;

  bool cuda_device_compact_available() const;

  std::size_t cuda_device_compact_ndarray(Ndarray *values,
                                          Ndarray *flags,
                                          Ndarray *output,
                                          Ndarray *count,
                                          int value_type);

  std::size_t cuda_device_compact_dense_field(SNode *values,
                                              SNode *flags,
                                              SNode *output,
                                              SNode *count,
                                              int value_type,
                                              std::size_t n);

  void cuda_device_compact_clear_workspace();

  std::size_t cuda_device_compact_workspace_bytes() const;

  bool cuda_cub_select_available() const;

  std::size_t cuda_cub_select_ndarray(Ndarray *values,
                                      Ndarray *flags,
                                      Ndarray *output,
                                      Ndarray *count,
                                      int value_type);

  std::size_t cuda_cub_select_dense_field(SNode *values,
                                          SNode *flags,
                                          SNode *output,
                                          SNode *count,
                                          int value_type,
                                          std::size_t n);

  std::size_t cuda_cub_select_i32_ndarray(Ndarray *values,
                                          Ndarray *flags,
                                          Ndarray *output,
                                          Ndarray *count);

  void cuda_cub_select_clear_workspace();

  std::size_t cuda_cub_select_workspace_bytes() const;

  bool cuda_device_histogram_available() const;

  std::size_t cuda_device_histogram_ndarray(Ndarray *values,
                                             Ndarray *bins,
                                             int value_type,
                                             int bin_type);

  std::size_t cuda_device_histogram_dense_field(SNode *values,
                                                 SNode *bins,
                                                 int value_type,
                                                 int bin_type,
                                                 std::size_t n,
                                                 std::size_t num_bins);

  void cuda_device_histogram_clear_workspace();

  std::size_t cuda_device_histogram_workspace_bytes() const;

  bool cuda_cub_histogram_available() const;

  std::size_t cuda_cub_histogram_ndarray(Ndarray *values,
                                         Ndarray *bins,
                                         int value_type,
                                         int bin_type);

  std::size_t cuda_cub_histogram_i32_ndarray(Ndarray *values, Ndarray *bins);

  std::size_t cuda_cub_histogram_dense_field(SNode *values,
                                             SNode *bins,
                                             int value_type,
                                             int bin_type,
                                             std::size_t n,
                                             std::size_t num_bins);

  void cuda_cub_histogram_clear_workspace();

  std::size_t cuda_cub_histogram_workspace_bytes() const;

  bool cuda_device_reduce_available() const;

  std::size_t cuda_device_reduce_ndarray(Ndarray *values,
                                          Ndarray *output,
                                          int value_type,
                                          int op);

  std::size_t cuda_device_reduce_member_ndarray(Ndarray *values,
                                                 Ndarray *output,
                                                 int value_type,
                                                 std::size_t offset,
                                                 std::size_t stride,
                                                 int op);

  std::size_t cuda_device_reduce_strided_ndarray(
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  std::size_t cuda_device_reduce_dense_field(SNode *values,
                                              SNode *output,
                                              int value_type,
                                              std::size_t n,
                                              int op);

  std::size_t cuda_device_reduce_dense_field_packed(SNode *values,
                                                     SNode *output,
                                                     int value_type,
                                                     std::size_t n,
                                                     int lane_count,
                                                     int op);

  void cuda_device_reduce_clear_workspace();

  std::size_t cuda_device_reduce_workspace_bytes() const;

  bool cuda_cub_reduce_available() const;

  std::size_t cuda_cub_reduce_ndarray(Ndarray *values,
                                       Ndarray *output,
                                       int value_type,
                                       int op);

  std::size_t cuda_cub_reduce_member_ndarray(Ndarray *values,
                                             Ndarray *output,
                                             int value_type,
                                             std::size_t offset,
                                             std::size_t stride,
                                             int op);

  std::size_t cuda_cub_reduce_strided_ndarray(Ndarray *values,
                                              Ndarray *output,
                                              int value_type,
                                              std::size_t values_offset,
                                              std::size_t values_stride,
                                              std::size_t output_offset,
                                              std::size_t output_stride,
                                              int op);

  std::size_t cuda_cub_reduce_dense_field(SNode *values,
                                          SNode *output,
                                          int value_type,
                                          std::size_t n,
                                          int op);

  std::size_t cuda_cub_reduce_dense_field_packed(SNode *values,
                                                 SNode *output,
                                                 int value_type,
                                                 std::size_t n,
                                                 int lane_count,
                                                 int op);

  void cuda_cub_reduce_clear_workspace();

  std::size_t cuda_cub_reduce_workspace_bytes() const;

  bool cuda_device_check_count_available() const;

  std::size_t cuda_device_check_count_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           int check_op,
                                           int lower,
                                           int upper);

  std::size_t cuda_device_check_count_strided_ndarray(Ndarray *values,
                                                   Ndarray *output,
                                                   int value_type,
                                                   std::size_t offset,
                                                   std::size_t stride,
                                                   int check_op,
                                                   int lower,
                                                   int upper);

  std::size_t cuda_device_check_count_dense_field(SNode *values,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t n,
                                               int check_op,
                                               int lower,
                                               int upper);

  void cuda_device_check_count_clear_workspace();

  std::size_t cuda_device_check_count_workspace_bytes() const;

  bool cuda_device_metric_reduce_available() const;

  bool cuda_device_metric_reduce_value_type_available(int value_type) const;

  std::size_t cuda_device_metric_reduce_ndarray(Ndarray *values,
                                             Ndarray *other,
                                             Ndarray *output,
                                             int value_type,
                                             int metric_op);

  std::size_t cuda_device_metric_reduce_strided_ndarray(
      Ndarray *values,
      Ndarray *other,
      Ndarray *output,
      int value_type,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t other_offset,
      std::size_t other_stride,
      int metric_op);

  std::size_t cuda_device_metric_reduce_dense_field(SNode *values,
                                                 SNode *other,
                                                 Ndarray *output,
                                                 int value_type,
                                                 std::size_t n,
                                                 int metric_op);

  std::size_t cuda_device_metric_reduce_dense_field_strided_ndarray(
      SNode *field,
      Ndarray *array,
      Ndarray *output,
      int value_type,
      std::size_t n,
      std::size_t array_offset,
      std::size_t array_stride,
      bool field_is_values,
      int metric_op);

  void cuda_device_metric_reduce_clear_workspace();

  std::size_t cuda_device_metric_reduce_workspace_bytes() const;

  bool cuda_cub_check_count_available() const;

  std::size_t cuda_cub_check_count_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           int check_op,
                                           int lower,
                                           int upper);

  std::size_t cuda_cub_check_count_strided_ndarray(Ndarray *values,
                                                   Ndarray *output,
                                                   int value_type,
                                                   std::size_t offset,
                                                   std::size_t stride,
                                                   int check_op,
                                                   int lower,
                                                   int upper);

  std::size_t cuda_cub_check_count_dense_field(SNode *values,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t n,
                                               int check_op,
                                               int lower,
                                               int upper);

  void cuda_cub_check_count_clear_workspace();

  std::size_t cuda_cub_check_count_workspace_bytes() const;

  bool cuda_cub_metric_reduce_available() const;

  bool cuda_cub_metric_reduce_value_type_available(int value_type) const;

  std::size_t cuda_cub_metric_reduce_ndarray(Ndarray *values,
                                             Ndarray *other,
                                             Ndarray *output,
                                             int value_type,
                                             int metric_op);

  std::size_t cuda_cub_metric_reduce_strided_ndarray(
      Ndarray *values,
      Ndarray *other,
      Ndarray *output,
      int value_type,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t other_offset,
      std::size_t other_stride,
      int metric_op);

  std::size_t cuda_cub_metric_reduce_dense_field(SNode *values,
                                                 SNode *other,
                                                 Ndarray *output,
                                                 int value_type,
                                                 std::size_t n,
                                                 int metric_op);

  std::size_t cuda_cub_metric_reduce_dense_field_strided_ndarray(
      SNode *field,
      Ndarray *array,
      Ndarray *output,
      int value_type,
      std::size_t n,
      std::size_t array_offset,
      std::size_t array_stride,
      bool field_is_values,
      int metric_op);

  void cuda_cub_metric_reduce_clear_workspace();

  std::size_t cuda_cub_metric_reduce_workspace_bytes() const;

  bool cpu_scan_available() const;

  std::size_t cpu_inclusive_scan_ndarray(Ndarray *data, int value_type);

  std::size_t cpu_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                 int value_type);

  std::size_t cpu_inclusive_scan_member_ndarray(Ndarray *data,
                                                int value_type,
                                                std::size_t offset,
                                                std::size_t stride);

  std::size_t cpu_inclusive_reverse_scan_member_ndarray(Ndarray *data,
                                                        int value_type,
                                                        std::size_t offset,
                                                        std::size_t stride);

  std::size_t cpu_inclusive_scan_dense_field(SNode *data,
                                             int value_type,
                                             std::size_t n);

  std::size_t cpu_inclusive_reverse_scan_dense_field(SNode *data,
                                                     int value_type,
                                                     std::size_t n);

  std::size_t cpu_inclusive_scan_dense_field_packed(SNode *data,
                                                    int value_type,
                                                    std::size_t n,
                                                    int lane_count);

  std::size_t cpu_inclusive_reverse_scan_dense_field_packed(SNode *data,
                                                            int value_type,
                                                            std::size_t n,
                                                            int lane_count);

  std::size_t cpu_scan_workspace_bytes() const;

  bool cpu_compact_available() const;

  void fill_dense_field(SNode *dst,
                        int value_type,
                        uint64_t value_bits,
                        std::size_t n);

  void fill_dense_field_packed(SNode *dst,
                               int value_type,
                               uint64_t value_bits,
                               std::size_t n,
                               int lane_count);

  std::size_t transform_affine_dense_field_packed(SNode *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n,
                                                  int lane_count,
                                                  double scale,
                                                  double bias);

  void copy_dense_field(SNode *dst,
                        SNode *src,
                        int value_type,
                        std::size_t n);

  void copy_dense_field_packed(SNode *dst,
                               SNode *src,
                               int value_type,
                               std::size_t n,
                               int lane_count);

  void copy_dense_field_to_ndarray(Ndarray *dst,
                                   SNode *src,
                                   int value_type,
                                   std::size_t n,
                                   int lane_count);

  void copy_ndarray_to_dense_field(SNode *dst,
                                   Ndarray *src,
                                   int value_type,
                                   std::size_t n,
                                   int lane_count);

  void copy_dense_field_from_host(SNode *dst,
                                  std::uintptr_t src,
                                  std::size_t src_bytes,
                                  int value_type,
                                  std::size_t n);

  void copy_dense_field_packed_from_host(SNode *dst,
                                         std::uintptr_t src,
                                         std::size_t src_bytes,
                                         int value_type,
                                         std::size_t n,
                                         int lane_count);

  void copy_dense_field_to_host(SNode *src,
                                std::uintptr_t dst,
                                std::size_t dst_bytes,
                                int value_type,
                                std::size_t n);

  void copy_dense_field_packed_to_host(SNode *src,
                                       std::uintptr_t dst,
                                       std::size_t dst_bytes,
                                       int value_type,
                                       std::size_t n,
                                       int lane_count);

  std::size_t add_merge_dense_field_packed(SNode *src,
                                           SNode *dst,
                                           int value_type,
                                           std::size_t n,
                                           int lane_count);

  std::size_t scatter_add_dense_field_packed(SNode *src,
                                             Ndarray *indices,
                                             SNode *dst,
                                             int value_type,
                                             std::size_t src_n,
                                             std::size_t dst_n,
                                             int lane_count);

  std::size_t scatter_add_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t cpu_compact_ndarray(Ndarray *values,
                                  Ndarray *flags,
                                  Ndarray *output,
                                  Ndarray *count,
                                  int value_type);

  std::size_t cpu_compact_dense_field(SNode *values,
                                      SNode *flags,
                                      SNode *output,
                                      SNode *count,
                                      int value_type,
                                      std::size_t n);

  std::size_t cpu_compact_i32_ndarray(Ndarray *values,
                                      Ndarray *flags,
                                      Ndarray *output,
                                      Ndarray *count);

  std::size_t cpu_compact_workspace_bytes() const;

  bool cpu_histogram_available() const;

  std::size_t cpu_histogram_ndarray(Ndarray *values,
                                    Ndarray *bins,
                                    int value_type,
                                    int bin_type);

  std::size_t cpu_histogram_i32_ndarray(Ndarray *values, Ndarray *bins);

  std::size_t cpu_histogram_dense_field(SNode *values,
                                        SNode *bins,
                                        int value_type,
                                        int bin_type,
                                        std::size_t n,
                                        std::size_t num_bins);

  std::size_t cpu_histogram_workspace_bytes() const;

  bool cpu_reduce_available() const;

  std::size_t cpu_reduce_ndarray(Ndarray *values,
                                 Ndarray *output,
                                 int value_type,
                                 int op);

  std::size_t cpu_reduce_member_ndarray(Ndarray *values,
                                        Ndarray *output,
                                        int value_type,
                                        std::size_t offset,
                                        std::size_t stride,
                                        int op);

  std::size_t cpu_reduce_strided_ndarray(Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         std::size_t values_offset,
                                         std::size_t values_stride,
                                         std::size_t output_offset,
                                         std::size_t output_stride,
                                         int op);

  std::size_t cpu_reduce_dense_field(SNode *values,
                                     SNode *output,
                                     int value_type,
                                     std::size_t n,
                                     int op);

  std::size_t cpu_reduce_dense_field_packed(SNode *values,
                                            SNode *output,
                                            int value_type,
                                            std::size_t n,
                                            int lane_count,
                                            int op);

  std::size_t cpu_reduce_workspace_bytes() const;

  bool cpu_check_count_available() const;

  std::size_t cpu_check_count_ndarray(Ndarray *values,
                                      Ndarray *output,
                                      int value_type,
                                      int check_op,
                                      int lower,
                                      int upper);

  std::size_t cpu_check_count_strided_ndarray(Ndarray *values,
                                              Ndarray *output,
                                              int value_type,
                                              std::size_t offset,
                                              std::size_t stride,
                                              int check_op,
                                              int lower,
                                              int upper);

  std::size_t cpu_check_count_dense_field(SNode *values,
                                          Ndarray *output,
                                          int value_type,
                                          std::size_t n,
                                          int check_op,
                                          int lower,
                                          int upper);

  std::size_t cpu_check_count_workspace_bytes() const;

  bool cpu_metric_reduce_available() const;

  bool cpu_metric_reduce_value_type_available(int value_type) const;

  std::size_t cpu_metric_reduce_ndarray(Ndarray *values,
                                        Ndarray *other,
                                        Ndarray *output,
                                        int value_type,
                                        int metric_op);

  std::size_t cpu_metric_reduce_strided_ndarray(Ndarray *values,
                                                Ndarray *other,
                                                Ndarray *output,
                                                int value_type,
                                                std::size_t values_offset,
                                                std::size_t values_stride,
                                                std::size_t other_offset,
                                                std::size_t other_stride,
                                                int metric_op);

  std::size_t cpu_metric_reduce_dense_field(SNode *values,
                                            SNode *other,
                                            Ndarray *output,
                                            int value_type,
                                            std::size_t n,
                                            int metric_op);

  std::size_t cpu_metric_reduce_dense_field_strided_ndarray(
      SNode *field,
      Ndarray *array,
      Ndarray *output,
      int value_type,
      std::size_t n,
      std::size_t array_offset,
      std::size_t array_stride,
      bool field_is_values,
      int metric_op);

  std::size_t cpu_metric_reduce_workspace_bytes() const;

  bool cpu_transform_available() const;

  std::size_t cpu_transform_affine_ndarray(Ndarray *src,
                                           Ndarray *dst,
                                           int value_type,
                                           double scale,
                                           double bias);
  std::size_t cpu_transform_affine_member_ndarray(Ndarray *src,
                                                  Ndarray *dst,
                                                  int value_type,
                                                  std::size_t offset,
                                                  std::size_t stride,
                                                  double scale,
                                                  double bias);
  std::size_t cpu_transform_affine_strided_ndarray(Ndarray *src,
                                                   Ndarray *dst,
                                                   int value_type,
                                                   std::size_t src_offset,
                                                   std::size_t src_stride,
                                                   std::size_t dst_offset,
                                                   std::size_t dst_stride,
                                                   double scale,
                                                   double bias);
  std::size_t cpu_transform_affine_packed_strided_ndarray(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      int lane_count,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride,
      double scale,
      double bias);

  std::size_t cpu_transform_affine_dense_field(SNode *src,
                                               SNode *dst,
                                               int value_type,
                                               std::size_t n,
                                               double scale,
                                               double bias);

  std::size_t cpu_transform_workspace_bytes() const;

  bool cpu_add_merge_available() const;

  std::size_t cpu_add_merge_ndarray(Ndarray *src,
                                    Ndarray *dst,
                                    int value_type);

  std::size_t cpu_add_scaled_ndarray(Ndarray *src,
                                     Ndarray *dst,
                                     int value_type,
                                     double scale);

  std::size_t cpu_add_scalar_ndarray_to_ndarray(Ndarray *src,
                                                Ndarray *dst,
                                                int value_type,
                                                double scale);

  std::size_t cpu_add_merge_strided_ndarray(Ndarray *src,
                                            Ndarray *dst,
                                            int value_type,
                                            std::size_t src_offset,
                                            std::size_t src_stride,
                                            std::size_t dst_offset,
                                            std::size_t dst_stride);

  std::size_t cpu_add_merge_dense_field(Ndarray *src,
                                        SNode *dst,
                                        int value_type,
                                        std::size_t n);

  std::size_t cpu_add_scaled_dense_field(SNode *src,
                                         SNode *dst,
                                         int value_type,
                                         std::size_t n,
                                         double scale);

  std::size_t cpu_add_scalar_field_to_dense_field(SNode *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n);

  bool cpu_indexed_copy_available() const;

  std::size_t cpu_gather_ndarray(Ndarray *src, Ndarray *indices, Ndarray *dst);

  std::size_t cpu_gather_strided_ndarray(Ndarray *src,
                                         Ndarray *indices,
                                         Ndarray *dst,
                                         std::size_t item_bytes,
                                         std::size_t src_offset,
                                         std::size_t src_stride,
                                         std::size_t dst_offset,
                                         std::size_t dst_stride);

  std::size_t cpu_gather_dense_field(SNode *src,
                                     Ndarray *indices,
                                     SNode *dst,
                                     int value_type,
                                     std::size_t src_n,
                                     std::size_t dst_n);

  std::size_t cpu_gather_dense_field_packed(SNode *src,
                                            Ndarray *indices,
                                            SNode *dst,
                                            int value_type,
                                            std::size_t src_n,
                                            std::size_t dst_n,
                                            int lane_count);

  std::size_t cpu_gather_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t cpu_gather_dense_field_indices_field(SNode *src,
                                                   SNode *indices,
                                                   SNode *dst,
                                                   int value_type,
                                                   std::size_t src_n,
                                                   std::size_t indices_n,
                                                   std::size_t dst_n);

  std::size_t cpu_gather_add_ndarray(Ndarray *src,
                                     Ndarray *indices,
                                     Ndarray *dst,
                                     int value_type);

  std::size_t cpu_gather_add_dense_field(SNode *src,
                                         Ndarray *indices,
                                         SNode *dst,
                                         int value_type,
                                         std::size_t src_n,
                                         std::size_t dst_n);

  std::size_t cpu_gather_add_dense_field_indices_field(SNode *src,
                                                       SNode *indices,
                                                       SNode *dst,
                                                       int value_type,
                                                       std::size_t src_n,
                                                       std::size_t indices_n,
                                                       std::size_t dst_n);

  std::size_t cpu_scatter_ndarray(Ndarray *src, Ndarray *indices, Ndarray *dst);

  std::size_t cpu_scatter_strided_ndarray(Ndarray *src,
                                          Ndarray *indices,
                                          Ndarray *dst,
                                          std::size_t item_bytes,
                                          std::size_t src_offset,
                                          std::size_t src_stride,
                                          std::size_t dst_offset,
                                          std::size_t dst_stride);

  std::size_t cpu_scatter_dense_field(SNode *src,
                                      Ndarray *indices,
                                      SNode *dst,
                                      int value_type,
                                      std::size_t src_n,
                                      std::size_t dst_n);

  std::size_t cpu_scatter_dense_field_packed(SNode *src,
                                             Ndarray *indices,
                                             SNode *dst,
                                             int value_type,
                                             std::size_t src_n,
                                             std::size_t dst_n,
                                             int lane_count);

  std::size_t cpu_scatter_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t cpu_scatter_dense_field_indices_field(SNode *src,
                                                    SNode *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t indices_n,
                                                    std::size_t dst_n);

  std::size_t cpu_indexed_copy_workspace_bytes() const;

  bool cpu_scatter_add_available() const;

  std::size_t cpu_scatter_add_ndarray(Ndarray *src,
                                      Ndarray *indices,
                                      Ndarray *dst,
                                      int value_type);

  std::size_t cpu_scatter_add_member_ndarray(Ndarray *src,
                                             Ndarray *indices,
                                             Ndarray *dst,
                                             int value_type,
                                             std::size_t offset,
                                             std::size_t stride);

  std::size_t cpu_scatter_add_strided_ndarray(Ndarray *src,
                                              Ndarray *indices,
                                              Ndarray *dst,
                                              int value_type,
                                              std::size_t src_offset,
                                              std::size_t src_stride,
                                              std::size_t dst_offset,
                                              std::size_t dst_stride);

  std::size_t cpu_scatter_add_dense_field(SNode *src,
                                          Ndarray *indices,
                                          SNode *dst,
                                          int value_type,
                                          std::size_t src_n,
                                          std::size_t dst_n);

  std::size_t cpu_scatter_add_dense_field_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n);

  std::size_t cpu_scatter_add_workspace_bytes() const;

  void cpu_scatter_add_clear_workspace();

  bool cpu_bucket_builder_available() const;

  std::size_t cpu_bucket_builder_i32_ndarray(Ndarray *keys,
                                             Ndarray *values,
                                             Ndarray *offsets,
                                             Ndarray *output);

  std::size_t cpu_bucket_builder_ndarray(Ndarray *keys,
                                         Ndarray *values,
                                         Ndarray *offsets,
                                         Ndarray *output,
                                         int value_type);

  std::size_t cpu_bucket_builder_dense_field(SNode *keys,
                                             SNode *values,
                                             SNode *offsets,
                                             SNode *output,
                                             int value_type,
                                             std::size_t n,
                                             std::size_t num_bins);

  std::size_t cpu_bucket_builder_workspace_bytes() const;

  bool cpu_grouped_reduce_available() const;

  std::size_t cpu_grouped_reduce_ndarray(Ndarray *keys,
                                         Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         int op);

  std::size_t cpu_grouped_reduce_dense_field(SNode *keys,
                                             SNode *values,
                                             SNode *output,
                                             int value_type,
                                             std::size_t n,
                                             std::size_t num_groups,
                                             int op);

  std::size_t cpu_grouped_reduce_member_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                std::size_t offset,
                                                std::size_t stride,
                                                int op);

  std::size_t cpu_grouped_reduce_strided_ndarray(Ndarray *keys,
                                                 Ndarray *values,
                                                 Ndarray *output,
                                                 int value_type,
                                                 std::size_t values_offset,
                                                 std::size_t values_stride,
                                                 std::size_t output_offset,
                                                 std::size_t output_stride,
                                                 int op);

  std::size_t cpu_grouped_reduce_strided_keys_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t keys_offset,
      std::size_t keys_stride,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  std::size_t cpu_grouped_reduce_i32_ndarray(Ndarray *keys,
                                             Ndarray *values,
                                             Ndarray *output,
                                             int op);

  std::size_t cpu_grouped_reduce_workspace_bytes() const;

  void cpu_grouped_reduce_clear_workspace();

  bool vulkan_radix_sort_available() const;

  std::size_t vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                            Ndarray *values,
                                            int key_type,
                                            int value_type,
                                            std::size_t key_offset = 0,
                                            std::size_t value_offset = 0);

  std::size_t vulkan_radix_sort_u32_dense_field(SNode *keys,
                                                SNode *values,
                                                int key_type,
                                                int value_type,
                                                std::size_t n);

  void vulkan_radix_sort_clear_workspace();

  std::size_t vulkan_radix_sort_workspace_bytes() const;

  void vulkan_radix_sort_cpu_profile_clear();

  std::string vulkan_radix_sort_cpu_profile_report() const;

  bool vulkan_scan_available() const;

  bool vulkan_scan_value_type_available(int value_type) const;

  std::size_t vulkan_inclusive_scan_ndarray(Ndarray *data, int value_type);

  std::size_t vulkan_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                    int value_type);

  std::size_t vulkan_inclusive_scan_member_ndarray(Ndarray *data,
                                                   int value_type,
                                                   std::size_t offset,
                                                   std::size_t stride);

  std::size_t vulkan_inclusive_reverse_scan_member_ndarray(
      Ndarray *data,
      int value_type,
      std::size_t offset,
      std::size_t stride);

  std::size_t vulkan_inclusive_scan_dense_field(SNode *data,
                                                int value_type,
                                                std::size_t n);

  std::size_t vulkan_inclusive_reverse_scan_dense_field(SNode *data,
                                                        int value_type,
                                                        std::size_t n);

  std::size_t vulkan_inclusive_scan_dense_field_packed(SNode *data,
                                                       int value_type,
                                                       std::size_t n,
                                                       int lane_count);

  std::size_t vulkan_inclusive_reverse_scan_dense_field_packed(
      SNode *data,
      int value_type,
      std::size_t n,
      int lane_count);

  void vulkan_scan_clear_workspace();

  std::size_t vulkan_scan_workspace_bytes() const;

  bool vulkan_compact_available() const;

  std::size_t vulkan_compact_ndarray(Ndarray *values,
                                     Ndarray *flags,
                                     Ndarray *output,
                                     Ndarray *count,
                                     int value_type);

  std::size_t vulkan_compact_dense_field(SNode *values,
                                         SNode *flags,
                                         SNode *output,
                                         SNode *count,
                                         int value_type,
                                         std::size_t n);

  std::size_t vulkan_compact_i32_ndarray(Ndarray *values,
                                         Ndarray *flags,
                                         Ndarray *output,
                                         Ndarray *count);

  void vulkan_compact_clear_workspace();

  std::size_t vulkan_compact_workspace_bytes() const;

  bool vulkan_histogram_available() const;

  bool vulkan_histogram_value_type_available(int value_type,
                                             int bin_type) const;

  std::size_t vulkan_histogram_ndarray(Ndarray *values,
                                       Ndarray *bins,
                                       int value_type,
                                       int bin_type);

  std::size_t vulkan_histogram_i32_ndarray(Ndarray *values, Ndarray *bins);

  std::size_t vulkan_histogram_dense_field(SNode *values,
                                           SNode *bins,
                                           int value_type,
                                           int bin_type,
                                           std::size_t n,
                                           std::size_t num_bins);

  void vulkan_histogram_clear_workspace();

  std::size_t vulkan_histogram_workspace_bytes() const;

  bool vulkan_reduce_available() const;

  bool vulkan_reduce_value_type_available(int value_type) const;

  std::size_t vulkan_reduce_ndarray(Ndarray *values,
                                    Ndarray *output,
                                    int value_type,
                                    int op);

  std::size_t vulkan_reduce_member_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           std::size_t offset,
                                           std::size_t stride,
                                           int op);

  std::size_t vulkan_reduce_strided_ndarray(Ndarray *values,
                                            Ndarray *output,
                                            int value_type,
                                            std::size_t values_offset,
                                            std::size_t values_stride,
                                            std::size_t output_offset,
                                            std::size_t output_stride,
                                            int op);

  std::size_t vulkan_reduce_dense_field(SNode *values,
                                        SNode *output,
                                        int value_type,
                                        std::size_t n,
                                        int op);

  std::size_t vulkan_reduce_dense_field_packed(SNode *values,
                                               SNode *output,
                                               int value_type,
                                               std::size_t n,
                                               int lane_count,
                                               int op);

  std::size_t vulkan_reduce_i32_ndarray(Ndarray *values,
                                        Ndarray *output,
                                        int op);

  void vulkan_reduce_clear_workspace();

  std::size_t vulkan_reduce_workspace_bytes() const;

  bool vulkan_check_count_available() const;

  bool vulkan_check_count_value_type_available(int value_type) const;

  std::size_t vulkan_check_count_ndarray(Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         int check_op,
                                         int lower,
                                         int upper);

  std::size_t vulkan_check_count_strided_ndarray(Ndarray *values,
                                                 Ndarray *output,
                                                 int value_type,
                                                 std::size_t offset,
                                                 std::size_t stride,
                                                 int check_op,
                                                 int lower,
                                                 int upper);

  std::size_t vulkan_check_count_dense_field(SNode *values,
                                             Ndarray *output,
                                             int value_type,
                                             std::size_t n,
                                             int check_op,
                                             int lower,
                                             int upper);

  void vulkan_check_count_clear_workspace();

  std::size_t vulkan_check_count_workspace_bytes() const;

  bool vulkan_metric_reduce_available() const;

  bool vulkan_metric_reduce_value_type_available(int value_type) const;

  std::size_t vulkan_metric_reduce_ndarray(Ndarray *values,
                                           Ndarray *other,
                                           Ndarray *output,
                                           int value_type,
                                           int metric_op);

  std::size_t vulkan_metric_reduce_strided_ndarray(
      Ndarray *values,
      Ndarray *other,
      Ndarray *output,
      int value_type,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t other_offset,
      std::size_t other_stride,
      int metric_op);

  std::size_t vulkan_metric_reduce_dense_field(SNode *values,
                                               SNode *other,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t n,
                                               int metric_op);

  std::size_t vulkan_metric_reduce_dense_field_strided_ndarray(
      SNode *field,
      Ndarray *array,
      Ndarray *output,
      int value_type,
      std::size_t n,
      std::size_t array_offset,
      std::size_t array_stride,
      bool field_is_values,
      int metric_op);

  void vulkan_metric_reduce_clear_workspace();

  std::size_t vulkan_metric_reduce_workspace_bytes() const;

  bool vulkan_sparse_algebra_available() const;

  bool vulkan_sparse_assembly_available() const;

  VulkanSparseAssemblyDispatchInfo vulkan_sparse_assemble_csr(
      Ndarray *packed_triplets,
      Ndarray *triplet_rows,
      Ndarray *triplet_columns,
      Ndarray *triplet_values,
      Ndarray *sorted_keys,
      Ndarray *sorted_values,
      Ndarray *segment_ids,
      Ndarray *unique_keys,
      Ndarray *segment_offsets,
      Ndarray *unique_values,
      Ndarray *row_offsets,
      Ndarray *column_indices,
      Ndarray *active_count,
      Ndarray *control,
      std::size_t capacity,
      std::size_t rows,
      std::size_t cols);

  void vulkan_copy_ndarray_prefix(Ndarray *dst,
                                  Ndarray *src,
                                  std::size_t bytes);

  std::size_t vulkan_csr_spmv(Ndarray *row_offsets,
                              Ndarray *column_indices,
                              Ndarray *values,
                              Ndarray *x,
                              Ndarray *y,
                              std::size_t rows,
                              std::size_t cols,
                              std::size_t nnz);

  std::size_t vulkan_bsr_spmv(Ndarray *row_offsets,
                              Ndarray *column_indices,
                              Ndarray *values,
                              Ndarray *x,
                              Ndarray *y,
                              std::size_t block_rows,
                              std::size_t block_cols,
                              std::size_t block_nnz,
                              std::size_t block_size);

  std::size_t vulkan_sparse_axpy(Ndarray *x,
                                 Ndarray *y,
                                 std::size_t n,
                                 float alpha);

  std::size_t vulkan_sparse_diagonal_refresh(Ndarray *values,
                                             Ndarray *diagonal_offsets,
                                             Ndarray *staging_inverse,
                                             Ndarray *status,
                                             std::size_t rows,
                                             std::size_t nnz);

  std::size_t vulkan_sparse_diagonal_apply(Ndarray *inverse_diagonal,
                                           Ndarray *input,
                                           Ndarray *output,
                                           std::size_t n);

  std::size_t vulkan_sparse_block_cholesky_refresh(
      Ndarray *values,
      Ndarray *diagonal_block_offsets,
      Ndarray *staging_factors,
      Ndarray *status,
      std::size_t block_rows,
      std::size_t block_nnz,
      std::size_t block_size);

  std::size_t vulkan_sparse_block_diagonal_apply(
      Ndarray *factor_blocks,
      Ndarray *input,
      Ndarray *output,
      std::size_t block_rows,
      std::size_t block_size);

  std::size_t vulkan_sparse_dot(Ndarray *x,
                                Ndarray *y,
                                Ndarray *output,
                                std::size_t n);

  std::size_t vulkan_sparse_norm(Ndarray *x,
                                 Ndarray *output,
                                 std::size_t n);

  std::size_t vulkan_sparse_scalar_divide(Ndarray *numerator,
                                          Ndarray *denominator,
                                          Ndarray *quotient,
                                          Ndarray *status);

  std::size_t vulkan_sparse_cg_update(Ndarray *direction,
                                      Ndarray *applied_direction,
                                      Ndarray *alpha,
                                      Ndarray *solution,
                                      Ndarray *residual,
                                      std::size_t n);

  std::size_t vulkan_sparse_cg_direction(Ndarray *residual,
                                         Ndarray *beta,
                                         Ndarray *direction,
                                         std::size_t n);

  std::size_t vulkan_sparse_minres_scalar(
      Ndarray *initial_residual_squared,
      Ndarray *rhs_squared,
      Ndarray *dot,
      Ndarray *state,
      float absolute_tolerance,
      float relative_tolerance,
      std::uint32_t stage,
      bool limit_reached,
      bool has_preconditioner,
      bool stop_on_estimate);

  std::size_t vulkan_sparse_minres_vector_state(Ndarray *source,
                                                Ndarray *destination,
                                                Ndarray *state,
                                                std::size_t n,
                                                std::uint32_t coefficient,
                                                bool add);

  std::size_t vulkan_sparse_minres_commit(Ndarray *v,
                                          Ndarray *r1,
                                          Ndarray *r2,
                                          Ndarray *lanczos_residual,
                                          Ndarray *w_older,
                                          Ndarray *w_old,
                                          Ndarray *w,
                                          Ndarray *solution,
                                          Ndarray *state,
                                          std::size_t n);

  std::size_t vulkan_sparse_bicgstab_scalar(
      Ndarray *initial_residual_squared,
      Ndarray *rhs_squared,
      Ndarray *dot0,
      Ndarray *dot1,
      Ndarray *state,
      float absolute_tolerance,
      float relative_tolerance,
      std::uint32_t stage,
      bool limit_reached);

  std::size_t vulkan_sparse_bicgstab_direction(
      Ndarray *residual,
      Ndarray *direction,
      Ndarray *operator_direction,
      Ndarray *state,
      std::size_t n);

  std::size_t vulkan_sparse_bicgstab_intermediate(
      Ndarray *residual,
      Ndarray *operator_direction,
      Ndarray *intermediate,
      Ndarray *state,
      std::size_t n);

  std::size_t vulkan_sparse_bicgstab_commit(
      Ndarray *solution_direction,
      Ndarray *solution_intermediate,
      Ndarray *intermediate,
      Ndarray *operator_intermediate,
      Ndarray *solution,
      Ndarray *residual,
      Ndarray *state,
      std::size_t n);

  std::size_t vulkan_sparse_bicgstab_reconcile(
      Ndarray *true_residual,
      Ndarray *residual,
      Ndarray *shadow_residual,
      Ndarray *direction,
      Ndarray *operator_direction,
      Ndarray *solution,
      Ndarray *state,
      std::size_t n);

  std::size_t vulkan_sparse_gmres_multi_dot(
      Ndarray *basis,
      Ndarray *work,
      Ndarray *partials,
      Ndarray *projection,
      Ndarray *state,
      std::size_t n,
      std::size_t basis_stride,
      std::size_t basis_count,
      std::size_t group_count);

  std::size_t vulkan_sparse_gmres_projection(
      Ndarray *basis,
      Ndarray *work,
      Ndarray *projection,
      Ndarray *hessenberg,
      Ndarray *state,
      std::size_t n,
      std::size_t basis_stride,
      std::size_t restart,
      std::size_t step,
      std::size_t pass);

  std::size_t vulkan_sparse_gmres_basis(Ndarray *source,
                                        Ndarray *basis,
                                        Ndarray *current,
                                        Ndarray *state,
                                        std::size_t n,
                                        std::size_t basis_stride,
                                        std::size_t row,
                                        std::size_t mode);

  std::size_t vulkan_sparse_gmres_combine(
      Ndarray *basis,
      Ndarray *coefficients,
      Ndarray *update,
      Ndarray *state,
      std::size_t n,
      std::size_t basis_stride);

  std::size_t vulkan_sparse_gmres_scalar(
      Ndarray *initial_residual_squared,
      Ndarray *rhs_squared,
      Ndarray *dot0,
      Ndarray *dot1,
      Ndarray *hessenberg,
      Ndarray *cosines,
      Ndarray *sines,
      Ndarray *g,
      Ndarray *coefficients,
      Ndarray *state,
      float absolute_tolerance,
      float relative_tolerance,
      std::size_t restart,
      std::size_t max_iterations,
      std::size_t stage,
      std::size_t step,
      bool limit_reached);

  std::size_t vulkan_sparse_convergence(Ndarray *residual_squared,
                                        Ndarray *status,
                                        Ndarray *completed_iterations,
                                        Ndarray *rhs_squared,
                                        float absolute_tolerance,
                                        float relative_tolerance,
                                        std::uint32_t iteration);

  void vulkan_sparse_algebra_clear_workspace();

  std::size_t vulkan_sparse_algebra_workspace_bytes() const;
  std::uint64_t vulkan_sparse_algebra_replay_generation();

  bool vulkan_transform_available() const;

  bool vulkan_transform_value_type_available(int value_type) const;

  std::size_t vulkan_transform_affine_ndarray(Ndarray *src,
                                              Ndarray *dst,
                                              int value_type,
                                              double scale,
                                              double bias);
  std::size_t vulkan_transform_affine_ndarray_trusted(Ndarray *src,
                                                      Ndarray *dst,
                                                      int value_type,
                                                      double scale,
                                                      double bias);
  std::size_t vulkan_transform_indexed_affine_ndarray(Ndarray *src,
                                                       Ndarray *indices,
                                                       Ndarray *dst,
                                                       int value_type,
                                                       double scale,
                                                       double bias);

  std::size_t vulkan_transform_affine_member_ndarray(Ndarray *src,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     std::size_t offset,
                                                     std::size_t stride,
                                                     double scale,
                                                     double bias);
  std::size_t vulkan_transform_affine_strided_ndarray(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride,
      double scale,
      double bias);
  std::size_t vulkan_transform_affine_strided_ndarray_trusted(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride,
      double scale,
      double bias);
  std::size_t vulkan_transform_affine_packed_strided_ndarray(
      Ndarray *src,
      Ndarray *dst,
      int value_type,
      int lane_count,
      std::size_t src_offset,
      std::size_t src_stride,
      std::size_t dst_offset,
      std::size_t dst_stride,
      double scale,
      double bias);

  std::size_t vulkan_transform_affine_dense_field(SNode *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n,
                                                  double scale,
                                                  double bias);
  std::size_t vulkan_transform_affine_dense_field_trusted(SNode *src,
                                                          SNode *dst,
                                                          int value_type,
                                                          std::size_t n,
                                                          double scale,
                                                          double bias);
  std::size_t vulkan_transform_affine_dense_field_packed(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n,
                                                         int lane_count,
                                                         double scale,
                                                         double bias);

  std::size_t vulkan_zero_dense_field(SNode *dst,
                                      int value_type,
                                      std::size_t n);

  std::size_t vulkan_zero_dense_fields(
      const std::vector<SNode *> &dsts,
      const std::vector<int> &value_types,
      const std::vector<std::size_t> &ns);

  void vulkan_transform_clear_workspace();

  std::size_t vulkan_transform_workspace_bytes() const;

  bool vulkan_add_merge_available() const;

  bool vulkan_add_merge_value_type_available(int value_type) const;

  std::size_t vulkan_add_merge_ndarray(Ndarray *src,
                                       Ndarray *dst,
                                       int value_type);

  std::size_t vulkan_add_merge_strided_ndarray(Ndarray *src,
                                               Ndarray *dst,
                                               int value_type,
                                               std::size_t src_offset,
                                               std::size_t src_stride,
                                               std::size_t dst_offset,
                                               std::size_t dst_stride);

  std::size_t vulkan_add_merge_dense_field(Ndarray *src,
                                           SNode *dst,
                                           int value_type,
                                           std::size_t n);

  std::size_t vulkan_add_merge_dense_field_packed(SNode *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n,
                                                  int lane_count);

  std::size_t vulkan_add_scalar_field_to_dense_field(SNode *src,
                                                     SNode *dst,
                                                     int value_type,
                                                     std::size_t n);

  void vulkan_add_merge_clear_workspace();

  std::size_t vulkan_add_merge_workspace_bytes() const;

  bool vulkan_indexed_copy_available() const;

  std::size_t vulkan_gather_ndarray(Ndarray *src,
                                    Ndarray *indices,
                                    Ndarray *dst);

  std::size_t vulkan_gather_strided_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst,
                                            std::size_t item_bytes,
                                            std::size_t src_offset,
                                            std::size_t src_stride,
                                            std::size_t dst_offset,
                                            std::size_t dst_stride);

  std::size_t vulkan_gather_dense_field(SNode *src,
                                        Ndarray *indices,
                                        SNode *dst,
                                        int value_type,
                                        std::size_t src_n,
                                        std::size_t dst_n);

  std::size_t vulkan_gather_dense_field_packed(SNode *src,
                                               Ndarray *indices,
                                               SNode *dst,
                                               int value_type,
                                               std::size_t src_n,
                                               std::size_t dst_n,
                                               int lane_count);

  std::size_t vulkan_gather_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t vulkan_gather_dense_field_indices_field(SNode *src,
                                                      SNode *indices,
                                                      SNode *dst,
                                                      int value_type,
                                                      std::size_t src_n,
                                                      std::size_t indices_n,
                                                      std::size_t dst_n);

  std::size_t vulkan_scatter_ndarray(Ndarray *src,
                                     Ndarray *indices,
                                     Ndarray *dst);

  std::size_t vulkan_scatter_strided_ndarray(Ndarray *src,
                                             Ndarray *indices,
                                             Ndarray *dst,
                                             std::size_t item_bytes,
                                             std::size_t src_offset,
                                             std::size_t src_stride,
                                             std::size_t dst_offset,
                                             std::size_t dst_stride);

  std::size_t vulkan_scatter_dense_field(SNode *src,
                                         Ndarray *indices,
                                         SNode *dst,
                                         int value_type,
                                         std::size_t src_n,
                                         std::size_t dst_n);

  std::size_t vulkan_scatter_dense_field_packed(SNode *src,
                                                Ndarray *indices,
                                                SNode *dst,
                                                int value_type,
                                                std::size_t src_n,
                                                std::size_t dst_n,
                                                int lane_count);

  std::size_t vulkan_scatter_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t vulkan_scatter_dense_field_indices_field(SNode *src,
                                                       SNode *indices,
                                                       SNode *dst,
                                                       int value_type,
                                                       std::size_t src_n,
                                                       std::size_t indices_n,
                                                       std::size_t dst_n);

  void vulkan_indexed_copy_clear_workspace();

  std::size_t vulkan_indexed_copy_workspace_bytes() const;

  bool vulkan_scatter_add_available() const;

  bool vulkan_scatter_add_value_type_available(int value_type) const;

  std::size_t vulkan_scatter_add_ndarray(Ndarray *src,
                                         Ndarray *indices,
                                         Ndarray *dst,
                                         int value_type);

  std::size_t vulkan_scatter_add_member_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                int value_type,
                                                std::size_t offset,
                                                std::size_t stride);

  std::size_t vulkan_scatter_add_strided_ndarray(Ndarray *src,
                                                 Ndarray *indices,
                                                 Ndarray *dst,
                                                 int value_type,
                                                 std::size_t src_offset,
                                                 std::size_t src_stride,
                                                 std::size_t dst_offset,
                                                 std::size_t dst_stride);

  std::size_t vulkan_scatter_add_dense_field(SNode *src,
                                             Ndarray *indices,
                                             SNode *dst,
                                             int value_type,
                                             std::size_t src_n,
                                             std::size_t dst_n);
  std::size_t vulkan_scatter_add_dense_field_packed(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n,
                                                    int lane_count);

  std::size_t vulkan_scatter_add_dense_field_packed_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n,
      int lane_count);

  std::size_t vulkan_scatter_add_dense_field_indices_field(
      SNode *src,
      SNode *indices,
      SNode *dst,
      int value_type,
      std::size_t src_n,
      std::size_t indices_n,
      std::size_t dst_n);

  void vulkan_scatter_add_clear_workspace();

  std::size_t vulkan_scatter_add_workspace_bytes() const;

  bool vulkan_bucket_builder_available() const;

  bool vulkan_bucket_builder_value_type_available(int value_type) const;

  std::size_t vulkan_bucket_builder_i32_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *offsets,
                                                Ndarray *output,
                                                Ndarray *cursor);

  std::size_t vulkan_bucket_builder_ndarray(Ndarray *keys,
                                            Ndarray *values,
                                            Ndarray *offsets,
                                            Ndarray *output,
                                            Ndarray *cursor,
                                            int value_type);

  std::size_t vulkan_bucket_builder_dense_field(SNode *keys,
                                                SNode *values,
                                                SNode *offsets,
                                                SNode *output,
                                                Ndarray *cursor,
                                                int value_type,
                                                std::size_t n,
                                                std::size_t num_bins);

  void vulkan_bucket_builder_clear_workspace();

  std::size_t vulkan_bucket_builder_workspace_bytes() const;

  bool vulkan_grouped_reduce_available() const;

  bool vulkan_grouped_reduce_value_type_available(int value_type) const;

  bool vulkan_grouped_reduce_atomic_value_type_available(
      int value_type) const;

  std::size_t vulkan_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *output,
                                                   int value_type,
                                                   int op);

  std::size_t vulkan_grouped_reduce_atomic_dense_field(SNode *keys,
                                                       SNode *values,
                                                       SNode *output,
                                                       int value_type,
                                                       std::size_t n,
                                                       std::size_t num_groups,
                                                       int op);

  std::size_t vulkan_grouped_reduce_atomic_member_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t offset,
      std::size_t stride,
      int op);

  std::size_t vulkan_grouped_reduce_atomic_strided_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  std::size_t vulkan_grouped_reduce_atomic_strided_keys_ndarray(
      Ndarray *keys,
      Ndarray *values,
      Ndarray *output,
      int value_type,
      std::size_t keys_offset,
      std::size_t keys_stride,
      std::size_t values_offset,
      std::size_t values_stride,
      std::size_t output_offset,
      std::size_t output_stride,
      int op);

  std::size_t vulkan_grouped_reduce_i32_atomic_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       int op);

  std::size_t vulkan_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *output,
                                                Ndarray *offsets,
                                                Ndarray *scratch,
                                                Ndarray *cursor,
                                                int op);

  std::size_t vulkan_grouped_reduce_ndarray(Ndarray *keys,
                                            Ndarray *values,
                                            Ndarray *output,
                                            Ndarray *offsets,
                                            Ndarray *scratch,
                                            Ndarray *cursor,
                                            int value_type,
                                            int op);

  void vulkan_grouped_reduce_clear_workspace();

  std::size_t vulkan_grouped_reduce_workspace_bytes() const;

  void vulkan_clear_primitive_caches();

  Identifier get_next_global_id(const std::string &name = "") {
    return Identifier(global_id_counter_++, name);
  }

  /** Enqueue a custom compute op to the current program execution flow.
   *
   *  @params op The lambda that is invoked to construct the custom compute Op
   *  @params image_refs The image resource references used in this compute Op
   */
  void enqueue_compute_op_lambda(
      std::function<void(Device *device, CommandList *cmdlist)> op,
      const std::vector<ComputeOpImageRef> &image_refs);

  /**
   * TODO(zhanlue): Remove this interface
   *
   * Gets the underlying ProgramImpl object
   *
   * This interface is essentially a hack to temporarily accommodate
   * historical design issues with LLVM backend
   *
   * Please limit its use to LLVM backend only
   */
  ProgramImpl *get_program_impl() {
    TI_ASSERT(arch_uses_llvm(compile_config().arch));
    return program_impl_.get();
  }

  // TODO(zhanlue): Move these members and corresponding interfaces to
  // ProgramImpl Ideally, Program should serve as a pure interface class and all
  // the implementations should fall inside ProgramImpl
  //
  // Once we migrated these implementations to ProgramImpl, lower-level objects
  // could store ProgramImpl rather than Program.

 private:
  class RuntimeSubmissionWriteScope {
   public:
    explicit RuntimeSubmissionWriteScope(Program *program);
    RuntimeSubmissionWriteScope(const RuntimeSubmissionWriteScope &) = delete;
    RuntimeSubmissionWriteScope &operator=(
        const RuntimeSubmissionWriteScope &) = delete;
    ~RuntimeSubmissionWriteScope();

   private:
    Program *program_;
  };

  using ArgPackResourceRegistry = RuntimeResourceRegistry<ArgPack>;
  using ArgPackResourceHandle = ArgPackResourceRegistry::Handle;
  using ArgPackResourceLease = ArgPackResourceRegistry::Lease;

  static constexpr ArgPackResourceRegistry::Kind kArgPackResourceKind = 1;
  static constexpr std::size_t kInlineArgPackLaunchLeases = 8;

  class ArgPackLaunchLeases {
   public:
    ArgPackLaunchLeases() = default;
    ArgPackLaunchLeases(const ArgPackLaunchLeases &) = delete;
    ArgPackLaunchLeases &operator=(const ArgPackLaunchLeases &) = delete;
    ArgPackLaunchLeases(ArgPackLaunchLeases &&) noexcept = default;
    ArgPackLaunchLeases &operator=(ArgPackLaunchLeases &&) noexcept = default;

   private:
    friend class Program;
    bool contains(ArgPackResourceHandle handle) const noexcept;
    bool empty() const noexcept;
    void add(ArgPackResourceLease lease);

    std::array<std::optional<ArgPackResourceLease>,
               kInlineArgPackLaunchLeases>
        inline_leases_;
    std::size_t inline_count_{0};
    std::vector<ArgPackResourceLease> overflow_leases_;
  };

  struct ArgPackResourceView {
    ArgPackResourceHandle handle;
    ArgPackResourceLease lease;
  };

  using ArgPackInflightLeaseMap =
      std::unordered_map<std::uint64_t, ArgPackResourceLease>;

  static constexpr NdarrayResourceRegistry::Kind kNdarrayResourceKind = 2;
  static constexpr std::size_t kInlineNdarrayLaunchLeases = 8;

  class NdarrayLaunchLeases {
   public:
    NdarrayLaunchLeases() = default;
    NdarrayLaunchLeases(const NdarrayLaunchLeases &) = delete;
    NdarrayLaunchLeases &operator=(const NdarrayLaunchLeases &) = delete;
    NdarrayLaunchLeases(NdarrayLaunchLeases &&) noexcept = default;
    NdarrayLaunchLeases &operator=(NdarrayLaunchLeases &&) noexcept = default;

   private:
    friend class Program;
    bool contains(NdarrayResourceHandle handle) const noexcept;
    Ndarray *find(NdarrayResourceHandle handle) const noexcept;
    bool empty() const noexcept;
    void add(NdarrayResourceLease lease);

    std::array<std::optional<NdarrayResourceLease>,
               kInlineNdarrayLaunchLeases>
        inline_leases_;
    std::size_t inline_count_{0};
    std::vector<NdarrayResourceLease> overflow_leases_;
  };

  struct NdarrayResourceView {
    NdarrayResourceHandle handle;
    NdarrayResourceLease lease;
  };

  struct NdarrayResourceSlotView {
    const Ndarray *view{nullptr};
    NdarrayResourceHandle handle;
  };

  using NdarrayInflightLeaseMap =
      std::unordered_map<std::uint64_t, NdarrayResourceLease>;

  static constexpr TextureResourceRegistry::Kind kTextureResourceKind = 3;
  static constexpr std::size_t kInlineTextureLaunchLeases = 8;

  class TextureLaunchLeases {
   public:
    TextureLaunchLeases() = default;
    TextureLaunchLeases(const TextureLaunchLeases &) = delete;
    TextureLaunchLeases &operator=(const TextureLaunchLeases &) = delete;
    TextureLaunchLeases(TextureLaunchLeases &&) noexcept = default;
    TextureLaunchLeases &operator=(TextureLaunchLeases &&) noexcept = default;

   private:
    friend class Program;
    bool contains(TextureResourceHandle handle) const noexcept;
    bool empty() const noexcept;
    void add(TextureResourceLease lease);

    std::array<std::optional<TextureResourceLease>,
               kInlineTextureLaunchLeases>
        inline_leases_;
    std::size_t inline_count_{0};
    std::vector<TextureResourceLease> overflow_leases_;
  };

  struct TextureResourceView {
    TextureResourceHandle handle;
    TextureResourceLease lease;
  };

  struct TextureResourceSlotView {
    const Texture *view{nullptr};
    TextureResourceHandle handle;
  };

  using TextureInflightLeaseMap =
      std::unordered_map<std::uint64_t, TextureResourceLease>;

  class ExternalDenseStorageResource {
   public:
    ExternalDenseStorageResource(
        DeviceAllocation allocation,
        std::uint64_t allocation_bytes,
        ExternalDenseStorageRelease release,
        std::shared_ptr<ExternalSynchronizationDomain> synchronization_domain)
        : allocation(allocation),
          allocation_bytes(allocation_bytes),
          synchronization_domain(std::move(synchronization_domain)),
          release_(std::move(release)) {
    }
    ExternalDenseStorageResource(const ExternalDenseStorageResource &) = delete;
    ExternalDenseStorageResource &operator=(
        const ExternalDenseStorageResource &) = delete;
    ~ExternalDenseStorageResource() noexcept = default;

    void finalize();

    DeviceAllocation allocation{kDeviceNullAllocation};
    std::uint64_t allocation_bytes{0};
    std::shared_ptr<ExternalSynchronizationDomain> synchronization_domain;

   private:
    ExternalDenseStorageRelease release_;
    bool finalized_{false};
  };

  struct ExternalDenseStorageFinalizer {
    void operator()(ExternalDenseStorageResource &resource) const {
      resource.finalize();
    }
  };

  using ExternalDenseStorageRegistry =
      RuntimeResourceRegistry<ExternalDenseStorageResource,
                              ExternalDenseStorageFinalizer>;
  using ExternalDenseStorageHandle = ExternalDenseStorageRegistry::Handle;
  using ExternalDenseStorageLease = ExternalDenseStorageRegistry::Lease;
  static constexpr ExternalDenseStorageRegistry::Kind
      kExternalDenseStorageResourceKind = 5;
  static constexpr std::size_t kInlineExternalDenseStorageLaunchLeases = 8;

  class ExternalDenseStorageLaunchLeases {
   public:
    ExternalDenseStorageLaunchLeases() = default;
    ExternalDenseStorageLaunchLeases(const ExternalDenseStorageLaunchLeases &) =
        delete;
    ExternalDenseStorageLaunchLeases &operator=(
        const ExternalDenseStorageLaunchLeases &) = delete;
    ExternalDenseStorageLaunchLeases(
        ExternalDenseStorageLaunchLeases &&) noexcept = default;
    ExternalDenseStorageLaunchLeases &operator=(
        ExternalDenseStorageLaunchLeases &&) noexcept = default;

   private:
    friend class Program;
    ExternalDenseStorageResource *find(
        ExternalDenseStorageHandle handle) const noexcept;
    bool empty() const noexcept;
    void add(ExternalDenseStorageLease lease);
    const std::vector<std::shared_ptr<ExternalSynchronizationDomain>> &
    synchronization_domains() const noexcept;
    void track_synchronization_domain(
        const std::shared_ptr<ExternalSynchronizationDomain> &domain);

    std::array<std::optional<ExternalDenseStorageLease>,
               kInlineExternalDenseStorageLaunchLeases>
        inline_leases_;
    std::size_t inline_count_{0};
    std::vector<ExternalDenseStorageLease> overflow_leases_;
    std::vector<std::shared_ptr<ExternalSynchronizationDomain>>
        synchronization_domains_;
  };

  using ExternalDenseStorageInflightLeaseMap =
      std::unordered_map<std::uint64_t, ExternalDenseStorageLease>;

  struct RuntimeCompletionResourceBatch final
      : public RuntimeCompletionResources {
    ArgPackInflightLeaseMap argpacks;
    NdarrayInflightLeaseMap ndarrays;
    TextureInflightLeaseMap textures;
    ExternalDenseStorageInflightLeaseMap external_dense_storage;

    std::size_t retained_resource_count(
        std::uint32_t kind) const noexcept override;
    bool empty() const noexcept {
      return argpacks.empty() && ndarrays.empty() && textures.empty() &&
             external_dense_storage.empty();
    }
  };

  struct DenseFieldHostCopyStagingResource {
    DeviceAllocationUnique upload;
    std::size_t upload_capacity{0};
    DeviceAllocationUnique readback;
    std::size_t readback_capacity{0};
    std::mutex mutex;
  };

  using DenseFieldStagingRegistry =
      RuntimeResourceRegistry<DenseFieldHostCopyStagingResource>;
  using DenseFieldStagingHandle = DenseFieldStagingRegistry::Handle;
  using DenseFieldStagingLease = DenseFieldStagingRegistry::Lease;
  static constexpr DenseFieldStagingRegistry::Kind
      kDenseFieldStagingResourceKind = 4;

  ArgPackLaunchLeases acquire_argpack_launch_leases(
      const LaunchContextBuilder &ctx);
  void pin_argpack_launch_leases(ArgPackLaunchLeases &leases);
  void release_completed_argpack_leases();
  void close_argpack_resources();
  static std::uint64_t argpack_lease_key(ArgPackResourceHandle handle);
  NdarrayLaunchLeases acquire_ndarray_launch_leases(
      LaunchContextBuilder &ctx);
  void resolve_dense_storage_launch_context(
      LaunchContextBuilder &ctx,
      NdarrayLaunchLeases &ndarray_leases,
      ExternalDenseStorageLaunchLeases &external_leases);
  storage::ResolvedDenseBinding resolve_dense_storage_descriptor(
      const storage::DenseStorageDescriptor &descriptor,
      NdarrayLaunchLeases &ndarray_leases,
      ExternalDenseStorageLaunchLeases &external_leases,
      const storage::RuntimeStorageArgument *runtime_argument = nullptr);
  NdarrayLaunchLeases acquire_ndarray_leases(
      std::initializer_list<const Ndarray *> views);
  NdarrayLaunchLeases acquire_ndarray_leases(
      const std::vector<const Ndarray *> &views);
  NdarrayLaunchLeases acquire_ndarray_leases(const Ndarray *const *views,
                                             std::size_t count);
  void launch_kernel_impl(const CompiledKernelData &compiled_kernel_data,
                          LaunchContextBuilder &ctx,
                          const KernelLaunchHandle *registered_handle);
  void pin_ndarray_launch_leases(NdarrayLaunchLeases &leases);
  void release_completed_ndarray_leases();
  void close_ndarray_resources();
  static std::uint64_t ndarray_lease_key(NdarrayResourceHandle handle);
  TextureLaunchLeases acquire_texture_launch_leases(
      LaunchContextBuilder &ctx);
  TextureLaunchLeases acquire_texture_leases(
      std::initializer_list<const Texture *> views);
  TextureLaunchLeases acquire_texture_leases(
      const std::vector<const Texture *> &views);
  TextureLaunchLeases acquire_texture_leases(const Texture *const *views,
                                             std::size_t count);
  void pin_texture_launch_leases(TextureLaunchLeases &leases);
  void release_completed_texture_leases();
  void close_texture_resources();
  static std::uint64_t texture_lease_key(TextureResourceHandle handle);
  void pin_external_dense_storage_launch_leases(
      ExternalDenseStorageLaunchLeases &leases);
  void release_completed_external_dense_storage_leases();
  void begin_external_access_epoch(
      ExternalAccessEpoch &epoch,
      const ExternalDenseStorageLaunchLeases &leases);
  void close_external_dense_storage_resources();
  static std::uint64_t external_dense_storage_lease_key(
      ExternalDenseStorageHandle handle);
  ExternalDenseStorageHandle external_dense_storage_handle(
      const storage::StorageOwnerRef &owner) const noexcept;
  DenseFieldHostCopyStagingResource &dense_field_staging_resource();
  void close_dense_field_staging_resource();
  std::shared_ptr<RuntimeCompletionResourceBatch>
  detach_runtime_completion_resources();
  void track_runtime_completion(const RuntimeCompletion &completion);
  void collect_ready_runtime_completions();
  void complete_all_runtime_completions() noexcept;
  void fail_all_runtime_completions(const std::string &reason) noexcept;
  void initialize_runtime_backend_telemetry_baseline();
  void attach_runtime_fault_reporter();
  void detach_runtime_fault_reporter() noexcept;
  std::size_t runtime_completion_resource_count(
      std::uint32_t kind) const noexcept;
  void acquire_runtime_submission_reader() noexcept;
  void release_runtime_submission_reader() noexcept;
  void acquire_runtime_submission_writer() noexcept;
  void release_runtime_submission_writer() noexcept;

  CompileConfig compile_config_;

  uint64 ndarray_writer_counter_{0};
  uint64 ndarray_reader_counter_{0};
  int global_id_counter_{0};

  // SNode information that requires using Program.
  SNodeFieldMap snode_to_fields_;
  SNodeRwAccessorsBank snode_rw_accessors_bank_;

  std::vector<std::unique_ptr<SNodeTree>> snode_trees_;
  std::stack<int> free_snode_tree_ids_;
  std::vector<std::uint64_t> snode_tree_generations_;
  std::vector<std::uint8_t> snode_tree_active_;
  mutable std::shared_mutex snode_tree_lifecycle_mutex_;
  std::atomic<std::uint64_t> snode_tree_mutation_epoch_{1};

  std::vector<std::unique_ptr<Function>> functions_;
  std::unordered_map<FunctionKey, Function *> function_map_;

  std::unique_ptr<ProgramImpl> program_impl_;
  std::shared_ptr<void> lifetime_token_{std::make_shared<int>(0)};
  const std::uint64_t runtime_completion_domain_;
  std::shared_ptr<RuntimeFaultDomain> runtime_fault_domain_;
  RuntimeTraceRecorder runtime_trace_;
  PrimitiveWorkspaceArena primitive_workspace_arena_;
  struct RuntimeBackendTelemetryBaseline {
    std::uint64_t backend_waits{0};
    std::uint64_t backend_wait_ns{0};
    std::uint64_t backend_lock_samples{0};
    std::uint64_t backend_lock_contentions{0};
    std::uint64_t backend_lock_sampled_wait_ns{0};
  };
  RuntimeBackendTelemetryBaseline runtime_backend_telemetry_baseline_;
  // Default kernel/Graph execution only publishes a cheap dirty bit. The
  // reader/writer gate is activated permanently by the first completion
  // request, and only temporarily by a legacy Program::synchronize().
  alignas(64) std::atomic<bool> runtime_completion_tracking_enabled_{false};
  std::atomic<bool> runtime_submission_pending_{false};
  std::atomic<std::uint64_t> runtime_submission_gate_{0};
  static constexpr std::uint64_t kRuntimeSubmissionWriterBit =
      std::uint64_t{1} << 63;
  static constexpr std::uint64_t kRuntimeSubmissionReaderMask =
      ~kRuntimeSubmissionWriterBit;
  std::atomic<std::uint64_t> runtime_submission_epoch_{0};
  std::atomic<std::uint64_t> last_runtime_completion_submission_epoch_{0};
  std::atomic<std::uint64_t> next_runtime_completion_sequence_{1};
  mutable std::mutex runtime_completion_mutex_;
  std::deque<RuntimeCompletion> runtime_completions_;
  static constexpr std::size_t kMaxTrackedRuntimeCompletions = 64;
  DenseFieldStagingRegistry dense_field_staging_resources_;
  DenseFieldStagingHandle dense_field_staging_handle_;
  DenseFieldStagingLease dense_field_staging_lease_;
  bool dense_field_staging_open_{true};
  std::atomic<std::uint64_t> dense_storage_direct_submissions_{0};
  std::atomic<std::uint64_t> dense_storage_resolved_bindings_{0};
  std::atomic<std::uint64_t> dense_storage_resolved_bytes_{0};
  std::atomic<std::uint64_t> dense_storage_ndarray_bindings_{0};
  std::atomic<std::uint64_t> dense_storage_field_bindings_{0};
  std::atomic<std::uint64_t> dense_storage_external_bindings_{0};
  float64 total_compilation_time_{0.0};
  static std::atomic<int> num_instances_;
  bool finalized_{false};
  int hash_snode_tree_count_{0};

  ArgPackResourceRegistry argpack_resources_;
  mutable std::mutex argpack_lifecycle_mutex_;
  // One gate covers a submission containing any combination of high-level
  // resources. Separate per-type gates would require fragile lock ordering.
  std::recursive_mutex runtime_resource_submission_mutex_;
  bool argpack_resources_open_{true};
  std::unordered_map<const ArgPack *, ArgPackResourceView> argpack_views_;
  ArgPackInflightLeaseMap argpack_inflight_leases_;
  NdarrayResourceRegistry ndarray_resources_;
  mutable std::mutex ndarray_lifecycle_mutex_;
  bool ndarray_resources_open_{true};
  std::unordered_map<const Ndarray *, NdarrayResourceView> ndarray_views_;
  std::vector<NdarrayResourceSlotView> ndarray_view_slots_;
  NdarrayInflightLeaseMap ndarray_inflight_leases_;
  TextureResourceRegistry texture_resources_;
  mutable std::mutex texture_lifecycle_mutex_;
  bool texture_resources_open_{true};
  std::unordered_map<const Texture *, TextureResourceView> texture_views_;
  // Generation-qualified direct index used by ordinary kernel validation.
  // Pointer lookup remains for delete/external callers that do not carry a
  // captured handle.
  std::vector<TextureResourceSlotView> texture_view_slots_;
  TextureInflightLeaseMap texture_inflight_leases_;
  ExternalDenseStorageRegistry external_dense_storage_resources_;
  mutable std::mutex external_dense_storage_lifecycle_mutex_;
  bool external_dense_storage_resources_open_{true};
  ExternalDenseStorageInflightLeaseMap external_dense_storage_inflight_leases_;
};

TI_FORCE_INLINE Program::RuntimeSubmissionScope::RuntimeSubmissionScope(
    Program *program)
    : program_(program) {
  program_->ensure_runtime_submission_allowed("runtime submission");
  if (!program_->runtime_completion_tracking_enabled_.load(
          std::memory_order_acquire)) {
    // Before the first completion request there is no writer to exclude and
    // no nested scope that needs coalescing. Avoid even the TLS accessor so
    // ordinary kernel/Graph calls pay only the inlined atomic check and dirty
    // publication.
    program_ = nullptr;
    return;
  }
  Program *&active_program =
      runtime_completion_detail::active_runtime_submission_program();
  previous_program_ = active_program;
  TI_ASSERT(previous_program_ == nullptr || previous_program_ == program_);
  if (previous_program_ == nullptr) {
    program_->acquire_runtime_submission_reader();
    owns_reader_ = true;
  }
  active_program = program_;
}

TI_FORCE_INLINE Program::RuntimeSubmissionScope::RuntimeSubmissionScope(
    RuntimeSubmissionScope &&other) noexcept
    : program_(std::exchange(other.program_, nullptr)),
      previous_program_(std::exchange(other.previous_program_, nullptr)),
      owns_reader_(std::exchange(other.owns_reader_, false)) {
}

TI_FORCE_INLINE Program::RuntimeSubmissionScope::~RuntimeSubmissionScope() {
  if (program_ == nullptr) {
    return;
  }
  Program *&active_program =
      runtime_completion_detail::active_runtime_submission_program();
  TI_ASSERT(active_program == program_);
  active_program = previous_program_;
  if (owns_reader_) {
    program_->release_runtime_submission_reader();
  }
}

}  // namespace taichi::lang
