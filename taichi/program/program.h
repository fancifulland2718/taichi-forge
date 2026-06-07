// Program  - Taichi program execution context

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <atomic>
#include <stack>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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

  StreamSemaphore flush();
  StreamSemaphore flush_if_pending();
  bool has_pending_gfx_command_list() const;

  /**
   * Materializes the runtime.
   */
  void materialize_runtime();

  int get_snode_tree_size();

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

  // TODO: do we still need result_buffer?
  DeviceAllocation allocate_memory_on_device(std::size_t alloc_size,
                                             uint64 *result_buffer) {
    return program_impl_->allocate_memory_on_device(alloc_size, result_buffer);
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

  Texture *create_texture(BufferFormat buffer_format,
                          const std::vector<int> &shape);

  intptr_t get_ndarray_data_ptr_as_int(const Ndarray *ndarray);

  void fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val);

  void copy_ndarray_fast(Ndarray *dst, Ndarray *src);

  void copy_ndarray_from_host(Ndarray *dst,
                              const void *src,
                              std::size_t bytes);

  void copy_ndarray_to_host(Ndarray *src, void *dst, std::size_t bytes);

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

  void vulkan_metric_reduce_clear_workspace();

  std::size_t vulkan_metric_reduce_workspace_bytes() const;

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
  CompileConfig compile_config_;

  uint64 ndarray_writer_counter_{0};
  uint64 ndarray_reader_counter_{0};
  int global_id_counter_{0};

  // SNode information that requires using Program.
  SNodeFieldMap snode_to_fields_;
  SNodeRwAccessorsBank snode_rw_accessors_bank_;

  std::vector<std::unique_ptr<SNodeTree>> snode_trees_;
  std::stack<int> free_snode_tree_ids_;

  std::vector<std::unique_ptr<Function>> functions_;
  std::unordered_map<FunctionKey, Function *> function_map_;

  std::unique_ptr<ProgramImpl> program_impl_;
  struct DenseFieldHostCopyStagingCache {
    DeviceAllocationUnique upload;
    std::size_t upload_capacity{0};
    DeviceAllocationUnique readback;
    std::size_t readback_capacity{0};
    std::mutex mutex;
  };

  DenseFieldHostCopyStagingCache dense_field_host_copy_staging_;
  float64 total_compilation_time_{0.0};
  static std::atomic<int> num_instances_;
  bool finalized_{false};
  int hash_snode_tree_count_{0};

  // TODO: Move ndarrays_, argpacks_ and textures_ to be managed by runtime
  std::unordered_map<void *, std::unique_ptr<Ndarray>> ndarrays_;
  std::unordered_map<void *, std::unique_ptr<ArgPack>> argpacks_;
  std::vector<std::unique_ptr<Texture>> textures_;
};

}  // namespace taichi::lang
