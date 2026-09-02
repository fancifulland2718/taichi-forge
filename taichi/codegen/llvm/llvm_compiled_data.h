#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>

#include "llvm/IR/Module.h"
#include "taichi/common/serialization.h"
#include "taichi/ir/offloaded_task_type.h"

namespace taichi::lang {

class OffloadedTask {
 public:
  static constexpr int kSparseListOpNone = 0;
  static constexpr int kSparseListOpClearList = 1;
  static constexpr int kSparseListOpListgen = 2;
  static constexpr int kSparseMutationNone = -1;
  static constexpr int kSparseMutationUnknown = -2;

  std::string name;
  // Host-only. The compilation manager reconstructs it from the stable cache
  // identity, so it must not become part of the offline-cache payload.
  std::string task_id;
  // Host-only CUDA entry point resolved once when the owning JIT module is
  // registered. It is deliberately excluded from TI_IO_DEF; deserialized
  // tasks are rebound after their module is loaded.
  void *native_function{nullptr};
  OffloadedTaskType task_type{OffloadedTaskType::serial};
  int requested_block_dim{0};
  int requested_grid_dim{0};
  bool source_block_dim_explicit{false};
  // Exact Forge task-plan requests. Defaults describe the ordinary compiler
  // path and keep old offline-cache payloads semantically unchanged.
  int requested_thread_local_mode{0};  // 0 auto, 1 on, 2 off
  int requested_cuda_min_blocks_per_sm{2};
  int requested_cuda_max_registers{-1};
  int requested_grid_residency_waves{0};
  int requested_range_work_per_thread_target{1};
  std::string requested_memory_strategy{"direct"};
  int block_dim{0};
  int grid_dim{0};
  std::int64_t constant_range_size{-1};
  int static_shared_array_bytes{0};
  int dynamic_shared_array_bytes{0};
  std::uint64_t thread_local_bytes{0};
  bool one_to_one{false};
  bool external_shared_staged{false};
  int external_shared_arg_index{-1};
  int external_shared_halo_low{0};
  int external_shared_halo_high{0};
  std::vector<int> external_shared_arg_indices;
  std::vector<int> external_shared_halo_lows;
  std::vector<int> external_shared_halo_highs;
  std::vector<int> external_shared_byte_offsets;
  std::vector<int> external_shared_element_bytes;
  std::vector<int> external_shared_scalar_bytes;
  std::vector<std::vector<int>> external_shared_element_shapes;
  std::vector<int> external_shared_iteration_shape;
  std::vector<int> external_shared_iteration_origin;
  std::vector<int> external_shared_tile_shape;
  std::vector<std::vector<int>> external_shared_halo_lows_nd;
  std::vector<std::vector<int>> external_shared_halo_highs_nd;
  std::vector<std::vector<std::vector<int>>>
      external_shared_access_offsets;
  int sparse_list_op{kSparseListOpNone};
  int sparse_list_snode_id{-1};
  int sparse_list_parent_snode_id{-1};
  bool may_mutate_sparse_topology{false};
  int sparse_mutation_snode_id{kSparseMutationNone};

  explicit OffloadedTask(const std::string &name = "",
                         int block_dim = 0,
                         int grid_dim = 0,
                         int dynamic_shared_array_bytes = 0)
      : name(name),
        block_dim(block_dim),
        grid_dim(grid_dim),
        dynamic_shared_array_bytes(dynamic_shared_array_bytes) {};
  TI_IO_DEF(name,
            task_type,
            requested_block_dim,
            requested_grid_dim,
            source_block_dim_explicit,
            requested_thread_local_mode,
            requested_cuda_min_blocks_per_sm,
            requested_cuda_max_registers,
            requested_grid_residency_waves,
            requested_range_work_per_thread_target,
            requested_memory_strategy,
            block_dim,
            grid_dim,
            constant_range_size,
            static_shared_array_bytes,
            dynamic_shared_array_bytes,
            thread_local_bytes,
            one_to_one,
            external_shared_staged,
            external_shared_arg_index,
            external_shared_halo_low,
            external_shared_halo_high,
            external_shared_arg_indices,
            external_shared_halo_lows,
            external_shared_halo_highs,
            external_shared_byte_offsets,
            external_shared_element_bytes,
            external_shared_scalar_bytes,
            external_shared_element_shapes,
            external_shared_iteration_shape,
            external_shared_iteration_origin,
            external_shared_tile_shape,
            external_shared_halo_lows_nd,
            external_shared_halo_highs_nd,
            external_shared_access_offsets,
            sparse_list_op,
            sparse_list_snode_id,
            sparse_list_parent_snode_id,
            may_mutate_sparse_topology,
            sparse_mutation_snode_id);
};

struct LLVMCompiledTask {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> struct_for_tls_sizes;
  LLVMCompiledTask() = default;
  LLVMCompiledTask(LLVMCompiledTask &&) = default;
  LLVMCompiledTask &operator=(LLVMCompiledTask &&) = default;
  LLVMCompiledTask(std::vector<OffloadedTask> tasks,
                   std::unique_ptr<llvm::Module> module,
                   std::unordered_set<int> used_tree_ids,
                   std::unordered_set<int> struct_for_tls_sizes)
      : tasks(std::move(tasks)),
        module(std::move(module)),
        used_tree_ids(std::move(used_tree_ids)),
        struct_for_tls_sizes(std::move(struct_for_tls_sizes)) {
  }
  LLVMCompiledTask clone() const;
  TI_IO_DEF(tasks);
};

struct LLVMCompiledKernel {
  std::vector<OffloadedTask> tasks;
  // -1 inherits the Program-level CUDA register cap. This lives with the
  // compiled kernel so an exact specialization keeps its artifact control
  // through offline-cache load and delayed launcher registration.
  int cuda_max_registers{-1};
  // Modules are tied to their LLVMContext. Parallel compilation can finish on
  // a short-lived worker thread, so retain that worker's context independently
  // of the thread-local registry until every module clone has been released.
  // Keep this member before `module`: reverse destruction order must destroy
  // the module before releasing its context owner.
  std::shared_ptr<llvm::LLVMContext> module_context_owner;
  std::unique_ptr<llvm::Module> module{nullptr};
  LLVMCompiledKernel() = default;
  LLVMCompiledKernel(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel &operator=(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel(std::vector<OffloadedTask> tasks,
                     std::unique_ptr<llvm::Module> module,
                     std::shared_ptr<llvm::LLVMContext> module_context_owner =
                         nullptr,
                     int cuda_max_registers = -1)
      : tasks(std::move(tasks)),
        cuda_max_registers(cuda_max_registers),
        module_context_owner(std::move(module_context_owner)),
        module(std::move(module)) {
  }
  LLVMCompiledKernel clone() const;
  TI_IO_DEF(tasks, cuda_max_registers);
};

}  // namespace taichi::lang
