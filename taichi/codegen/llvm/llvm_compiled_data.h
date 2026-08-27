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
  OffloadedTaskType task_type{OffloadedTaskType::serial};
  int requested_block_dim{0};
  int requested_grid_dim{0};
  int block_dim{0};
  int grid_dim{0};
  int static_shared_array_bytes{0};
  int dynamic_shared_array_bytes{0};
  std::uint64_t thread_local_bytes{0};
  bool one_to_one{false};
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
            block_dim,
            grid_dim,
            static_shared_array_bytes,
            dynamic_shared_array_bytes,
            thread_local_bytes,
            one_to_one,
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
                         nullptr)
      : tasks(std::move(tasks)),
        module_context_owner(std::move(module_context_owner)),
        module(std::move(module)) {
  }
  LLVMCompiledKernel clone() const;
  TI_IO_DEF(tasks);
};

}  // namespace taichi::lang
