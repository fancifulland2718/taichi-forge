#include "taichi/analysis/gather_snode_tree_dependencies.h"

#include <algorithm>
#include <map>
#include <unordered_set>

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"

namespace taichi::lang {
namespace irpass::analysis {
namespace {

bool is_layout_independent_cuda_intrinsic(const InternalFuncStmt &stmt) {
  // These intrinsics consume scalar lane values or impose ordering; none
  // interprets an ndarray pointer/descriptor. Do not infer this from scalar
  // argument types alone: other internal calls accept integer-encoded handles.
  static const std::unordered_set<std::string> names{
      "block_barrier", "block_barrier_and_i32", "block_barrier_or_i32",
      "block_barrier_count_i32", "grid_memfence", "warp_barrier",
      "cuda_all_sync_i32", "cuda_any_sync_i32", "cuda_uni_sync_i32",
      "cuda_ballot_i32", "cuda_active_mask", "cuda_match_any_sync_i32",
      "cuda_match_all_sync_i32", "cuda_shfl_sync_i32", "cuda_shfl_sync_f32",
      "cuda_shfl_up_sync_i32", "cuda_shfl_up_sync_f32",
      "cuda_shfl_down_sync_i32", "cuda_shfl_down_sync_f32",
      "cuda_shfl_xor_sync_i32"};
  return !stmt.with_runtime_context && names.count(stmt.func_name) &&
         !stmt.ret_type.is_pointer() &&
         std::none_of(stmt.args.begin(), stmt.args.end(), [](const Stmt *arg) {
           return arg->ret_type.is_pointer();
         });
}

class SNodeTreeDependencyCollector : public BasicStmtVisitor {
 public:
  explicit SNodeTreeDependencyCollector(bool include_source = false)
      : include_source_(include_source) {
  }
  void collect_array_layouts() {
    collect_array_layouts_ = true;
  }

  void visit(FuncCallStmt *stmt) override {
    visit_function(stmt->func);
  }

  void visit(FrontendFuncCallStmt *stmt) override {
    visit_function(stmt->func);
  }

  void visit(Block *block) override {
    for (SNode *snode : block->stop_gradients) {
      record(snode);
    }
    BasicStmtVisitor::visit(block);
  }

  void visit(FrontendSNodeOpStmt *stmt) override {
    record(stmt->snode);
    if (stmt->op_type == SNodeOpType::activate) {
      record_hash_activation_path(stmt->snode);
    }
  }

  void visit(FrontendForStmt *stmt) override {
    record(stmt->snode);
    BasicStmtVisitor::visit(stmt);
  }

  void visit(GlobalPtrStmt *stmt) override {
    record(stmt->snode);
    if (stmt->activate) {
      record_hash_activation_path(stmt->snode);
    }
  }

  void visit(MatrixOfGlobalPtrStmt *stmt) override {
    for (SNode *snode : stmt->snodes) {
      record(snode);
      if (stmt->activate) {
        record_hash_activation_path(snode);
      }
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    record(stmt->snode);
    if (stmt->op_type == SNodeOpType::activate) {
      record_hash_activation_path(stmt->snode);
    }
  }

  void visit(StructForStmt *stmt) override {
    record(stmt->snode);
    BasicStmtVisitor::visit(stmt);
  }

  void visit(GetRootStmt *stmt) override {
    record(stmt->root());
  }

  void visit(SNodeLookupStmt *stmt) override {
    record(stmt->snode);
    if (stmt->activate) {
      record_hash_activation_path(stmt->snode);
    }
  }

  void visit(GetChStmt *stmt) override {
    record(stmt->input_snode);
    record(stmt->output_snode);
  }

  void visit(OffloadedStmt *stmt) override {
    record(stmt->snode);
    if (stmt->task_type == OffloadedTaskType::mesh_for)
      runtime_state_ |= runtime_state_mesh;
    BasicStmtVisitor::visit(stmt);
  }

  void visit(RandStmt *) override {
    runtime_state_ |= runtime_state_random;
  }
  void visit(AssertStmt *) override {
    runtime_state_ |= runtime_state_error;
  }
  void visit(PrintStmt *) override {
    runtime_state_ |= runtime_state_print;
  }
  void visit(ExternalFuncCallStmt *) override {
    runtime_state_ |= runtime_state_external;
    array_layouts_known_ = false;
  }
  void visit(ExternalPtrStmt *stmt) override {
    if (!collect_array_layouts_) return;
    const auto *argument = stmt->base_ptr->cast<ArgLoadStmt>();
    if (!argument) {
      array_layouts_known_ = false;
      return;
    }
    const bool affine = stmt->runtime_affine && stmt->byte_stride == 0;
    auto [entry, inserted] = array_layouts_.emplace(argument->arg_id, affine);
    if (!inserted)
      entry->second &= affine;
  }
  void visit(InternalFuncStmt *stmt) override {
    // Internal pointer consumers need a separate argument-layout proof.
    if (!is_layout_independent_cuda_intrinsic(*stmt))
      array_layouts_known_ = false;
    if (stmt->with_runtime_context)
      runtime_state_ |= runtime_state_external;
  }

  void visit(ClearListStmt *stmt) override {
    record(stmt->snode);
  }

  std::vector<int> result() const {
    std::vector<int> ids(tree_ids_.begin(), tree_ids_.end());
    std::sort(ids.begin(), ids.end());
    return ids;
  }

  SNodeRelocationStructure relocation_structures() const {
    return relocation_structures_;
  }

  bool may_trigger_hash_overflow() const {
    return may_trigger_hash_overflow_;
  }

  std::uint32_t runtime_state() const {
    return runtime_state_ |
           (relocation_structures_ != SNodeRelocationStructure::none
                ? runtime_state_sparse
                : 0u) |
           (may_trigger_hash_overflow_ ? runtime_state_error : 0u);
  }

  std::vector<std::vector<int>> runtime_affine_arrays() const {
    std::vector<std::vector<int>> result;
    if (array_layouts_known_)
      for (const auto &[argument, affine] : array_layouts_)
        if (affine) result.push_back(argument);
    return result;
  }

 private:
  void visit_function(Function *func) {
    // Independent function arguments require their own caller mapping proof.
    array_layouts_known_ = false;
    // Field roots used only by a callee still belong to the caller's binding
    // table and lifetime contract. Mark before visiting to handle recursion.
    if (visited_functions_.insert(func).second) {
      TI_ERROR_IF(func->definition_retired(),
                  "Cannot compile a retired function definition after its "
                  "SNodeTree was destroyed; rebuild the function/kernel.");
      if (include_source_) {
        const auto &ids = func->source_snode_tree_dependencies();
        tree_ids_.insert(ids.begin(), ids.end());
      }
      func->ir->accept(this);
    }
  }

  void record_hash_activation_path(const SNode *snode) {
    for (const SNode *node = snode; node != nullptr; node = node->parent) {
      if (node->type == SNodeType::hash) {
        may_trigger_hash_overflow_ = true;
        return;
      }
    }
  }

  void record(const SNode *snode) {
    if (snode != nullptr) {
      tree_ids_.insert(snode->get_snode_tree_id());
      for (const SNode *node = snode; node != nullptr; node = node->parent) {
        switch (node->type) {
          case SNodeType::pointer:
            relocation_structures_ |= SNodeRelocationStructure::pointer;
            break;
          case SNodeType::bitmasked:
            relocation_structures_ |= SNodeRelocationStructure::bitmasked;
            break;
          case SNodeType::dynamic:
            relocation_structures_ |= SNodeRelocationStructure::dynamic;
            break;
          case SNodeType::hash:
            relocation_structures_ |= SNodeRelocationStructure::hash;
            break;
          default:
            break;
        }
      }
    }
  }

  std::unordered_set<int> tree_ids_;
  bool include_source_{false};
  std::unordered_set<Function *> visited_functions_;
  SNodeRelocationStructure relocation_structures_{
      SNodeRelocationStructure::none};
  bool may_trigger_hash_overflow_{false};
  std::uint32_t runtime_state_{runtime_state_known};
  bool array_layouts_known_{true};
  bool collect_array_layouts_{false};
  std::map<std::vector<int>, bool> array_layouts_;
};

bool tree_contains_non_dense_snode(const SNode &node) {
  if (!node.is_path_all_dense) {
    return true;
  }
  return std::any_of(
      node.ch.begin(), node.ch.end(), [](const std::unique_ptr<SNode> &child) {
        return tree_contains_non_dense_snode(*child);
      });
}

}  // namespace

std::vector<int> gather_snode_tree_dependencies(const Kernel &kernel) {
  TI_ASSERT(kernel.ir != nullptr);
  return gather_snode_tree_dependencies(*kernel.ir);
}

KernelRuntimeDependencies gather_kernel_runtime_dependencies(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  collector.collect_array_layouts();
  ir.accept(&collector);
  return {collector.result(), collector.relocation_structures(),
          collector.may_trigger_hash_overflow(), collector.runtime_state(),
          collector.runtime_affine_arrays()};
}

std::vector<int> gather_snode_tree_dependencies(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  ir.accept(&collector);
  return collector.result();
}

std::vector<int> gather_source_snode_tree_dependencies(IRNode &ir) {
  SNodeTreeDependencyCollector collector(/*include_source=*/true);
  ir.accept(&collector);
  return collector.result();
}

SNodeRelocationStructure gather_snode_relocation_structures(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  ir.accept(&collector);
  return collector.relocation_structures();
}

bool may_trigger_hash_overflow(IRNode &ir) {
  SNodeTreeDependencyCollector collector;
  ir.accept(&collector);
  return collector.may_trigger_hash_overflow();
}

bool has_non_dense_snode_tree_dependency(
    Program &program,
    const std::vector<SNodeTreeDependency> &dependencies) {
  for (const auto &dependency : dependencies) {
    const SNode *root = program.get_snode_root(dependency.tree_id);
    if (root != nullptr && tree_contains_non_dense_snode(*root)) {
      return true;
    }
  }
  return false;
}

}  // namespace irpass::analysis
}  // namespace taichi::lang
