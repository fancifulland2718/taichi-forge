#include "taichi/transforms/sink_serial_integer_prelude.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/stmt_use_index.h"
#include "taichi/ir/type_utils.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang::irpass {
namespace {

bool scalar_integer(Stmt *stmt) {
  return stmt && stmt->ret_type->is<PrimitiveType>() &&
         is_integral(stmt->ret_type);
}

bool dense_unpacked_leaf(SNode *snode) {
  return snode && snode->type == SNodeType::place && snode->is_path_all_dense &&
         !snode->is_bit_level && snode->dt->is<PrimitiveType>() &&
         (is_integral(snode->dt) || is_real(snode->dt));
}

bool scalar_snapshot_pointer(GlobalPtrStmt *ptr) {
  return dense_unpacked_leaf(ptr->snode) &&
         ptr->snode->dt == PrimitiveType::i32 &&
         ptr->snode->num_active_indices == 0 && ptr->indices.empty() &&
         !ptr->activate && !ptr->is_bit_vectorized && !ptr->is_cell_access;
}

// A snapshot may be moved only after the whole consumer has been proved not
// to change it. Existing alias_analysis treats ExternalPtr and GlobalPtr as
// different, but external ndarray storage can alias a captured field. This
// checker therefore admits a closed field/local-memory IR vocabulary instead.
class FieldSnapshotConsumer final : public BasicStmtVisitor {
 public:
  explicit FieldSnapshotConsumer(const std::unordered_set<SNode *> &sources)
      : sources_(sources) {
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(GlobalPtrStmt *stmt) override {
    safe_ &= dense_unpacked_leaf(stmt->snode) && !stmt->is_cell_access &&
             !stmt->is_bit_vectorized;
    for (auto *index : stmt->indices) {
      safe_ &= scalar_integer(index);
    }
  }

  void visit(AllocaStmt *stmt) override {
    safe_ &= !stmt->is_shared;
  }

  void visit(MatrixPtrStmt *stmt) override {
    // Byte-offset field matrix pointers may cross leaf boundaries. Keep them
    // out until a full address-range proof exists; local matrix pointers are
    // private. Unwrap every level, not only the immediate origin.
    safe_ &= local_pointer(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    safe_ &= memory_pointer(stmt->src, false) && !stmt->ret_type.is_pointer();
  }

  void visit(GlobalStoreStmt *stmt) override {
    safe_ &=
        memory_pointer(stmt->dest, true) && !stmt->val->ret_type.is_pointer();
  }

  void visit(AtomicOpStmt *stmt) override {
    safe_ &=
        memory_pointer(stmt->dest, true) && !stmt->val->ret_type.is_pointer();
  }

  void visit(LocalLoadStmt *stmt) override {
    safe_ &= local_pointer(stmt->src) && !stmt->ret_type.is_pointer();
  }

  void visit(LocalStoreStmt *stmt) override {
    safe_ &= local_pointer(stmt->dest) && !stmt->val->ret_type.is_pointer();
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    // Nested scalar loops and branches retain their exact control semantics.
    // Sparse/mesh loops and other container forms remain outside this proof.
    safe_ &= stmt->is<IfStmt>() || stmt->is<RangeForStmt>();
    scalar_operands(stmt);
  }

  void visit(Stmt *stmt) override {
    bool known = stmt->is<ConstStmt>() || stmt->is<UnaryOpStmt>() ||
                 stmt->is<BinaryOpStmt>() || stmt->is<TernaryOpStmt>() ||
                 stmt->is<LoopIndexStmt>() || stmt->is<ContinueStmt>() ||
                 stmt->is<MatrixInitStmt>();
    if (auto *arg = stmt->cast<ArgLoadStmt>()) {
      known = !arg->is_ptr && arg->create_load && arg->arg_depth == 0 &&
              arg->arg_id.size() == 1 && arg->ret_type->is<PrimitiveType>();
    }
    // Unknown/internal/external calls, raw addresses, external ndarray access,
    // sparse operations, AD stacks, assertions and random effects fail closed.
    safe_ &= known && !stmt->ret_type.is_pointer();
    scalar_operands(stmt);
  }

  bool safe() const {
    return safe_;
  }

 private:
  static bool local_pointer(Stmt *pointer) {
    std::unordered_set<Stmt *> seen;
    while (auto *matrix = pointer ? pointer->cast<MatrixPtrStmt>() : nullptr) {
      if (!seen.insert(pointer).second || !scalar_integer(matrix->offset)) {
        return false;
      }
      pointer = matrix->origin;
    }
    auto *alloca = pointer ? pointer->cast<AllocaStmt>() : nullptr;
    return alloca && !alloca->is_shared;
  }

  bool memory_pointer(Stmt *pointer, bool write) const {
    if (local_pointer(pointer)) {
      return true;
    }
    // Deliberately reject every MatrixPtr chain rooted in a field. Even an
    // apparently different field origin is insufficient with raw byte offsets.
    auto *field = pointer ? pointer->cast<GlobalPtrStmt>() : nullptr;
    return field && dense_unpacked_leaf(field->snode) &&
           !field->is_cell_access && !field->is_bit_vectorized &&
           (!write || !sources_.count(field->snode));
  }

  void scalar_operands(Stmt *stmt) {
    for (int i = 0; i < stmt->num_operands(); ++i) {
      auto *operand = stmt->operand(i);
      safe_ &= !operand || !operand->ret_type.is_pointer();
    }
  }

  const std::unordered_set<SNode *> &sources_;
  bool safe_{true};
};

bool immutable_integer_statement(Stmt *stmt) {
  // ExtractConstant hoists floating constants from otherwise integer-prefix
  // kernels too. Copying a literal does not recompute floating arithmetic.
  if (stmt->is<ConstStmt>()) {
    return stmt->ret_type->is<PrimitiveType>() &&
           (is_integral(stmt->ret_type) || is_real(stmt->ret_type));
  }
  if (!scalar_integer(stmt)) {
    return false;
  }
  if (auto *arg = stmt->cast<ArgLoadStmt>()) {
    return !arg->is_ptr && arg->create_load && arg->arg_depth == 0 &&
           arg->arg_id.size() == 1;
  }
  if (auto *shape = stmt->cast<ExternalTensorShapeAlongAxisStmt>()) {
    // Restrict this first candidate to a top-level ndarray argument. Nested
    // argpack metadata has a separate device-buffer lifetime/alias contract.
    return shape->arg_id.size() == 1 && shape->axis >= 0;
  }
  for (int i = 0; i < stmt->num_operands(); ++i) {
    if (!scalar_integer(stmt->operand(i))) {
      return false;
    }
  }
  if (auto *unary = stmt->cast<UnaryOpStmt>()) {
    switch (unary->op_type) {
      case UnaryOpType::neg:
      case UnaryOpType::bit_not:
      case UnaryOpType::logic_not:
      case UnaryOpType::cast_value:
        return true;
      default:
        return false;
    }
  }
  if (auto *binary = stmt->cast<BinaryOpStmt>()) {
    if (binary->is_bit_vectorized) {
      return false;
    }
    switch (binary->op_type) {
      case BinaryOpType::add:
      case BinaryOpType::sub:
      case BinaryOpType::mul:
      case BinaryOpType::min:
      case BinaryOpType::max:
      case BinaryOpType::bit_and:
      case BinaryOpType::bit_or:
      case BinaryOpType::bit_xor:
      case BinaryOpType::cmp_lt:
      case BinaryOpType::cmp_le:
      case BinaryOpType::cmp_gt:
      case BinaryOpType::cmp_ge:
      case BinaryOpType::cmp_eq:
      case BinaryOpType::cmp_ne:
      case BinaryOpType::logical_and:
      case BinaryOpType::logical_or:
        return true;
      default:
        // Division, remainder and shifts can have undefined inputs. On a
        // zero-trip range they must not silently lose their original execution.
        return false;
    }
  }
  if (auto *select = stmt->cast<TernaryOpStmt>()) {
    return select->op_type == TernaryOpType::select;
  }
  // In particular, GlobalLoad has no global side effect but reads a snapshot.
  // Also exclude pointers, alloca, RNG, calls, control flow and AD statements.
  return false;
}

bool plain_body(OffloadedStmt *stmt) {
  return stmt->body && !stmt->tls_prologue && !stmt->tls_epilogue &&
         !stmt->bls_prologue && !stmt->bls_epilogue && !stmt->mesh_prologue &&
         !stmt->snode && !stmt->mesh && !stmt->reversed &&
         !stmt->is_bit_vectorized && !stmt->begin_stmt && !stmt->end_stmt;
}

bool supported_prefix_statement(Stmt *stmt) {
  if (auto *pointer = stmt->cast<GlobalPtrStmt>()) {
    return scalar_snapshot_pointer(pointer);
  }
  if (auto *load = stmt->cast<GlobalLoadStmt>()) {
    auto *pointer = load->src ? load->src->cast<GlobalPtrStmt>() : nullptr;
    return pointer && scalar_snapshot_pointer(pointer) &&
           load->ret_type == PrimitiveType::i32;
  }
  return immutable_integer_statement(stmt);
}

bool possible_pair(OffloadedStmt *serial, OffloadedStmt *range) {
  if (serial->task_type != OffloadedTaskType::serial ||
      range->task_type != OffloadedTaskType::range_for ||
      serial->device != range->device || !plain_body(serial) ||
      !plain_body(range) || !range->const_begin || !range->const_end ||
      serial->body->statements.empty()) {
    return false;
  }
  bool nonconstant = false;
  for (const auto &stmt : serial->body->statements) {
    if (!supported_prefix_statement(stmt.get())) {
      return false;
    }
    // On CPU the snapshot avoids repeated field-address work in the loop.
    // Keep it there; GPU launch savings justify per-thread reads. Immutable
    // by-value/shape prefixes remain backend-independent, without a size cap.
    if (stmt->is<GlobalLoadStmt>() && range->device != Arch::cuda &&
        range->device != Arch::vulkan) {
      return false;
    }
    nonconstant |= !stmt->is<ConstStmt>();
  }
  return nonconstant;
}

bool in_body(Stmt *stmt, Block *body) {
  for (auto *block = stmt->parent; block; block = block->parent_block()) {
    if (block == body) {
      return true;
    }
  }
  return false;
}

void pin_special_references(OffloadedStmt *task,
                            std::unordered_set<Stmt *> &pinned) {
  if (task->begin_stmt) {
    pinned.insert(task->begin_stmt);
  }
  if (task->end_stmt) {
    pinned.insert(task->end_stmt);
  }
  for (const auto *references :
       {&task->owned_offset_local, &task->total_offset_local,
        &task->owned_num_local, &task->total_num_local}) {
    for (const auto &[element_type, value] : *references) {
      pinned.insert(value);
    }
  }
}

bool eligible(OffloadedStmt *serial,
              OffloadedStmt *range,
              const StmtUseIndex &uses,
              const std::unordered_set<Stmt *> &pinned) {
  if (!possible_pair(serial, range)) {
    return false;
  }
  std::unordered_set<Stmt *> prefix;
  std::unordered_set<Stmt *> live;
  std::unordered_set<SNode *> snapshots;
  bool useful = false;
  for (const auto &owned : serial->body->statements) {
    auto *stmt = owned.get();
    if (pinned.count(stmt)) {
      return false;
    }
    if (auto *load = stmt->cast<GlobalLoadStmt>()) {
      snapshots.insert(load->src->as<GlobalPtrStmt>()->snode);
    }
    // Closed, topologically ordered DAG: no reads from previous tasks and no
    // accidental forward/self references. Do not recursively walk expression
    // trees, which would rescan diamonds or overflow on long prefixes.
    for (int i = 0; i < stmt->num_operands(); ++i) {
      if (!prefix.count(stmt->operand(i))) {
        return false;
      }
    }
    prefix.insert(stmt);
  }
  for (const auto &owned : serial->body->statements) {
    auto *stmt = owned.get();
    for (auto *user : uses.users(stmt)) {
      if (prefix.count(user)) {
        continue;
      }
      if (in_body(user, range->body.get())) {
        live.insert(stmt);
        useful |= !stmt->is<ConstStmt>();
      } else if (!stmt->is<ConstStmt>()) {
        return false;
      }
      // Ordinary constant operands in other tasks retain the existing
      // FixCrossOffloadReferences literal-cloning path. Hidden references were
      // pinned above, including hidden references to constants.
    }
  }
  if (!useful) {
    return false;
  }
  if (!snapshots.empty()) {
    FieldSnapshotConsumer consumer(snapshots);
    range->body->accept(&consumer);
    if (!consumer.safe()) {
      return false;
    }
  }
  for (auto it = serial->body->statements.rbegin();
       it != serial->body->statements.rend(); ++it) {
    auto *stmt = it->get();
    if (!live.count(stmt)) {
      if (!stmt->is<ConstStmt>()) {
        return false;
      }
      continue;
    }
    for (int i = 0; i < stmt->num_operands(); ++i) {
      live.insert(stmt->operand(i));
    }
  }
  return true;
}
}  // namespace

bool sink_serial_integer_preludes(
    IRNode *root,
    AutodiffMode autodiff_mode,
    const std::unordered_set<Stmt *> &range_bound_statements) {
  if (autodiff_mode != AutodiffMode::kNone || !root->is<Block>()) {
    return false;
  }
  auto *block = root->as<Block>();
  auto pinned = range_bound_statements;
  OffloadedStmt *previous = nullptr;
  bool possible = false;
  for (const auto &owned : block->statements) {
    auto *task = owned->cast<OffloadedStmt>();
    if (!task) {
      return false;
    }
    pin_special_references(task, pinned);
    possible |= previous && possible_pair(previous, task);
    previous = task;
  }
  if (!possible) {
    return false;
  }
  StmtUseIndex uses(root);
  std::unordered_set<Stmt *> removed;
  for (size_t i = 1; i < block->statements.size(); ++i) {
    auto *serial = block->statements[i - 1]->as<OffloadedStmt>();
    auto *range = block->statements[i]->as<OffloadedStmt>();
    if (!eligible(serial, range, uses, pinned)) {
      continue;
    }
    VecStatement prefix;
    for (auto &stmt : serial->body->statements) {
      prefix.push_back(std::move(stmt));
    }
    serial->body->statements.clear();
    range->body->insert(std::move(prefix), 0);
    removed.insert(serial);
  }
  const bool changed = !removed.empty();
  if (changed) {
    block->erase(std::move(removed));
  }
  return changed;
}
}  // namespace taichi::lang::irpass
