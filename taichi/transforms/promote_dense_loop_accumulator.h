#pragma once

#include "taichi/transforms/parallelize_dense_grid.h"
#include "taichi/ir/analysis.h"

namespace taichi::lang::irpass {

// Recognize a dense cell indexed by both q = i / width and r = i - q * width.
// The pair identifies one outer iteration even if the axes are transposed.
inline bool has_private_cell_coordinates(GlobalPtrStmt *ptr,
                                         OffloadedStmt *task) {
  for (auto *index : ptr->indices) {
    auto *q = index->cast<BinaryOpStmt>();
    if (!q ||
        (q->op_type != BinaryOpType::floordiv &&
         q->op_type != BinaryOpType::div) ||
        q->ret_type != PrimitiveType::i32)
      continue;
    auto *i = q->lhs->cast<LoopIndexStmt>();
    auto *width = q->rhs->cast<ConstStmt>();
    if (!i || i->loop != task || i->index != 0 || !width ||
        width->ret_type != PrimitiveType::i32 || width->val.val_int32() <= 0)
      continue;
    bool remainder = false;
    bool invariant = true;
    for (auto *axis : ptr->indices) {
      if (axis == q || axis->is<ConstStmt>())
        continue;
      if (auto *arg = axis->cast<ArgLoadStmt>();
          arg && !arg->is_ptr && arg->create_load && !arg->arg_depth &&
          arg->ret_type == PrimitiveType::i32)
        continue;
      auto *r = axis->cast<BinaryOpStmt>();
      if (!r || r->lhs != i || r->ret_type != PrimitiveType::i32) {
        invariant = false;
        break;
      }
      if (r->op_type == BinaryOpType::mod &&
          analysis::same_value(r->rhs, width)) {
        remainder = true;
        continue;
      }
      auto *mul = r->rhs->cast<BinaryOpStmt>();
      if (r->op_type != BinaryOpType::sub || !mul ||
          mul->op_type != BinaryOpType::mul ||
          !((mul->lhs == q && analysis::same_value(mul->rhs, width)) ||
            (mul->rhs == q && analysis::same_value(mul->lhs, width)))) {
        invariant = false;
        break;
      }
      remainder = true;
    }
    if (invariant && remainder)
      return true;
  }
  return false;
}

// The address proof above admits only immutable indices and their pure
// arithmetic. Move existing definitions once, in dependency order.
inline void hoist_cell_address(Stmt *stmt, RangeForStmt *loop) {
  if (stmt->parent != loop->body.get())
    return;
  for (int i = 0; i < stmt->num_operands(); ++i)
    if (auto *arg = stmt->operand(i))
      hoist_cell_address(arg, loop);
  loop->insert_before_me(loop->body->extract(stmt));
}

inline bool private_cell_loop_is_isolated(Block *body,
                                          RangeForStmt *loop,
                                          AtomicOpStmt *add,
                                          GlobalPtrStmt *ptr) {
  for (const auto &stmt : body->statements) {
    Stmt *s = stmt.get();
    if (s == ptr || s == add)
      continue;
    if (s->has_operand(add) || s->has_operand(ptr))
      return false;
    if (s == loop) {
      if (!private_cell_loop_is_isolated(loop->body.get(), loop, add, ptr))
        return false;
    } else if (auto *branch = s->cast<IfStmt>()) {
      for (auto *arm :
           {branch->true_statements.get(), branch->false_statements.get()})
        if (arm && !private_cell_loop_is_isolated(arm, loop, add, ptr))
          return false;
    } else if (auto *p = s->cast<GlobalPtrStmt>()) {
      if (!dense_grid::CellAccesses::dense_place(p) || p->snode == ptr->snode)
        return false;
    } else if (auto *load = s->cast<GlobalLoadStmt>()) {
      if (!load->src->is<GlobalPtrStmt>())
        return false;
    } else if (auto *element = s->cast<MatrixPtrStmt>()) {
      if (!element->origin->is<AllocaStmt>() &&
          !element->origin->is<MatrixPtrStmt>())
        return false;
    } else if (auto *a = s->cast<AllocaStmt>()) {
      if (a->is_shared)
        return false;
    } else if (auto *arg = s->cast<ArgLoadStmt>()) {
      if (arg->is_ptr || !arg->create_load || arg->arg_depth)
        return false;
    } else if (!(s->is<LoopIndexStmt>() || s->is<ConstStmt>() ||
                 s->is<UnaryOpStmt>() || s->is<BinaryOpStmt>() ||
                 s->is<TernaryOpStmt>() || s->is<MatrixInitStmt>() ||
                 s->is<LocalLoadStmt>() || s->is<LocalStoreStmt>())) {
      return false;
    }
  }
  return true;
}

inline void promote_dense_loop_accumulator(OffloadedStmt *task) {
  if (task->task_type != OffloadedTaskType::range_for || !task->body ||
      task->body->size() == 0 || !task->const_begin || task->begin_value < 0)
    return;
  auto *loop = task->body->back()->cast<RangeForStmt>();
  if (dense_grid::range_extent(loop) < 2 || loop->body->size() == 0)
    return;
  auto *add = loop->body->back()->cast<AtomicOpStmt>();
  if (!add || add->op_type != AtomicOpType::add)
    return;
  auto *ptr = add->dest->cast<GlobalPtrStmt>();
  if (!dense_grid::CellAccesses::dense_place(ptr) ||
      ptr->snode->dt != PrimitiveType::f32 ||
      task->mem_access_opt.has_flag(ptr->snode, SNodeAccessFlag::block_local) ||
      task->mem_access_opt.has_flag(ptr->snode, SNodeAccessFlag::mesh_local) ||
      !has_private_cell_coordinates(ptr, task))
    return;

  // Only the candidate may access the output field. Other loads must come
  // from distinct dense fields. Opaque calls, external aliases, shared state
  // and observable atomic results cannot be moved across this loop.
  if (!private_cell_loop_is_isolated(task->body.get(), loop, add, ptr))
    return;

  hoist_cell_address(ptr, loop);
  auto *acc = loop->insert_before_me(Stmt::make<AllocaStmt>(ptr->snode->dt));
  auto *first = loop->insert_before_me(Stmt::make<GlobalLoadStmt>(ptr));
  loop->insert_before_me(Stmt::make<LocalStoreStmt>(acc, first));
  auto *old = add->insert_before_me(Stmt::make<LocalLoadStmt>(acc));
  auto *next = add->insert_before_me(Stmt::make<InternalFuncStmt>(
      "private_atomic_add_f32", std::vector<Stmt *>{old, add->val},
      PrimitiveType::f32, false));
  add->replace_with(VecStatement(Stmt::make<LocalStoreStmt>(acc, next)), false);
  auto *last = loop->insert_after_me(Stmt::make<LocalLoadStmt>(acc));
  last->insert_after_me(Stmt::make<GlobalStoreStmt>(ptr, last));
}

}  // namespace taichi::lang::irpass
