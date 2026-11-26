#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>

#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang::irpass {
namespace dense_grid {

inline int range_extent(RangeForStmt *loop) {
  if (!loop || loop->reversed || loop->strictly_serialized ||
      loop->is_bit_vectorized || loop->one_to_one || loop->block_dim ||
      loop->model_block_dim || !loop->device_options.empty() ||
      !loop->range_hint.empty())
    return 0;
  auto *begin = loop->begin->cast<ConstStmt>();
  auto *end = loop->end->cast<ConstStmt>();
  if (!begin || !end || begin->ret_type != PrimitiveType::i32 ||
      end->ret_type != PrimitiveType::i32 || begin->val.val_int32() != 0)
    return 0;
  return end->val.val_int32();
}

// A constant plane, or a scalar kernel argument plus a constant offset.
// These expressions are invariant across *all* cells, unlike a difference
// between two subscripts evaluated at a single iteration.
struct Plane {
  ArgLoadStmt *base;
  uint32_t offset;
};

inline std::optional<Plane> plane(Stmt *index) {
  if (index->ret_type != PrimitiveType::i32)
    return {};
  if (auto *c = index->cast<ConstStmt>())
    return Plane{nullptr, uint32_t(c->val.val_int32())};
  uint32_t offset = 0;
  if (auto *b = index->cast<BinaryOpStmt>()) {
    auto *c = b->rhs->cast<ConstStmt>();
    if (!c ||
        (b->op_type != BinaryOpType::sub && b->op_type != BinaryOpType::add))
      return {};
    offset = uint32_t(c->val.val_int32());
    if (b->op_type == BinaryOpType::sub)
      offset = 0u - offset;
    index = b->lhs;
  }
  auto *arg = index->cast<ArgLoadStmt>();
  if (!arg || arg->is_ptr || arg->arg_depth ||
      arg->ret_type != PrimitiveType::i32)
    return {};
  return Plane{arg, offset};
}

struct CellAccesses {
  GlobalPtrStmt *write{nullptr};
  std::vector<GlobalPtrStmt *> reads;
  std::unordered_set<Stmt *> locals;

  static bool dense_place(GlobalPtrStmt *p) {
    return p && p->snode->is_place() && p->snode->is_path_all_dense &&
           !p->snode->is_bit_level && !p->is_cell_access &&
           !p->is_bit_vectorized;
  }

  bool collect(Block *body, LoopIndexStmt *column) {
    if (!body)
      return true;
    for (const auto &stmt : body->statements) {
      if (stmt.get() == column || stmt->is<ConstStmt>() ||
          stmt->is<UnaryOpStmt>() || stmt->is<BinaryOpStmt>() ||
          stmt->is<TernaryOpStmt>() || stmt->is<ArgLoadStmt>())
        continue;
      if (auto *p = stmt->cast<GlobalPtrStmt>()) {
        if (!dense_place(p))
          return false;
      } else if (auto *a = stmt->cast<AllocaStmt>()) {
        if (a->is_shared)
          return false;
        locals.insert(a);
      } else if (auto *s = stmt->cast<LocalStoreStmt>()) {
        if (!locals.count(s->dest))
          return false;
      } else if (auto *s = stmt->cast<LocalLoadStmt>()) {
        if (!locals.count(s->src))
          return false;
      } else if (auto *s = stmt->cast<GlobalStoreStmt>()) {
        if (write || !dense_place(write = s->dest->cast<GlobalPtrStmt>()))
          return false;
      } else if (auto *s = stmt->cast<GlobalLoadStmt>()) {
        auto *ptr = s->src->cast<GlobalPtrStmt>();
        if (!dense_place(ptr))
          return false;
        reads.push_back(ptr);
      } else if (auto *branch = stmt->cast<IfStmt>()) {
        if (!collect(branch->true_statements.get(), column) ||
            !collect(branch->false_statements.get(), column))
          return false;
      } else {
        // Includes atomics, calls, barriers, thread IDs and control transfers.
        return false;
      }
    }
    return true;
  }

  bool independent(Stmt *row, Stmt *column) const {
    if (!write)
      return false;
    const auto &indices = write->indices;
    auto i = std::find(indices.begin(), indices.end(), row);
    auto j = std::find(indices.begin(), indices.end(), column);
    if (i == indices.end() || j == indices.end())
      return false;
    // Consecutive lanes should write consecutive dense cells. A transposed
    // store already favors row parallelism and would lose coalescing here.
    const auto *leaf = write->snode->parent;
    if (!leaf || leaf->type != SNodeType::dense)
      return false;
    int axis = write->snode->physical_index_position[j - indices.begin()];
    const auto &layout = leaf->extractors[axis];
    if (layout.shape < 32 || layout.acc_shape != 1)
      return false;
    for (auto *read : reads) {
      if (read->snode != write->snode)
        continue;
      const auto &other = read->indices;
      if (indices.size() != other.size())
        return false;
      if (other[i - indices.begin()] == row &&
          other[j - indices.begin()] == column)
        continue;
      bool disjoint = false;
      for (size_t k = 0; k < indices.size(); ++k) {
        auto a = plane(indices[k]);
        auto b = plane(other[k]);
        if (a && b && a->offset != b->offset &&
            ((!a->base && !b->base) ||
             (a->base && b->base && a->base->arg_id == b->base->arg_id))) {
          disjoint = true;
          break;
        }
      }
      if (!disjoint)
        return false;
    }
    return true;
  }
};

inline bool parallelize(RangeForStmt *outer) {
  int rows = range_extent(outer);
  // Target underfilled row-parallel grids. Ignore other geometries before
  // inspecting the cell body or constructing any dependency information.
  if (rows <= 0 || rows > 1024 || outer->body->size() < 2)
    return false;
  auto *inner = outer->body->back()->cast<RangeForStmt>();
  int cols = range_extent(inner);
  if (cols < 64 || int64_t(rows) * cols < 4096 ||
      cols > std::numeric_limits<int>::max() / rows || inner->body->size() == 0)
    return false;
  LoopIndexStmt *row = nullptr;
  for (const auto &s : outer->body->statements) {
    if (s.get() == inner || s->is<ConstStmt>())
      continue;
    auto *idx = s->cast<LoopIndexStmt>();
    if (row || !idx || idx->loop != outer || idx->index != 0)
      return false;
    row = idx;
  }
  LoopIndexStmt *column = nullptr;
  for (const auto &s : inner->body->statements) {
    if (auto *idx = s->cast<LoopIndexStmt>()) {
      if (column || idx->loop != inner || idx->index != 0)
        return false;
      column = idx;
    }
  }
  CellAccesses cells;
  if (!row || !column || !cells.collect(inner->body.get(), column) ||
      !cells.independent(row, column))
    return false;

  auto *width =
      outer->insert_before_me(Stmt::make<ConstStmt>(TypedConstant(cols)));
  outer->end = outer->insert_before_me(
      Stmt::make<ConstStmt>(TypedConstant(rows * cols)));
  auto *flat = row->insert_before_me(Stmt::make<LoopIndexStmt>(outer, 0));
  flat->ret_type = PrimitiveType::i32;
  auto decode = [&](LoopIndexStmt *idx, BinaryOpType op) {
    auto value = Stmt::make<BinaryOpStmt>(op, flat, width);
    value->ret_type = PrimitiveType::i32;
    replace_all_usages_with(inner->body.get(), idx, value.get());
    idx->parent->replace_with(idx, std::move(value), false);
  };
  decode(row, BinaryOpType::div);
  decode(column, BinaryOpType::mod);
  outer->body->replace_with(
      inner, VecStatement(std::move(inner->body->statements)), false);
  return true;
}

}  // namespace dense_grid
}  // namespace taichi::lang::irpass
