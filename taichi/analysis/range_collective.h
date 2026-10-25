#pragma once

#include <map>
#include <iterator>
#include <unordered_set>

#include "taichi/ir/statements.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/type_utils.h"

namespace taichi::lang {

using RangeReductionSlots = std::map<std::size_t, TypedConstant>;

inline bool valid_range_reduction_pointer(const ThreadLocalPtrStmt &ptr,
                                          std::size_t bytes) {
  const auto type = ptr.ret_type.ptr_removed();
  if (type != PrimitiveType::i32 && type != PrimitiveType::u32 &&
      type != PrimitiveType::i64 && type != PrimitiveType::u64 &&
      type != PrimitiveType::f32 && type != PrimitiveType::f64)
    return false;
  const auto size = data_type_size(type);
  return bytes >= size && ptr.offset <= bytes - size && ptr.offset % size == 0;
}

// Verify the final scalar reduction shape, after lowering and optimization.
// Merely having TLS storage is not a reduction proof. Unknown regions retain
// their baseline launch; this checker never changes the kernel or its grouping.
inline bool describe_range_reduction_slots(const OffloadedStmt &task,
                                           RangeReductionSlots &slots) {
  // CUDA's ordinary range helper exits whole empty blocks before either TLS
  // region. Keep that backend premise explicit; other backends need their own
  // execution proof before using this qualification for auxiliary regions.
  if (task.device != Arch::cuda || !task.tls_prologue || !task.tls_epilogue)
    return false;
  for (const auto &owned : task.tls_prologue->statements) {
    auto *stmt = owned.get();
    if (auto *ptr = stmt->cast<ThreadLocalPtrStmt>()) {
      if (!valid_range_reduction_pointer(*ptr, task.tls_size))
        return false;
    } else if (auto *store = stmt->cast<GlobalStoreStmt>()) {
      auto *ptr = store->dest->cast<ThreadLocalPtrStmt>();
      auto *value = store->val->cast<ConstStmt>();
      if (!ptr || !value || !valid_range_reduction_pointer(*ptr, task.tls_size) ||
          ptr->ret_type.ptr_removed() != value->val.dt)
        return false;
      const auto [position, inserted] = slots.emplace(ptr->offset, value->val);
      if (!inserted)
        return false;
      const auto next = std::next(position);
      if (next != slots.end() &&
          next->first - ptr->offset < data_type_size(value->val.dt))
        return false;
      if (position != slots.begin()) {
        const auto previous = std::prev(position);
        if (ptr->offset - previous->first < data_type_size(previous->second.dt))
          return false;
      }
    } else if (!stmt->is<ConstStmt>()) {
      return false;
    }
  }
  if (slots.empty())
    return false;

  const auto valid_slot = [&](ThreadLocalPtrStmt *ptr) {
    if (!ptr || !valid_range_reduction_pointer(*ptr, task.tls_size))
      return false;
    const auto found = slots.find(ptr->offset);
    return found != slots.end() && found->second.dt == ptr->ret_type.ptr_removed();
  };
  std::unordered_set<std::size_t> reduced;
  for (const auto &owned : task.tls_epilogue->statements) {
    auto *stmt = owned.get();
    if (auto *ptr = stmt->cast<ThreadLocalPtrStmt>()) {
      if (!valid_slot(ptr))
        return false;
    } else if (auto *load = stmt->cast<GlobalLoadStmt>()) {
      if (!valid_slot(load->src->cast<ThreadLocalPtrStmt>()))
        return false;
    } else if (auto *atomic = stmt->cast<AtomicOpStmt>()) {
      auto *load = atomic->val->cast<GlobalLoadStmt>();
      auto *ptr = load ? load->src->cast<ThreadLocalPtrStmt>() : nullptr;
      if (!atomic->is_reduction || !valid_slot(ptr) ||
          !(atomic->dest->is<GetChStmt>() ||
            atomic->dest->is<GlobalTemporaryStmt>() ||
            atomic->dest->is<MatrixPtrStmt>()) ||
          !reduced.insert(ptr->offset).second)
        return false;
      const auto type = ptr->ret_type.ptr_removed();
      TypedConstant neutral;
      if (atomic->op_type == AtomicOpType::add)
        neutral = TypedConstant(type, 0);
      else if (atomic->op_type == AtomicOpType::min)
        neutral = get_max_value(type);
      else if (atomic->op_type == AtomicOpType::max)
        neutral = get_min_value(type);
      else
        return false;
      if (!(slots.at(ptr->offset) == neutral))
        return false;
    } else if (auto *lookup = stmt->cast<SNodeLookupStmt>()) {
      if (lookup->activate || (lookup->snode->type != SNodeType::dense &&
                               lookup->snode->type != SNodeType::root))
        return false;
    } else if (auto *matrix = stmt->cast<MatrixPtrStmt>()) {
      if (!matrix->offset->is<ConstStmt>())
        return false;
    } else if (!(stmt->is<ConstStmt>() || stmt->is<GetRootStmt>() ||
                 stmt->is<GetChStmt>() || stmt->is<GlobalTemporaryStmt>())) {
      return false;
    }
  }
  return reduced.size() == slots.size();
}

}  // namespace taichi::lang
