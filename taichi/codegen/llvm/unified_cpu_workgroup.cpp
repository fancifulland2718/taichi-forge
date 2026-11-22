#include "taichi/codegen/llvm/unified_cpu_workgroup.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang::LLVM {
namespace {

bool builtin(Stmt *stmt, const char *name) {
  auto *call = stmt->cast<InternalFuncStmt>();
  return call && call->func_name == name;
}

Stmt *integer_cast(Block &block, Stmt *input, DataType dtype) {
  auto *cast = block.push_back<UnaryOpStmt>(UnaryOpType::cast_value, input)
                   ->as<UnaryOpStmt>();
  cast->cast_type = dtype;
  cast->ret_type = dtype;
  return cast;
}

bool is_address_expression(Stmt *s) {
  return s->is<MatrixPtrStmt>() || s->is<ExternalPtrStmt>() ||
         s->is<DynamicSharedPtrStmt>() || s->is<ArgLoadStmt>();
}

// A CPU worker executes a whole logical group. Each execution barrier starts
// a new loop over the same lanes. Values read before a barrier must retain
// their original value even if another lane subsequently changes the source.
class GroupPhases {
 public:
  GroupPhases(Block &group,
              RangeForStmt &range,
              Stmt *base,
              Stmt *begin,
              Stmt *count,
              int width)
      : group_(group),
        range_(range),
        base_(base),
        begin_(begin),
        count_(count),
        width_(width) {
  }

  void emit(stmt_vector statements, Block *branch = nullptr) {
    for (auto &statement : statements) {
      auto *s = statement.get();
      if (builtin(s, "workgroupBarrier") || builtin(s, "block_barrier")) {
        group_values_[s] = group_constant(0);
        if (!branch)
          lanes_ = nullptr;
        continue;
      }
      if (auto *allocation = s->cast<AllocaStmt>();
          allocation && allocation->is_shared) {
        allocation->is_shared = false;
        group_values_[s] = group_.insert(std::move(statement), 0);
        continue;
      }
      if (!lanes_)
        start_phase();
      Block &here = branch ? *branch : *lanes_;
      if (auto *choice = s->cast<IfStmt>()) {
        choice->cond = current(choice->cond, here);
        here.insert(std::move(statement));
        for (auto *arm :
             {choice->true_statements.get(), choice->false_statements.get()})
          if (arm)
            emit(std::move(arm->statements), arm);
        continue;
      }
      if (auto *index = s->cast<LoopIndexStmt>();
          index && index->loop == &range_) {
        indices_.insert(s);
        continue;
      }
      if (builtin(s, "localInvocationId")) {
        lane_ids_.insert(s);
        continue;
      }
      if (builtin(s, "vkGlobalThreadIdx")) {
        invocation_ids_.insert(s);
        continue;
      }
      if (builtin(s, "workgroupMemoryBarrier")) {
        // All accesses to this group's scratch run on the same CPU worker.
        group_values_[s] = group_constant(0);
        continue;
      }
      Stmt *defined = nullptr;
      if (auto *ptr = s->cast<DynamicSharedPtrStmt>()) {
        auto &local = allocations_[ptr->allocation_id];
        std::vector<int> shape;
        for (auto *extent : ptr->shape)
          shape.push_back(extent->as<ConstStmt>()->val.val_int32());
        if (!local) {
          local = group_.insert(
              Stmt::make<AllocaStmt>(shape, ptr->ret_type.ptr_removed()), 0);
        } else {
          DataType expected = TypeFactory::get_instance().get_tensor_type(
              shape, ptr->ret_type.ptr_removed());
          TI_ERROR_IF(local->ret_type.ptr_removed() != expected,
                      "Inconsistent CPU workgroup scratch layout");
        }
        defined =
            here.push_back<MatrixPtrStmt>(local, current(ptr->offset, here));
        expression_inputs_[defined] = {local, ptr->offset};
      } else {
        if (is_address_expression(s))
          expression_inputs_[s] = s->get_operands();
        for (int i = 0; i < s->num_operands(); ++i)
          s->set_operand(i, current(s->operand(i), here));
        if (auto *load = s->cast<GlobalLoadStmt>();
            load && is_group_address(load->src)) {
          defined = here.push_back<LocalLoadStmt>(load->src);
          defined->ret_type = s->ret_type;
        } else if (auto *store = s->cast<GlobalStoreStmt>();
                   store && is_group_address(store->dest)) {
          defined = here.push_back<LocalStoreStmt>(store->dest, store->val);
        } else {
          defined = here.insert(std::move(statement));
        }
      }
      if (defined->is<MatrixPtrStmt>() &&
          defined->as<MatrixPtrStmt>()->origin->is<AllocaStmt>())
        local_addresses_.insert(defined);
      definitions_[s] = {defined, lanes_, lane_};
      if (defined != s)
        definitions_[defined] = definitions_[s];
    }
    // Subexpressions are keyed by the original SSA definitions until every
    // phase has been emitted, including storage declarations in nested arms.
    for (auto &s : statements)
      if (s)
        replaced_.push_back(std::move(s));
  }

 private:
  struct Definition {
    Stmt *value;
    Block *phase;
    Stmt *lane;
  };
  Block &group_;
  RangeForStmt &range_;
  Stmt *base_, *begin_, *count_;
  int width_;
  Block *lanes_{nullptr};
  Stmt *lane_{}, *index_{};
  std::unordered_map<Stmt *, Definition> definitions_;
  std::unordered_map<Stmt *, Stmt *> group_values_, spills_;
  std::map<std::pair<Block *, Stmt *>, Stmt *> restored_;
  std::unordered_map<Stmt *, std::vector<Stmt *>> expression_inputs_;
  stmt_vector replaced_;
  std::unordered_map<int, Stmt *> allocations_;
  std::unordered_set<Stmt *> indices_, lane_ids_, invocation_ids_, local_addresses_;

  bool is_group_address(Stmt *address) const {
    return local_addresses_.count(address) != 0;
  }
  Stmt *group_constant(int n) {
    return group_.insert(Stmt::make<ConstStmt>(TypedConstant(n)), 0);
  }
  void start_phase() {
    auto *loop = group_
                     .push_back<RangeForStmt>(group_constant(0), count_,
                                              std::make_unique<Block>(), false,
                                              1, 0, true)
                     ->as<RangeForStmt>();
    lanes_ = loop->body.get();
    lane_ = lanes_->push_back<LoopIndexStmt>(loop, 0);
    index_ = lanes_->push_back<BinaryOpStmt>(BinaryOpType::add, base_, lane_);
    lane_ids_.insert(lane_);
    indices_.insert(index_);
    restored_.clear();
  }
  Stmt *current(Stmt *s, Block &here) {
    if (indices_.count(s))
      return index_;
    if (lane_ids_.count(s))
      return lane_;
    if (invocation_ids_.count(s))
      return here.push_back<BinaryOpStmt>(BinaryOpType::sub, index_, begin_);
    if (auto it = group_values_.find(s); it != group_values_.end())
      return it->second;
    auto found = definitions_.find(s);
    if (found == definitions_.end())
      return s;
    const auto &d = found->second;
    if (d.phase == lanes_)
      return d.value;
    const auto key = std::make_pair(&here, s);
    if (auto it = restored_.find(key); it != restored_.end())
      return it->second;
    Stmt *result;
    if (d.value->is<ConstStmt>() || is_address_expression(d.value)) {
      auto copy = d.value->clone();
      for (int i = 0; i < copy->num_operands(); ++i)
        copy->set_operand(i,
                          current(expression_inputs_.at(d.value).at(i), here));
      result = here.insert(std::move(copy));
      if (is_group_address(d.value))
        local_addresses_.insert(result);
    } else {
      auto type = d.value->ret_type;
      TI_ERROR_IF(
          type.is_pointer() || !type->is<PrimitiveType>(),
          "Unsupported CPU workgroup value across an execution barrier: {}",
          type->to_string());
      auto &storage = spills_[s];
      if (!storage) {
        storage = group_.insert(
            Stmt::make<AllocaStmt>(std::vector<int>{width_}, type), 0);
        auto *ptr = d.phase->push_back<MatrixPtrStmt>(storage, d.lane);
        d.phase->push_back<LocalStoreStmt>(ptr, d.value);
      }
      result = here.push_back<LocalLoadStmt>(
          here.push_back<MatrixPtrStmt>(storage, lane_));
      result->ret_type = type;
    }
    return restored_[key] = result;
  }
};
}  // namespace

bool materialize_cpu_workgroup(RangeForStmt &range,
                               const CompileConfig &config,
                               bool unbound) {
  int width = range.block_dim;
  if (range.model_block_dim) {
    auto *constant = range.model_block_dim->cast<ConstStmt>();
    if (!constant && unbound)
      return false;
    TI_ERROR_IF(!constant,
                "CPU workgroup width requires a bound model dimension");
    width = constant->val.val_int32();
    TI_ERROR_IF(width <= 0, "Invalid bound CPU workgroup width {}", width);
  }
  if (!width) {
    width = range.strictly_serialized ? 1 : config.default_cpu_block_dim;
    TI_ERROR_IF(width <= 0, "Invalid default CPU workgroup width {}", width);
  }
  std::vector<Block *> blocks{range.body.get()};
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (auto &s : blocks[i]->statements) {
      if (auto *choice = s->cast<IfStmt>()) {
        for (auto *arm :
             {choice->true_statements.get(), choice->false_statements.get()})
          if (arm)
            blocks.push_back(arm);
      } else if (s->is_container_statement() || s->is<ContinueStmt>() ||
                 (i != 0 && width > 1 &&
                  (builtin(s.get(), "workgroupBarrier") ||
                   builtin(s.get(), "block_barrier")))) {
        if (unbound)
          return false;
        TI_ERROR(
            "CPU workgroup requires convergent top-level phases at a reachable "
            "workgroup barrier");
      }
      if (auto *ptr = s->cast<DynamicSharedPtrStmt>()) {
        int elements = 1;
        for (auto *expr : ptr->shape) {
          auto *extent = expr->cast<ConstStmt>();
          if (!extent && unbound)
            return false;
          TI_ERROR_IF(!extent,
                      "CPU workgroup scratch requires bound dimensions");
          const int value = extent->val.val_int32();
          TI_ERROR_IF(value <= 0 || elements > INT_MAX / value,
                      "Invalid CPU workgroup scratch extent");
          elements *= value;
        }
      }
    }
  }
  irpass::type_check(&range, config);
  auto statements = std::move(range.body->statements);
  auto *original_begin = range.begin;
  range.block_dim = 0;
  range.model_block_dim = nullptr;
  range.one_to_one = false;
  auto &group = *range.body;
  if (width == 1) {
    auto *index = group.push_back<LoopIndexStmt>(&range, 0);
    auto *one = group.push_back<ConstStmt>(TypedConstant(1));
    GroupPhases phases(group, range, index, original_begin, one, width);
    phases.emit(std::move(statements));
    return true;
  }
  Block bounds;
  auto *w = bounds.push_back<ConstStmt>(TypedConstant(int64(width)));
  auto *begin = integer_cast(bounds, range.begin, PrimitiveType::i64);
  auto *end = integer_cast(bounds, range.end, PrimitiveType::i64);
  auto *length = bounds.push_back<BinaryOpStmt>(BinaryOpType::sub, end, begin);
  length = bounds.push_back<BinaryOpStmt>(
      BinaryOpType::max, length,
      bounds.push_back<ConstStmt>(TypedConstant(int64(0))));
  auto *padding = bounds.push_back<ConstStmt>(TypedConstant(int64(width - 1)));
  auto *rounded =
      bounds.push_back<BinaryOpStmt>(BinaryOpType::add, length, padding);
  auto *count = bounds.push_back<BinaryOpStmt>(BinaryOpType::div, rounded, w);
  range.begin = bounds.push_back<ConstStmt>(TypedConstant(0));
  range.end = integer_cast(bounds, count, PrimitiveType::i32);
  for (auto &s : bounds.statements)
    range.insert_before_me(std::move(s));
  auto *tile = integer_cast(group, group.push_back<LoopIndexStmt>(&range, 0),
                            PrimitiveType::i64);
  auto *offset = group.push_back<BinaryOpStmt>(BinaryOpType::mul, tile, w);
  auto *base = group.push_back<BinaryOpStmt>(BinaryOpType::add, begin, offset);
  auto *left = group.push_back<BinaryOpStmt>(BinaryOpType::sub, length, offset);
  auto *active = group.push_back<BinaryOpStmt>(BinaryOpType::min, left, w);
  GroupPhases phases(group, range,
                     integer_cast(group, base, PrimitiveType::i32), original_begin,
                     integer_cast(group, active, PrimitiveType::i32), width);
  phases.emit(std::move(statements));
  return true;
}
}  // namespace taichi::lang::LLVM
