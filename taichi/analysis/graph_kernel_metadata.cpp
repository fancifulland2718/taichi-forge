#include "taichi/analysis/graph_kernel_metadata.h"

#include <algorithm>
#include <map>
#include <tuple>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace {

using ResourceKey =
    std::tuple<std::string, std::vector<int>, int, int, bool>;

bool constant_integer(Stmt *stmt, std::int64_t *value) {
  auto *constant = stmt ? stmt->cast<ConstStmt>() : nullptr;
  if (constant == nullptr || !is_integral(constant->val.dt)) {
    return false;
  }
  *value = constant->val.val_as_int64();
  return true;
}

bool constant_one(Stmt *stmt) {
  std::int64_t value = 0;
  return constant_integer(stmt, &value) && value == 1;
}

bool external_shape(Stmt *stmt,
                    std::vector<int> *arg_id,
                    int *axis) {
  if (auto *shape = stmt ? stmt->cast<ExternalTensorShapeAlongAxisStmt>()
                         : nullptr) {
    *arg_id = shape->arg_id;
    *axis = shape->axis;
    return true;
  }
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  if (binary == nullptr || binary->op_type != BinaryOpType::mul) {
    return false;
  }
  if (constant_one(binary->lhs)) {
    return external_shape(binary->rhs, arg_id, axis);
  }
  if (constant_one(binary->rhs)) {
    return external_shape(binary->lhs, arg_id, axis);
  }
  return false;
}

GraphKernelIterationDomain iteration_domain(RangeForStmt *loop) {
  GraphKernelIterationDomain result;
  std::int64_t begin = 0;
  if (!constant_integer(loop->begin, &begin)) {
    return result;
  }
  std::int64_t end = 0;
  if (constant_integer(loop->end, &end)) {
    result.kind = "constant_range";
    result.begin = begin;
    result.end = end;
    return result;
  }
  std::vector<int> arg_id;
  int axis = -1;
  if (begin == 0 && external_shape(loop->end, &arg_id, &axis) && axis == 0) {
    result.kind = "external_tensor";
    result.arg_id = std::move(arg_id);
    result.axis = axis;
    return result;
  }
  if (begin == 0) {
    if (auto *arg = loop->end->cast<ArgLoadStmt>()) {
      if (!arg->is_ptr) {
        result.kind = "scalar_argument";
        result.arg_id = arg->arg_id;
      }
    }
  }
  return result;
}

bool same_external_shape(Stmt *stmt,
                         const GraphKernelIterationDomain &domain) {
  std::vector<int> arg_id;
  int axis = -1;
  return domain.kind == "external_tensor" &&
         external_shape(stmt, &arg_id, &axis) && arg_id == domain.arg_id &&
         axis == domain.axis;
}

bool is_loop_index(Stmt *stmt,
                   RangeForStmt *loop,
                   const GraphKernelIterationDomain &domain) {
  auto *index = stmt ? stmt->cast<LoopIndexStmt>() : nullptr;
  if (index != nullptr) {
    return index->loop == loop && index->index == 0;
  }
  auto *binary = stmt ? stmt->cast<BinaryOpStmt>() : nullptr;
  return binary != nullptr && binary->op_type == BinaryOpType::mod &&
         is_loop_index(binary->lhs, loop, domain) &&
         same_external_shape(binary->rhs, domain);
}

Stmt *pointer_origin(Stmt *stmt) {
  while (auto *matrix = stmt ? stmt->cast<MatrixPtrStmt>() : nullptr) {
    stmt = matrix->origin;
  }
  return stmt;
}

std::string merge_access(const std::string &lhs, const std::string &rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs == "opaque" || rhs == "opaque") {
    return "opaque";
  }
  if (lhs == "atomic" || rhs == "atomic") {
    return "atomic";
  }
  return "read_write";
}

class MetadataVisitor final : public BasicStmtVisitor {
 public:
  MetadataVisitor(RangeForStmt *loop,
                  const GraphKernelIterationDomain &domain)
      : loop_(loop), domain_(domain) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(Stmt *stmt) override {
    if (stmt->has_global_side_effect()) {
      block("unsupported_side_effect", "opaque");
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (stmt == loop_ || stmt->is<IfStmt>()) {
      return;
    }
    block("nested_control_flow", "control_flow");
  }

  void visit(GlobalLoadStmt *stmt) override {
    record(stmt->src, "read");
  }

  void visit(GlobalStoreStmt *stmt) override {
    record(stmt->dest, "write");
  }

  void visit(AtomicOpStmt *stmt) override {
    block("atomic_effect", "atomic");
    record(stmt->dest, "atomic");
  }

  bool blocked() const {
    return !blocker_.empty();
  }

  const std::string &blocker() const {
    return blocker_;
  }

  const std::vector<std::string> &side_effects() const {
    return side_effects_;
  }

  std::vector<GraphKernelResourceEffect> effects() const {
    std::vector<GraphKernelResourceEffect> result;
    result.reserve(effects_.size());
    for (const auto &[key, access] : effects_) {
      GraphKernelResourceEffect effect;
      effect.resource_kind = std::get<0>(key);
      effect.arg_id = std::get<1>(key);
      effect.snode_tree_id = std::get<2>(key);
      effect.snode_id = std::get<3>(key);
      effect.is_grad = std::get<4>(key);
      effect.access = access;
      result.push_back(std::move(effect));
    }
    return result;
  }

 private:
  void block(const std::string &reason, const std::string &side_effect) {
    if (blocker_.empty()) {
      blocker_ = reason;
    }
    if (!side_effect.empty() &&
        std::find(side_effects_.begin(), side_effects_.end(), side_effect) ==
            side_effects_.end()) {
      side_effects_.push_back(side_effect);
    }
  }

  void add_effect(const ResourceKey &key, const std::string &access) {
    auto [it, inserted] = effects_.emplace(key, access);
    if (!inserted) {
      it->second = merge_access(it->second, access);
    }
  }

  bool pointwise_indices(const std::vector<Stmt *> &indices,
                         int external_dimensions) const {
    // A physics map commonly iterates particles in the first dimension while
    // accessing a fixed vector/matrix component in the remaining dimensions.
    // This remains pointwise across loop iterations. Dynamic secondary
    // indices are rejected below, so stencil/gather accesses stay opaque.
    if (indices.empty() || external_dimensions < 1 ||
        indices.size() < static_cast<std::size_t>(external_dimensions) ||
        !is_loop_index(indices.front(), loop_, domain_)) {
      return false;
    }
    for (std::size_t index = 1; index < indices.size(); ++index) {
      std::int64_t ignored = 0;
      if (!constant_integer(indices[index], &ignored)) {
        return false;
      }
    }
    return true;
  }

  void record(Stmt *pointer, const std::string &access) {
    pointer = pointer_origin(pointer);
    if (auto *external =
            pointer ? pointer->cast<ExternalPtrStmt>() : nullptr) {
      auto *base = external->base_ptr->cast<ArgLoadStmt>();
      if (base == nullptr ||
          !pointwise_indices(external->indices, external->ndim)) {
        add_effect({"opaque", {}, -1, -1, false}, "opaque");
        block("non_pointwise_access", "opaque_access");
        return;
      }
      add_effect({"argument", base->arg_id, -1, -1, external->is_grad},
                 access);
      return;
    }
    if (auto *global = pointer ? pointer->cast<GlobalPtrStmt>() : nullptr) {
      if (global->activate || global->indices.size() != 1 ||
          !is_loop_index(global->indices.front(), loop_, domain_)) {
        add_effect({"opaque", {}, -1, -1, false}, "opaque");
        block(global->activate ? "sparse_activation" :
                                 "non_pointwise_access",
              "opaque_access");
        return;
      }
      add_effect({"snode", {}, global->snode->get_snode_tree_id(),
                  global->snode->id, false},
                 access);
      return;
    }
    add_effect({"opaque", {}, -1, -1, false}, "opaque");
    block("unsupported_pointer", "opaque_access");
  }

  RangeForStmt *loop_{nullptr};
  const GraphKernelIterationDomain &domain_;
  std::map<ResourceKey, std::string> effects_;
  std::vector<std::string> side_effects_;
  std::string blocker_;
};

}  // namespace

GraphKernelMetadata analyze_graph_kernel_metadata(IRNode *root,
                                                  const Kernel *kernel) {
  GraphKernelMetadata result;
  result.available = true;
  result.blocker.clear();
  if (root == nullptr || kernel == nullptr || !root->is<Block>()) {
    result.blocker = "invalid_ir";
    return result;
  }
  if (kernel->autodiff_mode != AutodiffMode::kNone || !kernel->rets.empty()) {
    result.blocker = "callable_boundary";
    return result;
  }

  RangeForStmt *loop = nullptr;
  for (const auto &statement : root->as<Block>()->statements) {
    if (auto *candidate = statement->cast<RangeForStmt>()) {
      if (loop != nullptr) {
        result.blocker = "multiple_iteration_domains";
        return result;
      }
      loop = candidate;
      continue;
    }
    if (statement->is_container_statement() ||
        statement->has_global_side_effect()) {
      result.blocker = "top_level_side_effect";
      return result;
    }
  }
  if (loop == nullptr) {
    result.blocker = "missing_range_for";
    return result;
  }
  if (loop->strictly_serialized || loop->reversed || loop->is_bit_vectorized) {
    result.blocker = "unsupported_iteration_mode";
    return result;
  }

  result.iteration_domain = iteration_domain(loop);
  if (result.iteration_domain.kind == "unknown") {
    result.blocker = "unknown_iteration_domain";
    return result;
  }

  MetadataVisitor visitor(loop, result.iteration_domain);
  root->accept(&visitor);
  result.effects = visitor.effects();
  result.side_effects = visitor.side_effects();
  if (visitor.blocked()) {
    result.blocker = visitor.blocker();
    result.synchronization = true;
    return result;
  }
  result.opaque = false;
  result.elementwise = true;
  result.synchronization = false;
  result.blocker.clear();
  return result;
}

}  // namespace taichi::lang
