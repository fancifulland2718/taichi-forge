#pragma once

#include <map>
#include <set>

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/runtime/llvm/aot_prepared_range.h"

namespace taichi::lang::LLVM {

namespace prepared_local_tensor {

// Budget for the *whole callback*. A large model retains its original backing
// store instead of scaling the per-thread device frame with model capacity.
inline constexpr unsigned kPrivateBytes = 1024;

struct Workspace {
  unsigned bytes;
  unsigned rows;
  llvm::SmallVector<llvm::LoadInst *, 1> pointers;
};
using Workspaces = std::map<int, Workspace>;

// Read every way the entry context reaches its ABI buffer. A nonconstant
// address, forwarding the buffer to a helper, or reading a part of a pointer
// can hide a second use, so an incomplete inventory disqualifies the rewrite.
inline bool collect_buffer_uses(llvm::Value *buffer, Workspaces &spaces) {
  const auto &layout =
      llvm::cast<llvm::Instruction>(buffer)->getModule()->getDataLayout();
  llvm::SmallVector<llvm::Value *> pending{buffer};
  while (!pending.empty()) {
    auto *address = pending.pop_back_val();
    for (auto *user : address->users()) {
      if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {
        if (!gep->hasAllConstantIndices())
          return false;
        pending.push_back(gep);
        continue;
      }
      auto *load = llvm::dyn_cast<llvm::LoadInst>(user);
      int64_t offset = 0;
      if (!load || !load->isSimple() ||
          llvm::GetPointerBaseWithConstantOffset(address, offset, layout) !=
              buffer ||
          offset < 0 || offset > INT_MAX)
        return false;
      const auto size = layout.getTypeStoreSize(load->getType());
      if (size.isScalable() || size.getFixedValue() > INT_MAX)
        return false;
      // Include a preceding slot if the load begins inside its pointer.
      auto it = spaces.lower_bound(static_cast<int>(offset) -
                                   static_cast<int>(layout.getPointerSize()) + 1);
      for (; it != spaces.end() && it->first < offset + int64_t(size.getFixedValue()); ++it) {
        if (it->first != offset || !load->getType()->isPointerTy() ||
            load->getPointerAddressSpace() != 0 ||
            load->getType()->getPointerAddressSpace() != 0)
          return false;
        it->second.pointers.push_back(load);
      }
    }
  }
  return true;
}

// Collect all argument-buffer reads, including serial tasks and ordinary
// wrappers. Only the canonical CUDA range body ABI supplies an iteration ID.
using Bodies = std::map<llvm::Function *, uint64_t>;
inline bool collect_task_uses(llvm::Function &entry, Workspaces &spaces,
                              Bodies &bodies) {
  if (entry.arg_size() != 1 || !entry.use_empty())
    return false;
  auto collect_context_load = [&](llvm::User *user) {
    auto *load = llvm::dyn_cast<llvm::LoadInst>(user);
    return load && load->isSimple() && load->getType()->isPointerTy() &&
           collect_buffer_uses(load, spaces);
  };
  for (auto &use : entry.getArg(0)->uses()) {
    auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
    if (!call) {
      if (!collect_context_load(use.getUser()))
        return false;
      continue;
    }
    if (entry.size() != 1 || !is_prepared_cuda_range_dispatch(*call, entry) ||
        !call->isArgOperand(&use) || call->getArgOperandNo(&use) != 0)
      return false;
    auto *body = llvm::dyn_cast<llvm::Function>(call->getArgOperand(4));
    if (!body || !body->hasLocalLinkage() || !body->hasOneUse() ||
        body->isDeclaration() || body->arg_size() != 3 ||
        !body->getArg(2)->getType()->isIntegerTy(32))
      return false;
    // A fixed dispatch entirely inside the captured outer extent establishes
    // a unique valid row for every logical callback, including grid-striding.
    auto *begin = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
    auto *end = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
    const auto extent = begin && end && !begin->isNegative() && !end->isNegative()
                            ? end->getZExtValue()
                            : UINT64_MAX;
    if (!bodies.emplace(body, extent).second)
      return false;
    for (auto *user : body->getArg(0)->users())
      if (!collect_context_load(user))
        return false;
  }
  return true;
}

// The producer emits byte strides and model extent expressions before LLVM
// sees the prepared constants. Fold only their pure integer operations.
inline bool integer_expression(const llvm::Instruction &inst) {
  return inst.getType()->isIntegerTy() &&
         llvm::isa<llvm::BinaryOperator, llvm::CastInst, llvm::ICmpInst,
                   llvm::SelectInst>(inst);
}

inline void fold_integer_constants(llvm::Function &body) {
  for (auto &block : body)
    for (auto it = block.begin(); it != block.end();) {
      auto &inst = *it++;
      if (!integer_expression(inst))
        continue;
      if (auto *value = llvm::ConstantFoldInstruction(
              &inst, body.getParent()->getDataLayout())) {
        inst.replaceAllUsesWith(value);
        inst.eraseFromParent();
      }
    }
}

inline bool iteration_term(llvm::Value *value, llvm::Argument *iteration,
                           uint64_t stride) {
  using namespace llvm::PatternMatch;
  llvm::Value *index = value;
  if (stride != 1) {
    uint64_t scale;
    if (match(value, m_c_Mul(m_Value(index), m_ConstantInt(scale)))) {
      if (scale != stride)
        return false;
    } else if (!match(value, m_Shl(m_Value(index), m_ConstantInt(scale))) ||
               scale >= 64 || (uint64_t(1) << scale) != stride) {
      return false;
    }
  }
  return match(index, m_SExt(m_Specific(iteration)));
}

struct Index {
  llvm::Value *value;
  uint64_t scale;
};

// Prove small dynamic offsets exactly. A canonical zero-based, unit-step
// loop has a finite set of indices on its true edge. Evaluating only the
// address expression retains correlations such as quotient and remainder
// when a matrix constructor flattens its zero-fill loop. Other recurrences,
// unbounded live indices and excessive analysis work keep generic storage.
class RowBounds {
 public:
  RowBounds(llvm::DominatorTree &dominance, llvm::Instruction &access)
      : dominance_(dominance), access_(access) {}

  bool contains(llvm::ArrayRef<Index> offsets, unsigned limit,
                unsigned align) {
    for (const auto &offset : offsets)
      if (offset.value && !collect(offset.value))
        return false;
    llvm::DenseMap<llvm::Value *, llvm::Constant *> values;
    const auto &layout = access_.getModule()->getDataLayout();
    for (unsigned i = 0; i < iterations_; ++i) {
      if (phi_)
        values[phi_] = llvm::ConstantInt::get(phi_->getType(), i);
      for (auto *inst : operations_) {
        llvm::SmallVector<llvm::Constant *, 3> operands;
        for (auto &operand : inst->operands())
          operands.push_back(constant(operand.get(), values));
        auto *result = llvm::ConstantFoldInstOperands(inst, operands, layout,
                                                     nullptr, false);
        if (!llvm::isa_and_nonnull<llvm::ConstantInt>(result))
          return false;
        values[inst] = result;
      }
      uint64_t bytes = 0;
      for (const auto &offset : offsets) {
        if (!offset.value)
          continue;
        auto index = llvm::cast<llvm::ConstantInt>(constant(offset.value, values))
                         ->getZExtValue();
        if (index > (limit - bytes) / offset.scale)
          return false;
        bytes += index * offset.scale;
      }
      if (bytes % align)
        return false;
    }
    return true;
  }

 private:
  static llvm::Constant *constant(
      llvm::Value *value,
      const llvm::DenseMap<llvm::Value *, llvm::Constant *> &values) {
    if (auto *literal = llvm::dyn_cast<llvm::ConstantInt>(value))
      return literal;
    return values.lookup(value);
  }

  bool collect(llvm::Value *value) {
    if (llvm::isa<llvm::ConstantInt>(value) || !visited_.insert(value).second)
      return true;
    auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
    if (!inst || visited_.size() > 48)
      return false;
    if (auto *phi = llvm::dyn_cast<llvm::PHINode>(inst))
      return !phi_ && bind_induction(*phi);
    if (!integer_expression(*inst))
      return false;
    for (auto &operand : inst->operands())
      if (!collect(operand.get()))
        return false;
    operations_.push_back(inst);
    return true;
  }

  bool bind_induction(llvm::PHINode &phi) {
    using namespace llvm::PatternMatch;
    if (!phi.getType()->isIntegerTy(32) || phi.getNumIncomingValues() != 2)
      return false;
    const int step = match(phi.getIncomingValue(0), m_Zero()) ? 1 : 0;
    if (!match(phi.getIncomingValue(1 - step), m_Zero()) ||
        !match(phi.getIncomingValue(step),
               m_c_Add(m_Specific(&phi), m_One())))
      return false;
    auto *branch = llvm::dyn_cast<llvm::BranchInst>(phi.getParent()->getTerminator());
    llvm::CmpPredicate predicate;
    uint64_t bound;
    if (!branch || !branch->isConditional() ||
        !match(branch->getCondition(),
               m_ICmp(predicate, m_Specific(&phi), m_ConstantInt(bound))) ||
        predicate != llvm::CmpInst::ICMP_SLT || !bound || bound > 256)
      return false;
    llvm::BasicBlockEdge edge(phi.getParent(), branch->getSuccessor(0));
    if (!dominance_.dominates(edge, phi.getIncomingBlock(step)) ||
        !dominance_.dominates(edge, access_.getParent()))
      return false;
    phi_ = &phi;
    iterations_ = bound;
    return true;
  }

  llvm::DominatorTree &dominance_;
  llvm::Instruction &access_;
  llvm::PHINode *phi_{nullptr};
  unsigned iterations_{1};
  llvm::SmallPtrSet<llvm::Value *, 16> visited_;
  llvm::SmallVector<llvm::Instruction *, 16> operations_;
};

inline std::optional<Index> gep_index(llvm::GetElementPtrInst &gep) {
  const auto &layout = gep.getModule()->getDataLayout();
  if (gep.getNumIndices() != 1 || !gep.getOperand(1)->getType()->isIntegerTy(64))
    return {};
  auto size = layout.getTypeAllocSize(gep.getSourceElementType());
  if (size.isScalable() || !size.getFixedValue())
    return {};
  return Index{gep.getOperand(1), size.getFixedValue()};
}

inline bool check_row_uses(llvm::Value &address, unsigned bytes,
                           llvm::SmallVectorImpl<Index> &offsets,
                           llvm::DominatorTree &dominance, unsigned &alignment) {
  for (auto &use : address.uses()) {
    auto *user = llvm::cast<llvm::Instruction>(use.getUser());
    if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {
      auto index = gep_index(*gep);
      if (!index || use.getOperandNo() != 0 || offsets.size() >= 8)
        return false;
      offsets.push_back(*index);
      if (!check_row_uses(*gep, bytes, offsets, dominance, alignment))
        return false;
      offsets.pop_back();
      continue;
    }
    llvm::Type *type;
    unsigned align;
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(user);
        load && load->isSimple()) {
      type = load->getType();
      align = load->getAlign().value();
    } else if (auto *store = llvm::dyn_cast<llvm::StoreInst>(user);
               store && store->isSimple() &&
               use.getOperandNo() == store->getPointerOperandIndex()) {
      type = store->getValueOperand()->getType();
      align = store->getAlign().value();
    } else {
      return false;
    }
    if ((!type->isIntegerTy() && !type->isFloatingPointTy()) || align > 16)
      return false;
    auto size = user->getModule()->getDataLayout().getTypeStoreSize(type);
    if (size > bytes ||
        !RowBounds(dominance, *user).contains(offsets, bytes - size.getFixedValue(), align))
      return false;
    alignment = std::max(alignment, align);
  }
  return true;
}

inline unsigned privatize(Workspace &space, llvm::Function &body,
                          llvm::DominatorTree &dominance, unsigned remaining) {
  unsigned alignment = 1;
  llvm::SmallVector<std::pair<llvm::GetElementPtrInst *, llvm::Value *>> rows;
  for (auto *pointer : space.pointers) {
    if (pointer->getFunction() != &body)
      return 0;
    for (auto *user : pointer->users()) {
      auto *row = llvm::dyn_cast<llvm::GetElementPtrInst>(user);
      if (!row || row->getPointerOperand() != pointer)
        return 0;
      auto index = gep_index(*row);
      if (!index || space.bytes % index->scale)
        return 0;
      const auto stride = space.bytes / index->scale;
      llvm::Value *inner = nullptr;
      if (!iteration_term(index->value, body.getArg(2), stride)) {
        auto *sum = llvm::dyn_cast<llvm::BinaryOperator>(index->value);
        if (!sum || sum->getOpcode() != llvm::Instruction::Add)
          return 0;
        if (iteration_term(sum->getOperand(0), body.getArg(2), stride))
          inner = sum->getOperand(1);
        else if (iteration_term(sum->getOperand(1), body.getArg(2), stride))
          inner = sum->getOperand(0);
        else
          return 0;
      }
      llvm::SmallVector<Index, 4> offsets{{inner, index->scale}};
      if (!check_row_uses(*row, space.bytes, offsets, dominance, alignment))
        return 0;
      rows.emplace_back(row, inner);
    }
  }
  unsigned frame_bytes = llvm::alignTo(space.bytes, alignment);
  if (rows.empty() || frame_bytes > remaining)
    return 0;
  llvm::IRBuilder<> builder(&*body.getEntryBlock().getFirstInsertionPt());
  auto *storage = builder.CreateAlloca(
      llvm::ArrayType::get(builder.getInt8Ty(), space.bytes), nullptr, "local_tensor");
  storage->setAlignment(llvm::Align(alignment));
  for (const auto &[row, inner] : rows) {
    builder.SetInsertPoint(row);
    llvm::Value *local = storage;
    if (inner)
      local = builder.CreateGEP(row->getSourceElementType(), storage, inner);
    row->replaceAllUsesWith(local);
    row->eraseFromParent();
  }
  return frame_bytes;
}

}  // namespace prepared_local_tensor

inline void specialize_prepared_local_tensors(
    llvm::Module &module, const std::vector<OffloadedTask> &tasks,
    prepared_local_tensor::Workspaces &spaces) {
  using namespace prepared_local_tensor;
  if (spaces.empty())
    return;
  Bodies bodies;
  for (const auto &task : tasks) {
    auto *entry = module.getFunction(task.name);
    if (!entry || entry->hasFnAttribute(llvm::Attribute::OptimizeNone) ||
        !collect_task_uses(*entry, spaces, bodies))
      return;
  }
  for (auto [body, extent] : bodies) {
    if (body->hasFnAttribute(llvm::Attribute::OptimizeNone))
      continue;
    std::optional<llvm::DominatorTree> dominance;
    unsigned remaining = kPrivateBytes;
    for (auto &[slot, space] : spaces) {
      if (extent > space.rows || space.bytes > remaining || space.pointers.empty() ||
          space.pointers.front()->getFunction() != body)
        continue;
      if (!dominance) {
        fold_integer_constants(*body);
        dominance.emplace(*body);
      }
      remaining -= privatize(space, *body, *dominance, remaining);
    }
  }
}

}  // namespace taichi::lang::LLVM
