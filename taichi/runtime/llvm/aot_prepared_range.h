#pragma once

#include <unordered_map>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"

namespace taichi::lang {

// A model-dependent launch is archived with an upper block bound. Once every
// input to that bound is a prepared constant, backend register allocation must
// see the selected block too. Keep the original task binding/validation and
// all other entry attributes (including explicit register/occupancy policy).
inline bool specialize_prepared_cuda_block_bound(llvm::Function &entry,
                                                 int upper, int selected) {
  if (selected <= 0 || selected > upper)
    return false;
  auto *annotations = entry.getParent()->getNamedMetadata("nvvm.annotations");
  if (!annotations)
    return false;
  for (auto *node : annotations->operands()) {
    if (node->getNumOperands() != 3)
      continue;
    auto *function = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(node->getOperand(0));
    auto *name = llvm::dyn_cast_or_null<llvm::MDString>(node->getOperand(1));
    auto *value = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(node->getOperand(2));
    if (function && function->getValue() == &entry && name &&
        name->getString() == "maxntidx" && value && value->getSExtValue() == upper) {
      node->replaceOperandWith(2, llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(value->getType(), selected)));
      return true;
    }
  }
  return false;
}

inline bool is_prepared_cuda_range_dispatch(const llvm::CallBase &call,
                                            const llvm::Function &entry) {
  const auto *callee = call.getCalledFunction();
  return callee && callee->getName() == "gpu_parallel_range_for_with_tls" &&
         entry.arg_size() == 1 && call.arg_size() == 7 &&
         call.getArgOperand(0) == entry.getArg(0) &&
         call.getArgOperand(1)->getType()->isIntegerTy(32) &&
         call.getArgOperand(2)->getType()->isIntegerTy(32) &&
         llvm::isa<llvm::ConstantPointerNull>(call.getArgOperand(3)) &&
         llvm::isa<llvm::ConstantPointerNull>(call.getArgOperand(5));
}

// A consumer-side proof over the linked library. The producer cannot inspect
// an extracted callee when it records unproven_call. Do not turn that absence
// of a proof into a permanent launch policy, or assume the call is harmless.
// This analysis runs only on a specialization miss and retains no IR objects.
class AotPreparedRangeAnalysis {
 public:
  struct Result {
    bool empty_has_no_effects{false};
    bool independent_iterations{false};
    bool cuda_block_collective{false};
  };

  Result inspect(llvm::Function *entry) {
    // Recognize the existing range dispatch ABI, not generated SSA names or
    // parameter-load patterns. Auxiliary callbacks and effects outside the
    // range require their original launch policy.
    if (!entry || entry->arg_size() != 1 || entry->size() != 1)
      return {};
    llvm::CallBase *range = nullptr;
    for (auto &instruction : entry->front()) {
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
        if (range || !is_prepared_cuda_range_dispatch(*call, *entry))
          return {};
        range = call;
      } else if (instruction.mayHaveSideEffects() ||
                 (instruction.isTerminator() &&
                  !llvm::isa<llvm::ReturnInst>(instruction))) {
        return {};
      }
    }
    if (!range)
      return {};
    auto *body = llvm::dyn_cast<llvm::Function>(range->getArgOperand(4));
    if (!body || body->isDeclaration() || body->arg_size() != 3)
      return {};
    // Even a body containing collectives or unknown calls has no effects when
    // this canonical range is empty. Nonempty ranges additionally need a proof
    // that their body cannot observe the removed physical groups.
    if (!body->getArg(1)->use_empty())
      return {true, false, false};
    const auto proof = inspect_body(body);
    return {true, proof == BodyProof::independent,
            proof == BodyProof::block_collective};
  }

 private:
  enum class BodyProof { unproven, independent, block_collective };

  static bool local_collective(const llvm::Function &function) {
    switch (function.getIntrinsicID()) {
#define PER_CUDA_LOCAL_COLLECTIVE(op, intrinsic) case llvm::Intrinsic::intrinsic:
#include "taichi/inc/cuda_local_collectives.inc.h"
#undef PER_CUDA_LOCAL_COLLECTIVE
        return true;
      default:
        return false;
    }
  }

  static bool scalar_intrinsic(const llvm::Function &function) {
    switch (function.getIntrinsicID()) {
      case llvm::Intrinsic::abs:
      case llvm::Intrinsic::smin:
      case llvm::Intrinsic::smax:
      case llvm::Intrinsic::umin:
      case llvm::Intrinsic::umax:
      case llvm::Intrinsic::fabs:
      case llvm::Intrinsic::minnum:
      case llvm::Intrinsic::maxnum:
      case llvm::Intrinsic::minimum:
      case llvm::Intrinsic::maximum:
      case llvm::Intrinsic::sqrt:
      case llvm::Intrinsic::fma:
      case llvm::Intrinsic::fmuladd:
      case llvm::Intrinsic::sin:
      case llvm::Intrinsic::cos:
      case llvm::Intrinsic::exp:
      case llvm::Intrinsic::exp2:
      case llvm::Intrinsic::log:
      case llvm::Intrinsic::log2:
      case llvm::Intrinsic::log10:
      case llvm::Intrinsic::pow:
      case llvm::Intrinsic::powi:
      case llvm::Intrinsic::floor:
      case llvm::Intrinsic::ceil:
      case llvm::Intrinsic::trunc:
      case llvm::Intrinsic::round:
      case llvm::Intrinsic::roundeven:
      case llvm::Intrinsic::ctpop:
      case llvm::Intrinsic::ctlz:
      case llvm::Intrinsic::cttz:
      case llvm::Intrinsic::bswap:
      case llvm::Intrinsic::bitreverse:
      case llvm::Intrinsic::lifetime_start:
      case llvm::Intrinsic::lifetime_end:
      case llvm::Intrinsic::memcpy:
      case llvm::Intrinsic::memmove:
      case llvm::Intrinsic::memset:
        return true;
      default:
        break;
    }
    // Do not admit NVVM intrinsics by memory attributes: register queries are
    // readnone too. These libdevice leaves depend only on scalar inputs and
    // fixed compiler options. All other target intrinsics remain unproved.
    const auto name = function.getName();
    return name == "llvm.nvvm.sqrt.f" || name == "llvm.nvvm.sqrt.rn.d" ||
           name == "llvm.nvvm.fabs.f" || name == "llvm.nvvm.fabs.ftz.f" ||
           name == "__nvvm_reflect";
  }

  BodyProof inspect_body(llvm::Function *function) {
    const auto known = proofs_.find(function);
    if (known != proofs_.end())
      return known->second;
    // A back edge in the call graph stays unproved. Memoize shared callees so
    // a multi-task library is inspected once, without inlining or cloning it.
    proofs_[function] = BodyProof::unproven;
    if (function->isIntrinsic() || function->isDeclaration()) {
      return proofs_[function] = local_collective(*function)
          ? BodyProof::block_collective
          : scalar_intrinsic(*function) ? BodyProof::independent
                                        : BodyProof::unproven;
    }
    auto proof = BodyProof::independent;
    for (auto &block : *function) {
      for (auto &instruction : block) {
        // Shared storage requires the existing whole-block qualification,
        // never independent-iteration/device-range dispatch permission.
        auto shared = [](llvm::Type *type) {
          return type->isPointerTy() && type->getPointerAddressSpace() == 3;
        };
        if (shared(instruction.getType()))
          proof = BodyProof::block_collective;
        for (const auto &operand : instruction.operands())
          if (uses_shared_pointer(operand.get()))
            proof = BodyProof::block_collective;
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
          auto *callee = call->getCalledFunction();
          if (!callee)
            return BodyProof::unproven;
          const auto called = inspect_body(callee);
          if (called == BodyProof::unproven ||
              (call->isConvergent() && called != BodyProof::block_collective))
            return BodyProof::unproven;
          if (called == BodyProof::block_collective)
            proof = called;
        }
      }
    }
    return proofs_[function] = proof;
  }

  static bool uses_shared_pointer(const llvm::Value *value) {
    if (value->getType()->isPointerTy() &&
        value->getType()->getPointerAddressSpace() == 3)
      return true;
    if (const auto *expression = llvm::dyn_cast<llvm::ConstantExpr>(value))
      for (const auto &operand : expression->operands())
        if (uses_shared_pointer(operand.get()))
          return true;
    return false;
  }

  std::unordered_map<llvm::Function *, BodyProof> proofs_;
};

}  // namespace taichi::lang
