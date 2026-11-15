#pragma once

#include <unordered_map>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

namespace taichi::lang {

// A consumer-side proof over the linked library. The producer cannot inspect
// an extracted callee when it records unproven_call. Do not turn that absence
// of a proof into a permanent launch policy, or assume the call is harmless.
// This analysis runs only on a specialization miss and retains no IR objects.
class AotPreparedRangeAnalysis {
 public:
  struct Result {
    bool empty_has_no_effects{false};
    bool independent_iterations{false};
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
        auto *callee = call->getCalledFunction();
        if (range || !callee ||
            callee->getName() != "gpu_parallel_range_for_with_tls" ||
            call->arg_size() != 7 || call->getArgOperand(0) != entry->getArg(0) ||
            !llvm::isa<llvm::ConstantPointerNull>(call->getArgOperand(3)) ||
            !llvm::isa<llvm::ConstantPointerNull>(call->getArgOperand(5)))
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
    return {true, body->getArg(1)->use_empty() && independent(body)};
  }

 private:
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

  bool independent(llvm::Function *function) {
    const auto known = proofs_.find(function);
    if (known != proofs_.end())
      return known->second;
    // A back edge in the call graph stays unproved. Memoize shared callees so
    // a multi-task library is inspected once, without inlining or cloning it.
    proofs_[function] = false;
    if (function->isIntrinsic() || function->isDeclaration())
      return proofs_[function] = scalar_intrinsic(*function);
    for (auto &block : *function) {
      for (auto &instruction : block) {
        // Shared/local cooperation is qualified by a different producer proof.
        // Do not recover it from partially lowered library code here.
        auto shared = [](llvm::Type *type) {
          return type->isPointerTy() && type->getPointerAddressSpace() == 3;
        };
        if (shared(instruction.getType()))
          return false;
        for (const auto &operand : instruction.operands())
          if (uses_shared_pointer(operand.get()))
            return false;
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
          auto *callee = call->getCalledFunction();
          if (!callee || call->isConvergent() || !independent(callee))
            return false;
        }
      }
    }
    return proofs_[function] = true;
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

  std::unordered_map<llvm::Function *, bool> proofs_;
};

}  // namespace taichi::lang
