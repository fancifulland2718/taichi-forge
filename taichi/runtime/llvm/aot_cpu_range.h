#pragma once

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "taichi/codegen/cpu/range_partition.h"
#include "taichi/codegen/range_for_bounds.h"
#include "taichi/common/core.h"

namespace taichi::lang::LLVM {

inline constexpr char kCpuChunkCount[] = "taichi.aot.cpu_chunk_count.v1";

// CPU export optimizes and inlines the scheduler before the binding exists.
// An external constant keeps the single chunk-count operand relocatable
// through that pipeline, including its uses in the scheduler's callback data.
inline llvm::Value *emit_cpu_chunk_count(llvm::IRBuilder<> &builder,
                                         llvm::Function &entry,
                                         int threads) {
  auto *count = new llvm::GlobalVariable(
      *entry.getParent(), builder.getInt32Ty(), true,
      llvm::GlobalValue::ExternalLinkage, nullptr, entry.getName() + ".chunks");
  entry.setMetadata(
      kCpuChunkCount,
      llvm::MDNode::get(entry.getContext(), {llvm::ValueAsMetadata::get(count),
                                             llvm::ConstantAsMetadata::get(
                                                 builder.getInt32(threads))}));
  return builder.CreateLoad(builder.getInt32Ty(), count);
}

// A serialized module is self-contained for ordinary, unspecialized use.
// The entry references the constant by LLVM identity, surviving task renames.
inline void seal_cpu_chunk_counts(llvm::Module &module) {
  for (auto &entry : module) {
    if (auto *node = entry.getMetadata(kCpuChunkCount)) {
      auto *count = llvm::mdconst::dyn_extract_or_null<llvm::GlobalVariable>(
          node->getOperand(0));
      if (!count) {
        // No scheduler operand survived producer optimization.
        entry.setMetadata(kCpuChunkCount, nullptr);
        continue;
      }
      count->setInitializer(
          llvm::mdconst::extract<llvm::ConstantInt>(node->getOperand(1)));
      count->setLinkage(llvm::GlobalValue::PrivateLinkage);
    }
  }
}

// An optional archive optimization runs after CPU codegen has sealed these
// defaults. Restore the declaration used during codegen so the optimizer
// cannot fold the load whose value prepare will specialize.
inline void unseal_cpu_chunk_counts(llvm::Module &module) {
  for (auto &entry : module) {
    if (auto *node = entry.getMetadata(kCpuChunkCount)) {
      auto *count = llvm::mdconst::extract<llvm::GlobalVariable>(
          node->getOperand(0));
      count->setLinkage(llvm::GlobalValue::ExternalLinkage);
      count->setInitializer(nullptr);
    }
  }
}

// Change only this preparation's copy. Both the ThreadPool submission and
// its callback see the same end even if the scheduler was already inlined.
inline void specialize_cpu_chunk_count(llvm::Function &entry,
                                       const RangeForBounds &range,
                                       int begin,
                                       int end) {
  auto *node = entry.getMetadata(kCpuChunkCount);
  if (!node)
    return;
  TI_ERROR_IF(node->getNumOperands() != 2, "Malformed CPU chunk metadata");
  auto *count = llvm::mdconst::dyn_extract_or_null<llvm::GlobalVariable>(
      node->getOperand(0));
  auto *generic = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(
      node->getOperand(1));
  TI_ERROR_IF(
      !count || !count->hasPrivateLinkage() || !count->isConstant() ||
          !count->hasInitializer() || !generic ||
          count->getInitializer() != generic ||
          !generic->getType()->isIntegerTy(32) ||
          range.begin.kind != RangeBoundKind::constant_i32 ||
          range.begin.value != 0 ||
          range.end.kind != RangeBoundKind::constant_i32 ||
          range.end.value <= 0 || range.end.value != generic->getSExtValue(),
      "Invalid prepared CPU chunk partition for {}", entry.getName().str());
  count->setInitializer(llvm::ConstantInt::get(
      generic->getType(), cpu_range_chunk_count(begin, end, range.end.value)));
}

}  // namespace taichi::lang::LLVM
