#include "taichi/runtime/llvm/aot_cpu_private_frames.h"

#include <limits>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

namespace taichi::lang::LLVM {

// After helper inlining the same private frame can contain vector stores and
// narrower, overlapping reads. Expose their scalar words to SROA. Uniformly
// accessed vectors and the caller's argument and array storage keep their ABI.
namespace {
bool split_cpu_frame_vectors(llvm::Function &function) {
  const auto &layout = function.getParent()->getDataLayout();
  llvm::DenseMap<llvm::AllocaInst *, llvm::Type *> first_type;
  llvm::SmallPtrSet<llvm::AllocaInst *, 16> mixed;
  llvm::SmallVector<std::pair<llvm::AllocaInst *, llvm::Instruction *>, 16>
      vectors;
  for (auto &block : function) {
    for (auto &inst : block) {
      auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst);
      auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst);
      if (!(load && load->isSimple()) && !(store && store->isSimple()))
        continue;
      auto *ptr = load ? load->getPointerOperand() : store->getPointerOperand();
      int64_t offset = 0;
      auto *frame = llvm::dyn_cast<llvm::AllocaInst>(
          llvm::GetPointerBaseWithConstantOffset(ptr, offset, layout));
      if (!frame || !frame->isStaticAlloca() ||
          !frame->getAllocatedType()->isAggregateType())
        continue;
      auto *type = load ? load->getType() : store->getValueOperand()->getType();
      auto inserted = first_type.try_emplace(frame, type);
      if (!inserted.second && inserted.first->second != type)
        mixed.insert(frame);
      auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(type);
      if (vec && (vec->getElementType()->isIntegerTy(32) ||
                  vec->getElementType()->isIntegerTy(64) ||
                  vec->getElementType()->isFloatTy() ||
                  vec->getElementType()->isDoubleTy()))
        vectors.emplace_back(frame, &inst);
    }
  }
  if (vectors.empty())
    return false;
  // One capture check per allocation, irrespective of the number of lanes.
  llvm::SmallPtrSet<llvm::AllocaInst *, 16> local;
  for (auto *frame : mixed) {
    if (!llvm::PointerMayBeCaptured(frame, true, true,
                                    std::numeric_limits<unsigned>::max()))
      local.insert(frame);
  }
  bool changed = false;
  for (auto [frame, inst] : vectors) {
    if (!local.contains(frame))
      continue;
    auto *load = llvm::dyn_cast<llvm::LoadInst>(inst);
    auto *store = llvm::dyn_cast<llvm::StoreInst>(inst);
    auto *ptr = load ? load->getPointerOperand() : store->getPointerOperand();
    auto *type = llvm::cast<llvm::FixedVectorType>(
        load ? load->getType() : store->getValueOperand()->getType());
    auto *word = type->getElementType();
    auto alignment = load ? load->getAlign() : store->getAlign();
    const auto bytes = layout.getTypeStoreSize(word).getFixedValue();
    llvm::IRBuilder<> builder(inst);
    llvm::Value *value = llvm::PoisonValue::get(type);
    for (unsigned i = 0; i < type->getNumElements(); ++i) {
      auto *address = builder.CreateGEP(word, ptr, builder.getInt32(i));
      auto align = llvm::commonAlignment(alignment, i * bytes);
      if (load)
        value = builder.CreateInsertElement(
            value, builder.CreateAlignedLoad(word, address, align), i);
      else
        builder.CreateAlignedStore(
            builder.CreateExtractElement(store->getValueOperand(), i), address,
            align);
    }
    if (load)
      load->replaceAllUsesWith(value);
    inst->eraseFromParent();
    changed = true;
  }
  return changed;
}

}  // namespace

void promote_cpu_private_frames(llvm::Module &module) {
  llvm::SmallVector<llvm::Function *, 4> changed;
  for (auto &fn : module) {
    if (!fn.isDeclaration() && !fn.hasOptNone() &&
        llvm::any_of(
            fn.getEntryBlock(),
            [](auto &inst) { return llvm::isa<llvm::AllocaInst>(inst); }) &&
        split_cpu_frame_vectors(fn))
      changed.push_back(&fn);
  }
  if (changed.empty())
    return;
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager cgscc;
  llvm::ModuleAnalysisManager modules;
  llvm::PassBuilder passes;
  passes.registerModuleAnalyses(modules);
  passes.registerCGSCCAnalyses(cgscc);
  passes.registerFunctionAnalyses(functions);
  passes.registerLoopAnalyses(loops);
  passes.crossRegisterProxies(loops, functions, cgscc, modules);
  llvm::FunctionPassManager cleanup;
  cleanup.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  cleanup.addPass(llvm::EarlyCSEPass(true));
  cleanup.addPass(llvm::InstCombinePass());
  cleanup.addPass(llvm::SimplifyCFGPass());
  for (auto *fn : changed)
    cleanup.run(*fn, functions);
}

}  // namespace taichi::lang::LLVM
