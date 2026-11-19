#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/runtime/llvm/aot_prepared_range.h"

namespace taichi::lang::LLVM {

// Only ordinary, finite task wrappers are transparent. A volatile read,
// auxiliary callback or observable operation outside the loop keeps the task.
inline bool prepared_scratch_address(const llvm::Instruction &instruction) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
  return call && call->getCalledFunction() &&
         call->getCalledFunction()->getName() ==
             "RuntimeContext_get_temporary_pointer";
}

inline bool plain_schedule_instruction(const llvm::Instruction &instruction) {
  if (instruction.isTerminator())
    return llvm::isa<llvm::BranchInst, llvm::ReturnInst>(instruction);
  if (llvm::isa<llvm::CallBase>(instruction))
    return prepared_scratch_address(instruction);
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&instruction))
    return load->isSimple();
  return !instruction.mayHaveSideEffects();
}

inline llvm::CallInst *prepared_range_dispatch(llvm::Function &entry) {
  llvm::CallInst *dispatch = nullptr;
  for (auto &block : entry) {
    for (auto &inst : block) {
      if (plain_schedule_instruction(inst))
        continue;
      auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
      if (dispatch || !call || !is_prepared_cuda_range_dispatch(*call, entry))
        return nullptr;
      dispatch = call;
    }
  }
  for (auto it = llvm::scc_begin(&entry); it != llvm::scc_end(&entry); ++it)
    if (it.hasCycle())
      return nullptr;
  return dispatch;
}

inline bool prepared_constant_stores(llvm::Function &entry,
                                     const std::set<int> &slots,
                                     std::map<int, int> &writes) {
  llvm::DominatorTree dominance(entry);
  llvm::SmallVector<llvm::StoreInst *> stores;
  for (auto it = llvm::scc_begin(&entry); it != llvm::scc_end(&entry); ++it)
    if (it.hasCycle())
      return false;
  for (auto &block : entry) {
    for (auto &inst : block) {
      if (plain_schedule_instruction(inst))
        continue;
      auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst);
      auto *tag = inst.getMetadata("taichi.scratch.store");
      if (!store || !store->isSimple() || !tag || tag->getNumOperands() != 1)
        return false;
      auto *slot =
          llvm::mdconst::dyn_extract<llvm::ConstantInt>(tag->getOperand(0));
      auto *value = llvm::dyn_cast<llvm::ConstantInt>(store->getValueOperand());
      if (!slot || !slot->getType()->isIntegerTy(32) ||
          !slots.count(slot->getSExtValue()) || !value ||
          !value->getType()->isIntegerTy(32))
        return false;
      auto [written, inserted] =
          writes.emplace(slot->getSExtValue(), value->getSExtValue());
      if (!inserted && written->second != value->getSExtValue())
        return false;
      stores.push_back(store);
    }
  }
  for (auto &block : entry)
    if (llvm::isa<llvm::ReturnInst>(block.getTerminator()))
      for (auto *store : stores)
        if (!dominance.dominates(store, block.getTerminator()))
          return false;
  return true;
}

inline void erase_prepared_tasks(llvm::Module &module,
                                 std::vector<OffloadedTask> &tasks,
                                 const std::set<llvm::Function *> &dead) {
  if (dead.empty())
    return;
  if (auto *annotations = module.getNamedMetadata("nvvm.annotations")) {
    llvm::SmallVector<llvm::MDNode *> keep;
    for (auto *node : annotations->operands()) {
      auto *value = node->getNumOperands()
                        ? llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(
                              node->getOperand(0).get())
                        : nullptr;
      if (!value ||
          !dead.count(llvm::dyn_cast<llvm::Function>(value->getValue())))
        keep.push_back(node);
    }
    annotations->clearOperands();
    for (auto *node : keep)
      annotations->addOperand(node);
  }
  tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
                             [&](const OffloadedTask &task) {
                               return dead.count(module.getFunction(task.name));
                             }),
              tasks.end());
  for (auto *entry : dead)
    entry->eraseFromParent();
}

// On a private specialization miss, propagate immutable i32 writes through
// the compiler's per-invocation scratch. Ordinary resources never enter this
// map. A missing inventory, an opaque use or a live read keeps its producer.
inline void specialize_prepared_task_schedule(
    llvm::Module &module,
    std::vector<OffloadedTask> &tasks) {
  using Kind = RangeBoundKind;
  if (tasks.size() < 2 ||
      std::none_of(tasks.begin(), tasks.end(), [](const auto &task) {
        const auto &r = task.range_bounds;
        return r && (r->begin.kind == Kind::global_tmp_i32 ||
                     r->end.kind == Kind::global_tmp_i32);
      }))
    return;
  struct TaskFacts {
    llvm::Function *entry;
    std::set<int> slots;
    std::map<int, int> writes;
    bool pure_writer{false};
  };
  std::vector<TaskFacts> facts;
  std::set<llvm::Function *> unique;
  for (const auto &task : tasks) {
    auto *entry = module.getFunction(task.name);
    auto *inventory =
        entry ? entry->getMetadata("taichi.scratch.slots") : nullptr;
    if (!inventory || !entry->use_empty() ||
        entry->hasFnAttribute(llvm::Attribute::OptimizeNone) ||
        !unique.insert(entry).second)
      return;
    facts.push_back({entry});
    for (const auto &operand : inventory->operands()) {
      auto *offset = llvm::mdconst::dyn_extract<llvm::ConstantInt>(operand);
      if (!offset || !offset->getType()->isIntegerTy(32))
        return;
      facts.back().slots.insert(offset->getSExtValue());
    }
  }

  llvm::PassBuilder builder;
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager analyses;
  llvm::CGSCCAnalysisManager calls;
  llvm::ModuleAnalysisManager modules;
  builder.registerLoopAnalyses(loops);
  builder.registerFunctionAnalyses(analyses);
  builder.registerCGSCCAnalyses(calls);
  builder.registerModuleAnalyses(modules);
  builder.crossRegisterProxies(loops, analyses, calls, modules);
  llvm::FunctionPassManager fold;
  fold.addPass(llvm::SCCPPass());
  fold.addPass(llvm::SimplifyCFGPass());
  fold.addPass(llvm::ADCEPass());
  std::set<llvm::Function *> dead;
  std::map<int, int> values;
  for (size_t i = 0; i < tasks.size(); ++i) {
    auto &task = tasks[i];
    auto &[entry, slots, writes, pure_writer] = facts[i];
    if (task.task_type == OffloadedTaskType::serial && !slots.empty()) {
      fold.run(*entry, analyses);
      pure_writer = prepared_constant_stores(*entry, slots, writes);
    } else if (auto &bounds = task.range_bounds) {
      auto *dispatch = prepared_range_dispatch(*entry);
      bool forwarded = false;
      if (dispatch) {
        auto forward = [&](RangeBound &bound, unsigned argument) {
          if (bound.kind != Kind::global_tmp_i32)
            return;
          auto known = values.find(bound.value);
          if (known == values.end())
            return;
          dispatch->setArgOperand(
              argument,
              llvm::ConstantInt::get(
                  dispatch->getArgOperand(argument)->getType(), known->second));
          bound = {Kind::constant_i32, known->second};
          forwarded = true;
        };
        forward(bounds->begin, 1);
        forward(bounds->end, 2);
        if (forwarded && bounds->const_range()) {
          task.device_range_dispatch_version = 0;
          bounds->host_launch.emplace();
          bounds->host_launch->begin.factor = bounds->begin.value;
          bounds->host_launch->end.factor = bounds->end.value;
        }
        if (bounds->constant_size() == 0) {
          dead.insert(entry);
          continue;
        }
        if (bounds->constant_size() == 1) {
          auto *body =
              llvm::dyn_cast<llvm::Function>(dispatch->getArgOperand(4));
          if (body && !body->isDeclaration() &&
              !body->hasFnAttribute(llvm::Attribute::OptimizeNone)) {
            fold.run(*body, analyses);
            if (body->size() == 1 && body->front().size() == 1 &&
                llvm::isa<llvm::ReturnInst>(body->front().front())) {
              dead.insert(entry);
              continue;
            }
          }
        }
      }
    }
    for (int slot : slots)
      values.erase(slot);
    if (pure_writer)
      values.insert(writes.begin(), writes.end());
  }

  std::set<int> live;
  for (size_t i = tasks.size(); i-- > 0;) {
    const auto &[entry, slots, writes, pure_writer] = facts[i];
    if (dead.count(entry))
      continue;
    if (pure_writer &&
        std::none_of(writes.begin(), writes.end(), [&](const auto &item) {
          return live.count(item.first);
        })) {
      dead.insert(entry);
      continue;
    }
    if (pure_writer) {
      for (const auto &[slot, value] : writes)
        live.erase(slot);
    } else {
      live.insert(slots.begin(), slots.end());
    }
    if (const auto &range = tasks[i].range_bounds)
      for (const auto &bound : {range->begin, range->end})
        if (bound.kind == Kind::global_tmp_i32)
          live.insert(bound.value);
  }
  if (dead.size() == tasks.size())
    dead.erase(facts.front().entry);  // Preserve the single-task launch ABI.
  erase_prepared_tasks(module, tasks, dead);
}

}  // namespace taichi::lang::LLVM
