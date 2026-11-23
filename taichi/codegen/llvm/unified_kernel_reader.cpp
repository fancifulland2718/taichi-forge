#include "taichi/codegen/llvm/unified_kernel_reader.h"
#include "taichi/transforms/cpu_workgroup.h"

#include <thread>

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Support/MemoryBufferRef.h"

#include "taichi/ir/statements.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/ir/type_system.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/kernel.h"
#include "taichi/program/extension.h"
#include "taichi/program/program.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/system/profiler.h"

namespace taichi::lang::LLVM {
namespace {

DataType scalar_type(llvm::Type *type, bool is_signed = true) {
  auto &types = TypeFactory::get_instance();
  if (type->isIntegerTy(1))
    return PrimitiveType::u1;
  if (type->isIntegerTy())
    return types.get_primitive_int_type(type->getIntegerBitWidth(), is_signed);
  if (type->isFloatingPointTy())
    return types.get_primitive_real_type(type->getScalarSizeInBits());
  if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type))
    return types.get_tensor_type({int(vector->getNumElements())},
                                 scalar_type(vector->getElementType(), is_signed));
  TI_ERROR("Unsupported unified LLVM value type");
}

int literal(const llvm::Value *value) {
  auto *integer = llvm::dyn_cast<llvm::ConstantInt>(value);
  TI_ERROR_IF(!integer || !integer->getValue().isSignedIntN(32),
              "Invalid integer in a unified LLVM port");
  return integer->getBitWidth() == 1 ? int(!integer->isZero()) :
                                        int(integer->getSExtValue());
}

DataType port_type(llvm::StringRef text) {
  std::vector<int> shape;
  if (text.consume_front("[Tensor (")) {
    auto [dimensions, element] = text.split(") ");
    TI_ERROR_IF(!element.consume_back("]"), "Invalid unified tensor element");
    llvm::SmallVector<llvm::StringRef, 4> axes;
    dimensions.split(axes, ", ");
    for (const auto &axis : axes) {
      int size;
      TI_ERROR_IF(axis.getAsInteger(10, size) || size <= 0, "Invalid unified tensor extent");
      shape.push_back(size);
    }
    text = element;
  }
  DataType element = PrimitiveType::unknown;
#define PER_TYPE(x) if (text == #x) element = PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  TI_ERROR_IF(element == PrimitiveType::unknown || element == PrimitiveType::gen,
              "Unknown unified scalar type {}", text.str());
  if (shape.empty()) {
    return element;
  }
  return TypeFactory::get_instance().get_tensor_type(shape, element);
}

unsigned portable_target_bit(Arch arch) {
  return arch == Arch::x64 ? 1 : arch == Arch::cuda ? 2 : arch == Arch::vulkan ? 4 : 0;
}

// Apply bound scalar inputs while workgroup operations still belong to their
// conditional ranges. Target codegen cannot compile an inactive GPU range
// on a CPU, even when its dispatched grid would contain zero work.
void substitute_bound_arguments(llvm::Function &fn, const CallableBase &kernel,
                                const std::map<std::vector<int>, int> &facts) {
  if (facts.empty())
    return;
  std::vector<std::pair<llvm::WeakVH, int>> loads;
  for (auto &bb : fn) {
    for (auto &inst : bb) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
      if (!call || !call->getCalledFunction() ||
          !call->getCalledFunction()->getName().starts_with("forge.arg.i32.") ||
          !call->getType()->isIntegerTy(32))
        continue;
      std::vector<int> path;
      for (unsigned i = 2; i < call->arg_size(); ++i)
        path.push_back(literal(call->getArgOperand(i)));
      auto it = facts.find(path);
      if (it != facts.end()) {
        TI_ERROR_IF(kernel.args_type->get_element_type(path) != PrimitiveType::i32,
                    "Invalid unified LLVM specialization argument");
        loads.emplace_back(call, it->second);
      }
    }
  }
  for (auto &[handle, value] : loads) {
    if (auto *load = llvm::dyn_cast_or_null<llvm::CallInst>(handle)) {
      llvm::replaceAndRecursivelySimplify(load, llvm::ConstantInt::get(load->getType(), value));
      if (handle)
        load->eraseFromParent();
    }
  }
}

// Both target predicates and bound arguments can make a parallel range
// empty. Drop its body before checking target-specific resources.
void fold_empty_ranges(llvm::Function &fn) {
  std::unique_ptr<llvm::DominatorTree> dominators;
  auto range_constant = [&](llvm::Value *value) -> llvm::ConstantInt * {
    if (auto *literal = llvm::dyn_cast<llvm::ConstantInt>(value))
      return literal;
    // Conditional launch bounds can be stored in a private scalar before
    // reaching the range preheader. After target folding, every remaining
    // store must agree and one must actually initialize this load.
    auto *load = llvm::dyn_cast<llvm::LoadInst>(value);
    if (!load || !load->isSimple())
      return nullptr;
    auto *slot = llvm::dyn_cast<llvm::AllocaInst>(load->getPointerOperand());
    if (!slot || !llvm::isAllocaPromotable(slot))
      return nullptr;
    llvm::ConstantInt *known = nullptr;
    bool initialized = false;
    for (auto *user : slot->users()) {
      auto *store = llvm::dyn_cast<llvm::StoreInst>(user);
      if (!store)
        continue;
      auto *constant = llvm::dyn_cast<llvm::ConstantInt>(store->getValueOperand());
      if (!store->isSimple() || !constant || (known && constant != known))
        return nullptr;
      known = constant;
      if (!dominators)
        dominators = std::make_unique<llvm::DominatorTree>(fn);
      initialized |= dominators->dominates(store, load);
    }
    return initialized ? known : nullptr;
  };
  for (auto &bb : fn) {
    auto *branch = llvm::dyn_cast<llvm::BranchInst>(bb.getTerminator());
    if (!branch || !branch->isConditional() || !branch->getMetadata("forge.range"))
      continue;
    auto *test = llvm::dyn_cast<llvm::ICmpInst>(branch->getCondition());
    auto *index = llvm::dyn_cast<llvm::PHINode>(&bb.front());
    if (!test || !index || index->getNumIncomingValues() != 2 ||
        test->getPredicate() != llvm::ICmpInst::ICMP_SLT || test->getOperand(0) != index)
      continue;
    llvm::Value *limit = test->getOperand(1);
    if (auto *opts = llvm::dyn_cast<llvm::CallInst>(limit); opts && opts->getCalledFunction() &&
        opts->getCalledFunction()->getName().starts_with("forge.range.device_options."))
      limit = opts->getArgOperand(0);
    auto *end = range_constant(limit);
    if (!end)
      continue;
    for (unsigned i = 0; i < 2; ++i) {
      auto *start = range_constant(index->getIncomingValue(i));
      auto *step = llvm::dyn_cast<llvm::BinaryOperator>(index->getIncomingValue(1 - i));
      if (start && step && step->getOpcode() == llvm::Instruction::Add &&
          (step->getOperand(0) == index || step->getOperand(1) == index) &&
          start->getValue().sge(end->getValue())) {
        branch->setCondition(llvm::ConstantInt::getFalse(fn.getContext()));
        break;
      }
    }
  }
  for (auto &bb : fn)
    llvm::ConstantFoldTerminator(&bb, true);
  llvm::removeUnreachableBlocks(fn);
}

template <typename Evaluate>
bool fold_device_query(llvm::Function &fn, llvm::StringRef name, Evaluate evaluate) {
  auto *predicate = fn.getParent()->getFunction(name);
  if (!predicate)
    return false;
  TI_ERROR_IF(!predicate->isDeclaration(), "Invalid portable target predicate");
  llvm::SmallVector<llvm::CallInst *, 4> calls;
  for (auto *user : predicate->users()) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(user);
    TI_ERROR_IF(!call || call->arg_size() != 1, "Invalid portable target test");
    if (call->getFunction() == &fn)
      calls.push_back(call);
  }
  for (auto *call : calls) {
    auto *id = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
    TI_ERROR_IF(!id, "Expected a constant device query");
    auto *value = llvm::ConstantInt::get(call->getType(), evaluate(id->getZExtValue()));
    llvm::WeakVH remaining(call);
    llvm::replaceAndRecursivelySimplify(call, value);
    if (remaining)
      call->eraseFromParent();
  }
  return !calls.empty();
}

// Resolve only the selected root's callees. Every unit is verified once for
// this materialization. Expand callees after their own dependencies so nested
// calls do not cause repeated reads.
void import_helpers(llvm::Module &module, llvm::Function &entry,
                    const UnifiedUnitLoader &loader, Arch arch,
                    const DeviceCapabilityConfig &caps) {
  TI_COMPILE_PROFILER("cpp.unified.prepare.import_helpers");
  std::unordered_set<std::string> active;
  llvm::Linker linker(module);
  std::function<void(llvm::Function &)> visit = [&](llvm::Function &fn) {
    bool folded = fold_device_query(fn, "forge.builtin.unified_target_matches.i1.i32", [&](unsigned mask) {
      TI_ERROR_IF(!mask || mask > 7, "Invalid portable target mask");
      return (mask & portable_target_bit(arch)) != 0;
    });
    folded |= fold_device_query(fn, "forge.builtin.unified_has_device_capability.i1.i32", [&](unsigned id) {
      return caps.get(static_cast<DeviceCapability>(id)) != 0;
    });
    folded |= fold_device_query(fn, "forge.builtin.unified_has_extension.i1.i32", [&](unsigned id) {
      return is_extension_supported(arch, static_cast<Extension>(id));
    });
    if (folded) {
      for (auto &bb : fn)
        llvm::ConstantFoldTerminator(&bb, true);
      llvm::removeUnreachableBlocks(fn);
    }
    std::vector<llvm::CallInst *> calls;
    for (auto &block : fn)
      for (auto &inst : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *target = call->getCalledFunction();
              target && target->getName().starts_with("forge.fn."))
            calls.push_back(call);
    for (auto *call : calls) {
      auto *target = call->getCalledFunction();
      auto name = target->getName().str();
      TI_ERROR_IF(active.count(name), "Recursive unified LLVM helper {}", name);
      if (target->isDeclaration()) {
        auto digest = name.substr(sizeof("forge.fn.") - 1);
        TI_ERROR_IF(!loader || digest.size() != 64 ||
                        digest.find_first_not_of("0123456789abcdef") != std::string::npos,
                    "Missing unified LLVM helper {}", name);
        auto bytes = loader(digest);
        TI_ERROR_IF(aot_artifact_sha256(bytes) != digest, "Changed unified LLVM helper {}", digest);
        auto body = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bytes, name), module.getContext());
        if (!body)
          TI_ERROR("Invalid unified LLVM helper: {}", llvm::toString(body.takeError()));
        auto *implementation = (*body)->getFunction("entry");
        TI_ERROR_IF(!implementation || implementation->isDeclaration() ||
                        implementation->getFunctionType() != target->getFunctionType() ||
                        llvm::verifyModule(**body), "Invalid unified LLVM helper ABI {}", name);
        implementation->setName(name);
        TI_ERROR_IF(linker.linkInModule(std::move(*body)), "Cannot link unified LLVM helper {}", name);
        target = module.getFunction(name);
        active.insert(name);
        visit(*target);
        active.erase(name);
      }
      llvm::InlineFunctionInfo info;
      auto result = llvm::InlineFunction(*call, info, false, nullptr, false);
      TI_ERROR_IF(!result.isSuccess(), "Cannot inline unified LLVM helper {}", name);
    }
  };
  visit(entry);
}

bool local_address(Stmt *address) {
  if (auto *component = address->cast<MatrixPtrStmt>())
    return local_address(component->origin);
  return address->is<AllocaStmt>();
}

// After helper inlining, many private slots are only used within one
// basic block. Forward their initialized values before generating target
// IR. Loop-carried state, escaping slots and observable memory stay intact.
void promote_block_locals(llvm::Function &entry) {
  std::vector<llvm::AllocaInst *> locals;
  for (auto &inst : entry.getEntryBlock()) {
    auto *slot = llvm::dyn_cast<llvm::AllocaInst>(&inst);
    if (!slot || slot->use_empty() || !llvm::isAllocaPromotable(slot))
      continue;
    llvm::Instruction *first = nullptr;
    for (auto *user : slot->users()) {
      auto *access = llvm::cast<llvm::Instruction>(user);
      if ((!llvm::isa<llvm::StoreInst>(access) && !llvm::isa<llvm::LoadInst>(access)) ||
          access->isAtomic() || (first && first->getParent() != access->getParent())) {
        first = nullptr;
        break;
      }
      if (!first || access->comesBefore(first))
        first = access;
    }
    if (first && llvm::isa<llvm::StoreInst>(first))
      locals.push_back(slot);
  }
  if (!locals.empty()) {
    llvm::DominatorTree dom(entry);
    llvm::PromoteMemToReg(locals, dom);
  }
}

// LLVM hoists fixed-size allocas, including inlined call frames, to function
// entry. A Forge range can become a parallel task. Restore lane-local storage
// before offload when all its accesses, including the producer's explicit
// initialization, belong to that range. Outer-scope state has an access
// outside the range and keeps its original lifetime.
void restore_parallel_locals(llvm::Function &fn) {
  llvm::DominatorTree dominators(fn);
  llvm::LoopInfo loops(dominators);
  std::vector<llvm::AllocaInst *> locals;
  for (auto &inst : fn.getEntryBlock())
    if (auto *local = llvm::dyn_cast<llvm::AllocaInst>(&inst))
      locals.push_back(local);
  for (auto *local : locals) {
    llvm::Loop *range = nullptr;
    bool shared = false;
    std::vector<llvm::Instruction *> addresses{local};
    for (size_t i = 0; i < addresses.size() && !shared; ++i) {
      for (auto *use : addresses[i]->users()) {
        auto *inst = llvm::dyn_cast<llvm::Instruction>(use);
        if (!inst) {
          shared = true;
          break;
        }
        if (llvm::isa<llvm::GetElementPtrInst>(inst)) {
          addresses.push_back(inst);
          continue;
        }
        llvm::Loop *owner = nullptr;
        for (auto *loop = loops.getLoopFor(inst->getParent()); loop; loop = loop->getParentLoop())
          if (loop->getHeader()->getTerminator()->getMetadata("forge.range"))
            owner = loop;
        if (!owner || (range && range != owner)) {
          shared = true;
          break;
        }
        range = owner;
      }
    }
    if (shared || !range)
      continue;
    auto *body = range->getHeader()->getTerminator()->getSuccessor(0);
    auto position = body->getFirstInsertionPt();
    for (auto *address : addresses)
      if (!range->contains(address->getParent()))
        address->moveBefore(*body, position);
  }
}

class KernelReader {
 public:
  KernelReader(Kernel &kernel, llvm::Function &function,
               const std::map<std::vector<int>, int> &constant_int_args,
               const std::vector<std::vector<SNode *>> &storage, bool *needs_bound_device)
      : kernel_(kernel), joins_(function), dominators_(function), loops_(dominators_),
        constants_(constant_int_args), storage_(storage), needs_bound_device_(needs_bound_device) {
    unsigned order = 0;
    for (auto *bb : llvm::ReversePostOrderTraversal<llvm::Function *>(&function))
      block_order_[bb] = order++;
    region(&function.getEntryBlock(), nullptr, nullptr, *kernel.ir->as<Block>());
    for (auto *range : cpu_workgroups_) {
      if (!irpass::materialize_cpu_workgroup(*range, kernel.program->compile_config(), needs_bound_device != nullptr))
        *needs_bound_device = true;
    }
  }

 private:
  using BB = llvm::BasicBlock;
  Kernel &kernel_;
  llvm::PostDominatorTree joins_;
  llvm::DominatorTree dominators_;
  llvm::LoopInfo loops_;
  const std::map<std::vector<int>, int> &constants_;
  const std::vector<std::vector<SNode *>> &storage_;
  bool *needs_bound_device_;
  std::unordered_map<const llvm::Value *, Stmt *> values_;
  std::unordered_set<const BB *> visited_;
  std::unordered_map<BB *, unsigned> block_order_;
  std::vector<BB *> range_latches_;
  std::unordered_map<const llvm::Value *, StructForStmt *> struct_loops_;
  std::vector<Stmt *> results_;
  Stmt *model_block_dim_{nullptr};
  std::vector<RangeForStmt *> cpu_workgroups_;

  void note_workgroup(Block &block) {
    if (!arch_is_cpu(kernel_.arch))
      return;
    Stmt *owner = block.parent_stmt();
    while (owner && owner->parent->parent_stmt())
      owner = owner->parent->parent_stmt();
    auto *range = owner ? owner->cast<RangeForStmt>() : nullptr;
    if (!range) {
      TI_ERROR_IF(!needs_bound_device_, "CPU workgroup operation requires a parallel range");
      *needs_bound_device_ = true;
      return;
    }
    if (cpu_workgroups_.empty() || cpu_workgroups_.back() != range)
      cpu_workgroups_.push_back(range);
  }
  struct WhileScope {
    llvm::Loop *loop;
    BB *exit;
  };
  std::vector<WhileScope> while_scopes_;

  // LLVM aggregate GEPs describe the typed helper frame. Allocate its
  // numerical members in the target IR. A descriptor is an immutable
  // reference forwarded by the caller, never a copied allocation.
  struct Slot {
    llvm::Type *type{};
    Stmt *local{};
    Stmt *descriptor{};
    std::vector<Slot> fields;
  };
  std::map<const llvm::Value *, Slot> frames_;
  std::unordered_map<const llvm::Value *, Slot *> slots_;
  std::unordered_map<const llvm::Value *, std::pair<Stmt *, std::vector<int>>> descriptor_members_;

  void allocate_slot(Block &block, Slot &slot, llvm::Type *type) {
    slot.type = type;
    if (auto *record = llvm::dyn_cast<llvm::StructType>(type)) {
      slot.fields.resize(record->getNumElements());
      for (unsigned i = 0; i < slot.fields.size(); ++i)
        allocate_slot(block, slot.fields[i], record->getElementType(i));
    } else if (!type->isPointerTy()) {
      slot.local = block.push_back<AllocaStmt>(scalar_type(type));
    }
  }

  Slot *member(const llvm::Value *address, llvm::Type *access) {
    auto found = slots_.find(address);
    if (found == slots_.end())
      return nullptr;
    auto *slot = found->second;
    // LLVM can fold a zero-offset GEP to the containing record pointer.
    while (slot->type != access && !slot->fields.empty())
      slot = &slot->fields.front();
    auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(slot->type);
    if (slot->type != access && !(vector && vector->getElementType() == access)) {
      std::string detail;
      llvm::raw_string_ostream out(detail);
      slot->type->print(out);
      out << " accessed as ";
      access->print(out);
      out << " at ";
      address->print(out);
      TI_ERROR("Invalid access to unified helper frame in {}: {}", kernel_.get_name(), out.str());
    }
    return slot;
  }

  Stmt *slot_address(Block &block, Slot &slot, llvm::Type *access) {
    if (slot.type == access)
      return slot.local;
    // A zero-offset GEP into lane zero can be folded into its record pointer.
    auto *zero = block.push_back<ConstStmt>(TypedConstant(0));
    return block.push_back<MatrixPtrStmt>(slot.local, zero);
  }

  Stmt *load_argument_member(Block &block, Stmt *base, std::vector<int> path,
                             llvm::Type *access) {
    auto *record = base->ret_type.ptr_removed()->as<StructType>();
    DataType type = path.empty() ? DataType(record) : DataType(record->get_element_type(path));
    // An opaque pointer at byte zero also addresses the first member of a
    // record. Reconstruct that member from the consumer ABI and access type.
    while (type->is<StructType>() && !access->isStructTy()) {
      const auto *nested = type->as<StructType>();
      TI_ERROR_IF(nested->elements().empty(), "Cannot load a member of an empty unified argument");
      path.push_back(0);
      type = nested->get_element_type({0});
    }
    auto *result = block.push_back<GetElementStmt>(base, path);
    result->ret_type = type;
    return result;
  }

  Stmt *numeric_address(Block &block, const llvm::Value *pointer, llvm::Type *access) {
    if (auto *slot = member(pointer, access))
      return slot_address(block, *slot, access);
    auto *address = value(block, pointer);
    // An opaque LLVM pointer to the first lane can be identical to the
    // address of its tensor. The access type distinguishes the two.
    if (!access->isVectorTy() && address->ret_type.ptr_removed()->is<TensorType>()) {
      auto *zero = block.push_back<ConstStmt>(TypedConstant(0));
      return block.push_back<MatrixPtrStmt>(address, zero);
    }
    return address;
  }

  Stmt *matrix(Block &block, DataType type, const std::vector<Stmt *> &lanes) {
    auto *result = block.push_back<MatrixInitStmt>(lanes);
    result->ret_type = type;
    return result;
  }

  // LLVM vectors have one lane axis. A resource can expose the same lanes
  // as a rank-two tensor with an affine component layout. Reshape the value
  // at this boundary while leaving the resource descriptor intact.
  Stmt *tensor_value(Block &block, Stmt *source, DataType from, DataType to) {
    if (from == to)
      return source;
    auto lanes = from->as<TensorType>()->get_num_elements();
    TI_ERROR_IF(lanes != to->as<TensorType>()->get_num_elements(),
                "Different tensor lane counts in unified LLVM memory access");
    if (auto *value = source->cast<MatrixInitStmt>())
      return matrix(block, to, value->values);
    auto *temp = block.push_back<AllocaStmt>(from);
    block.push_back<LocalStoreStmt>(temp, source);
    std::vector<Stmt *> values;
    for (int i = 0; i < lanes; ++i) {
      auto *index = block.push_back<ConstStmt>(TypedConstant(i));
      auto *pointer = block.push_back<MatrixPtrStmt>(temp, index);
      values.push_back(block.push_back<LocalLoadStmt>(pointer));
    }
    return matrix(block, to, values);
  }

  DataType flat_tensor(DataType type, llvm::Type *vector) {
    return TypeFactory::get_instance().get_tensor_type(
        {int(llvm::cast<llvm::FixedVectorType>(vector)->getNumElements())}, type.get_element_type());
  }

  Stmt *cast(Block &block, Stmt *operand, DataType type,
             UnaryOpType kind = UnaryOpType::cast_value) {
    auto *result = block.push_back<UnaryOpStmt>(kind, operand)->as<UnaryOpStmt>();
    result->cast_type = type.get_element_type();
    result->ret_type = type;
    return result;
  }

  Stmt *value(Block &block, const llvm::Value *operand) {
    auto found = values_.find(operand);
    if (found != values_.end() && found->second)
      return found->second;
    // Constants can occur in unrelated branches. Materialize at the use
    // so a value in one loop body never escapes its dominance scope.
    if (auto *integer = llvm::dyn_cast<llvm::ConstantInt>(operand))
      return block.push_back<ConstStmt>(TypedConstant(
          scalar_type(operand->getType()), integer->getZExtValue()));
    if (auto *real = llvm::dyn_cast<llvm::ConstantFP>(operand))
      return block.push_back<ConstStmt>(TypedConstant(
          scalar_type(operand->getType()), real->getValueAPF().convertToDouble()));
    if (auto *constant = llvm::dyn_cast<llvm::Constant>(operand)) {
      if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(operand->getType())) {
        std::vector<Stmt *> lanes;
        for (unsigned i = 0; i < vector->getNumElements(); ++i) {
          auto *lane = constant->getAggregateElement(i);
          TI_ERROR_IF(!lane || llvm::isa<llvm::UndefValue>(lane),
                      "Undefined lane in a unified LLVM value");
          lanes.push_back(value(block, lane));
        }
        return matrix(block, scalar_type(vector), lanes);
      }
    }
    std::string detail;
    llvm::raw_string_ostream out(detail);
    operand->print(out);
    TI_ERROR("Unsupported unified LLVM operand in {}: {}", kernel_.get_name(), out.str());
  }

  Stmt *port(Block &block, const llvm::CallInst &call) {
    const auto *fn = call.getCalledFunction();
    TI_ERROR_IF(!fn, "Indirect unified LLVM port");
    auto name = fn->getName();
    auto arg = [&](unsigned i) { return value(block, call.getArgOperand(i)); };
    auto number = [&](unsigned i) { return literal(call.getArgOperand(i)); };
    auto args = [&](unsigned first) {
      std::vector<Stmt *> operands;
      for (unsigned i = first; i < call.arg_size(); ++i)
        operands.push_back(arg(i));
      return operands;
    };
    if (name.starts_with("forge.arg.")) {
      std::vector<int> path;
      for (unsigned i = 2; i < call.arg_size(); ++i)
        path.push_back(number(i));
      const bool address = name.starts_with("forge.arg.address.");
      auto fixed = constants_.find(path);
      DataType type = kernel_.args_type->get_element_type(path);
      auto parameter = kernel_.nested_parameters.find(path);
      if (address && parameter != kernel_.nested_parameters.end() &&
          parameter->second.ptype == ParameterType::kNdarray) {
        const auto &array = parameter->second;
        DataType element = type->as<StructType>()->get_element_type({1});
        element = element.ptr_removed();
        auto &factory = TypeFactory::get_instance();
        if (!array.element_shape.empty())
          element = factory.get_tensor_type(array.element_shape, element);
        type = factory.get_ndarray_struct_type(element, array.total_dim - array.element_shape.size(),
                                                array.needs_grad);
      }
      if (fixed != constants_.end()) {
        TI_ERROR_IF(address || type != PrimitiveType::i32,
                    "Invalid unified LLVM specialization argument");
        return block.push_back<ConstStmt>(TypedConstant(fixed->second));
      }
      if (address)
        type.set_is_pointer(true);
      return block.push_back<ArgLoadStmt>(path, type, address || type.is_pointer(),
                                           !address, number(1));
    }
    if (name.starts_with("forge.array.extent.")) {
      auto *base = arg(0)->as<ArgLoadStmt>();
      auto path = base->arg_id;
      path.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      path.push_back(number(1));
      if (auto it = constants_.find(path); it != constants_.end())
        return block.push_back<ConstStmt>(TypedConstant(it->second));
      return block.push_back<ExternalTensorShapeAlongAxisStmt>(number(1), base->arg_id);
    }
    if (name.starts_with("forge.array.base."))
      return block.push_back<ExternalTensorBasePtrStmt>(arg(0)->as<ArgLoadStmt>()->arg_id,
                                                        bool(number(1)));
    if (name.consume_front("forge.array.address.")) {
      auto *base = arg(0);
      int rank = number(7);
      TI_ERROR_IF(rank < 0 || 8ull + rank > call.arg_size(), "Invalid unified tensor rank");
      std::vector<int> shape;
      for (int i = 0; i < rank; ++i)
        shape.push_back(number(8 + i));
      const auto offset = llvm::cast<llvm::ConstantInt>(call.getArgOperand(4))->getZExtValue();
      const auto stride = llvm::cast<llvm::ConstantInt>(call.getArgOperand(5))->getZExtValue();
      auto *pointer = block.push_back<ExternalPtrStmt>(
          base, args(8 + rank), number(1), shape, bool(number(2)),
          static_cast<BoundaryMode>(number(3)), offset, stride, bool(number(6)));
      pointer->ret_type = base->ret_type.ptr_removed()->as<StructType>()->get_element_type({1});
      const auto element = name.split('.').first;
      if (pointer->ret_type.ptr_removed()->to_string() != element) {
        pointer->ret_type = port_type(element);
        pointer->as<ExternalPtrStmt>()->overrided_dtype = true;
      }
      pointer->ret_type.set_is_pointer(true);
      return pointer;
    }
    if (name.starts_with("forge.array.component."))
      return block.push_back<MatrixPtrStmt>(arg(0), arg(1), DebugInfo(), args(2));
    if (name.starts_with("forge.hint.decoration.")) {
      std::vector<uint32_t> items;
      for (unsigned i = 1; i < call.arg_size(); ++i)
        items.push_back(number(i));
      return block.push_back<DecorationStmt>(arg(0), items);
    }
    if (name.starts_with("forge.hint.range."))
      return block.push_back<RangeAssumptionStmt>(arg(0), arg(1), number(2), number(3));
    if (name.starts_with("forge.range.block_size.")) {
      model_block_dim_ = arg(0);
      return nullptr;
    }
    if (name.starts_with("forge.range.device_options."))
      return arg(0);  // Scheduling is interpreted where the range consumes this bound.
    if (name.starts_with("forge.field.active_count."))
      return nullptr;  // Interpreted as native list generation by the enclosing loop.
    if (name.starts_with("forge.field.coordinate.")) {
      auto found = struct_loops_.find(call.getArgOperand(0));
      TI_ERROR_IF(found == struct_loops_.end(), "Unified field coordinate outside its active list");
      return block.push_back<LoopIndexStmt>(found->second, number(1));
    }
    if (name.starts_with("forge.field.activation.")) {
      auto op = static_cast<SNodeOpType>(number(3));
      TI_ERROR_IF(!SNodeOpStmt::activation_related(op), "Invalid unified activation operation");
      return block.push_back<SNodeOpStmt>(op, storage_.at(number(1)).at(number(2)), arg(4));
    }
    if (name.starts_with("forge.field.address.")) {
      auto *node = storage_.at(number(1)).at(number(2));
      TI_ERROR_IF(!node, "Missing unified field member");
      auto *ptr = block.push_back<GlobalPtrStmt>(node, args(6), bool(number(3)),
                                                 bool(number(4)))->as<GlobalPtrStmt>();
      ptr->is_bit_vectorized = number(5);
      return ptr;
    }
    if (name.starts_with("forge.field.component.")) {
      int count = number(3);
      TI_ERROR_IF(count <= 0 || 4ull + 2ull * count > call.arg_size(), "Invalid unified field tensor");
      std::vector<SNode *> members;
      for (int i = 0; i < count; ++i)
        members.push_back(storage_.at(number(4 + i * 2)).at(number(5 + i * 2)));
      auto *tensor = TypeFactory::get_instance().get_tensor_type({count}, members.front()->dt);
      auto *matrix = block.push_back<MatrixOfGlobalPtrStmt>(
          members, args(4 + 2 * count), false, 0, tensor, bool(number(2)));
      return block.push_back<MatrixPtrStmt>(matrix, arg(1));
    }
    if (name.starts_with("forge.assert.")) {
      auto *metadata = call.getMetadata("forge.message");
      TI_ERROR_IF(!metadata || metadata->getNumOperands() != 1, "Missing unified assertion message");
      return block.push_back<AssertStmt>(arg(1),
          llvm::cast<llvm::MDString>(metadata->getOperand(0))->getString().str(), args(2));
    }
    if (name.starts_with("forge.print.")) {
      auto *items = call.getMetadata("forge.print.items");
      TI_ERROR_IF(!items, "Missing unified print operands");
      std::vector<PrintStmt::EntryType> contents;
      std::vector<PrintStmt::FormatType> formats;
      unsigned next_arg = 1;
      for (const auto &operand : items->operands()) {
        auto *item = llvm::cast<llvm::MDNode>(operand);
        auto field = [&](unsigned pos) {
          return llvm::cast<llvm::MDString>(item->getOperand(pos))->getString();
        };
        TI_ERROR_IF(item->getNumOperands() < 2 || item->getNumOperands() > 3,
                    "Invalid unified print item");
        if (field(0) == "text")
          contents.emplace_back(field(1).str());
        else {
          TI_ERROR_IF(field(0) != "value" || next_arg >= call.arg_size(),
                      "Invalid unified print value");
          contents.emplace_back(cast(block, arg(next_arg++), port_type(field(1))));
        }
        formats.emplace_back(item->getNumOperands() == 3
                                 ? std::make_optional(field(2).str()) : std::nullopt);
      }
      TI_ERROR_IF(next_arg != call.arg_size(), "Unused unified print values");
      return block.push_back<PrintStmt>(contents, formats);
    }
    if (name.consume_front("forge.workgroup.storage.")) {
      note_workgroup(block);
      auto dtype = port_type(name.split('.').first);
      auto *local = block.push_back<AllocaStmt>(dtype)->as<AllocaStmt>();
      local->is_shared = true;
      return local;
    }
    if (name.consume_front("forge.workgroup.element.")) {
      note_workgroup(block);
      return block.push_back<DynamicSharedPtrStmt>(number(1), port_type(name.split('.').first),
                                                   args(3), arg(2));
    }
    if (name.starts_with("forge.acceleration_structure."))
      return block.push_back<AccelerationStructurePtrStmt>(arg(0));
    if (name.starts_with("forge.image.binding.")) {
      if (number(5) > 0)
        return block.push_back<TexturePtrStmt>(arg(0), number(1), number(5), arg(6));
      return block.push_back<TexturePtrStmt>(arg(0), number(1), bool(number(2)),
                                             static_cast<BufferFormat>(number(3)), number(4));
    }
    if (name.consume_front("forge.image.")) {
      auto tag = name.split('.').first;
      for (auto op : {TextureOpType::kSampleLod, TextureOpType::kFetchTexel,
                      TextureOpType::kLoad, TextureOpType::kStore,
                      TextureOpType::kSampleGrad, TextureOpType::kSampleCompare})
        if (tag == texture_op_type_name(op))
          return block.push_back<TextureOpStmt>(op, arg(0), args(1));
    }
    if (name.consume_front("forge.atomic.")) {
      const auto tag = name.split('.').first;
      for (auto op : {AtomicOpType::add, AtomicOpType::sub, AtomicOpType::min,
                      AtomicOpType::max, AtomicOpType::bit_and, AtomicOpType::bit_or,
                      AtomicOpType::bit_xor, AtomicOpType::mul}) {
        if (tag == atomic_op_type_name(op)) {
          auto *address = numeric_address(block, call.getArgOperand(0), call.getType());
          auto *operand = arg(1);
          auto logical = address->ret_type.ptr_removed();
          bool tensor = call.getType()->isVectorTy() && logical->is<TensorType>();
          if (tensor)
            operand = tensor_value(block, operand, flat_tensor(logical, call.getType()), logical);
          auto *atomic = block.push_back<AtomicOpStmt>(op, address, operand)->as<AtomicOpStmt>();
          atomic->is_reduction = number(2);
          return tensor ? tensor_value(block, atomic, logical, flat_tensor(logical, call.getType()))
                        : atomic;
        }
      }
    }
    if (name.starts_with("forge.thread.index."))
      return block.push_back<GlobalThreadIndexStmt>();
    if (name.starts_with("forge.random."))
      return block.push_back<RandStmt>(scalar_type(call.getType(), !name.starts_with("forge.random.u")));
    if (name.consume_front("forge.builtin.")) {
      std::string operation = name.split('.').first.str();
      if (operation == "unified_require_device") {
        const auto mask = number(0);
        TI_ERROR_IF(mask <= 0 || mask > 7 || !(mask & portable_target_bit(kernel_.arch)),
                    "Unified entry requires another device (target mask {}, consumer {})", mask, arch_name(kernel_.arch));
        return block.push_back<ConstStmt>(TypedConstant(0));
      }
      if (operation == "workgroupBarrier" || operation == "workgroupMemoryBarrier") {
        note_workgroup(block);
        if (kernel_.arch == Arch::cuda)
          operation = "block_barrier";
      }
      if (operation == "localInvocationId" || operation == "vkGlobalThreadIdx")
        note_workgroup(block);
      if (operation == "vkGlobalThreadIdx" && kernel_.arch == Arch::cuda)
        return block.push_back<InternalFuncStmt>("linear_thread_idx", std::vector<Stmt *>{},
                                                 PrimitiveType::i32, true);
      const auto builtin = llvm::StringRef(operation);
      const char *required_device = nullptr;
      if (kernel_.arch != Arch::cuda &&
          (builtin.starts_with("cuda_") || builtin.starts_with("block_barrier_") ||
           builtin == "warp_barrier"))
        required_device = "CUDA";
      else if (kernel_.arch != Arch::vulkan && builtin.starts_with("subgroup"))
        required_device = "Vulkan";
      if (required_device) {
        // A model dimension may still eliminate this branch. Retain its
        // typed result for specialization; never execute a device stub.
        TI_ERROR_IF(!needs_bound_device_, "Unified operation {} requires {}", operation, required_device);
        *needs_bound_device_ = true;
      }
      const bool context = call.arg_size() && call.getArgOperand(0) == call.getFunction()->getArg(0);
      DataType result;
      if (auto *record = call.getMetadata("forge.record.type")) {
        auto text = llvm::cast<llvm::MDString>(record->getOperand(0))->getString().str();
        liong::json::deserialize(liong::json::parse(text), result, true);
      } else if (call.getType()->isPointerTy()) {
        result = Operations::fixed_return_type(name.split('.').first.str());
        TI_ERROR_IF(!result.is_pointer(), "Missing pointer return type for unified builtin {}", name.str());
      } else {
        result = scalar_type(call.getType());
      }
      return block.push_back<InternalFuncStmt>(operation, args(context ? 1 : 0),
                                               result, context);
    }
    if (name.starts_with("forge.result.")) {
      TI_ERROR_IF(number(1) != results_.size(), "Discontinuous unified return value");
      results_.push_back(arg(2));
      return nullptr;
    }
    if (name.consume_front("forge.binary.")) {
      const auto op = name.split('.').first;
#define PER_BINARY_OP(x) if (op == #x) return block.push_back<BinaryOpStmt>(BinaryOpType::x, arg(0), arg(1));
#include "taichi/inc/binary_op.inc.h"
#undef PER_BINARY_OP
    }
    if (name.consume_front("forge.unary.")) {
      const auto op = name.split('.').first;
#define PER_UNARY_OP(x) if (op == #x) return block.push_back<UnaryOpStmt>(UnaryOpType::x, arg(0));
#include "taichi/inc/unary_op.inc.h"
#undef PER_UNARY_OP
    }
    TI_ERROR("Unsupported unified LLVM port {} in {}", fn->getName().str(), kernel_.get_name());
  }

  Stmt *instruction(Block &block, const llvm::Instruction &inst) {
    auto arg = [&](unsigned i) { return value(block, inst.getOperand(i)); };
    auto binary = [&](BinaryOpType op) { return block.push_back<BinaryOpStmt>(op, arg(0), arg(1)); };
    using I = llvm::Instruction;
    switch (inst.getOpcode()) {
      case I::Call: return port(block, llvm::cast<llvm::CallInst>(inst));
      case I::ExtractValue: {
        const auto indices = llvm::cast<llvm::ExtractValueInst>(inst).getIndices();
        auto *base = arg(0);
        std::vector<int> path(indices.begin(), indices.end());
        auto *field = block.push_back<GetElementStmt>(base, path);
        field->ret_type = base->ret_type.ptr_removed()->as<StructType>()->get_element_type(path);
        return field;
      }
      case I::Alloca: {
        auto *type = llvm::cast<llvm::AllocaInst>(inst).getAllocatedType();
        if (type->isStructTy() || type->isPointerTy() || type->isVectorTy()) {
          auto &slot = frames_[&inst];
          allocate_slot(block, slot, type);
          slots_[&inst] = &slot;
          return slot.local;
        }
        return block.push_back<AllocaStmt>(scalar_type(type));
      }
      case I::Load: {
        if (auto found = descriptor_members_.find(inst.getOperand(0)); found != descriptor_members_.end()) {
          const auto &[base, indices] = found->second;
          return load_argument_member(block, base, indices, inst.getType());
        }
        if (auto *slot = member(inst.getOperand(0), inst.getType())) {
          if (slot->local)
            return block.push_back<LocalLoadStmt>(slot_address(block, *slot, inst.getType()));
          TI_ERROR_IF(!slot->descriptor, "Missing unified helper array descriptor");
          return slot->descriptor;
        }
        auto *address = numeric_address(block, inst.getOperand(0), inst.getType());
        if (address->is<ArgLoadStmt>() && address->ret_type.ptr_removed()->is<StructType>() &&
            !inst.getType()->isStructTy())
          return load_argument_member(block, address, {}, inst.getType());
        auto *loaded = local_address(address) ? block.push_back<LocalLoadStmt>(address)
                                              : block.push_back<GlobalLoadStmt>(address);
        auto logical = address->ret_type.ptr_removed();
        return inst.getType()->isVectorTy() && logical->is<TensorType>()
                   ? tensor_value(block, loaded, logical, flat_tensor(logical, inst.getType()))
                   : loaded;
      }
      case I::Store: {
        if (auto *slot = member(inst.getOperand(1), inst.getOperand(0)->getType())) {
          if (slot->local)
            return block.push_back<LocalStoreStmt>(slot_address(block, *slot, inst.getOperand(0)->getType()), arg(0));
          TI_ERROR_IF(!slot->type->isPointerTy() || slot->descriptor,
                      "Invalid assignment to unified helper descriptor");
          slot->descriptor = arg(0);
          return nullptr;
        }
        auto *address = numeric_address(block, inst.getOperand(1), inst.getOperand(0)->getType());
        auto *source = arg(0);
        auto logical = address->ret_type.ptr_removed();
        auto *type = inst.getOperand(0)->getType();
        if (type->isVectorTy() && logical->is<TensorType>())
          source = tensor_value(block, source, flat_tensor(logical, type), logical);
        return local_address(address) ? block.push_back<LocalStoreStmt>(address, source)
                                      : block.push_back<GlobalStoreStmt>(address, source);
      }
      case I::GetElementPtr: {
        const auto &gep = llvm::cast<llvm::GetElementPtrInst>(inst);
        if (auto *slot = member(gep.getPointerOperand(), gep.getSourceElementType())) {
          if (slot->type != gep.getSourceElementType()) {
            TI_ERROR_IF(gep.getNumIndices() != 1, "Invalid unified vector frame offset");
            return block.push_back<MatrixPtrStmt>(slot->local, arg(1));
          }
          TI_ERROR_IF(literal(gep.getOperand(1)) != 0, "Invalid unified frame offset");
          for (unsigned i = 2; i < gep.getNumOperands(); ++i) {
            if (slot->type->isVectorTy()) {
              TI_ERROR_IF(i + 1 != gep.getNumOperands() || !slot->local, "Invalid unified frame lane");
              return block.push_back<MatrixPtrStmt>(slot->local, arg(i));
            }
            slot = &slot->fields.at(literal(gep.getOperand(i)));
          }
          slots_[&inst] = slot;
          return slot->local;
        }
        auto *base = arg(0);
        const int indices = gep.getNumIndices();
        if (base->ret_type.ptr_removed()->is<StructType>() && indices >= 2) {
          TI_ERROR_IF(literal(gep.getOperand(1)) != 0, "Invalid unified descriptor offset");
          std::vector<int> fields;
          for (unsigned i = 2; i < gep.getNumOperands(); ++i)
            fields.push_back(literal(gep.getOperand(i)));
          descriptor_members_[&gep] = {base, std::move(fields)};
          return nullptr;
        }
        TI_ERROR_IF(indices < 1 || indices > 2 ||
                    (indices == 2 && literal(gep.getOperand(1)) != 0) ||
                    !base->ret_type.ptr_removed()->is<TensorType>(),
                    "Unsupported unified LLVM address in {}", kernel_.get_name());
        return block.push_back<MatrixPtrStmt>(base, arg(indices));
      }
      case I::InsertElement: {
        // The producer writes whole tensor values as insertions starting at
        // poison. Only materialize complete values, never read a poison lane.
        if (llvm::all_of(inst.users(), [&](auto *user) {
              auto *next = llvm::dyn_cast<llvm::InsertElementInst>(user);
              return next && next->getOperand(0) == &inst;
            }))
          return nullptr;
        auto *vector = llvm::cast<llvm::FixedVectorType>(inst.getType());
        std::vector<const llvm::Value *> lanes(vector->getNumElements());
        auto *source = static_cast<const llvm::Value *>(&inst);
        while (auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(source)) {
          auto index = literal(insert->getOperand(2));
          TI_ERROR_IF(index < 0 || index >= lanes.size(), "Invalid unified vector lane");
          if (!lanes[index])
            lanes[index] = insert->getOperand(1);
          source = insert->getOperand(0);
        }
        std::vector<Stmt *> components;
        for (unsigned i = 0; i < lanes.size(); ++i) {
          if (!lanes[i] && llvm::isa<llvm::UndefValue>(source)) {
            for (auto *user : inst.users())
              TI_ERROR_IF(!llvm::isa<llvm::InsertElementInst>(user), "Incomplete unified LLVM vector");
            return nullptr;
          }
          if (!lanes[i]) {
            auto *base = llvm::dyn_cast<llvm::Constant>(source);
            TI_ERROR_IF(!base || !(lanes[i] = base->getAggregateElement(i)),
                        "Invalid unified LLVM vector base");
          }
          components.push_back(value(block, lanes[i]));
        }
        if (vector->getElementType()->isPointerTy()) {
          auto *type = TypeFactory::get_instance().get_tensor_type(
              {int(components.size())}, components.front()->ret_type.ptr_removed());
          return block.push_back<MatrixOfMatrixPtrStmt>(components, type);
        }
        return matrix(block, scalar_type(vector), components);
      }
      case I::ExtractElement: {
        auto *source = arg(0);
        if (auto *tensor = source->cast<MatrixInitStmt>()) {
          if (auto *index = llvm::dyn_cast<llvm::ConstantInt>(inst.getOperand(1)))
            return tensor->values.at(index->getZExtValue());
        }
        if (source->is<MatrixOfMatrixPtrStmt>())
          return block.push_back<MatrixPtrStmt>(source, arg(1));
        auto *storage = block.push_back<AllocaStmt>(scalar_type(inst.getOperand(0)->getType()));
        block.push_back<LocalStoreStmt>(storage, source);
        auto *lane = block.push_back<MatrixPtrStmt>(storage, arg(1));
        return block.push_back<LocalLoadStmt>(lane);
      }
      case I::BitCast:
        return cast(block, arg(0), scalar_type(inst.getType()), UnaryOpType::cast_bits);
      case I::Trunc: case I::FPTrunc: case I::FPExt:
      case I::FPToSI: case I::FPToUI:
        return cast(block, arg(0), scalar_type(inst.getType(), inst.getOpcode() != I::FPToUI));
      case I::SExt: case I::ZExt: case I::SIToFP: case I::UIToFP: {
        const bool sign = inst.getOpcode() == I::SExt || inst.getOpcode() == I::SIToFP;
        auto *input = cast(block, arg(0), scalar_type(inst.getOperand(0)->getType(), sign));
        auto *converted = cast(block, input, scalar_type(inst.getType(), sign));
        if (sign && inst.getOperand(0)->getType()->getScalarType()->isIntegerTy(1))
          return block.push_back<UnaryOpStmt>(UnaryOpType::neg, converted);
        return converted;
      }
      case I::UDiv: {
        auto type = scalar_type(inst.getType(), false);
        return block.push_back<BinaryOpStmt>(BinaryOpType::div, cast(block, arg(0), type),
                                             cast(block, arg(1), type));
      }
      case I::Add: case I::FAdd: return binary(BinaryOpType::add);
      case I::Sub: case I::FSub: return binary(BinaryOpType::sub);
      case I::Mul: case I::FMul: return binary(BinaryOpType::mul);
      case I::SDiv: case I::FDiv: return binary(BinaryOpType::div);
      case I::And: return binary(BinaryOpType::bit_and);
      case I::Or: return binary(BinaryOpType::bit_or);
      case I::Xor: return binary(BinaryOpType::bit_xor);
      case I::Shl: return binary(BinaryOpType::bit_shl);
      case I::AShr: return binary(BinaryOpType::bit_sar);
      case I::LShr: return binary(BinaryOpType::bit_shr);
      case I::FNeg: return block.push_back<UnaryOpStmt>(UnaryOpType::neg, arg(0));
      case I::Select: return block.push_back<TernaryOpStmt>(TernaryOpType::select, arg(0), arg(1), arg(2));
      case I::ICmp: case I::FCmp: {
        using P = llvm::CmpInst;
        switch (llvm::cast<P>(inst).getPredicate()) {
          case P::ICMP_EQ: case P::FCMP_OEQ: return binary(BinaryOpType::cmp_eq);
          case P::ICMP_NE: case P::FCMP_ONE: return binary(BinaryOpType::cmp_ne);
          case P::ICMP_SLT: case P::FCMP_OLT: return binary(BinaryOpType::cmp_lt);
          case P::ICMP_SLE: case P::FCMP_OLE: return binary(BinaryOpType::cmp_le);
          case P::ICMP_SGT: case P::FCMP_OGT: return binary(BinaryOpType::cmp_gt);
          case P::ICMP_SGE: case P::FCMP_OGE: return binary(BinaryOpType::cmp_ge);
          case P::ICMP_ULT: case P::ICMP_ULE: case P::ICMP_UGT: case P::ICMP_UGE: {
            auto type = scalar_type(inst.getOperand(0)->getType(), false);
            auto pred = llvm::cast<P>(inst).getPredicate();
            auto op = pred == P::ICMP_ULT ? BinaryOpType::cmp_lt :
                      pred == P::ICMP_ULE ? BinaryOpType::cmp_le :
                      pred == P::ICMP_UGT ? BinaryOpType::cmp_gt : BinaryOpType::cmp_ge;
            return block.push_back<BinaryOpStmt>(op, cast(block, arg(0), type), cast(block, arg(1), type));
          }
          case P::FCMP_UNE:
            if (auto *zero = llvm::dyn_cast<llvm::Constant>(inst.getOperand(1)); zero && zero->isNullValue())
              return cast(block, arg(0), scalar_type(inst.getType()));
            break;
          default: break;
        }
        break;
      }
      default: break;
    }
    TI_ERROR("Unsupported unified LLVM instruction {} in {}", inst.getOpcodeName(), kernel_.get_name());
  }

  BB *range(BB &head, BB *predecessor, llvm::BranchInst &branch, Block &block) {
    auto *index = llvm::dyn_cast<llvm::PHINode>(&head.front());
    auto *condition = llvm::dyn_cast<llvm::ICmpInst>(branch.getCondition());
    TI_ERROR_IF(!index || !condition || condition->getPredicate() != llvm::ICmpInst::ICMP_SLT ||
                condition->getOperand(0) != index || index->getNumIncomingValues() != 2 ||
                !predecessor, "Invalid unified parallel range");
    const int start = index->getBasicBlockIndex(predecessor);
    TI_ERROR_IF(start < 0, "Missing unified range preheader");
    auto *latch = index->getIncomingBlock(1 - start);
    auto *step = llvm::dyn_cast<llvm::BinaryOperator>(index->getIncomingValue(1 - start));
    // The step may also compute a body expression such as data[i + 1].
    // LLVM requires its value to be available on the back edge, including
    // when common-subexpression elimination shares the earlier addition.
    auto is_increment = [&](unsigned slot) {
      return step->getOperand(slot) == index &&
             llvm::PatternMatch::match(step->getOperand(1 - slot), llvm::PatternMatch::m_One());
    };
    TI_ERROR_IF(!step || step->getOpcode() != llvm::Instruction::Add ||
                !(is_increment(0) || is_increment(1)) ||
                !dominators_.dominates(step, latch->getTerminator()) ||
                latch->getSingleSuccessor() != &head || head.size() != 3,
                "Invalid unified range induction");
    auto *policy = branch.getMetadata("forge.range");
    auto item = [&](unsigned i) {
      return literal(llvm::mdconst::extract<llvm::ConstantInt>(policy->getOperand(i)));
    };
    int requested = item(1);
    int width = item(2);
    bool serialized = bool(item(0));
    auto *options = llvm::dyn_cast<llvm::CallInst>(condition->getOperand(1));
    if (options && options->getCalledFunction() &&
        options->getCalledFunction()->getName().starts_with("forge.range.device_options.")) {
      TI_ERROR_IF(options->arg_size() < 4 || (options->arg_size() - 1) % 3 != 0,
                  "Invalid device scheduling options on a unified range");
      for (unsigned i = 1; i < options->arg_size(); i += 3) {
        const auto mask = literal(options->getArgOperand(i));
        TI_ERROR_IF(mask <= 0 || mask > 7, "Invalid device mask on a unified range");
        if (!(mask & portable_target_bit(kernel_.arch)))
          continue;
        auto *operand = options->getArgOperand(i + 2);
        switch (static_cast<PortableLoopOption>(literal(options->getArgOperand(i + 1)))) {
          case PortableLoopOption::block_dim:
            if (llvm::isa<llvm::ConstantInt>(operand)) {
              width = literal(operand);
              TI_ERROR_IF(width <= 0 || (width % 32 && !bit::is_power_of_two(width)),
                          "Invalid unified workgroup size");
              model_block_dim_ = nullptr;
            } else {
              model_block_dim_ = value(block, operand);
            }
            break;
          case PortableLoopOption::parallelize:
            requested = literal(operand);
            break;
          case PortableLoopOption::serialize:
            serialized = true;
            break;
          default:
            TI_ERROR("Unknown unified loop scheduling option");
        }
      }
    }
    // On Vulkan a width resolved from a prepare fact can go directly into
    // LocalSize. Keep an unbound width in the model-workgroup ABI.
    if (kernel_.arch == Arch::vulkan && model_block_dim_) {
      if (auto *fixed = model_block_dim_->cast<ConstStmt>()) {
        const int lanes = fixed->val.val_int32();
        TI_ERROR_IF(lanes <= 0 || (lanes % 32 && !bit::is_power_of_two(lanes)),
                    "Invalid model workgroup size");
        if (lanes <= taichi_max_gpu_block_dim) {
          width = lanes;
          model_block_dim_ = nullptr;
        }
      }
    }
    int threads = kernel_.arch == Arch::cuda ? 1 :
                  requested ? requested : int(std::max(1u, std::thread::hardware_concurrency()));
    Block *body;
    if (options && options->getCalledFunction() &&
        options->getCalledFunction()->getName().starts_with("forge.field.active_count.")) {
      TI_ERROR_IF(options->arg_size() < 3 || options->arg_size() % 3 ||
                      !llvm::PatternMatch::match(index->getIncomingValue(start), llvm::PatternMatch::m_Zero()),
                  "Invalid unified active-list iteration");
      auto node = [&](unsigned i) {
        return storage_.at(literal(options->getArgOperand(i)))
                       .at(literal(options->getArgOperand(i + 1)));
      };
      auto *loop = block.push_back<StructForStmt>(node(1), std::make_unique<Block>(),
                                                  false, threads, width)->as<StructForStmt>();
      for (unsigned i = 3; i < options->arg_size(); i += 3)
        loop->mem_access_opt.add_flag(node(i),
            static_cast<SNodeAccessFlag>(literal(options->getArgOperand(i + 2))));
      struct_loops_.emplace(index, loop);
      body = loop->body.get();
    } else {
      auto *loop = block.push_back<RangeForStmt>(
          value(block, index->getIncomingValue(start)), value(block, condition->getOperand(1)),
          std::make_unique<Block>(), false, threads, width, serialized,
          llvm::cast<llvm::MDString>(policy->getOperand(4))->getString().str())->as<RangeForStmt>();
      loop->requested_cpu_threads = requested;
      loop->model_block_dim = model_block_dim_;
      loop->one_to_one = item(3);
      values_[index] = loop->body->push_back<LoopIndexStmt>(loop, 0);
      body = loop->body.get();
    }
    model_block_dim_ = nullptr;
    range_latches_.push_back(latch);
    region(branch.getSuccessor(0), latch, &head, *body);
    range_latches_.pop_back();
    // Inlining a helper can merge its last body block with its latch. Those
    // instructions still belong to each iteration, before the induction step.
    for (const auto &inst : *latch)
      if (!inst.isTerminator() && (&inst != step || !step->hasOneUse()))
        values_[&inst] = instruction(*body, inst);
    struct_loops_.erase(index);
    return branch.getSuccessor(1);
  }

  BB *branch_join(BB *head, BB *left, BB *right) {
    auto *join = joins_.findNearestCommonDominator(left, right);
    // A continue or break on one nested arm makes the loop latch or exit
    // the postdominator of the entire if. Its two fallthrough paths can
    // still rejoin earlier. Find that first merge of the two dominated arms.
    if (range_latches_.empty() && while_scopes_.empty())
      return join;
    const auto *scope_exit = while_scopes_.empty() ? nullptr : while_scopes_.back().exit;
    const auto *scope_latch = range_latches_.empty() ? nullptr : range_latches_.back();
    // DFS can visit a break-only path before the fallthrough merge. The
    // enclosing exit is a control transfer, not an ordering bound on the body.
    auto best = join && join != scope_exit && join != scope_latch &&
                        dominators_.properlyDominates(head, join)
                    ? block_order_[join] : UINT_MAX;
    for (auto *child : *dominators_.getNode(head)) {
      auto *candidate = child->getBlock();
      auto order = block_order_[candidate];
      if (candidate == scope_exit || candidate == scope_latch ||
          order <= block_order_[head] || order >= best)
        continue;
      bool from_left = candidate == left, from_right = candidate == right;
      for (auto *pred : llvm::predecessors(candidate)) {
        if (block_order_[pred] < order) {
          from_left |= dominators_.dominates(left, pred);
          from_right |= dominators_.dominates(right, pred);
        }
      }
      if (from_left && from_right) {
        best = order;
        join = candidate;
      }
    }
    return join;
  }

  void region(BB *current, BB *end, BB *previous, Block &block) {
    while (current && current != end) {
      if (!range_latches_.empty() && current == range_latches_.back()) {
        block.push_back<ContinueStmt>();
        return;
      }
      auto *loop = loops_.getLoopFor(current);
      if (loop && loop->getHeader() == current &&
          !current->getTerminator()->getMetadata("forge.range") &&
          (while_scopes_.empty() || while_scopes_.back().loop != loop)) {
        // LLVM LoopInfo excludes the instructions on a break-only path.
        // Keep those instructions in this iteration until all exits rejoin.
        llvm::SmallVector<BB *, 4> exits;
        loop->getUniqueExitBlocks(exits);
        BB *exit = exits.empty() ? nullptr : exits.front();
        for (size_t i = 1; i < exits.size() && exit; ++i)
          exit = joins_.findNearestCommonDominator(exit, exits[i]);
        TI_ERROR_IF(!exits.empty() && (!exit || loop->contains(exit)),
                    "Unified LLVM loop exits do not converge in {}", kernel_.get_name());
        auto *body = block.push_back<WhileStmt>(std::make_unique<Block>())->as<WhileStmt>();
        while_scopes_.push_back({loop, exit});
        region(current, exit, previous, *body->body);
        while_scopes_.pop_back();
        previous = nullptr;
        current = exit;
        continue;
      }
      if (!visited_.insert(current).second) {
        std::string detail;
        llvm::raw_string_ostream out(detail);
        for (auto *bb : {current, previous, end}) {
          if (bb) bb->printAsOperand(out, false); else out << "none";
          out << ' ';
        }
        TI_ERROR("Repeated unified LLVM block (current, predecessor, region end) {} in {}",
                 out.str(), kernel_.get_name());
      }
      auto *term = current->getTerminator();
      auto *branch = llvm::dyn_cast<llvm::BranchInst>(term);
      if (branch && branch->getMetadata("forge.range")) {
        auto *next = range(*current, previous, *branch, block);
        previous = current;
        current = next;
        continue;
      }
      for (auto &inst : *current) {
        if (&inst != term)
          values_[&inst] = instruction(block, inst);
      }
      if (llvm::isa<llvm::ReturnInst>(term)) {
        if (!results_.empty())
          block.push_back<ReturnStmt>(std::exchange(results_, {}));
        return;
      }
      TI_ERROR_IF(!branch, "Invalid terminator in unified LLVM kernel");
      previous = current;
      if (branch->isUnconditional()) {
        auto *target = branch->getSuccessor(0);
        if (!while_scopes_.empty()) {
          auto scope = while_scopes_.back();
          if (target == scope.loop->getHeader()) {
            block.push_back<ContinueStmt>();
            return;
          }
          if (target == scope.exit) {
            auto *stop = block.push_back<ConstStmt>(TypedConstant(0));
            block.push_back<WhileControlStmt>(nullptr, stop);
            return;
          }
        }
        current = target;
      } else {
        auto *left = branch->getSuccessor(0), *right = branch->getSuccessor(1);
        if (!while_scopes_.empty()) {
          auto scope = while_scopes_.back();
          if (left == scope.exit || right == scope.exit) {
            auto *condition = value(block, branch->getCondition());
            if (left == scope.exit)
              condition = block.push_back<UnaryOpStmt>(UnaryOpType::logic_not, condition);
            block.push_back<WhileControlStmt>(nullptr, condition);
            current = left == scope.exit ? right : left;
            if (current == scope.loop->getHeader())
              return;
            continue;
          }
        }
        auto *join = branch_join(current, left, right);
        // A nested branch with an early continue can postdominate beyond the
        // enclosing arm. Its fallthrough ends at the enclosing merge.
        if (end && (!join || !dominators_.properlyDominates(current, join) ||
                    dominators_.properlyDominates(end, join)))
          join = end;
        auto *choice = block.push_back<IfStmt>(value(block, branch->getCondition()))->as<IfStmt>();
        choice->set_true_statements(std::make_unique<Block>());
        choice->set_false_statements(std::make_unique<Block>());
        region(left, join, previous, *choice->true_statements);
        region(right, join, previous, *choice->false_statements);
        if (!while_scopes_.empty() &&
            (!join || join == while_scopes_.back().loop->getHeader() ||
             join == while_scopes_.back().exit))
          return;
        current = join;
      }
    }
  }
};
}  // namespace

std::unique_ptr<Kernel> declare_unified_kernel(
    Program &program, const std::string &name, const std::string &root_digest,
    const std::string &signature, const std::map<std::vector<int>, int> &constant_int_args,
    const DeviceCapabilityConfig &device_caps,
    const std::vector<std::vector<SNode *>> &storage) {
  // Include the immutable closure (callee digests are encoded in the root),
  // ABI and the facts substituted by this prepare. Public aliases are bound
  // separately by the caller.
  std::string identity = root_digest + aot_artifact_sha256(signature);
  for (const auto &[path, replacement] : constant_int_args)
    identity += fmt::format("|arg[{}]={}", fmt::join(path, ","), replacement);
  for (const auto &tree : storage) {
    identity += "|tree";
    for (const auto *node : tree) {
      TI_ERROR_IF(!node, "Missing unified LLVM storage binding for {}", name);
      identity += fmt::format("[{},{}]", node->get_snode_tree_id(), node->id);
    }
  }
  auto kernel = std::make_unique<Kernel>(program, std::make_unique<Block>(),
                                         "unified_" + aot_artifact_sha256(identity));
  using namespace liong::json;
  auto description = parse(signature);
  deserialize(description["arguments"], kernel->parameter_list, true);
  const auto arch = program.compile_config().arch;
  for (const auto &param : kernel->parameter_list) {
    TI_ERROR_IF((param.ptype == ParameterType::kAccelerationStructure ||
                 param.ptype == ParameterType::kRWTexture) && arch != Arch::vulkan,
                "Unified entry {} requires Vulkan resource bindings", name);
    TI_ERROR_IF(param.ptype == ParameterType::kTexture &&
                    arch != Arch::cuda && arch != Arch::vulkan,
                "Unified entry {} requires a sampled texture binding", name);
  }
  for (int i = 0; i < kernel->parameter_list.size(); ++i)
    kernel->nested_parameters[{i}] = kernel->parameter_list[i];
  std::vector<std::pair<std::vector<int>, CallableBase::Parameter>> nested;
  deserialize(description["nested"], nested, true);
  for (auto &[path, parameter] : nested)
    kernel->nested_parameters.emplace(std::move(path), std::move(parameter));
  deserialize(description["returns"], kernel->rets, true);
  if (arch == Arch::vulkan)
    kernel->finalize_params_with_layout(spirv::kernel_argument_data_layout(device_caps));
  else
    kernel->finalize_params();
  kernel->finalize_rets();
  // The shared compiler cache adds effective config, device capabilities and
  // ABI to this body identity. Layout words are bound to a private copy of
  // the resulting shader. SNode IDs alone are not persistent layout keys.
  if (arch == Arch::vulkan && storage.empty()) {
    kernel->set_persistent_ir_identity(fmt::format(
        "forge-unified-vulkan-v1/{}/{}/{}/{}/{}", get_version_string(),
        get_commit_hash(), get_forge_native_abi_revision(), LLVM_VERSION_STRING,
        kernel->get_name()));
  }
  return kernel;
}

std::unique_ptr<Kernel> read_unified_kernel(
    Program &program, const std::string &name, const std::string &bitcode,
    const std::string &signature, const std::map<std::vector<int>, int> &constant_int_args,
    const DeviceCapabilityConfig &device_caps,
    const UnifiedUnitLoader &load_helper, const std::vector<std::vector<SNode *>> &storage,
    const std::vector<int> &absent_inputs, bool *needs_bound_device) {
  TI_COMPILE_PROFILER("cpp.unified.prepare.read_kernel");
  llvm::LLVMContext context;
  auto module = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bitcode, name), context);
  if (!module)
    TI_ERROR("Invalid unified LLVM unit: {}", llvm::toString(module.takeError()));
  auto *entry = (*module)->getFunction("entry");
  TI_ERROR_IF(!entry || entry->isDeclaration() || entry->arg_size() != 1 ||
              !entry->getReturnType()->isVoidTy() || llvm::verifyModule(**module),
              "Invalid unified LLVM kernel {}", name);
  auto kernel = declare_unified_kernel(program, name, aot_artifact_sha256(bitcode),
                                       signature, constant_int_args, device_caps, storage);
  const auto arch = kernel->arch;
  substitute_bound_arguments(*entry, *kernel, constant_int_args);
  if (!constant_int_args.empty())
    fold_empty_ranges(*entry);
  import_helpers(**module, *entry, load_helper, arch, device_caps);
  // A helper can contain a folded break or continue. Remove impossible
  // edges before computing loop exits and branch convergence.
  for (auto &bb : *entry)
    llvm::ConstantFoldTerminator(&bb, true);
  llvm::removeUnreachableBlocks(*entry);
  // Recovering a forwarded array can leave the original descriptor unpack
  // unused. Remove dead address computations before the target sees them.
  llvm::SmallVector<llvm::WeakTrackingVH, 16> dead;
  for (auto &bb : *entry)
    for (auto &inst : bb)
      if (llvm::isInstructionTriviallyDead(&inst))
        dead.emplace_back(&inst);
  llvm::RecursivelyDeleteTriviallyDeadInstructions(dead);
  promote_block_locals(*entry);
  fold_empty_ranges(*entry);
  restore_parallel_locals(*entry);
  if (!absent_inputs.empty()) {
    for (int slot : absent_inputs) {
      TI_ERROR_IF(slot < 0 || size_t(slot) >= kernel->parameter_list.size() ||
                      kernel->parameter_list[slot].ptype != ParameterType::kNdarray,
                  "Invalid unavailable resource slot {} in {}", slot, name);
    }
    // Check the actual specialized body, including every inlined helper.
    // Only dead bindings may receive a zero-work descriptor.
    for (auto &bb : *entry) {
      for (auto &inst : bb) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (!call || call->use_empty() || !call->getCalledFunction() ||
            !call->getCalledFunction()->getName().starts_with("forge.arg."))
          continue;
        const int slot = literal(call->getArgOperand(2));
        TI_ERROR_IF(std::find(absent_inputs.begin(), absent_inputs.end(), slot) != absent_inputs.end(),
                    "Unified entry {} uses unavailable resource slot {} on {}", name, slot, arch_name(arch));
      }
    }
  }
  if (program.compile_config().print_ir)
    entry->print(llvm::errs());
  {
    TI_COMPILE_PROFILER("cpp.unified.prepare.reconstruct_ir");
    KernelReader reader(*kernel, *entry, constant_int_args, storage, needs_bound_device);
  }
  return kernel;
}
}  // namespace taichi::lang::LLVM
