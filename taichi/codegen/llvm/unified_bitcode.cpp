#include "taichi/codegen/llvm/unified_bitcode.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/DCE.h"

#include "taichi/ir/analysis.h"
#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/analysis/ndarray_descriptor_origin.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/llvm/function_code_cache.h"
#include "taichi/transforms/lower_matrix_ptr.h"

namespace taichi::lang::LLVM {
namespace {

using UnitMap = std::map<std::string, std::string>;

// Function objects are valid throughout one root's capture. A blank digest
// means its body is currently being emitted. Never retain these pointers when
// the producer releases the root and its specialized definitions.
using LocalFunctions = std::unordered_map<const Function *, std::string>;

bool is_floating_point_computation(const llvm::Instruction &inst) {
  if (!inst.isBinaryOp() &&
      !llvm::isa<llvm::UnaryOperator, llvm::FCmpInst, llvm::CastInst>(inst))
    return false;
  return inst.getType()->isFPOrFPVectorTy() ||
         inst.getOperand(0)->getType()->isFPOrFPVectorTy();
}

void simplify_private_computations(llvm::Module &module) {
  auto &entry = *module.getFunction("entry");
  // A portable range still represents parallel work. Serial forwarding of
  // device memory operations requires the tasks formed later by prepare.
  llvm::SmallVector<llvm::Instruction *, 16> device_accesses;
  // Sharing an FP multiply before target lowering can change which add the
  // device contracts with it. Keep each source operation opaque to EarlyCSE,
  // while still allowing forwarding of its private inputs and outputs.
  llvm::DenseMap<llvm::Type *, llvm::Function *> markers;
  llvm::SmallVector<std::pair<llvm::CallInst *, llvm::Instruction *>, 16> math;
  auto *id_type = llvm::Type::getInt32Ty(module.getContext());
  auto protect = [&](auto *access) {
    if (access->isSimple() &&
        !llvm::isa<llvm::AllocaInst>(llvm::getUnderlyingObject(access->getPointerOperand()))) {
      access->setVolatile(true);
      device_accesses.push_back(access);
    }
  };
  for (auto &block : entry) {
    for (auto &inst : llvm::make_early_inc_range(block)) {
      if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
        protect(load);
      else if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
        protect(store);
      else if (is_floating_point_computation(inst)) {
        auto &marker = markers[inst.getType()];
        if (!marker) {
          marker = llvm::Function::Create(
              llvm::FunctionType::get(inst.getType(), {id_type}, true),
              llvm::GlobalValue::ExternalLinkage, "forge.preserve.fp", module);
          marker->setDoesNotAccessMemory();
          marker->setDoesNotThrow();
        }
        llvm::SmallVector<llvm::Value *, 4> operands{
            llvm::ConstantInt::get(id_type, math.size())};
        operands.append(inst.value_op_begin(), inst.value_op_end());
        auto *call = llvm::CallInst::Create(marker, operands, "", inst.getIterator());
        inst.replaceAllUsesWith(call);
        inst.removeFromParent();
        inst.dropAllReferences();
        math.emplace_back(call, &inst);
      }
    }
  }
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager scc;
  llvm::ModuleAnalysisManager modules;
  llvm::PassBuilder passes;
  passes.registerLoopAnalyses(loops);
  passes.registerFunctionAnalyses(functions);
  passes.registerCGSCCAnalyses(scc);
  passes.registerModuleAnalyses(modules);
  passes.crossRegisterProxies(loops, functions, scc, modules);
  llvm::FunctionPassManager pipeline;
  pipeline.addPass(llvm::EarlyCSEPass());
  pipeline.addPass(llvm::DCEPass());
  pipeline.run(entry, functions);
  for (auto [call, inst] : math) {
    for (unsigned i = 0; i < inst->getNumOperands(); ++i)
      inst->setOperand(i, call->getArgOperand(i + 1));
    inst->insertBefore(call->getIterator());
    call->replaceAllUsesWith(inst);
    call->eraseFromParent();
  }
  for (auto &marker : markers)
    marker.second->eraseFromParent();
  for (auto *access : device_accesses) {
    if (auto *load = llvm::dyn_cast<llvm::LoadInst>(access))
      load->setVolatile(false);
    else
      llvm::cast<llvm::StoreInst>(access)->setVolatile(false);
  }
}

std::string serialize(llvm::Module &module, bool simplify) {
  if (simplify)
    simplify_private_computations(module);
  std::string error;
  llvm::raw_string_ostream errors(error);
  TI_ERROR_IF(llvm::verifyModule(module, &errors),
              "Invalid unified LLVM IR: {}", error);
  std::string bytes;
  llvm::raw_string_ostream out(bytes);
  llvm::WriteBitcodeToFile(module, out);
  return bytes;
}

std::unique_ptr<IRNode> lower_body(Callable &callable, bool ast) {
  auto root = irpass::analysis::clone(callable.ir.get());
  if (ast) {
    irpass::frontend_type_check(root.get());
    irpass::lower_ast(root.get(), true);
  }
  const auto &config = callable.program->compile_config();
  irpass::eliminate_immutable_local_vars(root.get());
  irpass::type_check(root.get(), config);
  irpass::scalarize_matrix_ptr_accesses(root.get());
  irpass::type_check(root.get(), config);
  return root;
}

// Operate before offload scheduling, SNode lowering and target data layout.
// Port operations carry logical indices. The consumer supplies their ABI.
class UnitWriter final : public IRVisitor {
 public:
  UnitWriter(llvm::Module &module,
             UnitMap &helpers,
             UnitMap &identities,
             const std::vector<int> &tree_ids,
             const std::map<int, int> &node_ordinals,
             LocalFunctions &functions,
             bool simplify,
             Function *callee = nullptr)
      : module_(module),
        ctx_(module.getContext()),
        b_(ctx_),
        helpers_(helpers),
        identities_(identities),
        tree_ids_(tree_ids),
        node_ordinals_(node_ordinals),
        functions_(functions),
        callee_(callee),
        simplify_(simplify) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    // These contexts belong to the archive writer. Prepare uses CFG edges
    // and scheduling metadata; only global names identify callable units.
    ctx_.setDiscardValueNames(true);
    std::vector<llvm::Type *> arguments(callee ? 3 : 1, b_.getPtrTy());
    auto *signature = llvm::FunctionType::get(b_.getVoidTy(), arguments, false);
    entry_ = llvm::Function::Create(signature, llvm::Function::ExternalLinkage,
                                    "entry", module_);
    b_.SetInsertPoint(llvm::BasicBlock::Create(ctx_, "entry", entry_));
  }

  void write(Block &root) {
    root.accept(this);
    if (!b_.GetInsertBlock()->getTerminator())
      b_.CreateRetVoid();
  }

 private:
  llvm::Module &module_;
  llvm::LLVMContext &ctx_;
  llvm::IRBuilder<> b_;
  UnitMap &helpers_;
  UnitMap &identities_;
  const std::vector<int> &tree_ids_;
  const std::map<int, int> &node_ordinals_;
  LocalFunctions &functions_;
  Function *callee_;
  bool simplify_;
  llvm::Function *entry_;
  std::unordered_map<const Stmt *, llvm::Value *> values_;
  std::unordered_map<const Stmt *, llvm::Value *> indices_;
  std::vector<std::pair<llvm::BasicBlock *, llvm::BasicBlock *>> loops_;
  unsigned shared_allocations_{0};
  const Stmt *current_{nullptr};
  using IRVisitor::visit;

  llvm::Type *type(DataType dt) {
    if (dt.is_pointer())
      return b_.getPtrTy();
    if (auto *tensor = dt->cast<TensorType>())
      return llvm::FixedVectorType::get(type(tensor->get_element_type()),
                                        tensor->get_num_elements());
    if (auto *record = dt->cast<StructType>()) {
      std::vector<llvm::Type *> members;
      for (const auto &field : record->elements())
        members.push_back(type(field.type));
      return llvm::StructType::get(ctx_, members);
    }
    if (dt == PrimitiveType::f32)
      return b_.getFloatTy();
    if (dt == PrimitiveType::f64)
      return b_.getDoubleTy();
    if (dt == PrimitiveType::f16)
      return b_.getHalfTy();
    if (dt == PrimitiveType::u1)
      return b_.getInt1Ty();
    if (is_integral(dt))
      return b_.getIntNTy(data_type_bits(dt));
    TI_ERROR("Unified LLVM value type is not lowered: {} in {}", dt->to_string(),
             current_ ? typeid(*current_).name() : "function ABI");
  }

  llvm::Value *value(const Stmt *stmt) {
    return values_.at(stmt);
  }

  llvm::StructType *argument_frame(const Function &fn) {
    const auto &fields = fn.args_type->elements();
    TI_ASSERT(fields.size() == fn.parameter_list.size());
    std::vector<llvm::Type *> types;
    for (size_t i = 0; i < fields.size(); ++i) {
      types.push_back(fn.parameter_list[i].ptype == ParameterType::kNdarray
                          ? b_.getPtrTy()
                          : type(fields[i].type));
    }
    return llvm::StructType::get(ctx_, types);
  }

  llvm::Value *array_descriptor(const std::vector<int> &path) {
    if (!callee_) {
      std::vector<llvm::Value *> indices{entry_->getArg(0), b_.getInt32(0)};
      for (int slot : path)
        indices.push_back(b_.getInt32(slot));
      return operation("arg.address", b_.getPtrTy(), indices);
    }
    TI_ERROR_IF(path.size() != 1 || callee_->parameter_list.at(path[0]).ptype !=
                                        ParameterType::kNdarray,
                "Invalid unified LLVM helper array descriptor");
    auto *address = b_.CreateStructGEP(argument_frame(*callee_),
                                       entry_->getArg(1), path[0]);
    return b_.CreateLoad(b_.getPtrTy(), address);
  }

  llvm::Value *operation(std::string name,
                         llvm::Type *result,
                         llvm::ArrayRef<llvm::Value *> args) {
    std::vector<llvm::Type *> formals;
    std::string types;
    llvm::raw_string_ostream out(types);
    result->print(out);
    for (auto *arg : args) {
      formals.push_back(arg->getType());
      out << ".";
      arg->getType()->print(out);
    }
    auto fn = module_.getOrInsertFunction(
        "forge." + name + "." + types,
        llvm::FunctionType::get(result, formals, false));
    return b_.CreateCall(fn, args);
  }

  void visit(Stmt *stmt) override {
    TI_ERROR("Unified LLVM operation is not lowered: {}", typeid(*stmt).name());
  }
  void visit(Block *block) override {
    for (auto &stmt : block->statements) {
      if (b_.GetInsertBlock()->getTerminator())
        break;
      current_ = stmt.get();
      stmt->accept(this);
    }
  }
  void visit(ConstStmt *s) override {
    auto *t = type(s->ret_type);
    values_[s] = is_real(s->ret_type)
                     ? llvm::ConstantFP::get(t, s->val.val_float())
                     : llvm::ConstantInt::get(t, s->val.val_as_int64());
  }
  void visit(ArgLoadStmt *s) override {
    if (callee_) {
      TI_ERROR_IF(s->arg_depth,
                  "Nested argument packs in a unified LLVM helper require a "
                  "descriptor");
      if (!s->create_load && s->is_ptr && s->arg_id.size() == 1 &&
          callee_->parameter_list.at(s->arg_id[0]).ptype ==
              ParameterType::kNdarray) {
        values_[s] = array_descriptor(s->arg_id);
        return;
      }
      auto *record = argument_frame(*callee_);
      std::vector<llvm::Value *> indices{b_.getInt32(0)};
      for (int index : s->arg_id)
        indices.push_back(b_.getInt32(index));
      auto *address = b_.CreateGEP(record, entry_->getArg(1), indices);
      values_[s] =
          s->create_load
              ? b_.CreateLoad(
                    type(callee_->args_type->get_element_type(s->arg_id)),
                    address)
              : address;
      return;
    }
    std::vector<llvm::Value *> path{entry_->getArg(0),
                                    b_.getInt32(s->arg_depth)};
    for (auto slot : s->arg_id)
      path.push_back(b_.getInt32(slot));
    auto *ret =
        s->is_ptr || !s->create_load ? b_.getPtrTy() : type(s->ret_type);
    values_[s] = operation(s->create_load ? "arg" : "arg.address", ret, path);
  }
  void visit(AllocaStmt *s) override {
    auto *t = type(s->ret_type.ptr_removed());
    if (s->is_shared) {
      // Each declaration names distinct storage within a task. Its workgroup
      // address space and lifetime are selected when the task is materialized.
      values_[s] = operation(
          "workgroup.storage." + s->ret_type.ptr_removed()->to_string(),
          b_.getPtrTy(),
          {entry_->getArg(0), b_.getInt32(shared_allocations_++)});
      return;
    }
    llvm::IRBuilder<> allocation(&entry_->getEntryBlock(),
                                 entry_->getEntryBlock().begin());
    values_[s] = allocation.CreateAlloca(t);
    b_.CreateStore(llvm::Constant::getNullValue(t), value(s));
  }
  void visit(LocalLoadStmt *s) override {
    values_[s] = b_.CreateLoad(type(s->ret_type), value(s->src));
  }
  void visit(GlobalLoadStmt *s) override {
    values_[s] =
        b_.CreateAlignedLoad(type(s->ret_type), value(s->src), llvm::Align(1));
  }
  void visit(LocalStoreStmt *s) override {
    b_.CreateStore(value(s->val), value(s->dest));
  }
  void visit(GlobalStoreStmt *s) override {
    b_.CreateAlignedStore(value(s->val), value(s->dest), llvm::Align(1));
  }
  void visit(ReferenceStmt *s) override {
    values_[s] = value(s->var);
  }
  void visit(DynamicSharedPtrStmt *s) override {
    std::vector<llvm::Value *> args{
        entry_->getArg(0), b_.getInt32(s->allocation_id), value(s->offset)};
    for (auto *extent : s->shape)
      args.push_back(value(extent));
    values_[s] =
        operation("workgroup.element." + s->ret_type.ptr_removed()->to_string(),
                  b_.getPtrTy(), args);
  }
  void visit(GlobalThreadIndexStmt *s) override {
    values_[s] =
        operation("thread.index", type(s->ret_type), {entry_->getArg(0)});
  }
  void visit(AssertStmt *s) override {
    std::vector<llvm::Value *> args{entry_->getArg(0), value(s->cond)};
    for (const auto *arg : s->args)
      args.push_back(value(arg));
    auto *call =
        llvm::cast<llvm::CallInst>(operation("assert", b_.getVoidTy(), args));
    call->setMetadata(
        "forge.message",
        llvm::MDNode::get(ctx_, llvm::MDString::get(ctx_, s->text)));
  }
  void visit(PrintStmt *s) override {
    // Keep one ordered side effect. Target-specific printf promotion and
    // literal-string storage belong to the device materializer.
    std::vector<llvm::Value *> args{entry_->getArg(0)};
    std::vector<llvm::Metadata *> items;
    for (size_t i = 0; i < s->contents.size(); ++i) {
      std::vector<llvm::Metadata *> item;
      if (auto *operand = std::get_if<Stmt *>(&s->contents[i])) {
        item.push_back(llvm::MDString::get(ctx_, "value"));
        item.push_back(llvm::MDString::get(ctx_, (*operand)->ret_type->to_string()));
        args.push_back(value(*operand));
      } else {
        item.push_back(llvm::MDString::get(ctx_, "text"));
        item.push_back(llvm::MDString::get(ctx_, std::get<std::string>(s->contents[i])));
      }
      if (!s->formats.empty() && s->formats.at(i))
        item.push_back(llvm::MDString::get(ctx_, *s->formats[i]));
      items.push_back(llvm::MDNode::get(ctx_, item));
    }
    auto *call = llvm::cast<llvm::CallInst>(operation("print", b_.getVoidTy(), args));
    call->setMetadata("forge.print.items", llvm::MDNode::get(ctx_, items));
  }
  void visit(AccelerationStructurePtrStmt *s) override {
    values_[s] = operation("acceleration_structure", b_.getPtrTy(),
                           {value(s->arg_load_stmt)});
  }
  void visit(TexturePtrStmt *s) override {
    std::vector<llvm::Value *> args{
        value(s->arg_load_stmt),   b_.getInt32(s->dimensions),
        b_.getInt1(s->is_storage), b_.getInt32(static_cast<int>(s->format)),
        b_.getInt32(s->lod),       b_.getInt32(s->collection_capacity)};
    if (s->collection_index)
      args.push_back(value(s->collection_index));
    values_[s] = operation("image.binding", b_.getPtrTy(), args);
  }
  void visit(TextureOpStmt *s) override {
    std::vector<llvm::Value *> args{value(s->texture_ptr)};
    for (const auto *arg : s->args)
      args.push_back(value(arg));
    // TextureOpStmt is an opaque composite in CHI. Its consumers extract
    // channels; its scalar format comes from the binding, not stmt.ret_type.
    auto *binding = s->texture_ptr->as<TexturePtrStmt>();
    auto element = binding->is_storage
                       ? buffer_format2storage_image_sampled_type(binding->format)
                       : DataType(PrimitiveType::f32);
    llvm::Type *result = b_.getVoidTy();
    if (s->op != TextureOpType::kStore)
      result = llvm::FixedVectorType::get(type(element), 4);
    values_[s] = operation("image." + texture_op_type_name(s->op), result, args);
  }
  void visit(DecorationStmt *s) override {
    // Keep loop uniqueness available to prepare, including when its operand
    // is a constant or an argument rather than an attachable instruction.
    std::vector<llvm::Value *> args{value(s->operand)};
    for (auto item : s->decoration)
      args.push_back(b_.getInt32(item));
    operation("hint.decoration", b_.getVoidTy(), args);
  }
  void visit(RangeAssumptionStmt *s) override {
    auto *input = value(s->input);
    operation(
        "hint.range", b_.getVoidTy(),
        {input, value(s->base), b_.getInt32(s->low), b_.getInt32(s->high)});
    values_[s] = input;
  }
  void visit(InternalFuncStmt *s) override {
    std::vector<llvm::Value *> args;
    if (s->with_runtime_context)
      args.push_back(entry_->getArg(0));
    for (const auto *arg : s->args)
      args.push_back(value(arg));
    auto *call = llvm::cast<llvm::CallInst>(
        operation("builtin." + s->func_name, type(s->ret_type), args));
    if (s->ret_type->is<StructType>()) {
      // Hardware query results are records by value. Keep field types and
      // names for the backend's existing composite extraction and liveness.
      auto description = liong::json::print(liong::json::serialize(s->ret_type));
      call->setMetadata("forge.record.type", llvm::MDNode::get(ctx_, llvm::MDString::get(ctx_, description)));
    }
    values_[s] = call;
  }
  void visit(RandStmt *s) override {
    values_[s] = operation("random." + s->ret_type->to_string(),
                           type(s->ret_type), {entry_->getArg(0)});
  }
  llvm::AllocaInst *private_storage(llvm::Type *t) {
    llvm::IRBuilder<> at_entry(&entry_->getEntryBlock(),
                               entry_->getEntryBlock().begin());
    return at_entry.CreateAlloca(t);
  }

  // Real-function inputs and returns are flattened in frontend IR. Rebuild
  // each private frame by logical member index, never by a producer byte
  // offset.
  void fill_frame(llvm::Type *t,
                  llvm::Value *dst,
                  const std::vector<Stmt *> &leaves,
                  size_t &next) {
    if (auto *record = llvm::dyn_cast<llvm::StructType>(t)) {
      for (unsigned i = 0; i < record->getNumElements(); ++i)
        fill_frame(record->getElementType(i), b_.CreateStructGEP(t, dst, i),
                   leaves, next);
    } else if (auto *vec = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
      for (unsigned i = 0; i < vec->getNumElements(); ++i)
        fill_frame(vec->getElementType(),
                   b_.CreateGEP(t, dst, {b_.getInt32(0), b_.getInt32(i)}),
                   leaves, next);
    } else {
      TI_ERROR_IF(next >= leaves.size() || value(leaves[next])->getType() != t,
                  "Unified LLVM private frame argument type mismatch at {}",
                  next);
      b_.CreateStore(value(leaves[next++]), dst);
    }
  }
  void fill_frame(llvm::Type *t,
                  llvm::Value *dst,
                  const std::vector<Stmt *> &leaves) {
    size_t next = 0;
    fill_frame(t, dst, leaves, next);
    TI_ERROR_IF(next != leaves.size(),
                "Too many unified LLVM private frame arguments");
  }

  std::string helper_digest(Function &fn) {
    if (auto it = functions_.find(&fn); it != functions_.end()) {
      TI_ERROR_IF(it->second.empty(), "Recursive unified LLVM helper {}",
                  fn.get_name());
      return it->second;
    }
    TI_ERROR_IF(!fn.ir || fn.ir_stage() > Function::IRStage::InitialIR,
                "Unified LLVM requires a pre-device helper body: {}",
                fn.get_name());
    functions_.emplace(&fn, std::string{});
    auto key = fn.ir_stage() == Function::IRStage::AST
                   ? unified_leaf_identity(fn, simplify_)
                   : std::string{};
    if (!key.empty())
      key = "helper:" + key;
    auto found = identities_.find(key);
    std::string digest;
    if (!key.empty() && found != identities_.end()) {
      digest = found->second;
    } else {
      llvm::LLVMContext context;
      llvm::Module unit("forge.unified.helper", context);
      auto ir = lower_body(fn, fn.ir_stage() == Function::IRStage::AST);
      UnitWriter(unit, helpers_, identities_, tree_ids_, node_ordinals_,
                 functions_, simplify_, &fn)
          .write(*ir->as<Block>());
      auto bytes = serialize(unit, simplify_);
      digest = aot_artifact_sha256(bytes);
      helpers_.try_emplace(digest, std::move(bytes));
      if (!key.empty())
        identities_.emplace(key, digest);
    }
    functions_.at(&fn) = digest;
    return digest;
  }
  void visit(FuncCallStmt *s) override {
    auto &fn = *s->func;
    const auto digest = helper_digest(fn);
    auto *inputs = argument_frame(fn);
    auto *args = private_storage(inputs);
    auto *results = private_storage(type(fn.ret_type));
    size_t next = 0;
    for (unsigned i = 0; i < inputs->getNumElements(); ++i) {
      auto *dst = b_.CreateStructGEP(inputs, args, i);
      const auto &parameter = fn.parameter_list.at(i);
      if (parameter.ptype != ParameterType::kNdarray || fn.backend_inline()) {
        fill_frame(inputs->getElementType(i), dst, s->args, next);
        continue;
      }
      const int ndim = parameter.total_dim - parameter.element_shape.size();
      const size_t prefix = TypeFactory::affine_mode_pos_in_ndarray(ndim) + 1;
      const auto leaf = [&](std::vector<int> path) -> Stmt * {
        size_t offset = path.size() == 2 ? path[1] :
                        prefix + (path[0] == TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
        return next + offset < s->args.size() ? s->args[next + offset] : nullptr;
      };
      ExternalTensorBasePtrStmt *pointer = nullptr, *gradient = nullptr;
      TI_ERROR_IF(!irpass::analysis::forwarded_ndarray(parameter, leaf, pointer, gradient),
                  "Invalid ndarray descriptor in unified helper {}", fn.get_name());
      b_.CreateStore(array_descriptor(pointer->arg_id), dst);
      next += prefix + 1 + parameter.needs_grad;
    }
    TI_ERROR_IF(next != s->args.size(), "Unified LLVM helper argument count mismatch");
    auto *sig = llvm::FunctionType::get(
        b_.getVoidTy(), {b_.getPtrTy(), b_.getPtrTy(), b_.getPtrTy()}, false);
    b_.CreateCall(module_.getOrInsertFunction("forge.fn." + digest, sig),
                  {entry_->getArg(0), args, results});
    values_[s] = results;
  }
  void visit(GetElementStmt *s) override {
    auto *base = value(s->src);
    if (!base->getType()->isPointerTy()) {
      auto *element = base;
      for (int index : s->index) {
        if (element->getType()->isVectorTy())
          element = b_.CreateExtractElement(element, b_.getInt32(index));
        else
          element = b_.CreateExtractValue(element, {static_cast<unsigned>(index)});
      }
      if (s->is_ptr) {
        auto *address = private_storage(element->getType());
        b_.CreateStore(element, address);
        element = address;
      }
      values_[s] = element;
      return;
    }
    std::vector<llvm::Value *> indices{b_.getInt32(0)};
    for (int i : s->index)
      indices.push_back(b_.getInt32(i));
    auto *address = b_.CreateGEP(type(s->src->ret_type.ptr_removed()),
                                   base, indices);
    values_[s] = s->ret_type.is_pointer()
                     ? address
                     : b_.CreateLoad(type(s->ret_type), address);
  }
  void visit(ExternalTensorShapeAlongAxisStmt *s) override {
    values_[s] = operation("array.extent", type(s->ret_type),
                           {array_descriptor(s->arg_id), b_.getInt32(s->axis)});
  }
  void visit(ExternalTensorBasePtrStmt *s) override {
    values_[s] =
        operation("array.base", type(s->ret_type),
                  {array_descriptor(s->arg_id), b_.getInt1(s->is_grad)});
  }
  std::vector<llvm::Value *> tree_binding(const SNode &snode) {
    const int tree = snode.get_snode_tree_id();
    auto found = std::lower_bound(tree_ids_.begin(), tree_ids_.end(), tree);
    TI_ERROR_IF(found == tree_ids_.end() || *found != tree,
                "Missing unified LLVM storage binding");
    return {entry_->getArg(0), b_.getInt32(found - tree_ids_.begin()),
            b_.getInt32(node_ordinals_.at(snode.id))};
  }
  void visit(GlobalPtrStmt *s) override {
    auto args = tree_binding(*s->snode);
    args.push_back(b_.getInt1(s->activate));
    args.push_back(b_.getInt1(s->is_cell_access));
    args.push_back(b_.getInt1(s->is_bit_vectorized));
    for (auto *index : s->indices)
      args.push_back(value(index));
    values_[s] =
        operation("field.address." + s->ret_type.ptr_removed()->to_string(),
                  b_.getPtrTy(), args);
  }
  void visit(SNodeOpStmt *s) override {
    TI_ERROR_IF(!SNodeOpStmt::activation_related(s->op_type) || s->val,
                "Unsupported unified SNode operation {}", snode_op_type_name(s->op_type));
    auto args = tree_binding(*s->snode);
    args.push_back(b_.getInt32(static_cast<int>(s->op_type)));
    args.push_back(value(s->ptr));
    values_[s] = operation("field.activation", type(s->ret_type), args);
  }
  void visit(MatrixInitStmt *s) override {
    llvm::Value *tensor = llvm::PoisonValue::get(type(s->ret_type));
    for (size_t i = 0; i < s->values.size(); ++i)
      tensor = b_.CreateInsertElement(tensor, value(s->values[i]), i);
    values_[s] = tensor;
  }
  // Whole-value uses were scalarized. Indexing below selects only the
  // requested member; the other SNodes must not be activated by the lookup.
  void visit(MatrixOfGlobalPtrStmt *) override {
  }
  void visit(MatrixOfMatrixPtrStmt *) override {
  }
  void visit(MatrixPtrStmt *s) override {
    if (auto *matrix = s->origin->cast<MatrixOfGlobalPtrStmt>()) {
      std::vector<llvm::Value *> args{entry_->getArg(0), value(s->offset),
                                      b_.getInt1(matrix->activate),
                                      b_.getInt32(matrix->snodes.size())};
      for (const auto *member : matrix->snodes) {
        auto binding = tree_binding(*member);
        args.insert(args.end(), binding.begin() + 1, binding.end());
      }
      for (auto *coord : matrix->indices)
        args.push_back(value(coord));
      values_[s] =
          operation("field.component." + s->ret_type.ptr_removed()->to_string(),
                    b_.getPtrTy(), args);
    } else if (auto *matrix = s->origin->cast<MatrixOfMatrixPtrStmt>()) {
      auto *lanes =
          llvm::FixedVectorType::get(b_.getPtrTy(), matrix->stmts.size());
      llvm::Value *pointers = llvm::PoisonValue::get(lanes);
      for (size_t i = 0; i < matrix->stmts.size(); ++i)
        pointers = b_.CreateInsertElement(pointers, value(matrix->stmts[i]), i);
      values_[s] = b_.CreateExtractElement(pointers, value(s->offset));
    } else if (s->origin->is<ExternalPtrStmt>()) {
      // The binding may interleave the tensor components. Preserve their
      // logical coordinates until prepare has a concrete descriptor.
      std::vector<llvm::Value *> args{value(s->origin), value(s->offset)};
      for (auto *index : s->component_indices)
        args.push_back(value(index));
      values_[s] = operation(
          "array.component." + s->origin->ret_type.ptr_removed()->to_string(),
          b_.getPtrTy(), args);
    } else {
      auto *unit = s->offset_used_as_index() ? type(s->ret_type.ptr_removed())
                                             : b_.getInt8Ty();
      values_[s] = b_.CreateGEP(unit, value(s->origin), value(s->offset));
    }
  }
  void visit(TernaryOpStmt *s) override {
    TI_ASSERT(s->op_type == TernaryOpType::select);
    values_[s] = b_.CreateSelect(b_.CreateIsNotNull(value(s->op1)),
                                 value(s->op2), value(s->op3));
  }
  void visit(AtomicOpStmt *s) override {
    values_[s] =
        operation("atomic." + atomic_op_type_name(s->op_type) + "." +
                      s->ret_type->to_string(),
                  type(s->ret_type),
                  {value(s->dest), value(s->val), b_.getInt1(s->is_reduction)});
  }
  void visit(ExternalPtrStmt *s) override {
    std::vector<llvm::Value *> args{value(s->base_ptr),
                                    b_.getInt32(s->ndim),
                                    b_.getInt1(s->is_grad),
                                    b_.getInt32(static_cast<int>(s->boundary)),
                                    b_.getInt64(s->byte_offset),
                                    b_.getInt64(s->byte_stride),
                                    b_.getInt1(s->runtime_affine),
                                    b_.getInt32(s->element_shape.size())};
    for (auto extent : s->element_shape)
      args.push_back(b_.getInt32(extent));
    for (auto *index : s->indices)
      args.push_back(value(index));
    values_[s] =
        operation("array.address." + s->ret_type.ptr_removed()->to_string(),
                  b_.getPtrTy(), args);
  }
  void visit(UnaryOpStmt *s) override {
    auto *x = value(s->operand);
    if (s->op_type == UnaryOpType::cast_value) {
      auto *t = type(s->ret_type);
      auto src = s->operand->ret_type.get_element_type();
      auto dst = s->cast_type.get_element_type();
      if (t == x->getType())
        values_[s] = x;
      else if (dst == PrimitiveType::u1) {
        values_[s] = is_real(src)
                         ? b_.CreateFCmpUNE(
                               x, llvm::Constant::getNullValue(x->getType()))
                         : b_.CreateIsNotNull(x);
      } else if (is_real(src) && is_real(dst))
        values_[s] = b_.CreateFPCast(x, t);
      else if (is_real(src))
        values_[s] =
            is_signed(dst) ? b_.CreateFPToSI(x, t) : b_.CreateFPToUI(x, t);
      else if (is_real(dst))
        values_[s] =
            is_signed(src) ? b_.CreateSIToFP(x, t) : b_.CreateUIToFP(x, t);
      else
        values_[s] = b_.CreateIntCast(x, t, is_signed(src));
    } else if (s->op_type == UnaryOpType::cast_bits) {
      values_[s] = b_.CreateBitCast(x, type(s->ret_type));
    } else if (s->op_type == UnaryOpType::neg) {
      values_[s] =
          x->getType()->isFPOrFPVectorTy() ? b_.CreateFNeg(x) : b_.CreateNeg(x);
    } else if (s->op_type == UnaryOpType::bit_not) {
      values_[s] = b_.CreateNot(x);
    } else if (s->op_type == UnaryOpType::logic_not) {
      values_[s] = b_.CreateIsNull(x);
    } else {
      values_[s] = operation("unary." + unary_op_type_name(s->op_type) + "." +
                                 s->operand->ret_type->to_string(),
                             type(s->ret_type), {x});
    }
  }
  void visit(BinaryOpStmt *s) override {
    auto *x = value(s->lhs), *y = value(s->rhs);
    bool fp = x->getType()->isFPOrFPVectorTy();
    bool signed_int = !fp && is_signed(s->lhs->ret_type.get_element_type());
    auto compare = [&](llvm::CmpInst::Predicate floating,
                       llvm::CmpInst::Predicate signed_pred,
                       llvm::CmpInst::Predicate unsigned_pred) {
      return fp ? b_.CreateFCmp(floating, x, y)
                : b_.CreateICmp(signed_int ? signed_pred : unsigned_pred, x, y);
    };
    using P = llvm::CmpInst;
    switch (s->op_type) {
      case BinaryOpType::add:
        values_[s] = fp ? b_.CreateFAdd(x, y) : b_.CreateAdd(x, y);
        break;
      case BinaryOpType::sub:
        values_[s] = fp ? b_.CreateFSub(x, y) : b_.CreateSub(x, y);
        break;
      case BinaryOpType::mul:
        values_[s] = fp ? b_.CreateFMul(x, y) : b_.CreateMul(x, y);
        break;
      case BinaryOpType::div:
        values_[s] =
            fp ? b_.CreateFDiv(x, y)
               : (signed_int ? b_.CreateSDiv(x, y) : b_.CreateUDiv(x, y));
        break;
      case BinaryOpType::cmp_eq:
        values_[s] = compare(P::FCMP_OEQ, P::ICMP_EQ, P::ICMP_EQ);
        break;
      case BinaryOpType::cmp_ne:
        values_[s] = compare(P::FCMP_ONE, P::ICMP_NE, P::ICMP_NE);
        break;
      case BinaryOpType::cmp_lt:
        values_[s] = compare(P::FCMP_OLT, P::ICMP_SLT, P::ICMP_ULT);
        break;
      case BinaryOpType::cmp_le:
        values_[s] = compare(P::FCMP_OLE, P::ICMP_SLE, P::ICMP_ULE);
        break;
      case BinaryOpType::cmp_gt:
        values_[s] = compare(P::FCMP_OGT, P::ICMP_SGT, P::ICMP_UGT);
        break;
      case BinaryOpType::cmp_ge:
        values_[s] = compare(P::FCMP_OGE, P::ICMP_SGE, P::ICMP_UGE);
        break;
      case BinaryOpType::bit_shl:
        values_[s] = b_.CreateShl(x, y);
        break;
      case BinaryOpType::bit_sar:
        values_[s] = signed_int ? b_.CreateAShr(x, y) : b_.CreateLShr(x, y);
        break;
      case BinaryOpType::bit_and:
        values_[s] = b_.CreateAnd(x, y);
        break;
      case BinaryOpType::bit_or:
        values_[s] = b_.CreateOr(x, y);
        break;
      case BinaryOpType::bit_xor:
        values_[s] = b_.CreateXor(x, y);
        break;
      default:
        values_[s] = operation("binary." + binary_op_type_name(s->op_type) +
                                   "." + s->lhs->ret_type->to_string(),
                               type(s->ret_type), {x, y});
        break;
    }
  }
  void visit(IfStmt *s) override {
    auto *yes = llvm::BasicBlock::Create(ctx_, "then", entry_);
    auto *no = llvm::BasicBlock::Create(ctx_, "else", entry_);
    auto *join = llvm::BasicBlock::Create(ctx_, "join", entry_);
    b_.CreateCondBr(b_.CreateIsNotNull(value(s->cond)), yes, no);
    auto arm = [&](llvm::BasicBlock *bb, Block *block) {
      b_.SetInsertPoint(bb);
      if (block)
        block->accept(this);
      if (!bb_terminated())
        b_.CreateBr(join);
    };
    arm(yes, s->true_statements.get());
    arm(no, s->false_statements.get());
    b_.SetInsertPoint(join);
  }
  bool bb_terminated() const {
    return b_.GetInsertBlock()->getTerminator() != nullptr;
  }
  void visit(WhileStmt *s) override {
    auto *body = llvm::BasicBlock::Create(ctx_, "while", entry_);
    auto *end = llvm::BasicBlock::Create(ctx_, "while.end", entry_);
    b_.CreateBr(body);
    b_.SetInsertPoint(body);
    loops_.emplace_back(body, end);
    s->body->accept(this);
    loops_.pop_back();
    if (!bb_terminated())
      b_.CreateBr(body);
    b_.SetInsertPoint(end);
  }
  void visit(WhileControlStmt *s) override {
    TI_ASSERT(!loops_.empty());
    auto *next = llvm::BasicBlock::Create(ctx_, "while.next", entry_);
    b_.CreateCondBr(b_.CreateIsNotNull(value(s->cond)), next,
                    loops_.back().second);
    b_.SetInsertPoint(next);
  }
  void visit(ContinueStmt *) override {
    TI_ASSERT(!loops_.empty());
    b_.CreateBr(loops_.back().first);
  }
  void visit(RangeForStmt *s) override {
    TI_ERROR_IF(s->reversed || s->is_bit_vectorized,
                "Unified LLVM reversed or vectorized range is not lowered");
    if (s->model_block_dim)
      operation("range.block_size", b_.getVoidTy(),
                {value(s->model_block_dim)});
    auto *bound = value(s->end);
    if (!s->device_options.empty()) {
      // The identity on the bound associates options with this loop even
      // after LLVM moves or inlines instructions in its preheader.
      std::vector<llvm::Value *> options{bound};
      for (auto *argument : s->device_options)
        options.push_back(value(argument));
      bound = operation("range.device_options", bound->getType(), options);
    }
    parallel_loop(s, *s->body, value(s->begin), bound, s->strictly_serialized,
                  s->requested_cpu_threads, s->block_dim, s->one_to_one, s->range_hint);
  }
  void visit(StructForStmt *s) override {
    TI_ERROR_IF(s->is_bit_vectorized || s->block_initialization || s->block_finalization,
                "Unsupported vectorized or lowered unified struct-for");
    auto args = tree_binding(*s->snode);
    // Keep cache hints tied to logical fields. Pointer addresses and container
    // iteration order must not become part of the unit identity.
    std::vector<std::tuple<int, int, int>> hints;
    for (const auto &[node, flags] : s->mem_access_opt.get_all()) {
      const auto field = tree_binding(*node);
      for (auto flag : flags)
        hints.emplace_back(llvm::cast<llvm::ConstantInt>(field[1])->getZExtValue(),
                           llvm::cast<llvm::ConstantInt>(field[2])->getZExtValue(),
                           static_cast<int>(flag));
    }
    std::sort(hints.begin(), hints.end());
    for (auto [tree, member, flag] : hints) {
      args.push_back(b_.getInt32(tree));
      args.push_back(b_.getInt32(member));
      args.push_back(b_.getInt32(flag));
    }
    auto *count = operation("field.active_count", b_.getInt32Ty(), args);
    parallel_loop(s, *s->body, b_.getInt32(0), count, false,
                  s->num_cpu_threads, s->block_dim, false, "");
  }
  void parallel_loop(Stmt *owner, Block &statements, llvm::Value *begin, llvm::Value *bound,
                     bool serial, int threads, int width, bool one_to_one, llvm::StringRef hint) {
    auto *initial = b_.GetInsertBlock();
    auto *head = llvm::BasicBlock::Create(ctx_, "range", entry_);
    auto *body = llvm::BasicBlock::Create(ctx_, "body", entry_);
    auto *latch = llvm::BasicBlock::Create(ctx_, "range.next", entry_);
    auto *after = llvm::BasicBlock::Create(ctx_, "end", entry_);
    b_.CreateBr(head);
    b_.SetInsertPoint(head);
    auto *index = b_.CreatePHI(begin->getType(), 2);
    index->addIncoming(begin, initial);
    indices_[owner] = index;
    auto *branch = b_.CreateCondBr(b_.CreateICmpSLT(index, bound), body, after);
    auto md = [&](int n) { return llvm::ConstantAsMetadata::get(b_.getInt32(n)); };
    branch->setMetadata("forge.range", llvm::MDNode::get(ctx_,
        {md(serial), md(threads), md(width), md(one_to_one), llvm::MDString::get(ctx_, hint)}));
    b_.SetInsertPoint(body);
    loops_.emplace_back(latch, after);
    statements.accept(this);
    loops_.pop_back();
    if (!bb_terminated())
      b_.CreateBr(latch);
    b_.SetInsertPoint(latch);
    auto *next =
        b_.CreateAdd(index, llvm::ConstantInt::get(index->getType(), 1));
    index->addIncoming(next, latch);
    b_.CreateBr(head);
    indices_.erase(owner);
    b_.SetInsertPoint(after);
  }
  void visit(LoopIndexStmt *s) override {
    if (s->loop->is<StructForStmt>()) {
      values_[s] = operation("field.coordinate", b_.getInt32Ty(),
                             {indices_.at(s->loop), b_.getInt32(s->index)});
      return;
    }
    TI_ERROR_IF(s->index != 0, "Unified LLVM non-range index is not lowered");
    values_[s] = indices_.at(s->loop);
  }
  void visit(ReturnStmt *s) override {
    if (callee_) {
      fill_frame(type(callee_->ret_type), entry_->getArg(2), s->values);
    } else {
      for (size_t i = 0; i < s->values.size(); ++i)
        operation("result", b_.getVoidTy(),
                  {entry_->getArg(0), b_.getInt32(i), value(s->values[i])});
    }
    b_.CreateRetVoid();
  }
};
}  // namespace

std::string unified_leaf_identity(Callable &callable, bool simplify) {
  std::ostringstream ast;
  if (!serialize_portable_leaf(callable.ir.get(), ast))
    return {};
  BinaryOutputSerializer serializer;
  serializer.initialize();
  serializer(callable.parameter_list);
  serializer(callable.rets);
  const auto &config = callable.program->compile_config();
  serializer(config.default_fp.to_string());
  serializer(config.default_ip.to_string());
  serializer(config.debug);
  serializer(config.fast_math);
  serializer(simplify);
  serializer(ast.str());
  serializer.finalize();
  return aot_artifact_sha256(llvm::StringRef(
      reinterpret_cast<const char *>(serializer.data.data()), serializer.data.size()));
}

std::string emit_unified_bitcode(Kernel &kernel,
                                 GraphKernelMetadata &effects,
                                 UnitMap &helpers,
                                 UnitMap &identities,
                                 const std::vector<int> &tree_ids,
                                 const std::map<int, int> &node_ordinals,
                                 bool simplify,
                                 std::vector<std::string> *imports) {
  TI_ERROR_IF(!kernel.ir || !kernel.ir_is_ast(),
              "Unified LLVM requires live frontend IR");
  auto root = lower_body(kernel, true);
  effects = analyze_graph_kernel_metadata(root.get(), &kernel);
  llvm::LLVMContext context;
  llvm::Module module("forge.unified", context);
  LocalFunctions functions;
  UnitWriter(module, helpers, identities, tree_ids, node_ordinals, functions, simplify)
      .write(*root->as<Block>());
  if (imports) {
    for (const auto &[function, digest] : functions)
      imports->push_back(digest);
    std::sort(imports->begin(), imports->end());
    imports->erase(std::unique(imports->begin(), imports->end()), imports->end());
  }
  return serialize(module, simplify);
}
}  // namespace taichi::lang::LLVM
