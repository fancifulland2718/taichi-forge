#include "taichi/codegen/cpu/codegen_cpu.h"

#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/llvm_opt_pipeline.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
// PassManagerBuilder was removed in LLVM 17. See
// taichi/runtime/llvm/llvm_opt_pipeline.h for the New-PM replacement.
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

namespace taichi::lang {

namespace {

class TaskCodeGenCPU : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenCPU(int id,
                 const CompileConfig &config,
                 TaichiLLVMContext &tlctx,
                 const Kernel *kernel,
                 IRNode *ir)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir, nullptr) {
    TI_AUTO_PROF
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    int step = 1;

    // In parallel for-loops reversing the order doesn't make sense.
    // However, we may need to support serial offloaded range for's in the
    // future, so it still makes sense to reverse the order here.
    if (stmt->reversed) {
      step = -1;
    }

    auto *tls_prologue = create_xlogue(stmt->tls_prologue);
    auto [begin, end] = get_range_for_bounds(stmt);

    // A bounded CPU range is lowered as one contiguous loop per scheduler
    // chunk. Keeping this loop in JIT-generated code lets LLVM hoist loop
    // invariants and vectorize it; the old per-element callback made that
    // impossible and also interpreted GPU block_dim as CPU task grain. The
    // existing binding's reserved word carries the runtime chunk size, so the
    // lowering remains compatible with older split-runtime payloads.
    const bool chunked_bounded_range = stmt->one_to_one;

    // The loop body
    llvm::Function *body;
    if (chunked_bounded_range) {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::PointerType::get(*llvm_context, 0),
           tlctx->get_data_type<int>()},
          "bounded_range_chunk_body");

      emit_cpu_debug_fault_guard();

      auto [range_begin, range_end] = get_range_for_bounds(stmt);
      auto *i64_type = llvm::Type::getInt64Ty(*llvm_context);
      auto *range_begin_i64 = builder->CreateSExt(range_begin, i64_type);
      auto *range_end_i64 = builder->CreateSExt(range_end, i64_type);
      auto *count_i64 = builder->CreateSExt(
          load_cpu_bounded_extent_count(), i64_type);
      auto *logical_end_i64 = builder->CreateAdd(range_begin_i64, count_i64);
      logical_end_i64 = builder->CreateSelect(
          builder->CreateICmpSLT(logical_end_i64, range_end_i64),
          logical_end_i64, range_end_i64);
      auto *chunk_size_i64 =
          builder->CreateSExt(load_cpu_bounded_chunk_size(), i64_type);
      auto *task_index_i64 = builder->CreateSExt(get_arg(2), i64_type);
      auto *chunk_begin_i64 = builder->CreateAdd(
          range_begin_i64,
          builder->CreateMul(task_index_i64, chunk_size_i64));
      auto *chunk_end_i64 =
          builder->CreateAdd(chunk_begin_i64, chunk_size_i64);
      chunk_end_i64 = builder->CreateSelect(
          builder->CreateICmpSLT(chunk_end_i64, logical_end_i64),
          chunk_end_i64, logical_end_i64);
      auto *chunk_begin = builder->CreateTrunc(chunk_begin_i64,
                                               builder->getInt32Ty());
      auto *chunk_end =
          builder->CreateTrunc(chunk_end_i64, builder->getInt32Ty());

      auto *loop_test_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_test", func);
      auto *loop_body_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_body", func);
      auto *loop_inc_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_inc", func);
      auto *func_exit =
          llvm::BasicBlock::Create(*llvm_context, "loop_exit", func);
      auto *loop_index = create_entry_block_alloca(PrimitiveType::i32);
      if (step == 1) {
        builder->CreateStore(chunk_begin, loop_index);
      } else {
        builder->CreateStore(
            builder->CreateSub(chunk_end, tlctx->get_constant(1)), loop_index);
      }
      builder->CreateBr(loop_test_bb);

      builder->SetInsertPoint(loop_test_bb);
      auto *loop_index_load =
          builder->CreateLoad(builder->getInt32Ty(), loop_index);
      llvm::Value *condition = nullptr;
      if (step == 1) {
        condition = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                        loop_index_load, chunk_end);
      } else {
        condition = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE,
                                        loop_index_load, chunk_begin);
      }
      if (compile_config.debug) {
        auto *fault_free = builder->CreateIsNull(
            call("LLVMRuntime_has_error", get_runtime()));
        condition = builder->CreateAnd(condition, fault_free);
      }
      builder->CreateCondBr(condition, loop_body_bb, func_exit);

      builder->SetInsertPoint(loop_body_bb);
      loop_vars_llvm[stmt].push_back(loop_index);
      auto *saved_loop_reentry = current_loop_reentry;
      const bool saved_chunk_body = inside_bounded_range_chunk_;
      current_loop_reentry = loop_inc_bb;
      inside_bounded_range_chunk_ = true;
      stmt->body->accept(this);
      inside_bounded_range_chunk_ = saved_chunk_body;
      current_loop_reentry = saved_loop_reentry;
      if (!returned) {
        builder->CreateBr(loop_inc_bb);
      } else {
        returned = false;
      }

      builder->SetInsertPoint(loop_inc_bb);
      loop_index_load = builder->CreateLoad(builder->getInt32Ty(), loop_index);
      builder->CreateStore(
          builder->CreateAdd(loop_index_load, tlctx->get_constant(step)),
          loop_index);
      builder->CreateBr(loop_test_bb);
      builder->SetInsertPoint(func_exit);
      body = guard.body;
    } else {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::PointerType::get(*llvm_context, 0),
           tlctx->get_data_type<int>()});

      emit_cpu_debug_fault_guard();

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    llvm::Value *epilogue = create_xlogue(stmt->tls_epilogue);

    const char *scheduler = compile_config.debug
                                ? "cpu_parallel_range_for_cancellable"
                                : "cpu_parallel_range_for";
    if (chunked_bounded_range) {
      const int effective_num_threads = std::max(1, stmt->num_cpu_threads);
      auto *i64_type = llvm::Type::getInt64Ty(*llvm_context);
      auto *begin_i64 = builder->CreateSExt(begin, i64_type);
      auto *bounded_end =
          call("cpu_bounded_range_end", get_arg(0), begin, end);
      auto *bounded_end_i64 = builder->CreateSExt(bounded_end, i64_type);
      auto *raw_count = builder->CreateSub(bounded_end_i64, begin_i64);
      auto *count = builder->CreateSelect(
          builder->CreateICmpSGT(raw_count, tlctx->get_constant((int64)0)),
          raw_count, tlctx->get_constant((int64)0));
      auto *thread_count =
          tlctx->get_constant((int64)effective_num_threads);
      auto *per_thread = builder->CreateSDiv(
          builder->CreateAdd(count,
                             tlctx->get_constant(
                                 (int64)effective_num_threads - 1)),
          thread_count);
      auto *minimum_chunk = tlctx->get_constant((int64)512);
      auto *chunk_size = builder->CreateSelect(
          builder->CreateICmpSGT(per_thread, minimum_chunk), per_thread,
          minimum_chunk);
      auto *task_count = builder->CreateSDiv(
          builder->CreateAdd(
              count, builder->CreateSub(chunk_size,
                                        tlctx->get_constant((int64)1))),
          chunk_size);
      store_cpu_bounded_chunk_size(
          builder->CreateTrunc(chunk_size, builder->getInt32Ty()));
      call(scheduler, get_arg(0),
           tlctx->get_constant(effective_num_threads),
           tlctx->get_constant(0),
           builder->CreateTrunc(task_count, builder->getInt32Ty()),
           tlctx->get_constant(1), tlctx->get_constant(1), tls_prologue, body,
           epilogue, tlctx->get_constant(stmt->tls_size));
    } else {
      call(scheduler, get_arg(0), tlctx->get_constant(stmt->num_cpu_threads),
           begin, end, tlctx->get_constant(step),
           tlctx->get_constant(stmt->block_dim), tls_prologue, body, epilogue,
           tlctx->get_constant(stmt->tls_size));
    }
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    auto *tls_prologue = create_mesh_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::PointerType::get(*llvm_context, 0),
           tlctx->get_data_type<int>()});

      emit_cpu_debug_fault_guard();

      for (int i = 0; i < stmt->mesh_prologue->size(); i++) {
        auto &s = stmt->mesh_prologue->statements[i];
        s->accept(this);
      }

      if (stmt->bls_prologue) {
        stmt->bls_prologue->accept(this);
      }

      auto loop_test_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_test", func);
      auto loop_body_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_body", func);
      auto func_exit =
          llvm::BasicBlock::Create(*llvm_context, "func_exit", func);
      auto loop_index =
          create_entry_block_alloca(llvm::Type::getInt32Ty(*llvm_context));
      builder->CreateStore(tlctx->get_constant(0), loop_index);
      builder->CreateBr(loop_test_bb);

      {
        builder->SetInsertPoint(loop_test_bb);
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
        auto cond = builder->CreateICmp(
            llvm::CmpInst::Predicate::ICMP_SLT, loop_index_load,
            llvm_val[stmt->owned_num_local.find(stmt->major_from_type)
                         ->second]);
        builder->CreateCondBr(cond, loop_body_bb, func_exit);
      }

      {
        builder->SetInsertPoint(loop_body_bb);
        loop_vars_llvm[stmt].push_back(loop_index);
        for (int i = 0; i < stmt->body->size(); i++) {
          auto &s = stmt->body->statements[i];
          s->accept(this);
        }
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
        builder->CreateStore(
            builder->CreateAdd(loop_index_load, tlctx->get_constant(1)),
            loop_index);
        builder->CreateBr(loop_test_bb);
        builder->SetInsertPoint(func_exit);
      }

      if (stmt->bls_epilogue) {
        stmt->bls_epilogue->accept(this);
      }

      body = guard.body;
    }

    llvm::Value *epilogue = create_mesh_xlogue(stmt->tls_epilogue);

    call("cpu_parallel_mesh_for", get_arg(0),
         tlctx->get_constant(stmt->num_cpu_threads),
         tlctx->get_constant(stmt->mesh->num_patches),
         tlctx->get_constant(stmt->block_dim), tls_prologue, body, epilogue,
         tlctx->get_constant(stmt->tls_size));
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                     stmt->bls_size);
    bls_buffer = new llvm::GlobalVariable(
        *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "bls_buffer", nullptr, llvm::GlobalVariable::LocalExecTLSModel, 0);
    /* module->getOrInsertGlobal("bls_buffer", type);
    bls_buffer = module->getNamedGlobal("bls_buffer");
    bls_buffer->setAlignment(llvm::MaybeAlign(8));*/ // TODO(changyu): Fix JIT session error: Symbols not found: [ __emutls_get_address ] in python 3.10

    // initialize the variable with an undef value to ensure it is added to the
    // symbol table
    bls_buffer->setInitializer(llvm::UndefValue::get(type));
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (compile_config.kernel_profiler && arch_is_cpu(compile_config.arch)) {
      call("LLVMRuntime_profiler_start", get_runtime(),
           create_global_string(offloaded_task_name));
    }
    emit_cpu_debug_fault_guard();
    if (stmt->task_type == Type::serial) {
      stmt->body->accept(this);
    } else if (stmt->task_type == Type::range_for) {
      create_offload_range_for(stmt);
    } else if (stmt->task_type == Type::mesh_for) {
      create_offload_mesh_for(stmt);
    } else if (stmt->task_type == Type::struct_for) {
      stmt->block_dim = std::min(stmt->snode->parent->max_num_elements(),
                                 (int64)stmt->block_dim);
      create_offload_struct_for(stmt);
    } else if (stmt->task_type == Type::listgen) {
      emit_list_gen(stmt);
    } else if (stmt->task_type == Type::gc) {
      emit_gc(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
    if (compile_config.kernel_profiler && arch_is_cpu(compile_config.arch)) {
      llvm::IRBuilderBase::InsertPointGuard guard(*builder);
      builder->SetInsertPoint(final_block);
      call("LLVMRuntime_profiler_stop", get_runtime());
    }
    finalize_offloaded_task_function();
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
    current_offload = nullptr;
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      TaskCodeGenLLVM::visit_call_bitcode(stmt);
    } else if (stmt->type == ExternalFuncCallStmt::SHARED_OBJECT) {
      TaskCodeGenLLVM::visit_call_shared_object(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(ContinueStmt *stmt) override {
    if (!inside_bounded_range_chunk_) {
      TaskCodeGenLLVM::visit(stmt);
      return;
    }

    // A top-level offloaded range normally lowers `continue` as a return from
    // its one-element callback. A bounded chunk callback contains multiple
    // logical iterations, so returning would silently discard the remainder
    // of the chunk. Nested loops update current_loop_reentry while they are
    // emitted, making the same branch correct for both cases.
    TI_ASSERT(current_loop_reentry != nullptr);
    builder->CreateBr(current_loop_reentry);
    auto *after_continue =
        llvm::BasicBlock::Create(*llvm_context, "after_continue", func);
    builder->SetInsertPoint(after_continue);
  }

 private:
  // CpuBoundedRangeBinding is a private split-runtime ABI prefix. Its layout
  // is statically asserted in taichi/program/context.h; keep codegen's byte
  // offset explicit so this lowering does not need a host-only struct type.
  static constexpr int64 kCpuBoundedRangeBindingSize = 16;
  bool inside_bounded_range_chunk_{false};

  llvm::Value *get_cpu_bounded_range_binding() {
    auto *runtime_context_type = get_runtime_type("RuntimeContext");
    auto *zero = tlctx->get_constant(0);
    auto *arg_buffer_field = builder->CreateGEP(
        runtime_context_type, get_context(), {zero, zero});
    auto *i8_ptr_type =
        llvm::PointerType::get(llvm::Type::getInt8Ty(*llvm_context), 0);
    arg_buffer_field = builder->CreatePointerCast(
        arg_buffer_field, llvm::PointerType::get(i8_ptr_type, 0));
    auto *arg_buffer = builder->CreateLoad(i8_ptr_type, arg_buffer_field);
    auto *binding_address = builder->CreateGEP(
        llvm::Type::getInt8Ty(*llvm_context), arg_buffer,
        tlctx->get_constant(-kCpuBoundedRangeBindingSize));
    auto *binding_type = get_runtime_type("CpuBoundedRangeBinding");
    return builder->CreatePointerCast(
        binding_address, llvm::PointerType::get(binding_type, 0));
  }

  llvm::Value *load_cpu_bounded_extent_count() {
    auto *binding_type = llvm::cast<llvm::StructType>(
        get_runtime_type("CpuBoundedRangeBinding"));
    auto *binding = get_cpu_bounded_range_binding();
    auto *extent_address_field =
        builder->CreateStructGEP(binding_type, binding, 0);
    auto *extent_address = builder->CreateLoad(
        binding_type->getElementType(0), extent_address_field);
    auto *extent = builder->CreateIntToPtr(
        extent_address,
        llvm::PointerType::get(builder->getInt32Ty(), 0));
    return builder->CreateLoad(builder->getInt32Ty(), extent);
  }

  llvm::Value *load_cpu_bounded_chunk_size() {
    auto *binding_type = llvm::cast<llvm::StructType>(
        get_runtime_type("CpuBoundedRangeBinding"));
    auto *chunk_size_field = builder->CreateStructGEP(
        binding_type, get_cpu_bounded_range_binding(), 2);
    return builder->CreateLoad(builder->getInt32Ty(), chunk_size_field);
  }

  void store_cpu_bounded_chunk_size(llvm::Value *chunk_size) {
    auto *binding_type = llvm::cast<llvm::StructType>(
        get_runtime_type("CpuBoundedRangeBinding"));
    auto *chunk_size_field = builder->CreateStructGEP(
        binding_type, get_cpu_bounded_range_binding(), 2);
    builder->CreateStore(chunk_size, chunk_size_field);
  }

  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx = tlctx->get_constant(0);
    auto block_dim = tlctx->get_constant(1);
    return std::make_tuple(thread_idx, block_dim);
  }
};

static llvm::Triple get_host_target_triple() {
  auto expected_jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!expected_jtmb) {
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  }
  return expected_jtmb->getTargetTriple();
}

}  // namespace

#ifdef TI_WITH_LLVM
LLVMCompiledTask KernelCodeGenCPU::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenCPU gen(task_codegen_id, config, get_taichi_llvm_context(), kernel,
                     block);
  return gen.run_compilation();
}

void KernelCodeGenCPU::optimize_module(llvm::Module *module) {
  TI_AUTO_PROF
  const auto &compile_config = get_compile_config();
  auto triple = get_host_target_triple();

  std::string err_str;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  llvm::TargetOptions options;
  if (compile_config.fast_math) {
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;

  llvm::StringRef mcpu = llvm::sys::getHostCPUName();
  std::unique_ptr<llvm::TargetMachine> target_machine(
      target->createTargetMachine(triple.str(), mcpu.str(), "", options,
                                  llvm::Reloc::PIC_, llvm::CodeModel::Small,
                                  llvm::CodeGenOptLevel::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Run the LLVM optimization pipeline at the configured level (O0-O3).
  // llvm_opt_level=3 (default) matches the previous hardcoded O3 behaviour.
  // CPU keeps LoopVectorize + SLPVectorize enabled (same as the legacy
  // `PassManagerBuilder` setup).
  {
    TI_PROFILER("llvm_module_opt_pipeline");
    LLVMOptPipelineOptions opts;
    opts.opt_level = llvm_opt_level_from_int(effective_llvm_opt_level(
        compile_config.llvm_opt_level, compile_config.compile_tier));
    opts.loop_vectorize = true;
    opts.slp_vectorize = true;
    opts.run_post_gep_passes = true;
    run_module_opt_pipeline(*module, target_machine.get(), opts);
  }

  if (compile_config.print_kernel_asm) {
    // Generate assembly code if necessary. Codegen still uses the
    // legacy PassManager in LLVM 19/20+.
    llvm::SmallString<8> outstr;
    llvm::raw_svector_ostream ostream(outstr);
    ostream.SetUnbuffered();

    llvm::legacy::PassManager emit_pm;
    target_machine->addPassesToEmitFile(emit_pm, ostream, nullptr,
                                        llvm::CodeGenFileType::AssemblyFile);
    {
      TI_PROFILER("llvm_emit_asm");
      emit_pm.run(*module);
    }
    static FileSequenceWriter writer(
        "taichi_kernel_cpu_llvm_ir_optimized_asm_{:04d}.s",
        "optimized assembly code (CPU)");
    std::string buffer(outstr.begin(), outstr.end());
    writer.write(buffer);
  }

  if (compile_config.print_kernel_llvm_ir_optimized) {
    if (false) {
      TI_INFO("Functions with > 100 instructions in optimized LLVM IR:");
      TaichiLLVMContext::print_huge_functions(module);
    }
    static FileSequenceWriter writer(
        "taichi_kernel_cpu_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CPU)");
    writer.write(module);
  }
}

#endif  // TI_WITH_LLVM
}  // namespace taichi::lang
