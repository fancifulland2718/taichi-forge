// Driver class for kernel codegen

#include "codegen.h"

#if defined(TI_WITH_LLVM)
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif
#if defined(TI_WITH_CUDA)
#include "taichi/codegen/cuda/codegen_cuda.h"
#endif
#if defined(TI_WITH_DX12)
#include "taichi/codegen/dx12/codegen_dx12.h"
#endif
#if defined(TI_WITH_AMDGPU)
#include "taichi/codegen/amdgpu/codegen_amdgpu.h"
#endif
#include "taichi/system/timer.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/analysis/offline_cache_util.h"

namespace taichi::lang {

KernelCodeGen::KernelCodeGen(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             TaichiLLVMContext &tlctx)
    : prog(kernel->program),
      kernel(kernel),
      ir(ir),
      compile_config_(compile_config),
      tlctx_(tlctx) {
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(
    const CompileConfig &compile_config,
    const Kernel *kernel,
    IRNode *ir,
    TaichiLLVMContext &tlctx) {
#ifdef TI_WITH_LLVM
  const auto arch = compile_config.arch;
  if (arch_is_cpu(arch)) {
    return std::make_unique<KernelCodeGenCPU>(compile_config, kernel, ir,
                                              tlctx);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return std::make_unique<KernelCodeGenCUDA>(compile_config, kernel, ir,
                                               tlctx);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::dx12) {
#if defined(TI_WITH_DX12)
    return std::make_unique<KernelCodeGenDX12>(compile_config, kernel, ir,
                                               tlctx);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    return std::make_unique<KernelCodeGenAMDGPU>(compile_config, kernel, ir,
                                                 tlctx);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }
#else
  TI_ERROR("Llvm disabled");
#endif
}
#ifdef TI_WITH_LLVM

LLVMCompiledKernel KernelCodeGen::compile_kernel_to_module() {
  auto block = dynamic_cast<Block *>(ir);
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  TI_ASSERT(block);

  auto &offloads = block->statements;
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(offloads.size());

  // P-Compile-4: 单 offload kernel 直接 inline 编译，跳过 compilation_workers 入队 /
  // flush 的线程池开销。这类 kernel 占常见用例的多数（单层 range_for / 顺序 kernel），
  // 在 P5 引入的外层 kernel 级线程池上叠加一次内层池入队会带来 CV 唤醒、TLS 切换和
  // 队列锁竞争，实测 CPU 路径出现 0.89x 回归（见 compile_doc/P5_并行编译.md）。
  // 多 offload 场景仍走原有 compilation_workers 路径以保持并行编译收益。
  //
  // V7 (2026-04-26): 当 compile_config_.compile_dag_scheduler=True 且当前线程已是
  // Program::compile_kernels 的外层 worker 时，外层池已在 kernel 维度饱和，
  // 内层池再 fan-out offload 任务会造成 N*M 线程过载。此时把多 offload 也
  // 走 inline 串行路径。详见 compile_doc/优化总规划.md §3.4。
  const bool serial_offloads =
      offloads.size() <= 1 ||
      (compile_config_.compile_dag_scheduler &&
       Program::in_compile_kernels_worker());
  if (serial_offloads) {
    for (int i = 0; i < (int)offloads.size(); i++) {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      Block blk;
      blk.insert(std::move(offload));
      auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
    }
  } else {
    for (int i = 0; i < (int)offloads.size(); i++) {
      auto compile_func = [&, i] {
        tlctx_.fetch_this_thread_struct_module();
        auto offload = irpass::analysis::clone(offloads[i].get());
        irpass::re_id(offload.get());

        Block blk;
        blk.insert(std::move(offload));
        auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
        data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
      };
      worker.enqueue(compile_func);
    }
    worker.flush();
  }

  auto llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
  optimize_module(llvm_compiled_kernel.module.get());
  return llvm_compiled_kernel;
}

#endif
}  // namespace taichi::lang
