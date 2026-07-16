// The CUDA backend

#pragma once

#include "taichi/codegen/codegen.h"
#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi::lang {

class KernelCodeGenCUDA : public KernelCodeGen {
 public:
  explicit KernelCodeGenCUDA(const CompileConfig &compile_config,
                             const DeviceCapabilityConfig &device_caps,
                             const Kernel *kernel,
                             IRNode *ir,
                             TaichiLLVMContext &tlctx)
      : KernelCodeGen(compile_config, kernel, ir, tlctx),
        device_caps_(device_caps) {
  }

// TODO: Stop defining this macro guards in the headers
#ifdef TI_WITH_LLVM
  LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) override;
#endif  // TI_WITH_LLVM

 private:
  DeviceCapabilityConfig device_caps_;
};

}  // namespace taichi::lang
