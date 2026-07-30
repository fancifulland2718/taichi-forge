#include <memory>
#include <mutex>
#include <unordered_map>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
// PassManagerBuilder was removed in LLVM 17; the optimization pipeline
// now runs through taichi::lang::run_module_opt_pipeline.
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/jit/jit_session.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/file_sequence_writer.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {

#if defined(TI_WITH_CUDA)
class JITModuleCUDA : public JITModule {
 private:
  friend class JITSessionCUDA;
  struct FunctionEntry {
    void *handle{nullptr};
    std::size_t prepared_dynamic_shared_bytes{0};
  };

  void *module_;
  std::mutex functions_mutex_;
  std::unordered_map<std::string, FunctionEntry> functions_;

  void *lookup_function_with_dynamic_shared_memory(
      const std::string &name,
      std::size_t dynamic_shared_mem_bytes) {
    std::lock_guard<std::mutex> lock(functions_mutex_);
    auto &entry = functions_[name];
    if (entry.handle == nullptr) {
      // TODO: figure out why using the guard leads to wrong tests results
      // auto context_guard = CUDAContext::get_instance().get_guard();
      CUDAContext::get_instance().make_current();
      auto t = Time::get_time();
      auto err =
          CUDADriver::get_instance().module_get_function.call_with_warning(
              &entry.handle, module_, name.c_str());
      if (err) {
        TI_ERROR("Cannot look up function {}", name);
      }
      t = Time::get_time() - t;
      TI_TRACE("CUDA module_get_function {} costs {} ms", name, t * 1000);
      TI_ASSERT(entry.handle != nullptr);
    }
    if (dynamic_shared_mem_bytes >
        entry.prepared_dynamic_shared_bytes) {
      CUDAContext::get_instance().prepare_dynamic_shared_memory(
          entry.handle, dynamic_shared_mem_bytes);
      entry.prepared_dynamic_shared_bytes = dynamic_shared_mem_bytes;
    }
    return entry.handle;
  }

 public:
  explicit JITModuleCUDA(void *module) : module_(module) {
  }

  void *lookup_function(const std::string &name) override {
    return lookup_function_with_dynamic_shared_memory(name, 0);
  }

  void call(const std::string &name,
            const std::vector<void *> &arg_pointers,
            const std::vector<int> &arg_sizes) override {
    launch(name, 1, 1, 0, arg_pointers, arg_sizes);
  }

  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t dynamic_shared_mem_bytes,
              const std::vector<void *> &arg_pointers,
              const std::vector<int> &arg_sizes) override {
    launch_with_stream(name, grid_dim, block_dim, dynamic_shared_mem_bytes,
                       arg_pointers, arg_sizes, nullptr);
  }

  void launch_with_stream(const std::string &name,
                          std::size_t grid_dim,
                          std::size_t block_dim,
                          std::size_t dynamic_shared_mem_bytes,
                          const std::vector<void *> &arg_pointers,
                          const std::vector<int> &arg_sizes,
                          void *stream) {
    auto func = lookup_function_with_dynamic_shared_memory(
        name, dynamic_shared_mem_bytes);
    CUDAContext::get_instance().launch(func, name, arg_pointers, arg_sizes,
                                       grid_dim, block_dim,
                                       dynamic_shared_mem_bytes, stream,
                                       /*dynamic_shared_memory_prepared=*/true);
  }

  bool direct_dispatch() const override {
    return false;
  }
};

class JITSessionCUDA : public JITSession {
 public:
  llvm::DataLayout data_layout;

  JITSessionCUDA(TaichiLLVMContext *tlctx,
                 const CompileConfig &config,
                 llvm::DataLayout data_layout)
      : JITSession(tlctx, config), data_layout(data_layout) {
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override;

  bool remove_module(JITModule *module) override;

  llvm::DataLayout get_data_layout() override {
    return data_layout;
  }

 private:
  std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module);
};

#endif

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch);

}  // namespace taichi::lang
