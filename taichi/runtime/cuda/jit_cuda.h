#include <memory>
#include <string>
#include <vector>

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
enum class CUDAArtifactKind {
  ptx,
  cubin,
};

// Internal hand-off object between LLVM NVPTX emission, optional artifact
// providers, and the CUDA module loader.  Keeping this typed prevents binary
// cubins from accidentally flowing through text-only PTX assumptions.
struct CUDAKernelArtifact {
  CUDAArtifactKind kind{CUDAArtifactKind::ptx};
  std::vector<char> payload;
  std::vector<std::string> entry_names;
  std::string target_identity;
  std::string provider_identity;
  int max_registers{0};
  bool fast_math{false};
  int llvm_opt_level{0};

  const void *data() const {
    return payload.data();
  }

  std::size_t payload_size() const {
    return payload.size();
  }

  std::size_t code_size() const {
    if (kind == CUDAArtifactKind::ptx && !payload.empty() &&
        payload.back() == '\0') {
      return payload.size() - 1;
    }
    return payload.size();
  }
};

class JITModuleCUDA : public JITModule {
 private:
  friend class JITSessionCUDA;
  void *module_;

 public:
  explicit JITModuleCUDA(void *module) : module_(module) {
  }

  void *lookup_function(const std::string &name) override {
    // TODO: figure out why using the guard leads to wrong tests results
    // auto context_guard = CUDAContext::get_instance().get_guard();
    CUDAContext::get_instance().make_current();
    void *func = nullptr;
    auto t = Time::get_time();
    uint32 err;
    {
      TI_COMPILE_PROFILER("cuda_driver_function_lookup")
      err = CUDADriver::get_instance().module_get_function.call_with_warning(
          &func, module_, name.c_str());
    }
    if (err) {
      TI_ERROR("Cannot look up function {}", name);
    }
    t = Time::get_time() - t;
    TI_TRACE("CUDA module_get_function {} costs {} ms", name, t * 1000);
    TI_ASSERT(func != nullptr);
    return func;
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
                          void *stream,
                          const std::string *profiler_name = nullptr) {
    auto func = lookup_function(name);
    CUDAContext::get_instance().launch(
        func, profiler_name != nullptr ? *profiler_name : name, arg_pointers,
        arg_sizes, grid_dim, block_dim, dynamic_shared_mem_bytes, stream);
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
  CUDAKernelArtifact build_canonical_artifact(
      std::unique_ptr<llvm::Module> &module,
      int max_reg);
  CUDAKernelArtifact select_artifact(CUDAKernelArtifact artifact);
  void *load_artifact(const CUDAKernelArtifact &artifact);
  std::vector<char> emit_module_to_ptx(std::unique_ptr<llvm::Module> &module);
};

#endif

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch);

}  // namespace taichi::lang
