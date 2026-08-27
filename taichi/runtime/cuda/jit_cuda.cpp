#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/llvm/llvm_opt_pipeline.h"
#include "taichi/util/environ_config.h"

#include <cstdlib>

#include "llvm/IR/Metadata.h"

namespace taichi::lang {

#if defined(TI_WITH_CUDA)

namespace {

constexpr std::size_t kCudaJitLogCapacity = 16 * 1024;

std::string cuda_jit_log_text(const std::vector<char> &buffer) {
  const auto end = std::find(buffer.begin(), buffer.end(), '\0');
  return std::string(buffer.begin(), end);
}

const char *cuda_jit_env(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' ? value : "<driver-default>";
}

std::vector<std::string> cuda_kernel_entry_names(llvm::Module &module) {
  std::vector<std::string> entries;
  auto *annotations = module.getNamedMetadata("nvvm.annotations");
  if (annotations == nullptr) {
    return entries;
  }
  for (const auto *node : annotations->operands()) {
    if (node->getNumOperands() < 3) {
      continue;
    }
    const auto *key =
        llvm::dyn_cast_if_present<llvm::MDString>(node->getOperand(1).get());
    const auto *value = llvm::dyn_cast_if_present<llvm::ValueAsMetadata>(
        node->getOperand(0).get());
    if (key == nullptr || key->getString() != "kernel" || value == nullptr) {
      continue;
    }
    const auto *function = llvm::dyn_cast<llvm::Function>(value->getValue());
    if (function != nullptr) {
      entries.push_back(function->getName().str());
    }
  }
  return entries;
}

const char *cuda_artifact_kind_name(CUDAArtifactKind kind) {
  switch (kind) {
    case CUDAArtifactKind::ptx:
      return "ptx";
    case CUDAArtifactKind::cubin:
      return "cubin";
  }
  TI_NOT_IMPLEMENTED
}

}  // namespace

std::string convert(std::string new_name);

JITModule *JITSessionCUDA ::add_module(std::unique_ptr<llvm::Module> M,
                                       int max_reg) {
  auto artifact = select_artifact(build_canonical_artifact(M, max_reg));
  if (this->config_.print_kernel_asm &&
      artifact.kind == CUDAArtifactKind::ptx) {
    static FileSequenceWriter writer("taichi_kernel_nvptx_{:04d}.ptx",
                                     "module NVPTX");
    writer.write(std::string(artifact.payload.data(), artifact.code_size()));
  }
  auto *cuda_module = load_artifact(artifact);
  modules.push_back(std::make_unique<JITModuleCUDA>(cuda_module));
  return modules.back().get();
}

CUDAKernelArtifact JITSessionCUDA::build_canonical_artifact(
    std::unique_ptr<llvm::Module> &module,
    int max_reg) {
  CUDAKernelArtifact artifact;
  artifact.kind = CUDAArtifactKind::ptx;

  // Canonicalize names before capturing the entry manifest. LLVM's optimizer
  // may discard metadata nodes after code generation, so the manifest cannot
  // be reconstructed reliably from the emitted module.
  for (auto &global : module->globals()) {
    global.setName(convert(global.getName().str()));
  }
  for (auto &function : *module) {
    function.setName(convert(function.getName().str()));
  }
  artifact.entry_names = cuda_kernel_entry_names(*module);
  artifact.payload = emit_module_to_ptx(module);
  artifact.target_identity = CUDAContext::get_instance().get_mcpu() + "|" +
                             CUDAContext::get_instance().get_mattrs();
  artifact.provider_identity = "llvm_nvptx";
  artifact.max_registers = max_reg;
  artifact.fast_math = config_.fast_math;
  artifact.llvm_opt_level = effective_llvm_opt_level(
      config_.llvm_opt_level, config_.compile_tier, /*min_level=*/1);
  return artifact;
}

CUDAKernelArtifact JITSessionCUDA::select_artifact(
    CUDAKernelArtifact artifact) {
  // The driver-only PTX path is deliberately the default. Optional providers
  // can replace this artifact without changing LLVM lowering or wheel inputs.
  return artifact;
}

void *JITSessionCUDA::load_artifact(const CUDAKernelArtifact &artifact) {
  // TODO: figure out why using the guard leads to wrong tests results
  // auto context_guard = CUDAContext::get_instance().get_guard();
  CUDAContext::get_instance().make_current();
  // Create module for object
  void *cuda_module = nullptr;
  TI_TRACE("CUDA {} artifact size: {:.2f}KB",
           cuda_artifact_kind_name(artifact.kind),
           artifact.code_size() / 1024.0);
  auto t = Time::get_time();
  TI_TRACE("Loading CUDA artifact: kind={}, provider={}, target={}, entries={}",
           cuda_artifact_kind_name(artifact.kind), artifact.provider_identity,
           artifact.target_identity, artifact.entry_names.size());
  [[maybe_unused]] auto _ = CUDAContext::get_instance().get_lock_guard();

  constexpr int max_num_options = 8;
  int num_options = 0;
  uint32 options[max_num_options];
  void *option_values[max_num_options];
  int max_registers = artifact.max_registers;

  auto &driver = CUDADriver::get_instance();
  const bool jit_diagnostics =
      get_environ_config("TI_CUDA_JIT_DIAGNOSTICS", 0) != 0 &&
      driver.nvidia_extensions_available();
  float driver_wall_time_ms = 0.0f;
  std::vector<char> info_log;
  std::vector<char> error_log;
  if (jit_diagnostics) {
    info_log.resize(kCudaJitLogCapacity);
    error_log.resize(kCudaJitLogCapacity);
  }
  uint32 info_log_capacity = static_cast<uint32>(info_log.size());
  uint32 error_log_capacity = static_cast<uint32>(error_log.size());

  // Insert options
  if (artifact.kind == CUDAArtifactKind::ptx && max_registers != 0) {
    options[num_options] = CU_JIT_MAX_REGISTERS;
    option_values[num_options] = &max_registers;
    num_options++;
  }

  if (jit_diagnostics) {
    options[num_options] = CU_JIT_WALL_TIME;
    option_values[num_options] = &driver_wall_time_ms;
    num_options++;
    options[num_options] = CU_JIT_INFO_LOG_BUFFER;
    option_values[num_options] = info_log.data();
    num_options++;
    options[num_options] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    option_values[num_options] = &info_log_capacity;
    num_options++;
    options[num_options] = CU_JIT_ERROR_LOG_BUFFER;
    option_values[num_options] = error_log.data();
    num_options++;
    options[num_options] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    option_values[num_options] = &error_log_capacity;
    num_options++;
  }

  TI_ASSERT(num_options <= max_num_options);

  bool load_succeeded = false;
  try {
    TI_COMPILE_PROFILER("cuda_driver_module_load")
    driver.module_load_data_ex(&cuda_module,
                               reinterpret_cast<const char *>(artifact.data()),
                               num_options, options, option_values);
    load_succeeded = true;
  } catch (...) {
    const auto host_wall_ms = (Time::get_time() - t) * 1000.0;
    const auto info = cuda_jit_log_text(info_log);
    const auto error = cuda_jit_log_text(error_log);
    driver.record_jit_module_load(
        artifact.kind == CUDAArtifactKind::ptx ? artifact.code_size() : 0,
        static_cast<uint64_t>(host_wall_ms * 1000000.0),
        static_cast<uint64_t>(driver_wall_time_ms * 1000.0f), jit_diagnostics,
        info.size(), error.size());
    if (jit_diagnostics) {
      TI_WARN(
          "CUDA artifact load failed: kind={}, provider={}, target={}, "
          "host_wall_ms={}, driver_wall_ms={}, info='{}', error='{}'",
          cuda_artifact_kind_name(artifact.kind), artifact.provider_identity,
          artifact.target_identity, host_wall_ms, driver_wall_time_ms, info,
          error);
    }
    throw;
  }

  const auto host_wall_ms = (Time::get_time() - t) * 1000.0;
  const auto info = cuda_jit_log_text(info_log);
  const auto error = cuda_jit_log_text(error_log);
  driver.record_jit_module_load(
      artifact.kind == CUDAArtifactKind::ptx ? artifact.code_size() : 0,
      static_cast<uint64_t>(host_wall_ms * 1000000.0),
      static_cast<uint64_t>(driver_wall_time_ms * 1000.0f), jit_diagnostics,
      info.size(), error.size());
  TI_TRACE("CUDA module load time : {}ms", host_wall_ms);
  if (jit_diagnostics && load_succeeded) {
    TI_INFO(
        "CUDA JIT diagnostics: route=driver_{}, provider={}, target={}, "
        "entries={}, code_bytes={}, "
        "host_wall_ms={}, driver_wall_ms={}, CUDA_CACHE_DISABLE={}, "
        "CUDA_CACHE_PATH={}, CUDA_CACHE_MAXSIZE={}, "
        "CUDA_FORCE_PTX_JIT={}, info='{}', error='{}'",
        cuda_artifact_kind_name(artifact.kind), artifact.provider_identity,
        artifact.target_identity, artifact.entry_names.size(),
        artifact.code_size(), host_wall_ms, driver_wall_time_ms,
        cuda_jit_env("CUDA_CACHE_DISABLE"), cuda_jit_env("CUDA_CACHE_PATH"),
        cuda_jit_env("CUDA_CACHE_MAXSIZE"), cuda_jit_env("CUDA_FORCE_PTX_JIT"),
        info, error);
  }
  return cuda_module;
}

bool JITSessionCUDA::remove_module(JITModule *module) {
  auto module_it =
      std::find_if(modules.begin(), modules.end(),
                   [module](const std::unique_ptr<JITModule> &owned) {
                     return owned.get() == module;
                   });
  if (module_it == modules.end()) {
    return false;
  }
  auto *cuda_module = dynamic_cast<JITModuleCUDA *>(module_it->get());
  TI_ASSERT(cuda_module != nullptr);
  CUDAContext::get_instance().make_current();
  auto context_lock = CUDAContext::get_instance().get_lock_guard();
  CUDADriver::get_instance().module_unload(cuda_module->module_);
  cuda_module->module_ = nullptr;
  modules.erase(module_it);
  return true;
}

std::string cuda_mattrs() {
  return CUDAContext::get_instance().get_mattrs();
}

std::string convert(std::string new_name) {
  // Evil C++ mangling on Windows will lead to "unsupported characters in
  // symbol" error in LLVM PTX printer. Convert here.
  for (int i = 0; i < (int)new_name.size(); i++) {
    if (new_name[i] == '@') {
      new_name.replace(i, 1, "_at_");
    } else if (new_name[i] == '?') {
      new_name.replace(i, 1, "_qm_");
    } else if (new_name[i] == '$') {
      new_name.replace(i, 1, "_dl_");
    } else if (new_name[i] == '<') {
      new_name.replace(i, 1, "_lb_");
    } else if (new_name[i] == '>') {
      new_name.replace(i, 1, "_rb_");
    } else if (!std::isalpha(new_name[i]) && !std::isdigit(new_name[i]) &&
               new_name[i] != '_' && new_name[i] != '.') {
      new_name.replace(i, 1, "_xx_");
    }
  }
  if (!new_name.empty())
    TI_ASSERT(isalpha(new_name[0]) || new_name[0] == '_' || new_name[0] == '.');
  return new_name;
}

std::vector<char> JITSessionCUDA::emit_module_to_ptx(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("LLVM Module broken");
  }

  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_cuda_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  TargetOptions options;
  if (this->config_.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // See NVPTXISelLowering.cpp
    // Setting UnsafeFPMath true will result in approximations such as
    // sqrt.approx in PTX for both f32 and f64
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), CUDAContext::get_instance().get_mcpu(), cuda_mattrs(),
      options, llvm::Reloc::PIC_, llvm::CodeModel::Small,
      llvm::CodeGenOptLevel::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  // NVidia's libdevice library uses a __nvvm_reflect to choose
  // how to handle denormalized numbers. (The pass replaces calls
  // to __nvvm_reflect with a constant via a map lookup. The inliner
  // pass then resolves these situations to fast code, often a single
  // instruction per decision point.)
  //
  // The default is (more) IEEE like handling. FTZ mode flushes them
  // to zero. (This may only apply to single-precision.)
  //
  // The libdevice documentation covers other options for math accuracy
  // such as replacing division with multiply by the reciprocal and
  // use of fused-multiply-add, but they do not seem to be controlled
  // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
  // passes.
  const auto kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);

  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      /* nvptx-f32ftz was deprecated.
       *
       * https://github.com/llvm/llvm-project/commit/a4451d88ee456304c26d552749aea6a7f5154bde#diff-6fda74ef428299644e9f49a2b0994c0d850a760b89828f655030a114060d075a
       */
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");

      // Use unsafe fp math for sqrt.approx instead of sqrt.rn
      fn.addFnAttr("unsafe-fp-math", "true");
    }
  }

  // Run the standard O3 optimization pipeline via the New PassManager.
  // NVPTX does not use the generic LoopVectorize / SLPVectorize passes,
  // so we leave them disabled — matching the original PassManagerBuilder
  // configuration (`b.LoopVectorize = false; b.SLPVectorize = false;`).
  {
    TI_PROFILER("llvm_module_opt_pipeline");
    LLVMOptPipelineOptions opts;
    opts.opt_level = llvm_opt_level_from_int(
        effective_llvm_opt_level(config_.llvm_opt_level, config_.compile_tier,
                                 /*min_level=*/1));
    opts.loop_vectorize = false;
    opts.slp_vectorize = false;
    opts.run_post_gep_passes = true;
    run_module_opt_pipeline(*module, target_machine.get(), opts);
  }

  // Emit PTX via the Legacy PassManager — `addPassesToEmitFile` still
  // requires it in LLVM 19/20+ (codegen has not yet been ported).
  target_machine->Options.MCOptions.AsmVerbose = true;

  legacy::PassManager emit_pm;
  bool fail = target_machine->addPassesToEmitFile(
      emit_pm, ostream, nullptr, llvm::CodeGenFileType::AssemblyFile, true);

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  {
    TI_PROFILER("llvm_emit_ptx");
    emit_pm.run(*module);
  }

  if (this->config_.print_kernel_llvm_ir_optimized) {
    static FileSequenceWriter writer(
        "taichi_kernel_cuda_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CUDA)");
    writer.write(module.get());
  }

  std::vector<char> buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_ASSERT(arch == Arch::cuda);
  // https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-layout
  auto data_layout = TaichiLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionCUDA>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_NOT_IMPLEMENTED
}
#endif

}  // namespace taichi::lang
