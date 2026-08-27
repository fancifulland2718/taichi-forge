// A LLVM JIT compiler for CPU archs wrapper

#include <limits>
#include <memory>

#ifdef TI_WITH_LLVM
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
// From https://github.com/JuliaLang/julia/pull/43664
#if defined(__APPLE__) && defined(__aarch64__)
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#else
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#endif
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetMachine.h"
// PassManagerBuilder was removed in LLVM 17. jit_cpu.cpp never used it
// directly (ORC's ConcurrentIRCompiler handles optimization on its own).
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

#endif

#include "taichi/jit/jit_module.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/jit/jit_session.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi::lang {

#ifdef TI_WITH_LLVM
using namespace llvm;
using namespace llvm::orc;
#if defined(__APPLE__) && defined(__aarch64__)
typedef orc::ObjectLinkingLayer ObjLayerT;
#else
typedef orc::RTDyldObjectLinkingLayer ObjLayerT;
#endif
#endif

#if defined(TI_WITH_LLVM) && defined(_WIN32)
namespace {

class OrderedCoffSectionMemoryManager final : public RTDyldMemoryManager {
 public:
  ~OrderedCoffSectionMemoryManager() override {
    deregisterEHFrames();
  }

  bool needsToReserveAllocationSpace() override {
    return true;
  }

  void reserveAllocationSpace(uintptr_t code_size,
                              Align code_align,
                              uintptr_t ro_size,
                              Align ro_align,
                              uintptr_t rw_size,
                              Align rw_align) override {
    TI_ASSERT(allocation_.base() == nullptr);
    const auto page_size =
        static_cast<std::uintptr_t>(sys::Process::getPageSizeEstimate());
    code_reserved_ = align_up(code_size, page_size);
    ro_reserved_ = align_up(ro_size, page_size);
    rw_reserved_ = align_up(rw_size, page_size);
    const auto code_segment_align =
        std::max<std::uintptr_t>(page_size, code_align.value());
    const auto ro_segment_align =
        std::max<std::uintptr_t>(page_size, ro_align.value());
    const auto rw_segment_align =
        std::max<std::uintptr_t>(page_size, rw_align.value());
    std::uintptr_t total_size = 0;
    add_allocation_size(total_size, code_reserved_, code_segment_align);
    add_allocation_size(total_size, ro_reserved_, ro_segment_align);
    add_allocation_size(total_size, rw_reserved_, rw_segment_align);
    total_size = std::max(total_size, page_size);

    std::error_code error;
    auto block = sys::Memory::allocateMappedMemory(
        total_size, nullptr,
        sys::Memory::MF_READ | sys::Memory::MF_WRITE, error);
    TI_ERROR_IF(error || block.base() == nullptr,
                "Failed to reserve ordered COFF JIT memory: {}",
                error.message());
    allocation_ = sys::OwningMemoryBlock(block);

    auto *allocation_begin = static_cast<std::uint8_t *>(allocation_.base());
    auto *allocation_end = allocation_begin + total_size;
    auto *code_begin = aligned_pointer(allocation_begin, code_segment_align);
    auto *ro_begin = aligned_pointer(code_begin + code_reserved_,
                                     ro_segment_align);
    auto *rw_begin =
        aligned_pointer(ro_begin + ro_reserved_, rw_segment_align);
    TI_ASSERT(rw_begin <= allocation_end &&
              rw_reserved_ <=
                  static_cast<std::uintptr_t>(allocation_end - rw_begin));
    code_ = {code_begin, code_begin, code_begin + code_reserved_};
    ro_ = {ro_begin, ro_begin, ro_begin + ro_reserved_};
    rw_ = {rw_begin, rw_begin, rw_begin + rw_reserved_};
    code_.alignment = code_align.value();
    ro_.alignment = ro_align.value();
    rw_.alignment = rw_align.value();
  }

  std::uint8_t *allocateCodeSection(uintptr_t size,
                                    unsigned alignment,
                                    unsigned,
                                    StringRef) override {
    return allocate(code_, size, alignment);
  }

  std::uint8_t *allocateDataSection(uintptr_t size,
                                    unsigned alignment,
                                    unsigned,
                                    StringRef,
                                    bool read_only) override {
    return allocate(read_only ? ro_ : rw_, size, alignment);
  }

  bool finalizeMemory(std::string *error_message = nullptr) override {
    if (protect(code_.begin, code_reserved_,
                sys::Memory::MF_READ | sys::Memory::MF_EXEC,
                error_message)) {
      return true;
    }
    if (protect(ro_.begin, ro_reserved_, sys::Memory::MF_READ,
                error_message)) {
      return true;
    }
    if (code_reserved_ != 0) {
      sys::Memory::InvalidateInstructionCache(code_.begin, code_reserved_);
    }
    return false;
  }

 private:
  struct Segment {
    std::uint8_t *begin{nullptr};
    std::uint8_t *cursor{nullptr};
    std::uint8_t *end{nullptr};
    std::uintptr_t alignment{1};
  };

  static std::uintptr_t align_up(std::uintptr_t value,
                                 std::uintptr_t alignment) {
    TI_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    return (value + alignment - 1) & ~(alignment - 1);
  }

  static void add_allocation_size(std::uintptr_t &total,
                                  std::uintptr_t size,
                                  std::uintptr_t alignment) {
    TI_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    constexpr auto max_size = std::numeric_limits<std::uintptr_t>::max();
    TI_ERROR_IF(size > max_size - total ||
                    alignment - 1 > max_size - total - size,
                "Ordered COFF JIT allocation size overflow");
    total += size + alignment - 1;
  }

  static std::uint8_t *aligned_pointer(std::uint8_t *pointer,
                                       std::uintptr_t alignment) {
    return reinterpret_cast<std::uint8_t *>(
        align_up(reinterpret_cast<std::uintptr_t>(pointer), alignment));
  }

  static std::uint8_t *allocate(Segment &segment,
                                std::uintptr_t size,
                                std::uintptr_t alignment) {
    alignment = std::max<std::uintptr_t>(alignment, segment.alignment);
    auto address = align_up(reinterpret_cast<std::uintptr_t>(segment.cursor),
                            alignment);
    auto *result = reinterpret_cast<std::uint8_t *>(address);
    if (result < segment.begin || result > segment.end ||
        size > static_cast<std::uintptr_t>(segment.end - result)) {
      return nullptr;
    }
    segment.cursor = result + size;
    return result;
  }

  static bool protect(std::uint8_t *begin,
                      std::uintptr_t size,
                      unsigned flags,
                      std::string *error_message) {
    if (size == 0) {
      return false;
    }
    auto block = sys::MemoryBlock(begin, size);
    const auto error = sys::Memory::protectMappedMemory(block, flags);
    if (!error) {
      return false;
    }
    if (error_message != nullptr) {
      *error_message = error.message();
    }
    return true;
  }

  sys::OwningMemoryBlock allocation_;
  Segment code_;
  Segment ro_;
  Segment rw_;
  std::uintptr_t code_reserved_{0};
  std::uintptr_t ro_reserved_{0};
  std::uintptr_t rw_reserved_{0};
};

}  // namespace
#endif

std::pair<JITTargetMachineBuilder, llvm::DataLayout> get_host_target_info() {
  auto expected_jtmb = JITTargetMachineBuilder::detectHost();
  if (!expected_jtmb)
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  auto jtmb = *expected_jtmb;
  auto expected_data_layout = jtmb.getDefaultDataLayoutForTarget();
  if (!expected_data_layout) {
    TI_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }
  auto data_layout = *expected_data_layout;
  return std::make_pair(jtmb, data_layout);
}

class JITSessionCPU;

class JITModuleCPU : public JITModule {
 private:
  friend class JITSessionCPU;
  JITSessionCPU *session_;
  JITDylib *dylib_;

 public:
  JITModuleCPU(JITSessionCPU *session, JITDylib *dylib)
      : session_(session), dylib_(dylib) {
  }

  void *lookup_function(const std::string &name) override;

  bool direct_dispatch() const override {
    return true;
  }
};

class JITSessionCPU : public JITSession {
 private:
  ExecutionSession es_;
  ObjLayerT object_layer_;
  IRCompileLayer compile_layer_;
  DataLayout dl_;
  MangleAndInterner mangle_;
  std::mutex mut_;
  std::vector<llvm::orc::JITDylib *> all_libs_;
  int module_counter_;

 public:
  JITSessionCPU(TaichiLLVMContext *tlctx,
                std::unique_ptr<ExecutorProcessControl> EPC,
                const CompileConfig &config,
                JITTargetMachineBuilder JTMB,
                DataLayout DL)
      : JITSession(tlctx, config),
        es_(std::move(EPC)),
#if defined(__APPLE__) && defined(__aarch64__)
        object_layer_(es_),
#else
        object_layer_(es_,
                      [&]() {
#if defined(_WIN32)
                        return std::make_unique<
                            OrderedCoffSectionMemoryManager>();
#else
                        return std::make_unique<SectionMemoryManager>();
#endif
                      }),
#endif
        compile_layer_(es_,
                       object_layer_,
                       std::make_unique<ConcurrentIRCompiler>(JTMB)),
        dl_(DL),
        mangle_(es_, this->dl_),
        module_counter_(0) {
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
      object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
    }
  }

  ~JITSessionCPU() override {
    std::lock_guard<std::mutex> _(mut_);
    if (auto Err = es_.endSession())
      es_.reportError(std::move(Err));
  }

  DataLayout get_data_layout() override {
    return dl_;
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M,
                        int max_reg,
                        [[maybe_unused]] JITModuleRole role) override {
    TI_ASSERT(max_reg == 0);  // No need to specify max_reg on CPUs
    TI_ASSERT(M);
    std::lock_guard<std::mutex> _(mut_);
    auto dylib_expect = es_.createJITDylib(fmt::format("{}", module_counter_));
    TI_ASSERT(dylib_expect);
    auto &dylib = dylib_expect.get();
    dylib.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dl_.getGlobalPrefix())));
    auto *thread_safe_context =
        this->tlctx_->get_this_thread_thread_safe_context();
    cantFail(compile_layer_.add(
        dylib,
        llvm::orc::ThreadSafeModule(std::move(M), *thread_safe_context)));
    all_libs_.push_back(&dylib);
    auto new_module = std::make_unique<JITModuleCPU>(this, &dylib);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    module_counter_++;
    return new_module_raw_ptr;
  }

  bool remove_module(JITModule *module) override {
    std::lock_guard<std::mutex> _(mut_);
    auto module_it = std::find_if(
        modules.begin(), modules.end(),
        [module](const std::unique_ptr<JITModule> &owned) {
          return owned.get() == module;
        });
    if (module_it == modules.end()) {
      return false;
    }
    auto *cpu_module = dynamic_cast<JITModuleCPU *>(module_it->get());
    TI_ASSERT(cpu_module != nullptr);
    auto lib_it =
        std::find(all_libs_.begin(), all_libs_.end(), cpu_module->dylib_);
    TI_ASSERT(lib_it != all_libs_.end());
    cantFail(es_.removeJITDylib(*cpu_module->dylib_));
    // RTDyldObjectLinkingLayer owns the per-object memory manager and performs
    // EH-frame deregistration before releasing it.
    all_libs_.erase(lib_it);
    modules.erase(module_it);
    return true;
  }

  void *lookup(const std::string Name) override {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup(all_libs_, mangle_(Name));
#else
    auto symbol = es_.lookup(all_libs_, es_.intern(Name));
#endif
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return symbol->getAddress().toPtr<void *>();
  }

  void *lookup_in_module(JITDylib *lib, const std::string Name) {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup({lib}, mangle_(Name));
#else
    auto symbol = es_.lookup({lib}, es_.intern(Name));
#endif
    if (!symbol)
      TI_ERROR("Function \"{}\" not found", Name);
    return symbol->getAddress().toPtr<void *>();
  }
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session_->lookup_in_module(dylib_, name);
}

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(
    TaichiLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  TI_ASSERT(arch_is_cpu(arch));
  auto target_info = get_host_target_info();
  auto EPC = SelfExecutorProcessControl::Create();
  TI_ASSERT(EPC);
  return std::make_unique<JITSessionCPU>(tlctx, std::move(*EPC), config,
                                         target_info.first, target_info.second);
}

}  // namespace taichi::lang
