#include "jit_module.h"

#include <atomic>
#include <functional>
#include <map>
#include "taichi_core_impl.h"
#include "taichi/program/program.h"
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/runtime/gfx/kernel_launcher.h"
#endif

namespace capi {
using namespace taichi::lang;
using taichi::Arch;
namespace {
class JitCodeRetirement final : public RuntimeClient {
 public:
  explicit JitCodeRetirement(::Runtime &runtime) : runtime_(runtime) {
  }
  void retire() override {
    if (!module.load(std::memory_order_acquire))
      return;
    // Program tree destruction holds its resource guard; do not enter the
    // borrower's state mutex in the reverse order. C API mutation uses this
    // same Runtime gate, and owner retirement serializes registry callbacks.
    std::lock_guard<std::recursive_mutex> lock(runtime_.submission_mutex());
    if (auto *live = module.load(std::memory_order_acquire))
      live->retire_borrowed_code();
  }
  std::atomic<AotModule *> module{nullptr};

 private:
  ::Runtime &runtime_;
};

struct BorrowedKernel {
  std::shared_ptr<KernelExecutionHandle> execution;
  KernelLaunchHandle launch;
  aot::CompiledDispatch dispatch;
};

// Both backend adapters retain the selected dispatch's frozen invocation ABI.
// No frontend shell or second kernel compilation is needed after adoption.
class JitInvocation {
 public:
  JitInvocation(aot::Kernel &kernel, const BorrowedKernel &selected)
      : dispatch_(selected.dispatch) {
    const auto &callable = *dispatch_.jit_callable();
    kernel.argument_origins = callable.argument_origins;
    kernel.value_parameter_types = callable.value_parameter_types;
    kernel.argpack_types = callable.argpack_types;
    dispatch_.ti_kernel = nullptr;
  }
  virtual ~JitInvocation() = default;

  LaunchContextBuilder make_context(bool cpu_bounded_range) const {
    return dispatch_.make_launch_context(cpu_bounded_range);
  }

 private:
  aot::CompiledDispatch dispatch_;
};

#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
class CudaJitKernel final : public llvm_aot::KernelImpl,
                            public JitInvocation {
 public:
  CudaJitKernel(cuda::KernelLauncher &launcher,
                const std::string &name,
                const BorrowedKernel &kernel)
      : llvm_aot::KernelImpl(&launcher,
                             kernel.launch,
                             Arch::cuda,
                             name,
                             static_cast<const LLVM::CompiledKernelData &>(
                                 kernel.execution->compiled())
                                 .get_internal_data()),
        JitInvocation(*this, kernel) {
  }
};
#endif

#ifdef TI_WITH_VULKAN
class VulkanJitKernel final : public gfx::KernelImpl, public JitInvocation {
 public:
  VulkanJitKernel(gfx::KernelLauncher &launcher, const BorrowedKernel &kernel)
      : gfx::KernelImpl(launcher.runtime(), parameters(kernel), &kernel.launch),
        JitInvocation(*this, kernel) {
  }

 private:
  static gfx::GfxRuntime::RegisterParams parameters(
      const BorrowedKernel &kernel) {
    const auto &compiled = static_cast<const spirv::CompiledKernelData &>(
        kernel.execution->compiled());
    const auto &metadata = compiled.get_internal_data().metadata;
    gfx::GfxRuntime::RegisterParams params;
    params.kernel_attribs = metadata.kernel_attribs;
    params.num_snode_trees = metadata.num_snode_trees;
    params.snode_tree_ids = compiled.snode_tree_ids();
    // The selected handle already owns its pipelines. Deliberately keep the
    // shader source empty rather than copying/re-registering it for this loan.
    return params;
  }
};
#endif

class JitModule final : public aot::Module {
 public:
  using KernelFactory = std::function<std::unique_ptr<aot::Kernel>(
      const std::string &, const BorrowedKernel &)>;
  JitModule(Program &program,
            Arch arch,
            KernelFactory factory,
            std::map<std::string, BorrowedKernel> kernels,
            std::shared_ptr<JitCodeRetirement> retirement)
      : program_(program),
        arch_(arch),
        factory_(std::move(factory)),
        kernels_(std::move(kernels)),
        retirement_(std::move(retirement)) {
  }
  ~JitModule() override {
    retirement_->module.store(nullptr, std::memory_order_release);
    program_.detach_runtime_client(retirement_.get());
  }
  Arch arch() const override {
    return arch_;
  }
  std::uint64_t version() const override {
    return 0;
  }
  // Field storage stays in the Program. No module-owned roots are allocated.
  std::size_t get_root_size() const override {
    return 0;
  }
  std::vector<std::size_t> get_root_sizes() const override {
    return {};
  }

 protected:
  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    const auto found = kernels_.find(name);
    if (found == kernels_.end())
      throw std::invalid_argument("Unknown borrowed JIT kernel entry");
    const auto &entry = found->second;
    if (!entry.execution->active())
      throw std::invalid_argument(
          "Borrowed JIT execution artifact has retired");
    return factory_(name, entry);
  }
  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &) override {
    throw std::invalid_argument("A JIT code loan has no kernel templates");
  }
  std::unique_ptr<aot::Field> make_new_field(const std::string &) override {
    throw std::invalid_argument("A JIT code loan has no module-owned fields");
  }

 private:
  Program &program_;
  const Arch arch_;
  KernelFactory factory_;
  std::map<std::string, BorrowedKernel> kernels_;
  std::shared_ptr<JitCodeRetirement> retirement_;
};
}  // namespace

std::uintptr_t import_jit_module(
    Program &program,
    ::Runtime &runtime,
    const std::vector<AotRuntimeClient::JitKernelImport> &kernels) {
  const auto arch = program.compile_config().arch;
  if (kernels.empty())
    throw std::invalid_argument(
        "Selected JIT code borrowing requires nonempty kernel entries");
  auto &launcher = program.get_kernel_launcher();
  JitModule::KernelFactory factory;
  std::function<KernelLaunchHandle(const BorrowedKernel &, bool)> register_kernel;
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
  if (arch == Arch::cuda) {
    auto *cuda_launcher = dynamic_cast<cuda::KernelLauncher *>(&launcher);
    if (!cuda_launcher)
      throw std::invalid_argument("Selected JIT code has no CUDA launcher");
    factory = [cuda_launcher](const auto &name, const auto &entry) {
      return std::make_unique<CudaJitKernel>(*cuda_launcher, name, entry);
    };
    register_kernel = [cuda_launcher](const auto &entry, bool gated) {
      const auto &compiled = static_cast<const LLVM::CompiledKernelData &>(
          entry.execution->compiled());
      return gated ? cuda_launcher->register_llvm_kernel_graph_gated(
                         compiled, entry.execution->registration())
                   : cuda_launcher->register_llvm_kernel(
                         compiled, entry.execution->registration());
    };
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    auto *gfx_launcher = dynamic_cast<gfx::KernelLauncher *>(&launcher);
    if (!gfx_launcher)
      throw std::invalid_argument("Selected JIT code has no Vulkan launcher");
    factory = [gfx_launcher](const auto &, const auto &entry) {
      return std::make_unique<VulkanJitKernel>(*gfx_launcher, entry);
    };
    register_kernel = [gfx_launcher](const auto &entry, bool) {
      return gfx_launcher->get_or_register_kernel(
          entry.execution->compiled(), entry.execution->registration());
    };
  }
#endif
  if (!factory)
    throw std::invalid_argument(
        "Selected JIT code borrowing requires CUDA or Vulkan support");
  std::map<std::string, BorrowedKernel> selected;
  std::vector<SNodeTreeDependency> dependencies;
  // Validate the complete code closure before registering any executable.
  for (const auto &[name, graph, index, gated] : kernels) {
    if (name.empty() || name.find('\0') != std::string::npos || !graph ||
        index >= graph->dispatches.size())
      throw std::invalid_argument("Invalid selected JIT kernel import");
    const auto &dispatch = graph->dispatches[index];
    if (!dispatch.jit_belongs_to_program(program) ||
        dispatch.cuda_capture_command)
      throw std::invalid_argument("JIT kernel does not belong to this Program");
    const auto &execution = dispatch.jit_execution_handle();
    if (!execution || !execution->active())
      throw std::invalid_argument(
          "JIT Graph needs its live compiled execution artifact");
    const auto &compiled = execution->compiled();
    if (compiled.arch() != arch)
      throw std::invalid_argument("JIT Graph needs its selected backend code");
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
    if (arch == Arch::cuda) {
      const auto *llvm = dynamic_cast<const LLVM::CompiledKernelData *>(&compiled);
      if (!llvm)
        throw std::invalid_argument("JIT Graph needs its selected CUDA code");
      const auto *blocker =
          independent_runtime_state_blocker(LLVM::runtime_state_use(
              *llvm->get_internal_data().compiled_data.module));
      if (blocker)
        throw std::invalid_argument(blocker);
    }
#endif
#ifdef TI_WITH_VULKAN
    if (arch == Arch::vulkan) {
      if (!dynamic_cast<const spirv::CompiledKernelData *>(&compiled))
        throw std::invalid_argument("JIT Graph needs its selected Vulkan code");
      if (gated)
        throw std::invalid_argument(
            "Vulkan JIT code loans do not support CUDA gated launch ABI");
    }
#endif
    program.validate_snode_tree_dependencies(dispatch.snode_tree_dependencies,
                                             "JIT code loan");
    for (int tree_id : compiled.snode_tree_ids()) {
      const auto found = std::find_if(dispatch.snode_tree_dependencies.begin(),
                                      dispatch.snode_tree_dependencies.end(),
                                      [tree_id](const auto &dependency) {
                                        return dependency.tree_id == tree_id;
                                      });
      if (found == dispatch.snode_tree_dependencies.end())
        throw std::invalid_argument(
            "JIT code lacks its Field generation proof");
      dependencies.push_back(*found);
    }
    if (!selected.emplace(name, BorrowedKernel{execution, {}, dispatch}).second)
      throw std::invalid_argument("Duplicate borrowed JIT entry name");
  }
  std::sort(dependencies.begin(), dependencies.end());
  dependencies.erase(std::unique(dependencies.begin(), dependencies.end()),
                     dependencies.end());
  program.validate_snode_tree_dependencies(dependencies, "JIT code loan");
  for (const auto &[name, graph, index, gated] : kernels) {
    auto &entry = selected.at(name);
    entry.launch = register_kernel(entry, gated);
  }
  auto retirement = std::make_shared<JitCodeRetirement>(runtime);
  auto module = std::make_unique<JitModule>(program, arch, std::move(factory),
                                          std::move(selected), retirement);
  program.retain_runtime_client(retirement, dependencies);
  auto owner = std::make_unique<AotModule>(runtime, std::move(module));
  retirement->module.store(owner.get(), std::memory_order_release);
  return reinterpret_cast<std::uintptr_t>(owner.release());
}
std::unique_ptr<LaunchContextBuilder> make_borrowed_launch_context(
    aot::Kernel *kernel,
    bool cpu_bounded_range) {
  if (const auto *borrowed = dynamic_cast<const JitInvocation *>(kernel))
    return std::make_unique<LaunchContextBuilder>(
        borrowed->make_context(cpu_bounded_range));
  return std::make_unique<LaunchContextBuilder>(kernel, cpu_bounded_range);
}
}  // namespace capi
