#include "jit_module.h"

#include <atomic>
#include <map>
#include "taichi_core_impl.h"
#include "taichi/program/program.h"
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"
#endif

namespace capi {
using namespace taichi::lang;
using taichi::Arch;
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
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

class JitKernel final : public llvm_aot::KernelImpl {
 public:
  JitKernel(cuda::KernelLauncher &launcher,
            const std::string &name,
            const BorrowedKernel &kernel)
      : llvm_aot::KernelImpl(&launcher,
                             kernel.launch,
                             Arch::cuda,
                             name,
                             static_cast<const LLVM::CompiledKernelData &>(
                                 kernel.execution->compiled())
                                 .get_internal_data()),
        dispatch_(kernel.dispatch) {
    // Invocation metadata is frozen; never dereference a frontend shell
    // after the cold adoption transaction.
    dispatch_.ti_kernel = nullptr;
  }
  LaunchContextBuilder make_context(bool cpu_bounded_range) const {
    return dispatch_.make_launch_context(cpu_bounded_range);
  }

 private:
  aot::CompiledDispatch dispatch_;
};

class JitModule final : public aot::Module {
 public:
  JitModule(Program &program,
            cuda::KernelLauncher &launcher,
            std::map<std::string, BorrowedKernel> kernels,
            std::shared_ptr<JitCodeRetirement> retirement)
      : program_(program),
        launcher_(launcher),
        kernels_(std::move(kernels)),
        retirement_(std::move(retirement)) {
  }
  ~JitModule() override {
    retirement_->module.store(nullptr, std::memory_order_release);
    program_.detach_runtime_client(retirement_.get());
  }
  Arch arch() const override {
    return Arch::cuda;
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
    return std::make_unique<JitKernel>(launcher_, name, entry);
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
  cuda::KernelLauncher &launcher_;
  std::map<std::string, BorrowedKernel> kernels_;
  std::shared_ptr<JitCodeRetirement> retirement_;
};
}  // namespace
#endif

std::uintptr_t import_jit_module(
    Program &program,
    ::Runtime &runtime,
    const std::vector<AotRuntimeClient::JitKernelImport> &kernels) {
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
  if (program.compile_config().arch != Arch::cuda || kernels.empty())
    throw std::invalid_argument(
        "Selected JIT code borrowing requires CUDA kernels");
  auto *launcher = dynamic_cast<cuda::KernelLauncher *>(
      &program.get_program_impl()->get_kernel_launcher());
  if (!launcher)
    throw std::invalid_argument("Selected JIT code has no CUDA launcher");
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
    const auto *compiled =
        dynamic_cast<const LLVM::CompiledKernelData *>(&execution->compiled());
    if (!compiled || compiled->arch() != Arch::cuda)
      throw std::invalid_argument("JIT Graph needs its selected CUDA code");
    const auto *blocker =
        independent_runtime_state_blocker(LLVM::runtime_state_use(
            *compiled->get_internal_data().compiled_data.module));
    if (blocker)
      throw std::invalid_argument(blocker);
    program.validate_snode_tree_dependencies(dispatch.snode_tree_dependencies,
                                             "JIT code loan");
    for (int tree_id : compiled->snode_tree_ids()) {
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
    const auto &compiled = static_cast<const LLVM::CompiledKernelData &>(
        entry.execution->compiled());
    entry.launch = gated ? launcher->register_llvm_kernel_graph_gated(
                               compiled, entry.execution->registration())
                         : launcher->register_llvm_kernel(
                               compiled, entry.execution->registration());
  }
  auto retirement = std::make_shared<JitCodeRetirement>(runtime);
  auto module = std::make_unique<JitModule>(program, *launcher,
                                            std::move(selected), retirement);
  program.retain_runtime_client(retirement, dependencies);
  auto owner = std::make_unique<AotModule>(runtime, std::move(module));
  retirement->module.store(owner.get(), std::memory_order_release);
  return reinterpret_cast<std::uintptr_t>(owner.release());
#else
  throw std::invalid_argument("JIT code borrowing requires LLVM/CUDA support");
#endif
}
std::unique_ptr<LaunchContextBuilder> make_borrowed_launch_context(
    aot::Kernel *kernel,
    bool cpu_bounded_range) {
#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
  if (const auto *borrowed = dynamic_cast<const JitKernel *>(kernel))
    return std::make_unique<LaunchContextBuilder>(
        borrowed->make_context(cpu_bounded_range));
#endif
  return std::make_unique<LaunchContextBuilder>(kernel, cpu_bounded_range);
}
}  // namespace capi
