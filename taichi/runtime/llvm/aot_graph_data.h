#pragma once
#include <exception>
#include <functional>
#include <mutex>

#include "taichi/util/lang_util.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_loader.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#include "taichi/codegen/llvm/kernel_call_contract.h"
#include "taichi/runtime/llvm/snode_tree_layout.h"

namespace taichi::lang {
namespace llvm_aot {

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(LLVM::KernelLauncher *launcher,
                      KernelLaunchHandle handle,
                      Arch target_arch,
                      const std::string &kernel_name,
                      const LLVM::KernelCallContract &kernel_data,
                      std::vector<std::vector<int>> affine_arrays = {},
                      std::function<KernelLaunchHandle()> materialize = {})
      : launcher_(launcher), handle_(handle) {
    TI_ASSERT(launcher_ != nullptr);
    TI_ASSERT(handle_.get_launch_id() >= 0 || materialize);
    if (materialize)
      deferred_ = std::make_shared<DeferredRegistration>(std::move(materialize));
    TI_ASSERT(arch_uses_llvm(target_arch));
    rets = kernel_data.rets;
    ret_type = kernel_data.ret_type;
    ret_size = kernel_data.ret_size;
    nested_parameters.reserve(kernel_data.args.size());
    for (const auto &kv : kernel_data.args) {
      nested_parameters[kv.first] = kv.second;
    }
    args_type = kernel_data.args_type;
    args_size = kernel_data.args_size;
    arch = target_arch;
    name = kernel_name;
    runtime_affine_arrays = std::move(affine_arrays);
  }

  void launch(LaunchContextBuilder &ctx) override {
    TI_ERROR_IF(!device_range_tasks_.empty(),
                "AOT device range requires its Graph recording recipe");
    launcher_->launch_llvm_kernel(handle(), ctx);
  }

  // The source is an internal physical callable, before any public entry's
  // value/argument-origin annotations. Only code registration is shared.
  KernelImpl(const KernelImpl &source,
             const std::string &entry_name,
             std::vector<std::uint32_t> range_tasks)
      : launcher_(source.launcher_), handle_(source.handle_),
        deferred_(source.deferred_),
        device_range_tasks_(std::move(range_tasks)) {
    static_cast<CallableBase &>(*this) = static_cast<const CallableBase &>(source);
    runtime_affine_arrays = source.runtime_affine_arrays;
    name = entry_name;
  }

  const std::vector<std::uint32_t> &device_range_tasks() const {
    return device_range_tasks_;
  }

  LLVM::KernelLauncher &launcher() const {
    return *launcher_;
  }

  KernelLaunchHandle handle() const {
    return deferred_ ? deferred_->get() : handle_;
  }

 private:
  struct DeferredRegistration {
    explicit DeferredRegistration(std::function<KernelLaunchHandle()> factory)
        : materialize(std::move(factory)) {
    }

    KernelLaunchHandle get() {
      std::call_once(once, [&] {
        try {
          handle = materialize();
        } catch (...) {
          failure = std::current_exception();
        }
        // Retain neither compiler IR nor the factory after materialization.
        // A failed registration is also final, rather than recompiling on
        // every use of a second public alias.
        materialize = {};
      });
      if (failure)
        std::rethrow_exception(failure);
      return handle;
    }

    std::once_flag once;
    std::function<KernelLaunchHandle()> materialize;
    KernelLaunchHandle handle;
    std::exception_ptr failure;
  };

  // The module owns both the launcher and its registrations. Prepared users
  // must retain the module, just like ordinary AOT kernel/graph consumers.
  // Keep the loaded entry accessible without retaining compiler IR or
  // registering a second copy when preparing backend replay.
  LLVM::KernelLauncher *const launcher_;
  const KernelLaunchHandle handle_;
  std::shared_ptr<DeferredRegistration> deferred_;
  const std::vector<std::uint32_t> device_range_tasks_;
};

class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(const LLVM::SNodeTreeLayout &field) : field_(field) {
  }

  explicit FieldImpl(LLVM::SNodeTreeLayout &&field) : field_(std::move(field)) {
  }

  LLVM::SNodeTreeLayout get_snode_tree_cache() const {
    return field_;
  }

 private:
  LLVM::SNodeTreeLayout field_;
};

}  // namespace llvm_aot
}  // namespace taichi::lang
