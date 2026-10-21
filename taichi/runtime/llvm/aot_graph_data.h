#pragma once
#include "taichi/util/lang_util.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_loader.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"

namespace taichi::lang {
namespace llvm_aot {

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(LLVM::KernelLauncher *launcher,
                      KernelLaunchHandle handle,
                      Arch target_arch,
                      const std::string &kernel_name,
                      const LLVM::KernelCallContract &kernel_data)
      : launcher_(launcher), handle_(handle) {
    TI_ASSERT(launcher_ != nullptr);
    TI_ASSERT(handle_.get_launch_id() >= 0);
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
  }

  void launch(LaunchContextBuilder &ctx) override {
    launcher_->launch_llvm_kernel(handle_, ctx);
  }

  LLVM::KernelLauncher &launcher() const {
    return *launcher_;
  }

  KernelLaunchHandle handle() const {
    return handle_;
  }

 private:
  // The module owns both the launcher and its registrations. Prepared users
  // must retain the module, just like ordinary AOT kernel/graph consumers.
  // Keep the loaded entry accessible without retaining compiler IR or
  // registering a second copy when preparing backend replay.
  LLVM::KernelLauncher *const launcher_;
  const KernelLaunchHandle handle_;
};

class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(const LlvmOfflineCache::FieldCacheData &field)
      : field_(field) {
  }

  explicit FieldImpl(LlvmOfflineCache::FieldCacheData &&field)
      : field_(std::move(field)) {
  }

  LlvmOfflineCache::FieldCacheData get_snode_tree_cache() const {
    return field_;
  }

 private:
  LlvmOfflineCache::FieldCacheData field_;
};

}  // namespace llvm_aot
}  // namespace taichi::lang
