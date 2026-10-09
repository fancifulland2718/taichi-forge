#pragma once
#include "taichi/util/lang_util.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_loader.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"

namespace taichi::lang {
namespace llvm_aot {

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(FunctionType fn,
                      const std::string &kernel_name,
                      const LLVM::CompiledKernelData::InternalData &kernel_data)
      : fn_(std::move(fn)) {
    rets = kernel_data.rets;
    ret_type = kernel_data.ret_type;
    ret_size = kernel_data.ret_size;
    nested_parameters.reserve(kernel_data.args.size());
    for (const auto &kv : kernel_data.args) {
      nested_parameters[kv.first] = kv.second;
    }
    args_type = kernel_data.args_type;
    args_size = kernel_data.args_size;
    arch = Arch::x64;  // Only for letting the launch context builder know
                       // the arch uses LLVM.
                       // TODO: remove arch after the refactoring of
                       //  SPIR-V based backends completes.
    name = kernel_name;
  }

  void launch(LaunchContextBuilder &ctx) override {
    fn_(ctx);
  }

 private:
  FunctionType fn_;
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
