#pragma once
#ifdef TI_WITH_LLVM

#include "taichi_core_impl.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/native_resource.h"

namespace capi {

// Owned empty states for the exact fixed-tree schemas of one loaded module.
// The code lease also keeps the prototype's common allocator backing alive.
class SNodeInstance final : public taichi::lang::NativeResource {
 public:
  explicit SNodeInstance(AotModule &module);
  ~SNodeInstance() override;
  void initialize();
  void validate_owner(AotModule &module) const;
  std::vector<LlvmSparseRuntimeViewBinding> bindings_for(
      taichi::lang::LlvmRuntimeExecutor &executor,
      const std::vector<int> &code_trees) const;
  std::string info() const;
  taichi::Arch arch() const override;
  taichi::lang::Device &device() const override;
  std::size_t workspace_bytes() const override { return 0; }

 private:
  std::shared_ptr<taichi::lang::aot::Module> code_;
  taichi::lang::LLVM::LlvmAotModule *module_;
  taichi::lang::LlvmRuntimeExecutor *executor_;
  std::vector<int> reserved_;
  std::vector<taichi::lang::LLVM::SNodeTreeLayout> states_;
  std::vector<LlvmSparseRuntimeViewBinding> bindings_;
};

}  // namespace capi
#endif
