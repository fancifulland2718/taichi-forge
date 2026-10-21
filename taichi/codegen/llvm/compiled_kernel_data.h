#pragma once

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/codegen/llvm/kernel_call_contract.h"

#include "llvm/IR/LLVMContext.h"

namespace taichi::lang {

namespace LLVM {

class CompiledKernelData : public lang::CompiledKernelData {
 public:
  struct InternalData : KernelCallContract {
    LLVMCompiledKernel compiled_data;
    bool may_trigger_hash_overflow{false};

    TI_IO_DEF(args,
              rets,
              compiled_data,
              used_snode_tree_ids,
              may_trigger_hash_overflow,
              graph_metadata,
              ret_type,
              ret_size,
              args_type,
              args_size);

    InternalData() = default;

    InternalData(const InternalData &o)
        : KernelCallContract(o),
          compiled_data(o.compiled_data.clone()),
          may_trigger_hash_overflow(o.may_trigger_hash_overflow) {
    }

    InternalData(InternalData &&o) = default;
  };

  CompiledKernelData() = default;
  CompiledKernelData(Arch arch, InternalData data);

  Arch arch() const override;
  std::unique_ptr<lang::CompiledKernelData> clone() const override;
  std::vector<int> snode_tree_ids() const override;
  bool has_snode_tree_dependencies() const noexcept override;
  bool may_trigger_hash_overflow() const noexcept override;
  std::size_t task_count() const override;
  std::vector<OffloadedTaskManifest> task_manifest() const override;
  const GraphKernelMetadata &graph_metadata() const override {
    return data_.graph_metadata;
  }
  void set_graph_metadata(GraphKernelMetadata metadata) override {
    data_.graph_metadata = std::move(metadata);
  }

  Err check() const override;

  const InternalData &get_internal_data() const {
    return data_;
  }

 protected:
  void refresh_task_identities() override;
  Err load_impl(const CompiledKernelDataFile &file) override;
  Err dump_impl(CompiledKernelDataFile &file) const override;

 private:
  Arch arch_;
  InternalData data_;
};

}  // namespace LLVM

}  // namespace taichi::lang
