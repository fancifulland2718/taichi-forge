#pragma once

#include <optional>
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "taichi/rhi/device_capability.h"
#include "taichi/rhi/cuda/cuda_capability.h"

namespace taichi::lang {

// Requirements of already-lowered kernel IR. BC can be finalized for a newer
// consumer, whereas PTX/cubin export explicitly selects its deployment target.
// Runtime modules have their own current-device policy and no kernel record.
struct CUDALoweringTarget {
  int compute_capability;
  int ptx_version;

  void validate() const {
    const auto resolved = cuda::detail::resolve_compute_capability_target(compute_capability);
    TI_ERROR_IF(compute_capability < 20 ||
                    resolved.codegen_compute_capability != compute_capability ||
                    resolved.ptx_version != ptx_version,
                "Invalid CUDA kernel lowering target {} / PTX {}",
                compute_capability, ptx_version);
  }

  static CUDALoweringTarget from_caps(const DeviceCapabilityConfig &caps) {
    CUDALoweringTarget result{
        static_cast<int>(caps.get(DeviceCapability::cuda_compute_capability)),
        static_cast<int>(caps.get(DeviceCapability::cuda_ptx_version))};
    result.validate();
    return result;
  }

  void write(llvm::Module &module) const {
    validate();
    llvm::IRBuilder<> builder(module.getContext());
    auto *metadata = module.getOrInsertNamedMetadata("taichi.cuda.lowering_target.v1");
    metadata->clearOperands();
    metadata->addOperand(llvm::MDNode::get(
        module.getContext(),
        {llvm::ConstantAsMetadata::get(builder.getInt32(compute_capability)),
         llvm::ConstantAsMetadata::get(builder.getInt32(ptx_version))}));
    // The shared runtime template is specialized for the producer's device.
    // Only rewrite this owned kernel copy before inlining helpers; never
    // mutate the runtime template or the producer's other compiled kernels.
    if (auto *function = module.getFunction("cuda_compute_capability")) {
      function->deleteBody();
      builder.SetInsertPoint(llvm::BasicBlock::Create(module.getContext(), "entry", function));
      builder.CreateRet(builder.getInt32(compute_capability));
    }
  }

  static std::optional<CUDALoweringTarget> read(const llvm::Module &module) {
    auto *metadata = module.getNamedMetadata("taichi.cuda.lowering_target.v1");
    if (!metadata)
      return std::nullopt;
    TI_ERROR_IF(metadata->getNumOperands() != 1 || metadata->getOperand(0)->getNumOperands() != 2,
                "Malformed CUDA kernel lowering target");
    auto integer = [&](unsigned index) {
      auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(metadata->getOperand(0)->getOperand(index));
      TI_ERROR_IF(!value || value->getBitWidth() != 32, "Malformed CUDA kernel lowering target");
      return static_cast<int>(value->getSExtValue());
    };
    CUDALoweringTarget result{integer(0), integer(1)};
    result.validate();
    return result;
  }

  void require_finalization_target(llvm::StringRef cpu, llvm::StringRef features) const {
    unsigned final_sm = 0, final_ptx = 0;
    bool valid = cpu.consume_front("sm_") && !cpu.getAsInteger(10, final_sm);
    llvm::SmallVector<llvm::StringRef, 8> values;
    features.split(values, ',');
    for (auto value : values)
      if (value.consume_front("+ptx")) {
        valid &= !value.getAsInteger(10, final_ptx);
        break;
      }
    TI_ERROR_IF(!valid || final_sm < static_cast<unsigned>(compute_capability) ||
                    final_ptx < static_cast<unsigned>(ptx_version),
                "CUDA finalization target is below kernel lowering requirements sm_{} / PTX {}",
                compute_capability, ptx_version);
  }
};

}  // namespace taichi::lang
