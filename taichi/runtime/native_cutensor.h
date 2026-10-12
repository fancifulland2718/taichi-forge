#pragma once

#include <array>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

struct CutensorTensorDescription {
  std::vector<int64_t> shape;
  std::vector<int32_t> modes;
};

struct CutensorPlanDescription {
  static constexpr const char *kind = "cutensor_contraction_plan_f32";
  static constexpr uint32_t version = 1;
  std::array<CutensorTensorDescription, 4> tensors;
  uint32_t alignment_bytes{256};
  uint32_t compute_mode{0};
  uint32_t workspace_preference{2};
  uint64_t workspace_limit_bytes{0};
};

class CutensorPlanResource : public NativeResource {
 public:
  const CutensorPlanDescription &description() const {
    return description_;
  }

 protected:
  explicit CutensorPlanResource(CutensorPlanDescription description)
      : description_(std::move(description)) {
  }
  CutensorPlanDescription description_;
};

struct CutensorContractOperation {
  static constexpr const char *kind = "cutensor_contract_f32";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CutensorPlanResource> plan;
  std::array<NativeArrayBinding, 4> arrays;
  float alpha{1};
  float beta{0};
};

}  // namespace taichi::lang
