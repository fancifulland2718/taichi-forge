#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

enum class FftTransformKind { c2c, r2c, c2r };

struct FftLayoutDescription {
  std::vector<int> embed;
  int stride{1};
  int batch_distance{0};
};

// Physical extents use the existing cuFFT PlanMany int32 ABI. Plans are
// reconstructed per prepared owner; several operations may retain one plan.
struct FftPlanDescription {
  static constexpr const char *kind = "cufft_plan_f32";
  static constexpr uint32_t version = 1;
  std::vector<int> dimensions;
  int batch_count{1};
  FftTransformKind transform{FftTransformKind::c2c};
  FftLayoutDescription input_layout;
  FftLayoutDescription output_layout;
};

class FftPlanResource : public NativeResource {
 public:
  const FftPlanDescription &description() const {
    return description_;
  }

 protected:
  explicit FftPlanResource(FftPlanDescription description)
      : description_(std::move(description)) {
  }
  FftPlanDescription description_;
};

struct FftExecuteOperation {
  static constexpr const char *kind = "cufft_execute_f32";
  static constexpr uint32_t version = 1;
  std::shared_ptr<FftPlanResource> plan;
  NativeArrayBinding input;
  NativeArrayBinding output;
  bool inverse{false};
};

}  // namespace taichi::lang
