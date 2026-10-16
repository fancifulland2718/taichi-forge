#pragma once

#include <array>
#include <optional>
#include "taichi/runtime/native_operation_description.h"
#include "taichi/external_runtime/forge_runtime_provider.h"

namespace taichi::lang {

struct CusparseLtPlanDescription {
  static constexpr const char *kind = "cusparselt_matmul_plan_f16";
  static constexpr uint32_t version = 1;
  int64_t m{0}, n{0}, k{0};
  uint32_t alignment_bytes{16};
  // Frozen recipes require exact configuration and storage reconstruction.
  // Compressed A, compression scratch, then matmul workspace, in bytes.
  std::optional<std::array<uint64_t, 3>> expected_resources;
  TiForgeCusparseLtMatmulConfig configuration{
      sizeof(TiForgeCusparseLtMatmulConfig),
      -1,
      TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT,
      TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT,
      TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT,
      0};
};

class CusparseLtPlanResource : public NativeResource {
 public:
  const CusparseLtPlanDescription &description() const {
    return description_;
  }

 protected:
  explicit CusparseLtPlanResource(CusparseLtPlanDescription description)
      : description_(std::move(description)) {
  }
  CusparseLtPlanDescription description_;
};

struct CusparseLtCompressOperation {
  static constexpr const char *kind = "cusparselt_compress_f16";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CusparseLtPlanResource> plan;
  NativeArrayBinding a;
};

struct CusparseLtMatmulOperation {
  static constexpr const char *kind = "cusparselt_matmul_f16";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CusparseLtPlanResource> plan;
  std::array<NativeArrayBinding, 3> arrays;  // B(n,k), C(m,n), D(m,n).
  float alpha{1}, beta{0};
};

}  // namespace taichi::lang
