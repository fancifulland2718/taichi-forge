#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang::cuda {

struct CudaCGScalarState {
  float rr_current{0.0f};
  float rr_next{0.0f};
  float rho_current{0.0f};
  float rho_next{0.0f};
  float p_ap{0.0f};
  float alpha{0.0f};
  float negative_alpha{0.0f};
  float beta{1.0f};
  float source_scale{0.0f};
  float one{1.0f};
  float zero{0.0f};
  float negative_one{-1.0f};
  float tolerance_squared{0.0f};
  float rhs_squared{0.0f};
  float initial_rr{0.0f};
  float relative_reference_norm{0.0f};
  float effective_tolerance{0.0f};
  float absolute_tolerance{0.0f};
  float relative_tolerance{0.0f};
  std::int32_t status{0};
  std::int32_t completed_iterations{0};
  std::int32_t active{1};
  std::int32_t has_preconditioner{0};
};

static_assert(sizeof(CudaCGScalarState) == 92);
static_assert(offsetof(CudaCGScalarState, completed_iterations) == 80);
static_assert(offsetof(CudaCGScalarState, active) == 84);

bool driver_cg_scalar_available();
bool driver_cg_conditional_setter_compiled();
void driver_cg_prepare_conditional_setter();
void driver_cg_initialize(CudaCGScalarState *state, void *stream = nullptr);
void driver_cg_validate_rho(CudaCGScalarState *state,
                            void *stream = nullptr);
void driver_cg_prepare_alpha(CudaCGScalarState *state,
                             void *stream = nullptr);
void driver_cg_finish_iteration(CudaCGScalarState *state,
                                void *stream = nullptr);
void driver_cg_prepare_direction(CudaCGScalarState *state,
                                 void *stream = nullptr);
void driver_cg_set_conditional(CudaCGScalarState *state,
                               std::uint64_t conditional_handle,
                               int max_iterations,
                               void *stream = nullptr);

}  // namespace taichi::lang::cuda
