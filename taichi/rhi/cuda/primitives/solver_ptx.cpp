#include "taichi/rhi/cuda/primitives/solver_ptx.h"

#include "taichi/common/core.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

#include <mutex>

namespace taichi::lang::cuda {
namespace {

const char kCudaCGScalarPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry cg_initialize(.param .u64 state_param)
{
    .reg .pred %p<10>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<12>;

    ld.param.u64 %rd1, [state_param];
    ld.global.f32 %f1, [%rd1+0];
    ld.global.f32 %f2, [%rd1+52];
    ld.global.f32 %f3, [%rd1+68];
    ld.global.f32 %f4, [%rd1+72];
    st.global.f32 [%rd1+56], %f1;
    mov.u32 %r1, 0;
    mov.u32 %r2, 1;
    st.global.u32 [%rd1+76], %r1;
    st.global.u32 [%rd1+80], %r1;
    st.global.u32 [%rd1+84], %r2;

    mov.b32 %r3, %f1;
    and.b32 %r4, %r3, 2139095040;
    setp.eq.u32 %p1, %r4, 2139095040;
    setp.lt.f32 %p2, %f1, 0f00000000;
    or.pred %p3, %p1, %p2;
    @%p3 bra INIT_BREAKDOWN;

    mov.f32 %f5, 0f00000000;
    setp.le.f32 %p4, %f4, 0f00000000;
    @%p4 bra INIT_REFERENCE_READY;
    mov.b32 %r3, %f2;
    and.b32 %r4, %r3, 2139095040;
    setp.eq.u32 %p5, %r4, 2139095040;
    setp.lt.f32 %p6, %f2, 0f00000000;
    or.pred %p7, %p5, %p6;
    @%p7 bra INIT_BREAKDOWN;
    sqrt.rn.f32 %f5, %f2;

INIT_REFERENCE_READY:
    mul.rn.f32 %f6, %f4, %f5;
    max.f32 %f7, %f3, %f6;
    mul.rn.f32 %f8, %f7, %f7;
    mov.b32 %r3, %f7;
    and.b32 %r4, %r3, 2139095040;
    setp.eq.u32 %p8, %r4, 2139095040;
    mov.b32 %r5, %f8;
    and.b32 %r6, %r5, 2139095040;
    setp.eq.u32 %p9, %r6, 2139095040;
    or.pred %p8, %p8, %p9;
    @%p8 bra INIT_BREAKDOWN;
    st.global.f32 [%rd1+48], %f8;
    st.global.f32 [%rd1+60], %f5;
    st.global.f32 [%rd1+64], %f7;
    setp.le.f32 %p1, %f1, %f8;
    @%p1 bra INIT_CONVERGED;
    ret;

INIT_CONVERGED:
    mov.u32 %r1, 2;
    mov.u32 %r2, 0;
    st.global.u32 [%rd1+76], %r1;
    st.global.u32 [%rd1+84], %r2;
    ret;

INIT_BREAKDOWN:
    mov.u32 %r1, 1;
    mov.u32 %r2, 0;
    st.global.u32 [%rd1+76], %r1;
    st.global.u32 [%rd1+84], %r2;
    ret;
}

.visible .entry cg_validate_rho(.param .u64 state_param)
{
    .reg .pred %p<5>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<3>;
    ld.param.u64 %rd1, [state_param];
    ld.global.u32 %r1, [%rd1+84];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 ret;
    ld.global.f32 %f1, [%rd1+8];
    mov.b32 %r2, %f1;
    and.b32 %r3, %r2, 2139095040;
    setp.eq.u32 %p2, %r3, 2139095040;
    setp.le.f32 %p3, %f1, 0f00000000;
    or.pred %p4, %p2, %p3;
    @!%p4 ret;
    mov.u32 %r4, 1;
    mov.u32 %r5, 0;
    st.global.u32 [%rd1+76], %r4;
    st.global.u32 [%rd1+84], %r5;
    ret;
}

.visible .entry cg_prepare_alpha(.param .u64 state_param)
{
    .reg .pred %p<8>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<8>;
    ld.param.u64 %rd1, [state_param];
    ld.global.u32 %r1, [%rd1+84];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra ALPHA_MASKED;
    ld.global.f32 %f1, [%rd1+16];
    mov.b32 %r2, %f1;
    and.b32 %r3, %r2, 2139095040;
    setp.eq.u32 %p2, %r3, 2139095040;
    setp.le.f32 %p3, %f1, 0f00000000;
    or.pred %p4, %p2, %p3;
    @%p4 bra ALPHA_BREAKDOWN;
    ld.global.u32 %r4, [%rd1+88];
    setp.eq.u32 %p5, %r4, 0;
    @%p5 ld.global.f32 %f2, [%rd1+0];
    @!%p5 ld.global.f32 %f2, [%rd1+8];
    div.rn.f32 %f3, %f2, %f1;
    mov.b32 %r5, %f3;
    and.b32 %r6, %r5, 2139095040;
    setp.eq.u32 %p6, %r6, 2139095040;
    @%p6 bra ALPHA_BREAKDOWN;
    neg.f32 %f4, %f3;
    st.global.f32 [%rd1+20], %f3;
    st.global.f32 [%rd1+24], %f4;
    ret;

ALPHA_BREAKDOWN:
    mov.u32 %r7, 1;
    mov.u32 %r8, 0;
    st.global.u32 [%rd1+76], %r7;
    st.global.u32 [%rd1+84], %r8;
ALPHA_MASKED:
    mov.f32 %f5, 0f00000000;
    st.global.f32 [%rd1+20], %f5;
    st.global.f32 [%rd1+24], %f5;
    ret;
}

.visible .entry cg_finish_iteration(.param .u64 state_param)
{
    .reg .pred %p<7>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<5>;
    ld.param.u64 %rd1, [state_param];
    ld.global.u32 %r1, [%rd1+84];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 ret;
    ld.global.f32 %f1, [%rd1+4];
    mov.b32 %r2, %f1;
    and.b32 %r3, %r2, 2139095040;
    setp.eq.u32 %p2, %r3, 2139095040;
    setp.lt.f32 %p3, %f1, 0f00000000;
    or.pred %p4, %p2, %p3;
    @%p4 bra FINISH_BREAKDOWN;
    ld.global.u32 %r4, [%rd1+80];
    add.u32 %r4, %r4, 1;
    st.global.u32 [%rd1+80], %r4;
    ld.global.f32 %f2, [%rd1+48];
    setp.le.f32 %p5, %f1, %f2;
    @%p5 bra FINISH_CONVERGED;
    ret;

FINISH_CONVERGED:
    st.global.f32 [%rd1+0], %f1;
    mov.u32 %r5, 2;
    mov.u32 %r6, 0;
    st.global.u32 [%rd1+76], %r5;
    st.global.u32 [%rd1+84], %r6;
    ret;

FINISH_BREAKDOWN:
    st.global.f32 [%rd1+0], %f1;
    mov.u32 %r7, 1;
    mov.u32 %r8, 0;
    st.global.u32 [%rd1+76], %r7;
    st.global.u32 [%rd1+84], %r8;
    ret;
}

.visible .entry cg_prepare_direction(.param .u64 state_param)
{
    .reg .pred %p<10>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<10>;
    ld.param.u64 %rd1, [state_param];
    ld.global.u32 %r1, [%rd1+84];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra DIRECTION_MASKED;
    ld.global.u32 %r2, [%rd1+88];
    setp.eq.u32 %p2, %r2, 0;
    @%p2 ld.global.f32 %f1, [%rd1+4];
    @%p2 ld.global.f32 %f2, [%rd1+0];
    @!%p2 ld.global.f32 %f1, [%rd1+12];
    @!%p2 ld.global.f32 %f2, [%rd1+8];
    mov.b32 %r3, %f1;
    and.b32 %r4, %r3, 2139095040;
    setp.eq.u32 %p3, %r4, 2139095040;
    mov.b32 %r5, %f2;
    and.b32 %r6, %r5, 2139095040;
    setp.eq.u32 %p4, %r6, 2139095040;
    or.pred %p5, %p3, %p4;
    setp.le.f32 %p6, %f1, 0f00000000;
    or.pred %p5, %p5, %p6;
    setp.le.f32 %p7, %f2, 0f00000000;
    or.pred %p5, %p5, %p7;
    @%p5 bra DIRECTION_BREAKDOWN;
    div.rn.f32 %f3, %f1, %f2;
    mov.b32 %r7, %f3;
    and.b32 %r8, %r7, 2139095040;
    setp.eq.u32 %p8, %r8, 2139095040;
    @%p8 bra DIRECTION_BREAKDOWN;
    mov.f32 %f4, 0f3f800000;
    st.global.f32 [%rd1+28], %f3;
    st.global.f32 [%rd1+32], %f4;
    ld.global.f32 %f5, [%rd1+4];
    st.global.f32 [%rd1+0], %f5;
    @!%p2 ld.global.f32 %f6, [%rd1+12];
    @!%p2 st.global.f32 [%rd1+8], %f6;
    ret;

DIRECTION_BREAKDOWN:
    mov.u32 %r9, 1;
    mov.u32 %r10, 0;
    st.global.u32 [%rd1+76], %r9;
    st.global.u32 [%rd1+84], %r10;
DIRECTION_MASKED:
    mov.f32 %f7, 0f3f800000;
    mov.f32 %f8, 0f00000000;
    st.global.f32 [%rd1+28], %f7;
    st.global.f32 [%rd1+32], %f8;
    ret;
}
)ptx";

const char kCudaCGConditionalPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional(
    .param .b64 handle_param,
    .param .b32 value_param
);

.visible .entry cg_set_conditional(
    .param .u64 state_param,
    .param .u64 handle_param,
    .param .u32 max_iterations_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    .param .b64 call_handle;
    .param .b32 call_value;

    ld.param.u64 %rd1, [state_param];
    ld.param.u64 %rd2, [handle_param];
    ld.param.u32 %r1, [max_iterations_param];
    ld.global.u32 %r2, [%rd1+80];
    ld.global.u32 %r3, [%rd1+84];
    setp.ne.u32 %p1, %r3, 0;
    setp.lt.u32 %p2, %r2, %r1;
    and.pred %p3, %p1, %p2;
    selp.u32 %r4, 1, 0, %p3;
    st.param.b64 [call_handle], %rd2;
    st.param.b32 [call_value], %r4;
    call.uni cudaGraphSetConditional, (call_handle, call_value);
    ret;
}
)ptx";

std::once_flag module_once;
void *module{nullptr};
void *initialize_func{nullptr};
void *validate_rho_func{nullptr};
void *prepare_alpha_func{nullptr};
void *finish_iteration_func{nullptr};
void *prepare_direction_func{nullptr};
std::once_flag conditional_module_once;
void *conditional_module{nullptr};
void *set_conditional_func{nullptr};

void load_module_once() {
  auto &context = CUDAContext::get_instance();
  auto context_guard = context.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&module, kCudaCGScalarPtx, 0, nullptr, nullptr);
  driver.module_get_function(&initialize_func, module, "cg_initialize");
  driver.module_get_function(&validate_rho_func, module, "cg_validate_rho");
  driver.module_get_function(&prepare_alpha_func, module, "cg_prepare_alpha");
  driver.module_get_function(&finish_iteration_func, module,
                             "cg_finish_iteration");
  driver.module_get_function(&prepare_direction_func, module,
                             "cg_prepare_direction");
}

void load_conditional_module_once() {
  auto &context = CUDAContext::get_instance();
  auto context_guard = context.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&conditional_module, kCudaCGConditionalPtx, 0,
                             nullptr, nullptr);
  driver.module_get_function(&set_conditional_func, conditional_module,
                             "cg_set_conditional");
}

void launch_scalar(void *function,
                   const char *name,
                   CudaCGScalarState *state,
                   void *stream) {
  void *state_arg = state;
  CUDAContext::get_instance().launch(function, name, {&state_arg}, {}, 1, 1,
                                     0, stream);
}

void ensure_module() {
  std::call_once(module_once, load_module_once);
}

void ensure_conditional_module() {
  std::call_once(conditional_module_once, load_conditional_module_once);
}

}  // namespace

bool driver_cg_scalar_available() {
  return CUDADriver::get_instance_without_context().detected();
}

bool driver_cg_conditional_setter_compiled() {
  return true;
}

void driver_cg_prepare_conditional_setter() {
  ensure_conditional_module();
}

void driver_cg_initialize(CudaCGScalarState *state, void *stream) {
  ensure_module();
  launch_scalar(initialize_func, "cuda_cg_initialize", state, stream);
}

void driver_cg_validate_rho(CudaCGScalarState *state, void *stream) {
  ensure_module();
  launch_scalar(validate_rho_func, "cuda_cg_validate_rho", state, stream);
}

void driver_cg_prepare_alpha(CudaCGScalarState *state, void *stream) {
  ensure_module();
  launch_scalar(prepare_alpha_func, "cuda_cg_prepare_alpha", state, stream);
}

void driver_cg_finish_iteration(CudaCGScalarState *state, void *stream) {
  ensure_module();
  launch_scalar(finish_iteration_func, "cuda_cg_finish_iteration", state,
                stream);
}

void driver_cg_prepare_direction(CudaCGScalarState *state, void *stream) {
  ensure_module();
  launch_scalar(prepare_direction_func, "cuda_cg_prepare_direction", state,
                stream);
}

void driver_cg_set_conditional(CudaCGScalarState *state,
                               std::uint64_t conditional_handle,
                               int max_iterations,
                               void *stream) {
  ensure_conditional_module();
  void *state_arg = state;
  void *handle_arg = &conditional_handle;
  void *max_iterations_arg = &max_iterations;
  CUDAContext::get_instance().launch(
      set_conditional_func, "cuda_cg_set_conditional",
      {&state_arg, handle_arg, max_iterations_arg}, {}, 1, 1, 0, stream);
}

}  // namespace taichi::lang::cuda
