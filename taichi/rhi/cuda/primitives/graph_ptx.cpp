#include "taichi/rhi/cuda/primitives/graph_ptx.h"

#include <mutex>

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang::cuda {

namespace {

const char kCudaGraphConditionalPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional(
    .param .b64 handle_param,
    .param .b32 value_param
);

.visible .entry graph_set_conditional(
    .param .u64 control_param,
    .param .u64 handle_param
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<4>;
    .param .b64 call_handle;
    .param .b32 call_value;

    ld.param.u64 %rd1, [control_param];
    ld.param.u64 %rd2, [handle_param];
    ld.global.u64 %rd3, [%rd1+0];
    ld.global.u32 %r1, [%rd1+8];
    ld.global.u32 %r2, [%rd1+12];
    ld.global.u32 %r3, [%rd1+16];
    add.u32 %r4, %r1, 1;
    st.global.u32 [%rd1+8], %r4;
    ld.global.u32 %r5, [%rd3];

    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra CONTINUE_NONZERO;
    setp.eq.u32 %p2, %r5, 0;
    bra PREDICATE_READY;
CONTINUE_NONZERO:
    setp.ne.u32 %p2, %r5, 0;
PREDICATE_READY:
    setp.lt.u32 %p3, %r4, %r2;
    and.pred %p4, %p2, %p3;
    selp.u32 %r6, 1, 0, %p4;
    st.param.b64 [call_handle], %rd2;
    st.param.b32 [call_value], %r6;
    call.uni cudaGraphSetConditional, (call_handle, call_value);
    ret;
}

.visible .entry graph_set_branch_conditional(
    .param .u64 control_param,
    .param .u64 handle_param
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .param .b64 call_handle;
    .param .b32 call_value;

    ld.param.u64 %rd1, [control_param];
    ld.param.u64 %rd2, [handle_param];
    ld.global.u64 %rd3, [%rd1+0];
    ld.global.u32 %r1, [%rd1+8];
    ld.global.u32 %r2, [%rd1+12];
    ld.global.u32 %r3, [%rd1+16];
    ld.global.u32 %r4, [%rd3];

    setp.eq.u32 %p1, %r1, 0;
    @!%p1 bra SWITCH_SELECTOR;
    setp.ne.u32 %p2, %r4, 0;
    selp.u32 %r7, 1, 0, %p2;
    bra SELECTOR_READY;
SWITCH_SELECTOR:
    setp.lt.u32 %p3, %r4, %r2;
    setp.lt.u32 %p4, %r3, %r2;
    and.pred %p5, %p4, !%p3;
    selp.u32 %r5, %r3, %r2, %p5;
    selp.u32 %r7, %r4, %r5, %p3;
SELECTOR_READY:
    st.param.b64 [call_handle], %rd2;
    st.param.b32 [call_value], %r7;
    call.uni cudaGraphSetConditional, (call_handle, call_value);
    ret;
}
)ptx";

std::once_flag module_once;
void *conditional_module{nullptr};
void *set_conditional_func{nullptr};
void *set_branch_conditional_func{nullptr};

void load_module_once() {
  auto &context = CUDAContext::get_instance();
  auto context_guard = context.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&conditional_module, kCudaGraphConditionalPtx, 0,
                             nullptr, nullptr);
  driver.module_get_function(&set_conditional_func, conditional_module,
                             "graph_set_conditional");
  driver.module_get_function(&set_branch_conditional_func, conditional_module,
                             "graph_set_branch_conditional");
}

void ensure_module() {
  std::call_once(module_once, load_module_once);
}

}  // namespace

bool driver_graph_conditional_setter_compiled() {
  return true;
}

void driver_graph_prepare_conditional_setter() {
  ensure_module();
}

void driver_graph_set_conditional(
    CudaGraphConditionalControl *control,
    std::uint64_t conditional_handle,
    void *stream) {
  ensure_module();
  void *control_arg = control;
  void *handle_arg = &conditional_handle;
  CUDAContext::get_instance().launch(
      set_conditional_func, "cuda_graph_set_conditional",
      {&control_arg, handle_arg}, {}, 1, 1, 0, stream);
}

void driver_graph_set_branch_conditional(
    CudaGraphConditionalControl *control,
    std::uint64_t conditional_handle,
    void *stream) {
  ensure_module();
  void *control_arg = control;
  void *handle_arg = &conditional_handle;
  CUDAContext::get_instance().launch(
      set_branch_conditional_func, "cuda_graph_set_branch_conditional",
      {&control_arg, handle_arg}, {}, 1, 1, 0, stream);
}

}  // namespace taichi::lang::cuda
