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

const char kCudaGraphMaskPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry graph_latch_while(
    .param .u64 predicate_param,
    .param .u64 gate_param,
    .param .u32 continue_nonzero_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [predicate_param];
    ld.param.u64 %rd2, [gate_param];
    ld.param.u32 %r1, [continue_nonzero_param];
    ld.global.u32 %r2, [%rd1];
    setp.ne.u32 %p1, %r1, 0;
    @%p1 setp.ne.u32 %p2, %r2, 0;
    @!%p1 setp.eq.u32 %p2, %r2, 0;
    selp.u32 %r3, 1, 0, %p2;
    st.global.u32 [%rd2], %r3;
    ret;
}

.visible .entry graph_latch_branch(
    .param .u64 selector_param,
    .param .u64 gate_param,
    .param .u32 type_param,
    .param .u32 branch_count_param,
    .param .u32 default_branch_param
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [selector_param];
    ld.param.u64 %rd2, [gate_param];
    ld.param.u32 %r1, [type_param];
    ld.param.u32 %r2, [branch_count_param];
    ld.param.u32 %r3, [default_branch_param];
    ld.global.u32 %r4, [%rd1];
    mov.u32 %r8, 0;
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra IF_SELECTOR;

    setp.lt.u32 %p2, %r4, %r2;
    @%p2 add.u32 %r8, %r4, 1;
    @%p2 bra STORE_SELECTOR;
    setp.lt.u32 %p3, %r3, %r2;
    @%p3 add.u32 %r8, %r3, 1;
    bra STORE_SELECTOR;

IF_SELECTOR:
    setp.ne.u32 %p4, %r4, 0;
    @%p4 mov.u32 %r8, 1;
    @%p4 bra STORE_SELECTOR;
    setp.gt.u32 %p5, %r2, 1;
    @%p5 mov.u32 %r8, 2;

STORE_SELECTOR:
    st.global.u32 [%rd2], %r8;
    ret;
}
)ptx";

std::once_flag module_once;
void *conditional_module{nullptr};
void *set_conditional_func{nullptr};
void *set_branch_conditional_func{nullptr};

std::once_flag mask_module_once;
void *mask_module{nullptr};
void *latch_while_func{nullptr};
void *latch_branch_func{nullptr};

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

void load_mask_module_once() {
  auto &context = CUDAContext::get_instance();
  auto context_guard = context.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&mask_module, kCudaGraphMaskPtx, 0, nullptr,
                             nullptr);
  driver.module_get_function(&latch_while_func, mask_module,
                             "graph_latch_while");
  driver.module_get_function(&latch_branch_func, mask_module,
                             "graph_latch_branch");
}

void ensure_mask_module() {
  std::call_once(mask_module_once, load_mask_module_once);
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

bool driver_graph_mask_latch_compiled() {
  return true;
}

void driver_graph_prepare_mask_latch() {
  ensure_mask_module();
}

void driver_graph_latch_while(void *predicate,
                              void *gate,
                              bool continue_while_nonzero,
                              void *stream) {
  ensure_mask_module();
  std::uint32_t continue_arg = continue_while_nonzero ? 1u : 0u;
  CUDAContext::get_instance().launch(latch_while_func, "cuda_graph_latch_while",
                                     {&predicate, &gate, &continue_arg}, {}, 1,
                                     1, 0, stream);
}

void driver_graph_latch_branch(void *selector,
                               void *gate,
                               std::uint32_t conditional_type,
                               std::uint32_t branch_count,
                               std::uint32_t default_branch,
                               void *stream) {
  ensure_mask_module();
  CUDAContext::get_instance().launch(
      latch_branch_func, "cuda_graph_latch_branch",
      {&selector, &gate, &conditional_type, &branch_count, &default_branch}, {},
      1, 1, 0, stream);
}

}  // namespace taichi::lang::cuda
