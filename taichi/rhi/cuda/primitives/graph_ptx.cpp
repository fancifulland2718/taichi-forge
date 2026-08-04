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

const char kCudaGraphBoundedUpdatePtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.extern .func (.param .b32 result) cudaGraphKernelNodeSetEnabled(
    .param .b64 node,
    .param .b32 enabled
);

.extern .func (.param .b32 result) cudaGraphKernelNodeSetGridDim(
    .param .b64 node,
    .param .align 4 .b8 grid_dim[12]
);

.visible .entry graph_update_bounded_node(
    .param .u64 control_param
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<3>;
    .param .b64 call_node;
    .param .b32 call_enabled;
    .param .align 4 .b8 call_grid[12];
    .param .b32 call_result;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    or.b32 %r3, %r1, %r2;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra DONE;

    ld.param.u64 %rd1, [control_param];
    ld.global.u64 %rd2, [%rd1+0];
    ld.global.u32 %r4, [%rd1+8];
    ld.global.u32 %r5, [%rd1+12];
    setp.ne.u32 %p2, %r5, 0;
    selp.u32 %r6, 1, 0, %p2;

    st.param.b64 [call_node], %rd2;
    st.param.b32 [call_enabled], %r6;
    call.uni (call_result), cudaGraphKernelNodeSetEnabled,
        (call_node, call_enabled);
    ld.param.b32 %r7, [call_result];
    setp.ne.u32 %p3, %r7, 0;
    @%p3 bra STORE_STATUS;
    setp.eq.u32 %p4, %r6, 0;
    @%p4 bra STORE_STATUS;

    st.param.b64 [call_node], %rd2;
    st.param.b32 [call_grid+0], %r4;
    mov.u32 %r8, 1;
    st.param.b32 [call_grid+4], %r8;
    st.param.b32 [call_grid+8], %r8;
    call.uni (call_result), cudaGraphKernelNodeSetGridDim,
        (call_node, call_grid);
    ld.param.b32 %r7, [call_result];

STORE_STATUS:
    st.global.u32 [%rd1+16], %r7;
DONE:
    ret;
}

.visible .entry graph_update_bounded_extent(
    .param .u64 control_param
)
{
    .reg .pred %p<9>;
    .reg .b32 %r<17>;
    .reg .b64 %rd<4>;
    .param .b64 call_node;
    .param .b32 call_enabled;
    .param .align 4 .b8 call_grid[12];
    .param .b32 call_result;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    or.b32 %r3, %r1, %r2;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra EXTENT_DONE;

    ld.param.u64 %rd1, [control_param];
    ld.global.u64 %rd2, [%rd1+0];
    ld.global.u64 %rd3, [%rd1+8];
    ld.global.u32 %r4, [%rd1+16];
    ld.global.u32 %r5, [%rd1+20];
    ld.global.u32 %r16, [%rd1+28];
    ld.global.s32 %r6, [%rd3];
    setp.lt.s32 %p2, %r6, 0;
    setp.gt.s32 %p3, %r6, %r4;
    or.pred %p4, %p2, %p3;
    @%p2 mov.u32 %r7, 0;
    @!%p2 mov.u32 %r7, %r6;
    @%p3 mov.u32 %r7, %r4;
    mov.u32 %r15, 1;
    @%p4 st.global.u32 [%rd3+4], %r15;
    st.global.u32 [%rd3], %r7;
    mov.u32 %r12, 0;
    st.global.u32 [%rd1+24], %r12;

    setp.ne.u32 %p5, %r7, 0;
    selp.u32 %r8, 1, 0, %p5;
    add.u32 %r9, %r7, %r5;
    sub.u32 %r9, %r9, 1;
    div.u32 %r10, %r9, %r5;
    min.u32 %r10, %r10, %r16;

    st.param.b64 [call_node], %rd2;
    st.param.b32 [call_enabled], %r8;
    call.uni (call_result), cudaGraphKernelNodeSetEnabled,
        (call_node, call_enabled);
    ld.param.b32 %r11, [call_result];
    setp.ne.u32 %p6, %r11, 0;
    @%p6 st.global.u32 [%rd1+24], %r11;
    @%p6 bra EXTENT_DONE;
    setp.eq.u32 %p7, %r8, 0;
    @%p7 bra EXTENT_DONE;

    st.param.b64 [call_node], %rd2;
    st.param.b32 [call_grid+0], %r10;
    mov.u32 %r13, 1;
    st.param.b32 [call_grid+4], %r13;
    st.param.b32 [call_grid+8], %r13;
    call.uni (call_result), cudaGraphKernelNodeSetGridDim,
        (call_node, call_grid);
    ld.param.b32 %r14, [call_result];
    setp.ne.u32 %p8, %r14, 0;
    @%p8 st.global.u32 [%rd1+24], %r14;

EXTENT_DONE:
    ret;
}

.visible .entry graph_update_bounded_group(
    .param .u64 control_param
)
{
    .reg .pred %p<14>;
    .reg .b32 %r<23>;
    .reg .b64 %rd<8>;
    .param .b64 call_node;
    .param .b32 call_enabled;
    .param .align 4 .b8 call_grid[12];
    .param .b32 call_result;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    or.b32 %r3, %r1, %r2;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra GROUP_DONE;

    ld.param.u64 %rd1, [control_param];
    ld.global.u64 %rd2, [%rd1+0];
    ld.global.u64 %rd3, [%rd1+8];
    ld.global.u32 %r4, [%rd1+16];
    ld.global.u32 %r5, [%rd1+20];
    ld.global.u32 %r6, [%rd1+24];
    ld.global.u32 %r7, [%rd1+28];
    ld.global.s32 %r8, [%rd2];
    setp.lt.s32 %p2, %r8, 0;
    setp.gt.s32 %p3, %r8, %r5;
    or.pred %p4, %p2, %p3;
    @%p2 mov.u32 %r9, 0;
    @!%p2 mov.u32 %r9, %r8;
    @%p3 mov.u32 %r9, %r5;
    mov.u32 %r10, 1;
    @%p4 st.global.u32 [%rd2+4], %r10;
    st.global.u32 [%rd2], %r9;

    setp.ne.u32 %p5, %r9, 0;
    selp.u32 %r11, 1, 0, %p5;
    add.u32 %r12, %r9, %r6;
    sub.u32 %r12, %r12, 1;
    div.u32 %r12, %r12, %r6;
    min.u32 %r12, %r12, %r7;

    ld.global.u32 %r13, [%rd1+40];
    ld.global.u32 %r14, [%rd1+32];
    ld.global.u32 %r15, [%rd1+36];
    setp.eq.u32 %p6, %r13, 0;
    setp.ne.u32 %p7, %r11, %r15;
    setp.ne.u32 %p8, %r12, %r14;
    or.pred %p9, %p6, %p7;
    mov.u32 %r16, 0;
    st.global.u32 [%rd1+44], %r16;
    or.pred %p13, %p9, %p8;
    @!%p13 bra GROUP_DONE;
    mov.u32 %r17, 0;

GROUP_LOOP:
    setp.ge.u32 %p10, %r17, %r4;
    @%p10 bra GROUP_COMMIT;
    cvt.u64.u32 %rd4, %r17;
    shl.b64 %rd5, %rd4, 3;
    add.u64 %rd6, %rd3, %rd5;
    ld.global.u64 %rd7, [%rd6];

    @!%p9 bra GROUP_GRID_CHECK;
    st.param.b64 [call_node], %rd7;
    st.param.b32 [call_enabled], %r11;
    call.uni (call_result), cudaGraphKernelNodeSetEnabled,
        (call_node, call_enabled);
    ld.param.b32 %r16, [call_result];
    setp.ne.u32 %p11, %r16, 0;
    @%p11 bra GROUP_FAIL;

GROUP_GRID_CHECK:
    setp.eq.u32 %p12, %r11, 0;
    @%p12 bra GROUP_NEXT;

    st.param.b64 [call_node], %rd7;
    st.param.b32 [call_grid+0], %r12;
    mov.u32 %r18, 1;
    st.param.b32 [call_grid+4], %r18;
    st.param.b32 [call_grid+8], %r18;
    call.uni (call_result), cudaGraphKernelNodeSetGridDim,
        (call_node, call_grid);
    ld.param.b32 %r16, [call_result];
    setp.ne.u32 %p11, %r16, 0;
    @%p11 bra GROUP_FAIL;
GROUP_NEXT:
    add.u32 %r17, %r17, 1;
    bra GROUP_LOOP;

GROUP_COMMIT:
    st.global.u32 [%rd1+32], %r12;
    st.global.u32 [%rd1+36], %r11;
    mov.u32 %r19, 1;
    st.global.u32 [%rd1+40], %r19;
    bra GROUP_DONE;

GROUP_FAIL:
    st.global.u32 [%rd1+44], %r16;
GROUP_DONE:
    ret;
}

.visible .entry graph_bounded_probe_payload(
    .param .u64 visited_param
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [visited_param];
    atom.global.add.u32 %r1, [%rd1], 1;
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

std::once_flag bounded_module_once;
void *bounded_module{nullptr};
void *bounded_update_func{nullptr};
void *bounded_extent_update_func{nullptr};
void *bounded_group_update_func{nullptr};
void *bounded_probe_payload_func{nullptr};
std::uint32_t bounded_module_error{0};

std::mutex bounded_probe_mutex;
CudaGraphBoundedProbeResult bounded_probe_result;

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

void load_bounded_module_once() {
  auto &context = CUDAContext::get_instance();
  auto context_guard = context.get_guard();
  auto &driver = CUDADriver::get_instance();
  bounded_module_error = driver.module_load_data_ex.call(
      &bounded_module, kCudaGraphBoundedUpdatePtx, 0, nullptr, nullptr);
  if (bounded_module_error != CUDA_SUCCESS) {
    return;
  }
  bounded_module_error = driver.module_get_function.call(
      &bounded_update_func, bounded_module, "graph_update_bounded_node");
  if (bounded_module_error != CUDA_SUCCESS) {
    return;
  }
  bounded_module_error = driver.module_get_function.call(
      &bounded_extent_update_func, bounded_module,
      "graph_update_bounded_extent");
  if (bounded_module_error != CUDA_SUCCESS) {
    return;
  }
  bounded_module_error = driver.module_get_function.call(
      &bounded_group_update_func, bounded_module,
      "graph_update_bounded_group");
  if (bounded_module_error != CUDA_SUCCESS) {
    return;
  }
  bounded_module_error = driver.module_get_function.call(
      &bounded_probe_payload_func, bounded_module,
      "graph_bounded_probe_payload");
}

bool ensure_bounded_module(std::uint32_t *driver_error) {
  std::call_once(bounded_module_once, load_bounded_module_once);
  if (driver_error != nullptr) {
    *driver_error = bounded_module_error;
  }
  return bounded_module_error == CUDA_SUCCESS && bounded_module != nullptr &&
         bounded_update_func != nullptr &&
         bounded_extent_update_func != nullptr &&
         bounded_group_update_func != nullptr &&
         bounded_probe_payload_func != nullptr;
}

}  // namespace

bool driver_graph_conditional_setter_compiled() {
  return true;
}

void driver_graph_prepare_conditional_setter() {
  ensure_module();
}

void driver_graph_set_conditional(CudaGraphConditionalControl *control,
                                  std::uint64_t conditional_handle,
                                  void *stream) {
  ensure_module();
  void *control_arg = control;
  void *handle_arg = &conditional_handle;
  CUDAContext::get_instance().launch(
      set_conditional_func, "cuda_graph_set_conditional",
      {&control_arg, handle_arg}, {}, 1, 1, 0, stream);
}

void driver_graph_set_branch_conditional(CudaGraphConditionalControl *control,
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

bool driver_graph_bounded_update_compiled() {
  return true;
}

bool driver_graph_prepare_bounded_update(std::uint32_t *driver_error) {
  return ensure_bounded_module(driver_error);
}

void driver_graph_update_bounded(CudaGraphBoundedControl *control,
                                 void *stream) {
  std::uint32_t driver_error = CUDA_SUCCESS;
  TI_ERROR_IF(!ensure_bounded_module(&driver_error),
              "CUDA bounded Graph updater PTX failed to load: {}",
              get_cuda_error_message(driver_error));
  void *control_arg = control;
  CUDAContext::get_instance().launch(bounded_update_func,
                                     "cuda_graph_update_bounded_node",
                                     {&control_arg}, {}, 1, 1, 0, stream);
}

void driver_graph_update_bounded_extent(CudaGraphBoundedExtentControl *control,
                                        void *stream) {
  std::uint32_t driver_error = CUDA_SUCCESS;
  TI_ERROR_IF(!ensure_bounded_module(&driver_error),
              "CUDA bounded Graph updater PTX failed to load: {}",
              get_cuda_error_message(driver_error));
  void *control_arg = control;
  CUDAContext::get_instance().launch(bounded_extent_update_func,
                                     "cuda_graph_update_bounded_extent",
                                     {&control_arg}, {}, 1, 1, 0, stream);
}

void driver_graph_update_bounded_group(CudaGraphBoundedGroupControl *control,
                                       void *stream) {
  std::uint32_t driver_error = CUDA_SUCCESS;
  TI_ERROR_IF(!ensure_bounded_module(&driver_error),
              "CUDA bounded Graph updater PTX failed to load: {}",
              get_cuda_error_message(driver_error));
  void *control_arg = control;
  CUDAContext::get_instance().launch(bounded_group_update_func,
                                     "cuda_graph_update_bounded_group",
                                     {&control_arg}, {}, 1, 1, 0, stream);
}

void *driver_graph_bounded_probe_payload_function() {
  std::uint32_t driver_error = CUDA_SUCCESS;
  if (!ensure_bounded_module(&driver_error)) {
    return nullptr;
  }
  return bounded_probe_payload_func;
}

CudaGraphBoundedProbeResult driver_graph_bounded_probe(bool run) {
  std::lock_guard<std::mutex> lock(bounded_probe_mutex);
  if (!run || bounded_probe_result.attempted) {
    return bounded_probe_result;
  }
  auto &result = bounded_probe_result;
  result.attempted = true;

  auto &driver_without_context = CUDADriver::get_instance_without_context();
  if (!driver_without_context.detected()) {
    result.reason = "cuda_driver_not_loaded";
    return result;
  }
  const int driver_api_version =
      driver_without_context.get_version_major() * 1000 +
      driver_without_context.get_version_minor() * 10;
  if (driver_api_version < 12040) {
    result.reason = "cuda_driver_api_below_12040";
    return result;
  }
  if (!driver_without_context.launch_kernel_ex.available() ||
      !driver_without_context.graph_upload.available() ||
      !driver_without_context.stream_begin_capture.available() ||
      !driver_without_context.stream_end_capture.available() ||
      !driver_without_context.graph_instantiate_with_flags.available() ||
      !driver_without_context.graph_launch.available() ||
      !driver_without_context.graph_destroy.available() ||
      !driver_without_context.graph_exec_destroy.available()) {
    result.reason = "cuda_device_update_symbols_unavailable";
    return result;
  }

  CUDAContext::get_instance().make_current();
  auto &driver = CUDADriver::get_instance();
  if (!ensure_bounded_module(&result.driver_error)) {
    result.reason = "cuda_device_update_ptx_link_failed";
    return result;
  }
  result.ptx_linked = true;

  struct ProbeResources {
    void *stream{nullptr};
    void *control{nullptr};
    void *external_control{nullptr};
    void *visited{nullptr};
    CUgraph graph{nullptr};
    CUgraphExec graph_exec{nullptr};
    bool capture_active{false};

    ~ProbeResources() {
      auto &driver = CUDADriver::get_instance();
      if (capture_active && stream != nullptr) {
        CUgraph aborted_graph = nullptr;
        if (driver.stream_end_capture.call(stream, &aborted_graph) ==
                CUDA_SUCCESS &&
            aborted_graph != nullptr) {
          driver.graph_destroy.call(aborted_graph);
        }
      }
      if (stream != nullptr) {
        driver.stream_synchronize.call(stream);
      }
      if (graph_exec != nullptr) {
        driver.graph_exec_destroy.call(graph_exec);
      }
      if (graph != nullptr) {
        driver.graph_destroy.call(graph);
      }
      if (visited != nullptr) {
        driver.mem_free.call(visited);
      }
      if (control != nullptr) {
        driver.mem_free.call(control);
      }
      if (external_control != nullptr) {
        driver.mem_free.call(external_control);
      }
      if (stream != nullptr) {
        driver.stream_destroy.call(stream);
      }
    }
  } resources;

  auto fail = [&](const char *reason, std::uint32_t error) {
    result.reason = reason;
    result.driver_error = error;
    return result;
  };

  std::uint32_t error =
      driver.stream_create.call(&resources.stream, CU_STREAM_NON_BLOCKING);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_stream_create_failed", error);
  }
  error =
      driver.malloc.call(&resources.control, sizeof(CudaGraphBoundedControl));
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_control_alloc_failed", error);
  }
  error = driver.malloc.call(&resources.external_control,
                             sizeof(CudaGraphBoundedControl));
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_external_control_alloc_failed", error);
  }
  error = driver.malloc.call(&resources.visited, sizeof(std::uint32_t));
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_probe_alloc_failed", error);
  }
  error = driver.memsetd32.call(resources.visited, 0, 1);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_probe_clear_failed", error);
  }

  CudaGraphBoundedControl host_control;
  host_control.grid_x = 7;
  host_control.enabled = 1;
  host_control.status = 0xffffffffu;
  error = driver.memcpy_host_to_device.call(resources.control, &host_control,
                                            sizeof(host_control));
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_control_upload_failed", error);
  }

  error = driver.stream_begin_capture.call(resources.stream,
                                           CU_STREAM_CAPTURE_MODE_RELAXED);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_capture_begin_failed", error);
  }
  resources.capture_active = true;

  void *control_arg = resources.control;
  void *update_args[] = {&control_arg};
  error = driver.launch_kernel.call(bounded_update_func, 1, 1, 1, 1, 1, 1, 0,
                                    resources.stream, update_args, nullptr);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_updater_capture_failed", error);
  }

  TaichiCudaLaunchAttribute attribute{};
  attribute.id = TAICHI_CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
  attribute.value.device_updatable_kernel_node.device_updatable = 1;
  TaichiCudaLaunchConfig config{};
  config.grid_dim_x = 64;
  config.grid_dim_y = 1;
  config.grid_dim_z = 1;
  config.block_dim_x = 1;
  config.block_dim_y = 1;
  config.block_dim_z = 1;
  config.stream = resources.stream;
  config.attributes = &attribute;
  config.num_attributes = 1;
  void *visited_arg = resources.visited;
  void *payload_args[] = {&visited_arg};
  error = driver.launch_kernel_ex.call(&config, bounded_probe_payload_func,
                                       payload_args, nullptr);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_payload_capture_failed", error);
  }
  void *device_node = attribute.value.device_updatable_kernel_node.device_node;
  if (device_node == nullptr) {
    return fail("cuda_device_update_node_handle_missing",
                CUDA_ERROR_NOT_SUPPORTED);
  }

  error = driver.stream_end_capture.call(resources.stream, &resources.graph);
  resources.capture_active = false;
  if (error != CUDA_SUCCESS || resources.graph == nullptr) {
    return fail("cuda_device_update_capture_end_failed", error);
  }
  host_control.device_node = reinterpret_cast<std::uintptr_t>(device_node);
  error = driver.memcpy_host_to_device.call(resources.control, &host_control,
                                            sizeof(host_control));
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_handle_upload_failed", error);
  }
  error = driver.graph_instantiate_with_flags.call(&resources.graph_exec,
                                                   resources.graph, 0);
  if (error != CUDA_SUCCESS || resources.graph_exec == nullptr) {
    return fail("cuda_device_update_instantiate_failed", error);
  }
  error = driver.graph_upload.call(resources.graph_exec, resources.stream);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_graph_upload_failed", error);
  }
  result.graph_uploaded = true;

  auto run_case = [&](std::uint32_t grid_x, bool enabled,
                      std::uint32_t expected_visited,
                      std::uint32_t *observed_visited) {
    host_control.grid_x = grid_x;
    host_control.enabled = enabled ? 1u : 0u;
    host_control.status = 0xffffffffu;
    std::uint32_t case_error = driver.memcpy_host_to_device.call(
        resources.control, &host_control, sizeof(host_control));
    if (case_error == CUDA_SUCCESS) {
      case_error =
          driver.graph_launch.call(resources.graph_exec, resources.stream);
    }
    if (case_error == CUDA_SUCCESS) {
      case_error = driver.stream_synchronize.call(resources.stream);
    }
    if (case_error == CUDA_SUCCESS) {
      case_error = driver.memcpy_device_to_host.call(
          &host_control, resources.control, sizeof(host_control));
    }
    if (case_error == CUDA_SUCCESS) {
      case_error = driver.memcpy_device_to_host.call(
          observed_visited, resources.visited, sizeof(*observed_visited));
    }
    if (case_error != CUDA_SUCCESS) {
      result.driver_error = case_error;
      return false;
    }
    return host_control.status == CUDA_SUCCESS &&
           *observed_visited == expected_visited;
  };

  if (!run_case(7, true, 7, &result.sparse_visited)) {
    result.reason = "cuda_device_update_sparse_grid_mismatch";
    return result;
  }
  if (!run_case(1, false, 7, &result.zero_visited)) {
    result.reason = "cuda_device_update_zero_skip_mismatch";
    return result;
  }
  result.zero_count_skipped = true;
  if (!run_case(3, true, 10, &result.rebound_visited)) {
    result.reason = "cuda_device_update_reenable_mismatch";
    return result;
  }
  if (!run_case(64, true, 74, &result.baseline_visited)) {
    result.reason = "cuda_device_update_baseline_grid_mismatch";
    return result;
  }

  const auto device_node_handle = host_control.device_node;
  auto clear_visited = [&]() {
    const auto clear_error = driver.memsetd32.call(resources.visited, 0, 1);
    if (clear_error != CUDA_SUCCESS) {
      result.driver_error = clear_error;
      return false;
    }
    return true;
  };
  auto launch_without_updater = [&](std::uint32_t expected,
                                    std::uint32_t *observed) {
    host_control.device_node = 0;
    host_control.status = 0xffffffffu;
    std::uint32_t case_error = driver.memcpy_host_to_device.call(
        resources.control, &host_control, sizeof(host_control));
    if (case_error == CUDA_SUCCESS) {
      case_error = driver.graph_launch.call(resources.graph_exec,
                                            resources.stream);
    }
    if (case_error == CUDA_SUCCESS) {
      case_error = driver.stream_synchronize.call(resources.stream);
    }
    if (case_error == CUDA_SUCCESS) {
      case_error = driver.memcpy_device_to_host.call(
          observed, resources.visited, sizeof(*observed));
    }
    host_control.device_node = device_node_handle;
    if (case_error != CUDA_SUCCESS) {
      result.driver_error = case_error;
      return false;
    }
    return *observed == expected;
  };

  if (!clear_visited() ||
      !run_case(7, true, 7, &result.sparse_visited) ||
      !launch_without_updater(14, &result.persistent_sparse_visited)) {
    result.reason = "cuda_device_update_sparse_state_not_persistent";
    return result;
  }
  std::uint32_t persistent_disabled_setup_visited = 0;
  if (!clear_visited() ||
      !run_case(1, false, 0, &persistent_disabled_setup_visited) ||
      !launch_without_updater(0, &result.persistent_disabled_visited)) {
    result.reason = "cuda_device_update_disabled_state_not_persistent";
    return result;
  }
  result.launch_update_persists = true;

  if (!clear_visited()) {
    result.reason = "cuda_device_update_external_clear_failed";
    return result;
  }
  CudaGraphBoundedControl external_control;
  external_control.device_node = device_node_handle;
  external_control.grid_x = 5;
  external_control.enabled = 1;
  external_control.status = 0xffffffffu;
  error = driver.memcpy_host_to_device.call(resources.external_control,
                                            &external_control,
                                            sizeof(external_control));
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_external_control_upload_failed", error);
  }
  void *external_arg = resources.external_control;
  void *external_args[] = {&external_arg};
  error = driver.launch_kernel.call(bounded_update_func, 1, 1, 1, 1, 1, 1, 0,
                                    resources.stream, external_args, nullptr);
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_external_launch_failed", error);
  }
  if (!launch_without_updater(5, &result.external_update_visited)) {
    result.reason = "cuda_device_update_external_grid_mismatch";
    return result;
  }
  if (!launch_without_updater(10, &result.external_reset_visited)) {
    result.reason = "cuda_device_update_external_state_not_persistent";
    return result;
  }
  result.external_update_persists = true;

  if (!clear_visited() ||
      !run_case(7, true, 7, &result.partial_failure_visited)) {
    result.reason = "cuda_device_update_partial_failure_setup_failed";
    return result;
  }
  host_control.device_node = device_node_handle;
  host_control.grid_x = 0;
  host_control.enabled = 1;
  host_control.status = 0xffffffffu;
  error = driver.memcpy_host_to_device.call(resources.control, &host_control,
                                            sizeof(host_control));
  if (error == CUDA_SUCCESS) {
    error = driver.graph_launch.call(resources.graph_exec, resources.stream);
  }
  if (error == CUDA_SUCCESS) {
    error = driver.stream_synchronize.call(resources.stream);
  }
  if (error == CUDA_SUCCESS) {
    error = driver.memcpy_device_to_host.call(
        &host_control, resources.control, sizeof(host_control));
  }
  if (error == CUDA_SUCCESS) {
    error = driver.memcpy_device_to_host.call(
        &result.partial_failure_visited, resources.visited,
        sizeof(result.partial_failure_visited));
  }
  if (error != CUDA_SUCCESS) {
    return fail("cuda_device_update_partial_failure_launch_failed", error);
  }
  if (host_control.status == CUDA_SUCCESS ||
      result.partial_failure_visited != 14) {
    result.reason = "cuda_device_update_partial_failure_not_capacity_safe";
    return result;
  }
  result.partial_failure_capacity_safe = true;
  result.passed = true;
  result.reason = "none";
  result.driver_error = CUDA_SUCCESS;
  return result;
}

}  // namespace taichi::lang::cuda
