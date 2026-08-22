#include "taichi/rhi/cuda/primitives/matrix_ptx.h"

#include "taichi/common/core.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace taichi::lang::cuda {
namespace {

const char kCudaMatrixMmaPtx[] = R"ptx(
.version 6.3
.target sm_70
.address_size 64

.visible .entry matrix_mma_f16_f32_m16n16k16(
    .param .u64 a_param,
    .param .u64 b_param,
    .param .u64 output_param,
    .param .u32 batch_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<12>;
    .reg .b32 %a<8>;
    .reg .b32 %b<8>;
    .reg .f32 %c<8>;
    .reg .f32 %d<8>;

    ld.param.u64 %rd1, [a_param];
    ld.param.u64 %rd2, [b_param];
    ld.param.u64 %rd3, [output_param];
    ld.param.u32 %r1, [batch_param];

    mov.u32 %r2, %ctaid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r2, 512;
    mul.wide.u32 %rd5, %r2, 1024;
    add.u64 %rd6, %rd1, %rd4;
    add.u64 %rd7, %rd2, %rd4;
    add.u64 %rd8, %rd3, %rd5;

    wmma.load.a.sync.aligned.m16n16k16.global.row.f16
        {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, [%rd6], 16;
    wmma.load.b.sync.aligned.m16n16k16.global.row.f16
        {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}, [%rd7], 16;

    mov.f32 %c0, 0f00000000;
    mov.f32 %c1, 0f00000000;
    mov.f32 %c2, 0f00000000;
    mov.f32 %c3, 0f00000000;
    mov.f32 %c4, 0f00000000;
    mov.f32 %c5, 0f00000000;
    mov.f32 %c6, 0f00000000;
    mov.f32 %c7, 0f00000000;

    wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32
        {%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7},
        {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7},
        {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7},
        {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7};

    wmma.store.d.sync.aligned.m16n16k16.global.row.f32
        [%rd8], {%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7}, 16;

DONE:
    ret;
}
)ptx";

std::once_flag matrix_mma_module_once;
void *matrix_mma_module{nullptr};
void *matrix_mma_f16_f32_func{nullptr};

void load_matrix_mma_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&matrix_mma_module, kCudaMatrixMmaPtx, 0, nullptr,
                             nullptr);
  driver.module_get_function(&matrix_mma_f16_f32_func, matrix_mma_module,
                             "matrix_mma_f16_f32_m16n16k16");
}

}  // namespace

bool driver_matrix_mma_f16_f32_available() {
  if (!CUDADriver::get_instance_without_context()
           .nvidia_extensions_available()) {
    return false;
  }
  return CUDAContext::get_instance().get_compute_capability() >= 70;
}

std::size_t driver_matrix_mma_f16_f32(void *a,
                                      void *b,
                                      void *output,
                                      int batch_count,
                                      void *stream) {
  TI_ERROR_IF(batch_count < 0,
              "CUDA matrix MMA expects a non-negative batch count.");
  TI_ERROR_IF(!a || !b || !output,
              "CUDA matrix MMA received a null pointer.");
  if (batch_count == 0) {
    return 0;
  }
  TI_ERROR_IF(!driver_matrix_mma_f16_f32_available(),
              "CUDA matrix MMA requires NVIDIA compute capability 7.0 or "
              "newer.");
  TI_ERROR_IF((reinterpret_cast<std::uintptr_t>(a) % 32) != 0 ||
                  (reinterpret_cast<std::uintptr_t>(b) % 32) != 0 ||
                  (reinterpret_cast<std::uintptr_t>(output) % 32) != 0,
              "CUDA matrix MMA buffers must be 32-byte aligned.");

  std::call_once(matrix_mma_module_once, load_matrix_mma_module_once);
  void *a_arg = a;
  void *b_arg = b;
  void *output_arg = output;
  uint32_t batch_arg = static_cast<uint32_t>(batch_count);
  std::vector<void *> args{&a_arg, &b_arg, &output_arg, &batch_arg};
  CUDAContext::get_instance().launch(
      matrix_mma_f16_f32_func, "cuda_matrix_mma_f16_f32", args, {},
      static_cast<unsigned>(batch_count), 32, 0, stream);
  return 0;
}

}  // namespace taichi::lang::cuda
