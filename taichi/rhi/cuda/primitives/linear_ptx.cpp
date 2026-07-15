#include "taichi/rhi/cuda/primitives/linear_ptx.h"

#include "taichi/common/core.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace taichi::lang::cuda {
namespace {
const char kCudaTransformPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry transform_i32_affine(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 scale_param,
    .param .u32 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [scale_param];
    ld.param.u32 %r3, [bias_param];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra DONE_I32;

    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.u32 %r8, [%rd4];
    mul.lo.u32 %r9, %r8, %r2;
    add.u32 %r9, %r9, %r3;
    st.global.u32 [%rd5], %r9;

DONE_I32:
    ret;
}

.visible .entry transform_f32_affine(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .f32 scale_param,
    .param .f32 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.f32 %f1, [scale_param];
    ld.param.f32 %f2, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F32;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f3, [%rd4];
    mul.rn.f32 %f4, %f3, %f1;
    add.rn.f32 %f4, %f4, %f2;
    st.global.f32 [%rd5], %f4;

DONE_F32:
    ret;
}

.visible .entry transform_u64_affine(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u64 scale_param,
    .param .u64 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u64 %rd3, [scale_param];
    ld.param.u64 %rd4, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_U64;

    mul.wide.u32 %rd5, %r5, 8;
    add.u64 %rd6, %rd1, %rd5;
    add.u64 %rd7, %rd2, %rd5;
    ld.global.u64 %rd8, [%rd6];
    mul.lo.u64 %rd9, %rd8, %rd3;
    add.u64 %rd10, %rd9, %rd4;
    st.global.u64 [%rd7], %rd10;

DONE_U64:
    ret;
}

.visible .entry transform_f64_affine(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .f64 scale_param,
    .param .f64 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f64 %fd<5>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.f64 %fd1, [scale_param];
    ld.param.f64 %fd2, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F64;

    mul.wide.u32 %rd3, %r5, 8;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.f64 %fd3, [%rd4];
    mul.rn.f64 %fd4, %fd3, %fd1;
    add.rn.f64 %fd4, %fd4, %fd2;
    st.global.f64 [%rd5], %fd4;

DONE_F64:
    ret;
}

.visible .entry transform_u32_affine_strided(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u64 src_offset_param,
    .param .u64 src_stride_param,
    .param .u64 dst_offset_param,
    .param .u64 dst_stride_param,
    .param .u32 scale_param,
    .param .u32 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u64 %rd3, [src_offset_param];
    ld.param.u64 %rd4, [src_stride_param];
    ld.param.u64 %rd5, [dst_offset_param];
    ld.param.u64 %rd6, [dst_stride_param];
    ld.param.u32 %r2, [scale_param];
    ld.param.u32 %r3, [bias_param];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra DONE_U32_STRIDED;

    cvt.u64.u32 %rd7, %r7;
    mul.lo.u64 %rd8, %rd7, %rd4;
    add.u64 %rd8, %rd8, %rd3;
    add.u64 %rd9, %rd1, %rd8;
    mul.lo.u64 %rd10, %rd7, %rd6;
    add.u64 %rd10, %rd10, %rd5;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.u32 %r8, [%rd9];
    mul.lo.u32 %r9, %r8, %r2;
    add.u32 %r9, %r9, %r3;
    st.global.u32 [%rd11], %r9;

DONE_U32_STRIDED:
    ret;
}

.visible .entry transform_f32_affine_strided(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u64 src_offset_param,
    .param .u64 src_stride_param,
    .param .u64 dst_offset_param,
    .param .u64 dst_stride_param,
    .param .f32 scale_param,
    .param .f32 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u64 %rd3, [src_offset_param];
    ld.param.u64 %rd4, [src_stride_param];
    ld.param.u64 %rd5, [dst_offset_param];
    ld.param.u64 %rd6, [dst_stride_param];
    ld.param.f32 %f1, [scale_param];
    ld.param.f32 %f2, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F32_STRIDED;

    cvt.u64.u32 %rd7, %r5;
    mul.lo.u64 %rd8, %rd7, %rd4;
    add.u64 %rd8, %rd8, %rd3;
    add.u64 %rd9, %rd1, %rd8;
    mul.lo.u64 %rd10, %rd7, %rd6;
    add.u64 %rd10, %rd10, %rd5;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.f32 %f3, [%rd9];
    mul.rn.f32 %f4, %f3, %f1;
    add.rn.f32 %f4, %f4, %f2;
    st.global.f32 [%rd11], %f4;

DONE_F32_STRIDED:
    ret;
}

.visible .entry transform_u64_affine_strided(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u64 src_offset_param,
    .param .u64 src_stride_param,
    .param .u64 dst_offset_param,
    .param .u64 dst_stride_param,
    .param .u64 scale_param,
    .param .u64 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u64 %rd3, [src_offset_param];
    ld.param.u64 %rd4, [src_stride_param];
    ld.param.u64 %rd5, [dst_offset_param];
    ld.param.u64 %rd6, [dst_stride_param];
    ld.param.u64 %rd7, [scale_param];
    ld.param.u64 %rd8, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_U64_STRIDED;

    cvt.u64.u32 %rd9, %r5;
    mul.lo.u64 %rd10, %rd9, %rd4;
    add.u64 %rd10, %rd10, %rd3;
    add.u64 %rd11, %rd1, %rd10;
    mul.lo.u64 %rd12, %rd9, %rd6;
    add.u64 %rd12, %rd12, %rd5;
    add.u64 %rd13, %rd2, %rd12;
    ld.global.u64 %rd10, [%rd11];
    mul.lo.u64 %rd11, %rd10, %rd7;
    add.u64 %rd11, %rd11, %rd8;
    st.global.u64 [%rd13], %rd11;

DONE_U64_STRIDED:
    ret;
}

.visible .entry transform_f64_affine_strided(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u64 src_offset_param,
    .param .u64 src_stride_param,
    .param .u64 dst_offset_param,
    .param .u64 dst_stride_param,
    .param .f64 scale_param,
    .param .f64 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f64 %fd<5>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u64 %rd3, [src_offset_param];
    ld.param.u64 %rd4, [src_stride_param];
    ld.param.u64 %rd5, [dst_offset_param];
    ld.param.u64 %rd6, [dst_stride_param];
    ld.param.f64 %fd1, [scale_param];
    ld.param.f64 %fd2, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F64_STRIDED;

    cvt.u64.u32 %rd7, %r5;
    mul.lo.u64 %rd8, %rd7, %rd4;
    add.u64 %rd8, %rd8, %rd3;
    add.u64 %rd9, %rd1, %rd8;
    mul.lo.u64 %rd10, %rd7, %rd6;
    add.u64 %rd10, %rd10, %rd5;
    add.u64 %rd11, %rd2, %rd10;
    ld.global.f64 %fd3, [%rd9];
    mul.rn.f64 %fd4, %fd3, %fd1;
    add.rn.f64 %fd4, %fd4, %fd2;
    st.global.f64 [%rd11], %fd4;

DONE_F64_STRIDED:
    ret;
}
)ptx";

std::once_flag transform_module_once;
void *transform_module{nullptr};
void *transform_i32_func{nullptr};
void *transform_f32_func{nullptr};
void *transform_u64_func{nullptr};
void *transform_f64_func{nullptr};
void *transform_u32_strided_func{nullptr};
void *transform_f32_strided_func{nullptr};
void *transform_u64_strided_func{nullptr};
void *transform_f64_strided_func{nullptr};

const char kCudaIndexedCopyPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry gather_u32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_GATHER;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u32 %r6, [%rd5];
    setp.ge.u32 %p1, %r6, %r8;
    @%p1 bra INVALID_GATHER;
    mul.wide.u32 %rd6, %r6, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u32 %r7, [%rd7];
    add.u64 %rd8, %rd3, %rd4;
    st.global.u32 [%rd8], %r7;
    bra DONE_GATHER;

INVALID_GATHER:
    add.u64 %rd8, %rd3, %rd4;
    mov.u32 %r7, 0;
    st.global.u32 [%rd8], %r7;

DONE_GATHER:
    ret;
}

.visible .entry scatter_u32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_SCATTER;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r6, [%rd5];
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r7, [%rd6];
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra DONE_SCATTER;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd3, %rd7;
    st.global.u32 [%rd8], %r6;

DONE_SCATTER:
    ret;
}

.visible .entry gather_words_strided_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param,
    .param .u32 item_words_param,
    .param .u64 src_offset_words_param,
    .param .u64 src_stride_words_param,
    .param .u64 dst_offset_words_param,
    .param .u64 dst_stride_words_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [bound_param];
    ld.param.u32 %r3, [item_words_param];
    ld.param.u64 %rd4, [src_offset_words_param];
    ld.param.u64 %rd5, [src_stride_words_param];
    ld.param.u64 %rd6, [dst_offset_words_param];
    ld.param.u64 %rd7, [dst_stride_words_param];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra DONE_GATHER_WORDS;

    cvt.u64.u32 %rd8, %r7;
    mul.lo.u64 %rd12, %rd8, %rd7;
    add.u64 %rd12, %rd12, %rd6;
    shl.b64 %rd12, %rd12, 2;
    add.u64 %rd13, %rd3, %rd12;

    mul.wide.u32 %rd9, %r7, 4;
    add.u64 %rd9, %rd2, %rd9;
    ld.global.s32 %r8, [%rd9];
    setp.lt.s32 %p2, %r8, 0;
    @%p2 bra ZERO_GATHER_WORDS;
    setp.ge.s32 %p3, %r8, %r2;
    @%p3 bra ZERO_GATHER_WORDS;

    cvt.u64.u32 %rd9, %r8;
    mul.lo.u64 %rd10, %rd9, %rd5;
    add.u64 %rd10, %rd10, %rd4;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd11, %rd1, %rd10;
    mov.u32 %r9, 0;
COPY_GATHER_WORDS:
    setp.ge.u32 %p1, %r9, %r3;
    @%p1 bra DONE_GATHER_WORDS;
    mul.wide.u32 %rd8, %r9, 4;
    add.u64 %rd9, %rd11, %rd8;
    add.u64 %rd10, %rd13, %rd8;
    ld.global.u32 %r10, [%rd9];
    st.global.u32 [%rd10], %r10;
    add.u32 %r9, %r9, 1;
    bra COPY_GATHER_WORDS;

ZERO_GATHER_WORDS:
    mov.u32 %r9, 0;
ZERO_GATHER_WORDS_LOOP:
    setp.ge.u32 %p1, %r9, %r3;
    @%p1 bra DONE_GATHER_WORDS;
    mul.wide.u32 %rd8, %r9, 4;
    add.u64 %rd10, %rd13, %rd8;
    mov.u32 %r10, 0;
    st.global.u32 [%rd10], %r10;
    add.u32 %r9, %r9, 1;
    bra ZERO_GATHER_WORDS_LOOP;

DONE_GATHER_WORDS:
    ret;
}

.visible .entry scatter_words_strided_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param,
    .param .u32 item_words_param,
    .param .u64 src_offset_words_param,
    .param .u64 src_stride_words_param,
    .param .u64 dst_offset_words_param,
    .param .u64 dst_stride_words_param
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<14>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [bound_param];
    ld.param.u32 %r3, [item_words_param];
    ld.param.u64 %rd4, [src_offset_words_param];
    ld.param.u64 %rd5, [src_stride_words_param];
    ld.param.u64 %rd6, [dst_offset_words_param];
    ld.param.u64 %rd7, [dst_stride_words_param];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra DONE_SCATTER_WORDS;

    mul.wide.u32 %rd8, %r7, 4;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.s32 %r8, [%rd8];
    setp.lt.s32 %p2, %r8, 0;
    @%p2 bra DONE_SCATTER_WORDS;
    setp.ge.s32 %p3, %r8, %r2;
    @%p3 bra DONE_SCATTER_WORDS;

    cvt.u64.u32 %rd8, %r7;
    mul.lo.u64 %rd9, %rd8, %rd5;
    add.u64 %rd9, %rd9, %rd4;
    shl.b64 %rd9, %rd9, 2;
    add.u64 %rd10, %rd1, %rd9;
    cvt.u64.u32 %rd11, %r8;
    mul.lo.u64 %rd11, %rd11, %rd7;
    add.u64 %rd11, %rd11, %rd6;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd12, %rd3, %rd11;
    mov.u32 %r9, 0;
COPY_SCATTER_WORDS:
    setp.ge.u32 %p1, %r9, %r3;
    @%p1 bra DONE_SCATTER_WORDS;
    mul.wide.u32 %rd13, %r9, 4;
    add.u64 %rd8, %rd10, %rd13;
    add.u64 %rd9, %rd12, %rd13;
    ld.global.u32 %r10, [%rd8];
    st.global.u32 [%rd9], %r10;
    add.u32 %r9, %r9, 1;
    bra COPY_SCATTER_WORDS;

DONE_SCATTER_WORDS:
    ret;
}
)ptx";

std::once_flag indexed_copy_module_once;
void *indexed_copy_module{nullptr};
void *gather_u32_func{nullptr};
void *scatter_u32_func{nullptr};
void *gather_words_strided_func{nullptr};
void *scatter_words_strided_func{nullptr};

const char kCudaScatterAddPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry scatter_add_u32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_U32;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.s32 %r6, [%rd5];
    setp.lt.s32 %p1, %r6, 0;
    @%p1 bra DONE_U32;
    setp.ge.s32 %p2, %r6, %r8;
    @%p2 bra DONE_U32;
    add.u64 %rd6, %rd1, %rd4;
    ld.global.u32 %r7, [%rd6];
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd3, %rd7;
    atom.global.add.u32 %r9, [%rd8], %r7;

DONE_U32:
    ret;
}

.visible .entry scatter_add_f32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F32;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.s32 %r6, [%rd5];
    setp.lt.s32 %p1, %r6, 0;
    @%p1 bra DONE_F32;
    setp.ge.s32 %p2, %r6, %r8;
    @%p2 bra DONE_F32;
    add.u64 %rd6, %rd1, %rd4;
    ld.global.f32 %f1, [%rd6];
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd3, %rd7;
    atom.global.add.f32 %f2, [%rd8], %f1;

DONE_F32:
    ret;
}
)ptx";

const char kCudaDiagnosticPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry zero_u32(
    .param .u64 output_param
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    ld.param.u64 %rd1, [output_param];
    mov.u32 %r1, 0;
    st.global.u32 [%rd1], %r1;
    ret;
}

.visible .entry zero_u64(
    .param .u64 output_param
)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output_param];
    mov.u64 %rd2, 0;
    st.global.u64 [%rd1], %rd2;
    ret;
}

.visible .entry check_count(
    .param .u64 values_param,
    .param .u64 output_param,
    .param .u32 n_param,
    .param .u32 type_param,
    .param .u64 offset_param,
    .param .u64 stride_param,
    .param .u32 op_param,
    .param .s32 lower_param,
    .param .s32 upper_param
)
{
    .reg .pred %p<10>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<18>;

    ld.param.u64 %rd1, [values_param];
    ld.param.u64 %rd2, [output_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [type_param];
    ld.param.u64 %rd3, [offset_param];
    ld.param.u64 %rd4, [stride_param];
    ld.param.u32 %r3, [op_param];
    ld.param.s32 %r4, [lower_param];
    ld.param.s32 %r5, [upper_param];

    mov.u32 %r6, %ctaid.x;
    mov.u32 %r7, %ntid.x;
    mov.u32 %r8, %tid.x;
    mad.lo.u32 %r9, %r6, %r7, %r8;
    setp.ge.u32 %p1, %r9, %r1;
    @%p1 bra CHECK_DONE;

    cvt.u64.u32 %rd5, %r9;
    mul.lo.u64 %rd6, %rd5, %rd4;
    add.u64 %rd6, %rd6, %rd3;
    add.u64 %rd7, %rd1, %rd6;
    mov.u32 %r10, 0;
    setp.ge.u32 %p2, %r2, 3;
    @%p2 bra CHECK_LOAD_64;
    ld.global.u32 %r11, [%rd7];
    bra CHECK_DISPATCH;
CHECK_LOAD_64:
    ld.global.u64 %rd8, [%rd7];

CHECK_DISPATCH:
    setp.eq.u32 %p3, %r3, 0;
    @%p3 bra CHECK_NONZERO;
    setp.eq.u32 %p3, %r3, 1;
    @%p3 bra CHECK_ZERO;
    setp.eq.u32 %p3, %r3, 2;
    @%p3 bra CHECK_NAN;
    setp.eq.u32 %p3, %r3, 3;
    @%p3 bra CHECK_INF;
    setp.eq.u32 %p3, %r3, 4;
    @%p3 bra CHECK_NOT_FINITE;
    setp.eq.u32 %p3, %r3, 5;
    @%p3 bra CHECK_BOUNDS;
    bra CHECK_DONE;

CHECK_NONZERO:
    @%p2 setp.ne.u64 %p4, %rd8, 0;
    @!%p2 setp.ne.u32 %p4, %r11, 0;
    selp.u32 %r10, 1, 0, %p4;
    bra CHECK_ACCUMULATE;

CHECK_ZERO:
    @%p2 setp.eq.u64 %p4, %rd8, 0;
    @!%p2 setp.eq.u32 %p4, %r11, 0;
    selp.u32 %r10, 1, 0, %p4;
    bra CHECK_ACCUMULATE;

CHECK_NAN:
    setp.eq.u32 %p4, %r2, 1;
    @%p4 bra CHECK_NAN_F32;
    setp.eq.u32 %p4, %r2, 5;
    @%p4 bra CHECK_NAN_F64;
    bra CHECK_DONE;
CHECK_NAN_F32:
    and.b32 %r12, %r11, 2139095040;
    and.b32 %r13, %r11, 8388607;
    setp.eq.u32 %p5, %r12, 2139095040;
    setp.ne.u32 %p6, %r13, 0;
    and.pred %p7, %p5, %p6;
    selp.u32 %r10, 1, 0, %p7;
    bra CHECK_ACCUMULATE;
CHECK_NAN_F64:
    and.b64 %rd9, %rd8, 9218868437227405312;
    and.b64 %rd10, %rd8, 4503599627370495;
    setp.eq.u64 %p5, %rd9, 9218868437227405312;
    setp.ne.u64 %p6, %rd10, 0;
    and.pred %p7, %p5, %p6;
    selp.u32 %r10, 1, 0, %p7;
    bra CHECK_ACCUMULATE;

CHECK_INF:
    setp.eq.u32 %p4, %r2, 1;
    @%p4 bra CHECK_INF_F32;
    setp.eq.u32 %p4, %r2, 5;
    @%p4 bra CHECK_INF_F64;
    bra CHECK_DONE;
CHECK_INF_F32:
    and.b32 %r12, %r11, 2147483647;
    setp.eq.u32 %p5, %r12, 2139095040;
    selp.u32 %r10, 1, 0, %p5;
    bra CHECK_ACCUMULATE;
CHECK_INF_F64:
    and.b64 %rd9, %rd8, 9223372036854775807;
    setp.eq.u64 %p5, %rd9, 9218868437227405312;
    selp.u32 %r10, 1, 0, %p5;
    bra CHECK_ACCUMULATE;

CHECK_NOT_FINITE:
    setp.eq.u32 %p4, %r2, 1;
    @%p4 bra CHECK_NOT_FINITE_F32;
    setp.eq.u32 %p4, %r2, 5;
    @%p4 bra CHECK_NOT_FINITE_F64;
    bra CHECK_DONE;
CHECK_NOT_FINITE_F32:
    and.b32 %r12, %r11, 2139095040;
    setp.eq.u32 %p5, %r12, 2139095040;
    selp.u32 %r10, 1, 0, %p5;
    bra CHECK_ACCUMULATE;
CHECK_NOT_FINITE_F64:
    and.b64 %rd9, %rd8, 9218868437227405312;
    setp.eq.u64 %p5, %rd9, 9218868437227405312;
    selp.u32 %r10, 1, 0, %p5;
    bra CHECK_ACCUMULATE;

CHECK_BOUNDS:
    setp.eq.u32 %p4, %r2, 0;
    @%p4 bra CHECK_BOUNDS_I32;
    setp.eq.u32 %p4, %r2, 2;
    @%p4 bra CHECK_BOUNDS_U32;
    setp.eq.u32 %p4, %r2, 3;
    @%p4 bra CHECK_BOUNDS_U64;
    setp.eq.u32 %p4, %r2, 4;
    @%p4 bra CHECK_BOUNDS_I64;
    bra CHECK_DONE;
CHECK_BOUNDS_I32:
    setp.lt.s32 %p5, %r11, %r4;
    setp.ge.s32 %p6, %r11, %r5;
    or.pred %p7, %p5, %p6;
    selp.u32 %r10, 1, 0, %p7;
    bra CHECK_ACCUMULATE;
CHECK_BOUNDS_U32:
    setp.lt.s32 %p5, %r4, 0;
    @%p5 bra CHECK_BOUNDS_U32_UPPER;
    setp.lt.u32 %p6, %r11, %r4;
    @%p6 mov.u32 %r10, 1;
CHECK_BOUNDS_U32_UPPER:
    setp.ge.u32 %p7, %r11, %r5;
    @%p7 mov.u32 %r10, 1;
    bra CHECK_ACCUMULATE;
CHECK_BOUNDS_U64:
    cvt.u64.u32 %rd11, %r5;
    setp.lt.s32 %p5, %r4, 0;
    @%p5 bra CHECK_BOUNDS_U64_UPPER;
    cvt.u64.u32 %rd12, %r4;
    setp.lt.u64 %p6, %rd8, %rd12;
    @%p6 mov.u32 %r10, 1;
CHECK_BOUNDS_U64_UPPER:
    setp.ge.u64 %p7, %rd8, %rd11;
    @%p7 mov.u32 %r10, 1;
    bra CHECK_ACCUMULATE;
CHECK_BOUNDS_I64:
    cvt.s64.s32 %rd11, %r4;
    cvt.s64.s32 %rd12, %r5;
    setp.lt.s64 %p5, %rd8, %rd11;
    setp.ge.s64 %p6, %rd8, %rd12;
    or.pred %p7, %p5, %p6;
    selp.u32 %r10, 1, 0, %p7;

CHECK_ACCUMULATE:
    setp.eq.u32 %p8, %r10, 0;
    @%p8 bra CHECK_DONE;
    atom.global.add.u32 %r14, [%rd2], %r10;
CHECK_DONE:
    ret;
}

.visible .entry metric_reduce(
    .param .u64 values_param,
    .param .u64 other_param,
    .param .u64 output_param,
    .param .u32 n_param,
    .param .u32 type_param,
    .param .u64 values_offset_param,
    .param .u64 values_stride_param,
    .param .u64 other_offset_param,
    .param .u64 other_stride_param,
    .param .u32 op_param
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<4>;
    .reg .f64 %fd<4>;

    ld.param.u64 %rd1, [values_param];
    ld.param.u64 %rd2, [other_param];
    ld.param.u64 %rd3, [output_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [type_param];
    ld.param.u64 %rd4, [values_offset_param];
    ld.param.u64 %rd5, [values_stride_param];
    ld.param.u64 %rd6, [other_offset_param];
    ld.param.u64 %rd7, [other_stride_param];
    ld.param.u32 %r3, [op_param];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra METRIC_DONE;
    cvt.u64.u32 %rd8, %r7;
    mul.lo.u64 %rd9, %rd8, %rd5;
    add.u64 %rd9, %rd9, %rd4;
    add.u64 %rd10, %rd1, %rd9;
    setp.eq.u32 %p2, %r2, 1;
    @%p2 bra METRIC_F32;

METRIC_F64:
    ld.global.f64 %fd1, [%rd10];
    setp.eq.u32 %p3, %r3, 1;
    @!%p3 bra METRIC_F64_ABS;
    mul.lo.u64 %rd11, %rd8, %rd7;
    add.u64 %rd11, %rd11, %rd6;
    add.u64 %rd12, %rd2, %rd11;
    ld.global.f64 %fd2, [%rd12];
    sub.rn.f64 %fd1, %fd1, %fd2;
METRIC_F64_ABS:
    mov.b64 %rd13, %fd1;
    and.b64 %rd13, %rd13, 9223372036854775807;
    and.b64 %rd14, %rd13, 9218868437227405312;
    setp.eq.u64 %p4, %rd14, 9218868437227405312;
    @%p4 mov.u64 %rd13, 9218868437227405312;
    atom.global.max.u64 %rd15, [%rd3], %rd13;
    bra METRIC_DONE;

METRIC_F32:
    ld.global.f32 %f1, [%rd10];
    setp.eq.u32 %p3, %r3, 1;
    @!%p3 bra METRIC_F32_ABS;
    mul.lo.u64 %rd11, %rd8, %rd7;
    add.u64 %rd11, %rd11, %rd6;
    add.u64 %rd12, %rd2, %rd11;
    ld.global.f32 %f2, [%rd12];
    sub.rn.f32 %f1, %f1, %f2;
METRIC_F32_ABS:
    mov.b32 %r8, %f1;
    and.b32 %r8, %r8, 2147483647;
    and.b32 %r9, %r8, 2139095040;
    setp.eq.u32 %p4, %r9, 2139095040;
    @%p4 mov.u32 %r8, 2139095040;
    atom.global.max.u32 %r10, [%rd3], %r8;

METRIC_DONE:
    ret;
}
)ptx";

std::once_flag diagnostic_module_once;
void *diagnostic_module{nullptr};
void *zero_u32_func{nullptr};
void *zero_u64_func{nullptr};
void *check_count_func{nullptr};
void *metric_reduce_func{nullptr};

std::once_flag scatter_add_module_once;
void *scatter_add_module{nullptr};
void *scatter_add_u32_func{nullptr};
void *scatter_add_f32_func{nullptr};

void load_transform_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&transform_module, kCudaTransformPtx, 0, nullptr,
                             nullptr);
  driver.module_get_function(&transform_i32_func, transform_module,
                             "transform_i32_affine");
  driver.module_get_function(&transform_f32_func, transform_module,
                             "transform_f32_affine");
  driver.module_get_function(&transform_u64_func, transform_module,
                             "transform_u64_affine");
  driver.module_get_function(&transform_f64_func, transform_module,
                             "transform_f64_affine");
  driver.module_get_function(&transform_u32_strided_func, transform_module,
                             "transform_u32_affine_strided");
  driver.module_get_function(&transform_f32_strided_func, transform_module,
                             "transform_f32_affine_strided");
  driver.module_get_function(&transform_u64_strided_func, transform_module,
                             "transform_u64_affine_strided");
  driver.module_get_function(&transform_f64_strided_func, transform_module,
                             "transform_f64_affine_strided");
}

void *cuda_transform_function(CudaTransformValueType value_type) {
  std::call_once(transform_module_once, load_transform_module_once);
  switch (value_type) {
    case CudaTransformValueType::i32:
    case CudaTransformValueType::u32:
      return transform_i32_func;
    case CudaTransformValueType::f32:
      return transform_f32_func;
    case CudaTransformValueType::u64:
    case CudaTransformValueType::i64:
      return transform_u64_func;
    case CudaTransformValueType::f64:
      return transform_f64_func;
  }
  TI_ERROR("Unsupported CUDA transform value type.");
  return nullptr;
}

void *cuda_transform_strided_function(CudaTransformValueType value_type) {
  std::call_once(transform_module_once, load_transform_module_once);
  switch (value_type) {
    case CudaTransformValueType::i32:
    case CudaTransformValueType::u32:
      return transform_u32_strided_func;
    case CudaTransformValueType::f32:
      return transform_f32_strided_func;
    case CudaTransformValueType::u64:
    case CudaTransformValueType::i64:
      return transform_u64_strided_func;
    case CudaTransformValueType::f64:
      return transform_f64_strided_func;
  }
  TI_ERROR("Unsupported CUDA strided transform value type.");
  return nullptr;
}

void load_indexed_copy_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&indexed_copy_module, kCudaIndexedCopyPtx, 0,
                             nullptr, nullptr);
  driver.module_get_function(&gather_u32_func, indexed_copy_module,
                             "gather_u32_by_i32");
  driver.module_get_function(&scatter_u32_func, indexed_copy_module,
                             "scatter_u32_by_i32");
  driver.module_get_function(&gather_words_strided_func, indexed_copy_module,
                             "gather_words_strided_by_i32");
  driver.module_get_function(&scatter_words_strided_func, indexed_copy_module,
                             "scatter_words_strided_by_i32");
}

void load_scatter_add_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&scatter_add_module, kCudaScatterAddPtx, 0,
                             nullptr, nullptr);
  driver.module_get_function(&scatter_add_u32_func, scatter_add_module,
                             "scatter_add_u32_by_i32");
  driver.module_get_function(&scatter_add_f32_func, scatter_add_module,
                             "scatter_add_f32_by_i32");
}

void load_diagnostic_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&diagnostic_module, kCudaDiagnosticPtx, 0, nullptr,
                             nullptr);
  driver.module_get_function(&zero_u32_func, diagnostic_module, "zero_u32");
  driver.module_get_function(&zero_u64_func, diagnostic_module, "zero_u64");
  driver.module_get_function(&check_count_func, diagnostic_module,
                             "check_count");
  driver.module_get_function(&metric_reduce_func, diagnostic_module,
                             "metric_reduce");
}

void ensure_diagnostic_module() {
  std::call_once(diagnostic_module_once, load_diagnostic_module_once);
}

void *cuda_indexed_copy_function(CudaIndexedCopyOp op) {
  std::call_once(indexed_copy_module_once, load_indexed_copy_module_once);
  switch (op) {
    case CudaIndexedCopyOp::gather:
      return gather_u32_func;
    case CudaIndexedCopyOp::scatter:
      return scatter_u32_func;
  }
  TI_ERROR("Unsupported CUDA indexed copy op.");
  return nullptr;
}

void *cuda_scatter_add_function(CudaScatterAddValueType value_type) {
  std::call_once(scatter_add_module_once, load_scatter_add_module_once);
  switch (value_type) {
    case CudaScatterAddValueType::i32:
    case CudaScatterAddValueType::u32:
      return scatter_add_u32_func;
    case CudaScatterAddValueType::f32:
      return scatter_add_f32_func;
    case CudaScatterAddValueType::u64:
    case CudaScatterAddValueType::i64:
    case CudaScatterAddValueType::f64:
      return nullptr;
  }
  return nullptr;
}

}  // namespace
bool driver_transform_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_transform_affine(void *src,
                                    void *dst,
                                    int num_items,
                                    CudaTransformValueType value_type,
                                    double scale,
                                    double bias) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver transform expects non-negative num_items.");
  TI_ERROR_IF(!src || !dst, "CUDA driver transform received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *func = cuda_transform_function(value_type);
  void *src_arg = src;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  std::vector<void *> args;
  args.reserve(5);
  args.push_back(&src_arg);
  args.push_back(&dst_arg);
  args.push_back(&n_arg);
  uint32_t scale_u32 = 0;
  uint32_t bias_u32 = 0;
  uint64_t scale_u64 = 0;
  uint64_t bias_u64 = 0;
  float scale_f32 = 0.0f;
  float bias_f32 = 0.0f;
  double scale_f64 = 0.0;
  double bias_f64 = 0.0;
  if (value_type == CudaTransformValueType::i32) {
    scale_u32 = static_cast<uint32_t>(static_cast<int32_t>(scale));
    bias_u32 = static_cast<uint32_t>(static_cast<int32_t>(bias));
    args.push_back(&scale_u32);
    args.push_back(&bias_u32);
  } else if (value_type == CudaTransformValueType::u32) {
    scale_u32 = static_cast<uint32_t>(scale);
    bias_u32 = static_cast<uint32_t>(bias);
    args.push_back(&scale_u32);
    args.push_back(&bias_u32);
  } else if (value_type == CudaTransformValueType::f32) {
    scale_f32 = static_cast<float>(scale);
    bias_f32 = static_cast<float>(bias);
    args.push_back(&scale_f32);
    args.push_back(&bias_f32);
  } else if (value_type == CudaTransformValueType::i64) {
    scale_u64 = static_cast<uint64_t>(static_cast<int64_t>(scale));
    bias_u64 = static_cast<uint64_t>(static_cast<int64_t>(bias));
    args.push_back(&scale_u64);
    args.push_back(&bias_u64);
  } else if (value_type == CudaTransformValueType::u64) {
    scale_u64 = static_cast<uint64_t>(scale);
    bias_u64 = static_cast<uint64_t>(bias);
    args.push_back(&scale_u64);
    args.push_back(&bias_u64);
  } else {
    scale_f64 = scale;
    bias_f64 = bias;
    args.push_back(&scale_f64);
    args.push_back(&bias_f64);
  }
  CUDAContext::get_instance().launch(func, "cuda_transform_affine", args, {},
                                     grid_dim, kBlockDim, 0);
  return 0;
}

void *cuda_indexed_copy_strided_function(CudaIndexedCopyOp op) {
  std::call_once(indexed_copy_module_once, load_indexed_copy_module_once);
  switch (op) {
    case CudaIndexedCopyOp::gather:
      return gather_words_strided_func;
    case CudaIndexedCopyOp::scatter:
      return scatter_words_strided_func;
  }
  TI_ERROR("Unsupported CUDA strided indexed copy op.");
  return nullptr;
}

std::size_t driver_transform_affine_strided(
    void *src,
    void *dst,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver strided transform expects non-negative num_items.");
  TI_ERROR_IF(!src || !dst,
              "CUDA driver strided transform received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *func = cuda_transform_strided_function(value_type);
  void *src_arg = src;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint64_t src_offset_arg = static_cast<uint64_t>(src_offset);
  uint64_t src_stride_arg = static_cast<uint64_t>(src_stride);
  uint64_t dst_offset_arg = static_cast<uint64_t>(dst_offset);
  uint64_t dst_stride_arg = static_cast<uint64_t>(dst_stride);
  uint32_t scale_u32 = 0;
  uint32_t bias_u32 = 0;
  uint64_t scale_u64 = 0;
  uint64_t bias_u64 = 0;
  float scale_f32 = 0.0f;
  float bias_f32 = 0.0f;
  double scale_f64 = 0.0;
  double bias_f64 = 0.0;
  std::vector<void *> args{&src_arg,       &dst_arg,       &n_arg,
                           &src_offset_arg, &src_stride_arg, &dst_offset_arg,
                           &dst_stride_arg};
  if (value_type == CudaTransformValueType::i32) {
    scale_u32 = static_cast<uint32_t>(static_cast<int32_t>(scale));
    bias_u32 = static_cast<uint32_t>(static_cast<int32_t>(bias));
    args.push_back(&scale_u32);
    args.push_back(&bias_u32);
  } else if (value_type == CudaTransformValueType::u32) {
    scale_u32 = static_cast<uint32_t>(scale);
    bias_u32 = static_cast<uint32_t>(bias);
    args.push_back(&scale_u32);
    args.push_back(&bias_u32);
  } else if (value_type == CudaTransformValueType::f32) {
    scale_f32 = static_cast<float>(scale);
    bias_f32 = static_cast<float>(bias);
    args.push_back(&scale_f32);
    args.push_back(&bias_f32);
  } else if (value_type == CudaTransformValueType::i64) {
    scale_u64 = static_cast<uint64_t>(static_cast<int64_t>(scale));
    bias_u64 = static_cast<uint64_t>(static_cast<int64_t>(bias));
    args.push_back(&scale_u64);
    args.push_back(&bias_u64);
  } else if (value_type == CudaTransformValueType::u64) {
    scale_u64 = static_cast<uint64_t>(scale);
    bias_u64 = static_cast<uint64_t>(bias);
    args.push_back(&scale_u64);
    args.push_back(&bias_u64);
  } else {
    scale_f64 = scale;
    bias_f64 = bias;
    args.push_back(&scale_f64);
    args.push_back(&bias_f64);
  }
  CUDAContext::get_instance().launch(
      func, "cuda_transform_affine_strided", args, {}, grid_dim, kBlockDim, 0);
  return 0;
}

std::size_t driver_transform_affine_packed_strided(
    void *src,
    void *dst,
    int num_items,
    int lane_count,
    CudaTransformValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  TI_ERROR_IF(lane_count <= 0,
              "CUDA driver packed transform expects positive lane_count.");
  const std::size_t value_size =
      value_type == CudaTransformValueType::i32 ||
              value_type == CudaTransformValueType::f32 ||
              value_type == CudaTransformValueType::u32
          ? 4
          : 8;
  for (int lane = 0; lane < lane_count; ++lane) {
    driver_transform_affine_strided(
        src, dst, num_items, value_type,
        src_offset + static_cast<std::size_t>(lane) * value_size, src_stride,
        dst_offset + static_cast<std::size_t>(lane) * value_size, dst_stride,
        scale, bias);
  }
  return 0;
}

bool driver_indexed_copy_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                int item_words,
                                CudaIndexedCopyOp op,
                                void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver indexed copy expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA driver indexed copy expects non-negative index_bound.");
  TI_ERROR_IF(item_words <= 0,
              "CUDA driver indexed copy expects at least one word per item.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA driver indexed copy received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  if (item_words != 1) {
    return driver_indexed_copy_strided(
        src, indices, dst, num_items, index_bound, item_words, 0, item_words,
        0, item_words, op, stream);
  }
  void *func = cuda_indexed_copy_function(op);
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t bound_arg = static_cast<uint32_t>(index_bound);
  std::vector<void *> args;
  args.reserve(5);
  args.push_back(&src_arg);
  args.push_back(&indices_arg);
  args.push_back(&dst_arg);
  args.push_back(&n_arg);
  args.push_back(&bound_arg);
  CUDAContext::get_instance().launch(func, "cuda_indexed_copy", args, {},
                                     grid_dim, kBlockDim, 0, stream);
  return 0;
}

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                CudaIndexedCopyOp op) {
  return driver_indexed_copy(src, indices, dst, num_items, index_bound, 1, op,
                             nullptr);
}

std::size_t driver_indexed_copy_strided(void *src,
                                        void *indices,
                                        void *dst,
                                        int num_items,
                                        int index_bound,
                                        int item_words,
                                        std::size_t src_offset_words,
                                        std::size_t src_stride_words,
                                        std::size_t dst_offset_words,
                                        std::size_t dst_stride_words,
                                        CudaIndexedCopyOp op,
                                        void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver strided indexed copy expects non-negative items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA driver strided indexed copy expects a non-negative bound.");
  TI_ERROR_IF(item_words <= 0,
              "CUDA driver strided indexed copy expects a positive payload.");
  TI_ERROR_IF(src_stride_words < static_cast<std::size_t>(item_words) ||
                  dst_stride_words < static_cast<std::size_t>(item_words),
              "CUDA driver strided indexed copy payload exceeds its stride.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA driver strided indexed copy received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *func = cuda_indexed_copy_strided_function(op);
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t bound_arg = static_cast<uint32_t>(index_bound);
  uint32_t item_words_arg = static_cast<uint32_t>(item_words);
  uint64_t src_offset_arg = static_cast<uint64_t>(src_offset_words);
  uint64_t src_stride_arg = static_cast<uint64_t>(src_stride_words);
  uint64_t dst_offset_arg = static_cast<uint64_t>(dst_offset_words);
  uint64_t dst_stride_arg = static_cast<uint64_t>(dst_stride_words);
  std::vector<void *> args{&src_arg,
                           &indices_arg,
                           &dst_arg,
                           &n_arg,
                           &bound_arg,
                           &item_words_arg,
                           &src_offset_arg,
                           &src_stride_arg,
                           &dst_offset_arg,
                           &dst_stride_arg};
  CUDAContext::get_instance().launch(func, "cuda_indexed_copy_strided", args,
                                     {}, grid_dim, kBlockDim, 0, stream);
  return 0;
}

bool driver_scatter_add_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_scatter_add(void *src,
                               void *indices,
                               void *dst,
                               int num_items,
                               int index_bound,
                               CudaScatterAddValueType value_type) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver scatter-add expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA driver scatter-add expects non-negative index_bound.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA driver scatter-add received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  void *func = cuda_scatter_add_function(value_type);
  TI_ERROR_IF(!func,
              "CUDA driver scatter-add currently supports i32, u32, and f32.");
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t bound_arg = static_cast<uint32_t>(index_bound);
  std::vector<void *> args;
  args.reserve(5);
  args.push_back(&src_arg);
  args.push_back(&indices_arg);
  args.push_back(&dst_arg);
  args.push_back(&n_arg);
  args.push_back(&bound_arg);
  CUDAContext::get_instance().launch(func, "cuda_scatter_add", args, {},
                                     grid_dim, kBlockDim, 0);
  return 0;
}

bool driver_check_count_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_check_count(void *values,
                               void *output,
                               int num_items,
                               CudaTransformValueType value_type,
                               std::size_t offset,
                               std::size_t stride,
                               CudaCheckOp op,
                               int lower,
                               int upper,
                               void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver check expects non-negative num_items.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver check received a null pointer.");
  ensure_diagnostic_module();
  void *output_arg = output;
  CUDAContext::get_instance().launch(zero_u32_func, "cuda_check_zero",
                                     {&output_arg}, {}, 1, 1, 0, stream);
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *values_arg = values;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t type_arg = static_cast<uint32_t>(value_type);
  uint64_t offset_arg = static_cast<uint64_t>(offset);
  uint64_t stride_arg = static_cast<uint64_t>(stride);
  uint32_t op_arg = static_cast<uint32_t>(op);
  int32_t lower_arg = static_cast<int32_t>(lower);
  int32_t upper_arg = static_cast<int32_t>(upper);
  std::vector<void *> args{&values_arg, &output_arg, &n_arg,
                           &type_arg,   &offset_arg, &stride_arg,
                           &op_arg,     &lower_arg,  &upper_arg};
  CUDAContext::get_instance().launch(check_count_func, "cuda_check_count",
                                     args, {}, grid_dim, kBlockDim, 0, stream);
  return 0;
}

bool driver_metric_reduce_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_metric_reduce(void *values,
                                 void *other,
                                 void *output,
                                 int num_items,
                                 CudaTransformValueType value_type,
                                 std::size_t values_offset,
                                 std::size_t values_stride,
                                 std::size_t other_offset,
                                 std::size_t other_stride,
                                 CudaMetricOp op,
                                 void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver metric expects non-negative num_items.");
  TI_ERROR_IF(!values || !output,
              "CUDA driver metric received a null pointer.");
  TI_ERROR_IF(op == CudaMetricOp::max_abs_delta && !other,
              "CUDA driver max_abs_delta received a null rhs pointer.");
  TI_ERROR_IF(value_type != CudaTransformValueType::f32 &&
                  value_type != CudaTransformValueType::f64,
              "CUDA driver metric supports only f32 and f64.");
  ensure_diagnostic_module();
  void *output_arg = output;
  void *zero_func =
      value_type == CudaTransformValueType::f32 ? zero_u32_func : zero_u64_func;
  CUDAContext::get_instance().launch(zero_func, "cuda_metric_zero",
                                     {&output_arg}, {}, 1, 1, 0, stream);
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *values_arg = values;
  void *other_arg = other;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t type_arg = static_cast<uint32_t>(value_type);
  uint64_t values_offset_arg = static_cast<uint64_t>(values_offset);
  uint64_t values_stride_arg = static_cast<uint64_t>(values_stride);
  uint64_t other_offset_arg = static_cast<uint64_t>(other_offset);
  uint64_t other_stride_arg = static_cast<uint64_t>(other_stride);
  uint32_t op_arg = static_cast<uint32_t>(op);
  std::vector<void *> args{
      &values_arg,        &other_arg,        &output_arg, &n_arg,
      &type_arg,          &values_offset_arg, &values_stride_arg,
      &other_offset_arg,  &other_stride_arg, &op_arg};
  CUDAContext::get_instance().launch(metric_reduce_func,
                                     "cuda_metric_reduce", args, {}, grid_dim,
                                     kBlockDim, 0, stream);
  return 0;
}

}  // namespace taichi::lang::cuda
