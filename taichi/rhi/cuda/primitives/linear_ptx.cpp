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
)ptx";

std::once_flag transform_module_once;
void *transform_module{nullptr};
void *transform_i32_func{nullptr};
void *transform_f32_func{nullptr};

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
)ptx";

std::once_flag indexed_copy_module_once;
void *indexed_copy_module{nullptr};
void *gather_u32_func{nullptr};
void *scatter_u32_func{nullptr};

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
    case CudaTransformValueType::f64:
      TI_ERROR("64-bit CUDA transform requires CUDA toolkit runtime support.");
  }
  TI_ERROR("Unsupported CUDA transform value type.");
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
  float scale_f32 = 0.0f;
  float bias_f32 = 0.0f;
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
  } else {
    scale_f32 = static_cast<float>(scale);
    bias_f32 = static_cast<float>(bias);
    args.push_back(&scale_f32);
    args.push_back(&bias_f32);
  }
  CUDAContext::get_instance().launch(func, "cuda_transform_affine", args, {},
                                     grid_dim, kBlockDim, 0);
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
                                CudaIndexedCopyOp op) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver indexed copy expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA driver indexed copy expects non-negative index_bound.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA driver indexed copy received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
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
                                     grid_dim, kBlockDim, 0);
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

}  // namespace taichi::lang::cuda