#include "taichi/runtime/texture_storage.h"

#include <limits>
#include "taichi/program/texture.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

namespace taichi::lang {
#ifdef TI_WITH_CUDA
namespace {

CUaddress_mode cuda_address_mode(ImageAddressMode mode) {
  switch (mode) {
    case ImageAddressMode::repeat:
      return CU_TR_ADDRESS_MODE_WRAP;
    case ImageAddressMode::mirrored_repeat:
      return CU_TR_ADDRESS_MODE_MIRROR;
    case ImageAddressMode::clamp_to_edge:
      return CU_TR_ADDRESS_MODE_CLAMP;
  }
  TI_ERROR("Unsupported CUDA texture address mode");
}

CUfilter_mode cuda_filter_mode(ImageFilter filter) {
  switch (filter) {
    case ImageFilter::nearest:
      return CU_TR_FILTER_MODE_POINT;
    case ImageFilter::linear:
      return CU_TR_FILTER_MODE_LINEAR;
  }
  TI_ERROR("Unsupported CUDA texture filter mode");
}

bool cuda_sampled_format_supported(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8:
    case BufferFormat::rg8:
    case BufferFormat::rgba8:
    case BufferFormat::r16:
    case BufferFormat::rg16:
    case BufferFormat::rgba16:
    case BufferFormat::r16f:
    case BufferFormat::rg16f:
    case BufferFormat::rgba16f:
    case BufferFormat::r32f:
    case BufferFormat::rg32f:
    case BufferFormat::rgba32f:
      return true;
    default:
      return false;
  }
}

CUarray_format cuda_array_format(DataType type) {
  if (type == PrimitiveType::u8)
    return CU_AD_FORMAT_UNSIGNED_INT8;
  if (type == PrimitiveType::u16)
    return CU_AD_FORMAT_UNSIGNED_INT16;
  if (type == PrimitiveType::f16)
    return CU_AD_FORMAT_HALF;
  if (type == PrimitiveType::f32)
    return CU_AD_FORMAT_FLOAT;
  TI_ERROR("Unsupported CUDA sampled texture channel type {}",
           type.to_string());
}

}  // namespace
#endif

CudaTextureStorage::CudaTextureStorage(const ImageParams &params) {
#ifdef TI_WITH_CUDA
  const auto &sampler = params.sampler_config;
  TI_ERROR_IF(params.dimension != ImageDimension::d1D &&
                  params.dimension != ImageDimension::d2D &&
                  params.dimension != ImageDimension::d3D,
              "CUDA texture dimensionality is invalid");
  TI_ERROR_IF(!params.x || !params.y || !params.z ||
                  (params.dimension == ImageDimension::d1D && params.y != 1) ||
                  (params.dimension != ImageDimension::d3D && params.z != 1),
              "CUDA texture extents must be positive with unit inactive axes");
  TI_ERROR_IF(params.mip_levels != 1 || sampler.has_extended_sampling(),
              "CUDA sampled textures require one mip and basic sampler state");
  TI_ERROR_IF(params.export_sharing,
              "CUDA sampled texture storage does not export shared images");
  TI_ERROR_IF(!cuda_sampled_format_supported(params.format),
              "CUDA sampled textures do not support BufferFormat {} under "
              "the vec4-f32 sampling contract",
              static_cast<int>(params.format));
  TI_ERROR_IF(sampler.min_filter != sampler.mag_filter,
              "CUDA sampled textures require matching min and mag filters");
  const auto [type, channels] = buffer_format2type_channels(params.format);
  texel_bytes_ = std::size_t(data_type_size(type)) * channels;
  logical_bytes_ = texel_bytes_;
  for (auto extent : {params.x, params.y, params.z}) {
    TI_ERROR_IF(
        extent > std::numeric_limits<std::size_t>::max() / logical_bytes_,
        "CUDA texture logical storage size overflows");
    logical_bytes_ *= extent;
  }
  physical_extent_ =
      params.dimension == ImageDimension::d1D
          ? std::array<std::size_t, 3>{params.x, 1, 1}
      : params.dimension == ImageDimension::d2D
          ? std::array<std::size_t, 3>{params.y, params.x, 1}
          : std::array<std::size_t, 3>{params.z, params.y, params.x};
  auto &driver = CUDADriver::get_instance();
  TI_ERROR_IF(driver.get_provider() != CUDADriverProvider::nvidia_cuda,
              "CUDA texture resources require the NVIDIA CUDA Driver provider");
  TI_ERROR_IF(!driver.array_3d_create.available() ||
                  !driver.array_destroy.available() ||
                  !driver.tex_object_create.available() ||
                  !driver.tex_object_destroy.available() ||
                  !driver.memcpy_3d.available(),
              "CUDA texture resources require array, texture-object, and "
              "3D-copy Driver API symbols");
  CUDA_ARRAY3D_DESCRIPTOR array_desc{};
  array_desc.Width = physical_extent_[0];
  array_desc.Height =
      params.dimension == ImageDimension::d1D ? 0 : physical_extent_[1];
  array_desc.Depth =
      params.dimension == ImageDimension::d3D ? physical_extent_[2] : 0;
  array_desc.Format = cuda_array_format(type);
  array_desc.NumChannels = channels;
  CUDA_TEXTURE_DESC texture_desc{};
  texture_desc.addressMode[0] = cuda_address_mode(sampler.address_mode_u);
  texture_desc.addressMode[1] = cuda_address_mode(sampler.address_mode_v);
  texture_desc.addressMode[2] = cuda_address_mode(sampler.address_mode_w);
  texture_desc.filterMode = cuda_filter_mode(sampler.min_filter);
  texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
  texture_desc.maxAnisotropy = 1;
  texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
  auto guard = CUDAContext::get_instance().get_guard();
  CUarray array = nullptr;
  driver.array_3d_create(&array, &array_desc);
  try {
    CUDA_RESOURCE_DESC resource_desc{};
    resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    resource_desc.res.array.hArray = array;
    driver.tex_object_create(&sampled_object_, &resource_desc, &texture_desc,
                             nullptr);
  } catch (...) {
    driver.array_destroy.call_with_warning(array);
    throw;
  }
  array_ = array;
#else
  TI_ERROR("CUDA texture storage requires a CUDA-enabled runtime");
#endif
}

void CudaTextureStorage::upload_device(std::uint64_t source) {
  upload(source, nullptr, false);
}

void CudaTextureStorage::upload_device_async(std::uint64_t source,
                                             void *stream) {
  upload(source, stream, true);
}

void CudaTextureStorage::upload(std::uint64_t source,
                                void *stream,
                                bool asynchronous) {
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!source, "CUDA texture upload source is null");
  CUDA_MEMCPY3D copy{};
  copy.srcMemoryType = ::CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(source);
  copy.srcPitch = physical_extent_[0] * texel_bytes_;
  copy.srcHeight = physical_extent_[1];
  copy.dstMemoryType = ::CU_MEMORYTYPE_ARRAY;
  copy.dstArray = static_cast<CUarray>(array_);
  copy.WidthInBytes = copy.srcPitch;
  copy.Height = physical_extent_[1];
  copy.Depth = physical_extent_[2];
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  if (asynchronous)
    driver.memcpy_3d_async(&copy, stream);
  else
    driver.memcpy_3d(&copy);
#else
  TI_ERROR("CUDA texture upload requires a CUDA-enabled runtime");
#endif
}

CudaTextureStorage::~CudaTextureStorage() {
#ifdef TI_WITH_CUDA
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  if (sampled_object_ && driver.tex_object_destroy.available())
    driver.tex_object_destroy.call_with_warning(sampled_object_);
  if (array_ && driver.array_destroy.available())
    driver.array_destroy.call_with_warning(static_cast<CUarray>(array_));
#endif
}

}  // namespace taichi::lang
