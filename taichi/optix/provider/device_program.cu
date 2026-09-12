#if TI_FORGE_OPTIX_TRANSFORM_PACK

struct OptixInstanceWire {
  float transform[12];
  unsigned int instance_id;
  unsigned int sbt_offset;
  unsigned int visibility_mask;
  unsigned int flags;
  unsigned long long traversable;
  unsigned int pad[2];
};

static_assert(sizeof(OptixInstanceWire) == 80,
              "OptixInstance wire layout must remain 80 bytes");

extern "C" __global__ void forge_pack_instance_transforms(
    const float *transforms,
    unsigned int instance_count,
    OptixInstanceWire *instances) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= instance_count) {
    return;
  }
  const float *source = transforms + 12u * index;
  float *destination = instances[index].transform;
#pragma unroll
  for (unsigned int word = 0; word < 12; ++word) {
    destination[word] = source[word];
  }
}

#else

#include <optix.h>
#include <optix_device.h>

// Dense field subranges can be word-aligned without float4 alignment. Keep
// this wire layout at four-byte alignment; no packing allocation is required.
struct PackedFloat4 {
  float x, y, z, w;
  __device__ PackedFloat4() = default;
  __device__ PackedFloat4(float4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {
  }
};
struct PackedUint4 {
  unsigned int x, y, z, w;
  __device__ PackedUint4() = default;
  __device__ PackedUint4(uint4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {
  }
};
struct RayRecord {
  PackedFloat4 origin_tmin;
  PackedFloat4 direction_tmax;
};

struct HitRecord {
  PackedFloat4 value;
};

struct LaunchParams {
  const RayRecord *rays;
  HitRecord *hits;
  OptixTraversableHandle traversable;
  PackedUint4 *hit_indices;
#if TI_FORGE_OPTIX_TYPED == 2
  const struct AlphaMask *masks;
  unsigned int any_hit;
  unsigned int reserved;
#endif
};

#if TI_FORGE_OPTIX_TYPED == 2
struct AlphaMask {
  const float *uvs;
  const unsigned int *indices;
  cudaTextureObject_t texture;
  float cutoff;
  unsigned int channel;
};
static_assert(sizeof(AlphaMask) == 32, "Alpha-mask wire layout changed");
static_assert(sizeof(LaunchParams) == 48, "Alpha launch wire layout changed");
#endif

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __miss__forge_batch_ray() {
}

#if !TI_FORGE_OPTIX_TYPED
extern "C" __global__ void __raygen__forge_batch_ray() {
  const unsigned int index = optixGetLaunchIndex().x;
  const RayRecord ray = params.rays[index];
  unsigned int t_bits = __float_as_uint(-1.0f);
  unsigned int primitive = __float_as_uint(-1.0f);
  unsigned int instance = __float_as_uint(-1.0f);
  unsigned int hit = 0;
  optixTrace(
      params.traversable,
      make_float3(ray.origin_tmin.x, ray.origin_tmin.y, ray.origin_tmin.z),
      make_float3(ray.direction_tmax.x, ray.direction_tmax.y,
                  ray.direction_tmax.z),
      ray.origin_tmin.w, ray.direction_tmax.w, 0.0f, OptixVisibilityMask(0xff),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, t_bits, primitive, instance, hit);
  params.hits[index].value =
      make_float4(__uint_as_float(t_bits), __uint_as_float(primitive),
                  __uint_as_float(instance), __uint_as_float(hit));
}

extern "C" __global__ void __closesthit__forge_batch_ray() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(__float_as_uint(float(optixGetPrimitiveIndex())));
  optixSetPayload_2(__float_as_uint(float(optixGetInstanceId())));
  optixSetPayload_3(__float_as_uint(1.0f));
}

#else
extern "C" __global__ void __raygen__forge_batch_ray_typed() {
  const unsigned int index = optixGetLaunchIndex().x;
  const RayRecord ray = params.rays[index];
  unsigned int t = __float_as_uint(-1.0f);
  unsigned int primitive = ~0u, instance = ~0u, custom = ~0u;
  unsigned int u = 0, v = 0, hit = 0;
  unsigned int flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
#if TI_FORGE_OPTIX_TYPED == 2
  flags = OPTIX_RAY_FLAG_ENFORCE_ANYHIT;
  if (params.any_hit) {
    flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;
  }
#endif
  optixTrace(
      params.traversable,
      make_float3(ray.origin_tmin.x, ray.origin_tmin.y, ray.origin_tmin.z),
      make_float3(ray.direction_tmax.x, ray.direction_tmax.y,
                  ray.direction_tmax.z),
      ray.origin_tmin.w, ray.direction_tmax.w, 0.0f, OptixVisibilityMask(0xff),
      flags, 0, 1, 0, t, primitive, instance, custom, u,
      v, hit);
  params.hits[index].value = make_float4(__uint_as_float(t), __uint_as_float(u),
                                         __uint_as_float(v), 0.0f);
  params.hit_indices[index] = make_uint4(primitive, instance, custom, hit);
}

extern "C" __global__ void __closesthit__forge_batch_ray_typed() {
  const float2 barycentrics = optixGetTriangleBarycentrics();
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2(optixGetInstanceIndex());
  optixSetPayload_3(optixGetInstanceId());
  optixSetPayload_4(__float_as_uint(barycentrics.x));
  optixSetPayload_5(__float_as_uint(barycentrics.y));
  optixSetPayload_6(1u);
}

#if TI_FORGE_OPTIX_TYPED == 2
extern "C" __global__ void __anyhit__forge_alpha_mask() {
  const AlphaMask mask = params.masks[optixGetInstanceIndex()];
  if (!mask.texture) {
    return;
  }
  const unsigned int *triangle = mask.indices + 3u * optixGetPrimitiveIndex();
  const float *a = mask.uvs + 2u * triangle[0];
  const float *b = mask.uvs + 2u * triangle[1];
  const float *c = mask.uvs + 2u * triangle[2];
  const float2 bary = optixGetTriangleBarycentrics();
  const float w = 1.0f - bary.x - bary.y;
  const float u = w * a[0] + bary.x * b[0] + bary.y * c[0];
  const float v = w * a[1] + bary.x * b[1] + bary.y * c[1];
  // Match Forge Texture.sample_lod: logical axis 0 is the outer array axis,
  // while CUDA's texture x coordinate addresses the innermost array axis.
  const float4 sample = tex2D<float4>(mask.texture, v, u);
  const float alpha = mask.channel == 0 ? sample.x :
                      mask.channel == 1 ? sample.y :
                      mask.channel == 2 ? sample.z : sample.w;
  if (!(alpha >= mask.cutoff)) {
    optixIgnoreIntersection();
  }
}
#endif
#endif

#endif  // TI_FORGE_OPTIX_TRANSFORM_PACK
