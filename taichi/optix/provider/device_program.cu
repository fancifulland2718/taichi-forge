#include <optix.h>
#include <optix_device.h>

struct RayRecord {
  float4 origin_tmin;
  float4 direction_tmax;
};

struct HitRecord {
  float4 value;
};

struct LaunchParams {
  const RayRecord *rays;
  HitRecord *hits;
  OptixTraversableHandle traversable;
};

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __raygen__forge_batch_ray() {
  const unsigned int index = optixGetLaunchIndex().x;
  const RayRecord ray = params.rays[index];
  unsigned int t_bits = __float_as_uint(-1.0f);
  unsigned int primitive = __float_as_uint(-1.0f);
  unsigned int instance = __float_as_uint(-1.0f);
  unsigned int hit = 0;
  optixTrace(params.traversable,
             make_float3(ray.origin_tmin.x, ray.origin_tmin.y,
                         ray.origin_tmin.z),
             make_float3(ray.direction_tmax.x, ray.direction_tmax.y,
                         ray.direction_tmax.z),
             ray.origin_tmin.w,
             ray.direction_tmax.w, 0.0f, OptixVisibilityMask(0xff),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, t_bits, primitive,
             instance, hit);
  params.hits[index].value =
      make_float4(__uint_as_float(t_bits), __uint_as_float(primitive),
                  __uint_as_float(instance), __uint_as_float(hit));
}

extern "C" __global__ void __miss__forge_batch_ray() {
}

extern "C" __global__ void __closesthit__forge_batch_ray() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(__float_as_uint(float(optixGetPrimitiveIndex())));
  optixSetPayload_2(__float_as_uint(float(optixGetInstanceId())));
  optixSetPayload_3(__float_as_uint(1.0f));
}
