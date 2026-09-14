#include <optix.h>
#include <cuda_runtime.h>

struct Parameters {
  OptixTraversableHandle scene;
  unsigned int *output;
  unsigned int bias;
  unsigned int count;
};
struct HitData {
  unsigned int value;
  unsigned int accept;
};

extern "C" {
__constant__ Parameters params;
}

static __forceinline__ __device__ float3 point(float x, float y, float z) {
  float3 p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

extern "C" __global__ void __raygen__render() {
  const auto index = optixGetLaunchIndex().x;
  if (index >= params.count) return;
  const auto ray_type = (index / 2) % 2;
  unsigned int result = 0;
  optixTrace(params.scene, point(index % 2 ? 3.f : 0.f, 0.f, -1.f), point(0.f, 0.f, 1.f),
             0.f, 100.f, 0.f, 255, OPTIX_RAY_FLAG_NONE,
             ray_type, 2, ray_type, result);
  params.output[index] = result + params.bias;
}

extern "C" __global__ void __raygen__query() {
  const auto index = optixGetLaunchIndex().x;
  if (index >= params.count) return;
  unsigned int result = 0;
  optixTrace(params.scene, point(0.f, 0.f, -1.f), point(0.f, 0.f, 1.f),
             0.f, 100.f, 0.f, 255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             0, 1, 0, result);
  params.output[index] = result != 0 ? params.bias : 0;
}

extern "C" __global__ void __miss__value() {
  const auto *data = reinterpret_cast<const unsigned int *>(optixGetSbtDataPointer());
  optixSetPayload_0(*data);
}

extern "C" __global__ void __closesthit__value() {
  const auto *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
  optixSetPayload_0(data->value + optixGetInstanceId());
}

extern "C" __global__ void __anyhit__mask() {
  const auto *data = reinterpret_cast<const HitData *>(optixGetSbtDataPointer());
  if (!data->accept) optixIgnoreIntersection();
}

// The only application resource pointers are indirect, in raygen SBT data.
// This proves the public preparation retains owners beyond launch parameters.
struct IndirectData {
  unsigned int *output;
  const unsigned int *input;
  cudaTextureObject_t texture;
};
extern "C" __global__ void __raygen__indirect() {
  const auto *data = reinterpret_cast<const IndirectData *>(optixGetSbtDataPointer());
  const auto index = optixGetLaunchIndex().x;
  data->output[index] = data->input[index] + static_cast<unsigned int>(tex2D<float>(data->texture, 0.5f, 0.5f));
}
