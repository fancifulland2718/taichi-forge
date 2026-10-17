#pragma once

#include "forge_optix_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_OPTIX_FIXED_QUERY_REVISION 1u
#define TI_FORGE_OPTIX_FIXED_QUERY_API_SYMBOL "taichi_forge_optix_fixed_query_api"

typedef void *TiForgeOptixPreparedQuery;

// Fixed opaque typed/occlusion queries. Optional device face-rule records have
// the existing TiForgeOptixFaceRule layout, one per instance. Their table
// contents may change on the device; bindings and the batch size stay fixed.
typedef struct TiForgeOptixFixedQueryDesc {
  uint32_t struct_size;
  uint32_t scene_kind;  // 0 triangle scene, 1 instance scene
  void *scene;
  uint32_t ray_count;
  uint32_t occlusion;  // hits is uint32[N], hit_indices=0, any_hit=1
  uint32_t any_hit;
  uint32_t face_count;
  uint64_t rays;
  uint64_t hits;
  uint64_t hit_indices;
  uint64_t faces;  // zero and face_count=0 disables face filtering
} TiForgeOptixFixedQueryDesc;

typedef struct TiForgeOptixFixedQueryMemory {
  uint32_t struct_size;
  uint32_t parameter_bytes;
  uint64_t device_bytes;
} TiForgeOptixFixedQueryMemory;

typedef struct TiForgeOptixFixedQueryApi {
  uint32_t struct_size;
  uint32_t revision;
  // Cold: prepares the pipeline/SBT, retains the scene, allocates and uploads
  // immutable launch parameters. It never traces rays or captures commands.
  TiForgeOptixResult (*prepare)(TiForgeOptixContext,
                               const TiForgeOptixFixedQueryDesc *,
                               TiForgeOptixPreparedQuery *);
  // Ordered optixLaunch only; this extension does not admit CUDA stream capture.
  // Callers retain arrays, make the owning CUDA context current, and serialize
  // launches/refits sharing a context's pipeline. No per-launch host upload.
  TiForgeOptixResult (*launch)(TiForgeOptixPreparedQuery, uint64_t cuda_stream);
  // Retire all launches before destroying the query.
  // AS UPDATE preserves the scene handle; rebuilding/replacing AS needs prepare.
  TiForgeOptixResult (*destroy)(TiForgeOptixPreparedQuery);
  TiForgeOptixResult (*memory)(TiForgeOptixPreparedQuery,
                              TiForgeOptixFixedQueryMemory *);
} TiForgeOptixFixedQueryApi;

// Optional extension: the legacy provider ABI/table remains unchanged.
TI_FORGE_OPTIX_EXPORT TiForgeOptixResult taichi_forge_optix_fixed_query_api(
    uint32_t revision, size_t api_size, TiForgeOptixFixedQueryApi *out_api);

#ifdef __cplusplus
}
#endif
