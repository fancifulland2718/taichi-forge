#pragma once

#include "taichi/optix/forge_optix_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional provider ABI-1 extension. No CUDA/OptiX SDK types cross this ABI.
typedef void *TiForgeOptixProgram;
typedef void *TiForgeOptixPreparedLaunch;

typedef struct TiForgeOptixProgramModule {
  const char *ptx;
  size_t ptx_size;
} TiForgeOptixProgramModule;

typedef struct TiForgeOptixProgramEntry {
  uint32_t module_index;
  const char *name;  // null denotes an absent optional hit entry
} TiForgeOptixProgramEntry;

typedef struct TiForgeOptixProgramGroup {
  uint32_t kind;                   // 0 raygen, 1 miss, 2 triangle hit group
  TiForgeOptixProgramEntry entry;  // closest-hit for a hit group
  TiForgeOptixProgramEntry any_hit;
} TiForgeOptixProgramGroup;

typedef struct TiForgeOptixProgramDesc {
  uint32_t struct_size;
  uint32_t module_count;
  const TiForgeOptixProgramModule *modules;
  uint32_t group_count;
  const TiForgeOptixProgramGroup *groups;
  const char *parameter_name;
  uint32_t parameter_size;
  uint32_t payload_count;
  uint32_t attribute_count;
  uint32_t max_trace_depth;
  uint32_t allow_opacity_micromaps;
} TiForgeOptixProgramDesc;

typedef struct TiForgeOptixSbtRecord {
  uint32_t group_index;
  uint32_t data_size;
  const void *data;  // host payload; copied during preparation
} TiForgeOptixSbtRecord;

typedef struct TiForgeOptixProgramScene {
  void *scene;    // existing Forge scene owner, not an OptixTraversableHandle
  uint32_t kind;  // 0 triangle scene, 1 instance scene
} TiForgeOptixProgramScene;

typedef struct TiForgeOptixProgramSceneInfo {
  uint32_t struct_size;
  uint32_t max_sbt_offset;
  uint64_t traversable;
  uint32_t has_opacity_micromaps;
} TiForgeOptixProgramSceneInfo;

typedef struct TiForgeOptixPrepareLaunchDesc {
  uint32_t struct_size;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  const void *parameters;
  uint32_t parameter_size;
  TiForgeOptixSbtRecord raygen;
  const TiForgeOptixSbtRecord *miss;
  uint32_t miss_count;
  const TiForgeOptixSbtRecord *hit;
  uint32_t hit_count;
  const TiForgeOptixProgramScene *scenes;
  uint32_t scene_count;
} TiForgeOptixPrepareLaunchDesc;

typedef struct TiForgeOptixProgramMemory {
  uint32_t struct_size;
  uint32_t continuation_stack_bytes;
  uint64_t device_data_bytes;
  uint64_t pinned_host_bytes;
  uint32_t parameter_bytes;
  uint32_t miss_stride;
  uint32_t hit_stride;
} TiForgeOptixProgramMemory;

typedef struct TiForgeOptixProgramApi {
  uint32_t struct_size;
  uint32_t revision;  // 1
  TiForgeOptixResult (*create)(TiForgeOptixContext,
                               const TiForgeOptixProgramDesc *,
                               TiForgeOptixProgram *);
  TiForgeOptixResult (*destroy)(TiForgeOptixProgram);
  TiForgeOptixResult (*prepare)(TiForgeOptixProgram,
                                const TiForgeOptixPrepareLaunchDesc *,
                                TiForgeOptixPreparedLaunch *);
  // Pure preparation copies host data/allocates storage; initialization uploads
  // once on the declared stream. Replay only launches, never uploads
  // parameters.
  TiForgeOptixResult (*initialize)(TiForgeOptixPreparedLaunch,
                                   uint64_t cuda_stream);
  TiForgeOptixResult (*launch)(TiForgeOptixPreparedLaunch,
                               uint64_t cuda_stream);
  TiForgeOptixResult (*destroy_launch)(TiForgeOptixPreparedLaunch);
  TiForgeOptixResult (*memory)(TiForgeOptixPreparedLaunch,
                               TiForgeOptixProgramMemory *);
  TiForgeOptixResult (*scene_info)(TiForgeOptixContext,
                                   const TiForgeOptixProgramScene *,
                                   TiForgeOptixProgramSceneInfo *);
} TiForgeOptixProgramApi;

#ifdef __cplusplus
}
#endif
