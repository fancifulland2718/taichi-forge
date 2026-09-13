#include "taichi/optix/forge_optix_provider.h"

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <map>
#include <new>
#include <string>
#include <vector>

#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "device_program_0_ptx.h"
#include "device_program_1_ptx.h"
#include "device_program_2_ptx.h"
#include "instance_transform_pack_ptx.h"

#if OPTIX_ABI_VERSION != 93 && OPTIX_ABI_VERSION != 105 && \
    OPTIX_ABI_VERSION != 118
#error "Forge OptiX provider supports only SDK ABI 93, 105, and 118"
#endif

namespace {

thread_local std::string last_error;
thread_local std::string last_optix_log;
std::mutex optix_loader_mutex;
void *optix_library_handle{nullptr};
std::size_t optix_context_count{0};
std::string active_optix_runtime_library_path;

#define TI_FORGE_STRINGIFY_IMPL(value) #value
#define TI_FORGE_STRINGIFY(value) TI_FORGE_STRINGIFY_IMPL(value)

constexpr char kProviderName[] = "taichi-forge-optix";
constexpr char kBuildIdentity[] =
    "forge-optix-provider-abi1-optix-abi" TI_FORGE_STRINGIFY(
        OPTIX_ABI_VERSION) "-scene-refit2-typed1-instances1-alpha2-omm1";
constexpr uint64_t kFeatures = TI_FORGE_OPTIX_FEATURE_TRIANGLE_GAS |
                               TI_FORGE_OPTIX_FEATURE_SINGLE_INSTANCE_IAS |
                               TI_FORGE_OPTIX_FEATURE_GAS_UPDATE |
                               TI_FORGE_OPTIX_FEATURE_BATCH_CLOSEST_HIT |
                               TI_FORGE_OPTIX_FEATURE_RUNTIME_ORDERED_STREAM |
                               TI_FORGE_OPTIX_FEATURE_EXACT_DEVICE_MEMORY |
                               TI_FORGE_OPTIX_FEATURE_TYPED_HITS |
                               TI_FORGE_OPTIX_FEATURE_WORD_ALIGNED_QUERY_STORAGE |
                               TI_FORGE_OPTIX_FEATURE_SHARED_TRIANGLE_GAS |
                               TI_FORGE_OPTIX_FEATURE_MULTI_INSTANCE_IAS |
                               TI_FORGE_OPTIX_FEATURE_DEVICE_INSTANCE_TRANSFORM_UPDATE |
                               TI_FORGE_OPTIX_FEATURE_ALPHA_MASK |
                               TI_FORGE_OPTIX_FEATURE_INSTANCE_OPACITY |
                               TI_FORGE_OPTIX_FEATURE_OPACITY_MICROMAP_IMPORT;

void clear_error_state() {
  last_error.clear();
  last_optix_log.clear();
}

TiForgeOptixResult fail(TiForgeOptixResult result, std::string message) {
  last_error = std::move(message);
  return result;
}

TiForgeOptixResult cuda_check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return TI_FORGE_OPTIX_SUCCESS;
  }
  const char *name = nullptr;
  const char *description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  return fail(TI_FORGE_OPTIX_ERROR_CUDA_CALL,
              std::string(operation) + " failed: " +
                  (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) + " (" +
                  (description == nullptr ? "no description" : description) +
                  ")");
}

TiForgeOptixResult optix_check(OptixResult result, const char *operation) {
  if (result == OPTIX_SUCCESS) {
    last_optix_log.clear();
    return TI_FORGE_OPTIX_SUCCESS;
  }
  std::string message =
      std::string(operation) + " failed: " +
      (optixGetErrorName(result) == nullptr
           ? std::to_string(static_cast<int>(result))
           : optixGetErrorName(result));
  if (!last_optix_log.empty()) {
    message += "; validation log: " + last_optix_log;
  }
  last_optix_log.clear();
  return fail(TI_FORGE_OPTIX_ERROR_OPTIX_CALL, std::move(message));
}

struct DeviceBuffer {
  CUdeviceptr pointer{0};
  std::size_t bytes{0};

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    reset();
  }

  TiForgeOptixResult allocate(std::size_t requested) {
    reset();
    bytes = requested;
    if (bytes == 0) {
      return TI_FORGE_OPTIX_SUCCESS;
    }
    const auto result = cuda_check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      pointer = 0;
      bytes = 0;
    }
    return result;
  }

  void reset() {
    if (pointer != 0) {
      cuMemFree(pointer);
    }
    pointer = 0;
    bytes = 0;
  }
};

struct EmptySbtData {
  uint32_t reserved{0};
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct RayPipeline {
  OptixModule module{nullptr};
  OptixProgramGroup raygen{nullptr};
  OptixProgramGroup miss{nullptr};
  OptixProgramGroup hitgroup{nullptr};
  OptixPipeline pipeline{nullptr};
  DeviceBuffer raygen_record;
  DeviceBuffer miss_record;
  DeviceBuffer hitgroup_record;
  OptixShaderBindingTable sbt{};
};

struct Context {
  CUcontext cuda_context{nullptr};
  OptixDeviceContext optix_context{nullptr};
  CUmodule transform_module{nullptr};
  CUfunction transform_pack{nullptr};
  RayPipeline legacy;
  RayPipeline typed;
  std::atomic<bool> typed_ready{false};
  RayPipeline alpha;
  std::atomic<bool> alpha_ready{false};
  RayPipeline micromap;
  std::atomic<bool> micromap_ready{false};
  std::mutex prepare_mutex;
  std::atomic<std::size_t> scene_count{0};
  std::atomic<std::size_t> gas_count{0};
  std::atomic<std::size_t> instance_scene_count{0};
};

struct LaunchParams {
  CUdeviceptr rays;
  CUdeviceptr hits;
  OptixTraversableHandle traversable;
  CUdeviceptr hit_indices;
};

struct AlphaLaunchParams {
  LaunchParams query;
  CUdeviceptr masks;
  uint32_t any_hit;
  uint32_t reserved;
};
static_assert(sizeof(AlphaLaunchParams) == 48,
              "Caller-owned alpha launch workspace must be 48 bytes");
static_assert(sizeof(TiForgeOptixAlphaMask) == 32,
              "Alpha-mask wire layout must be 32 bytes");

struct Scene {
  Context *context{nullptr};
  uint32_t vertex_count{0};
  uint32_t triangle_count{0};
  bool allow_update{false};
  DeviceBuffer gas;
  DeviceBuffer ias;
  DeviceBuffer scratch;
  DeviceBuffer ias_scratch;
  DeviceBuffer instance;
  DeviceBuffer launch_params;
  OptixTraversableHandle gas_handle{0};
  OptixTraversableHandle ias_handle{0};
};

struct TriangleGas {
  Context *context{nullptr};
  uint32_t vertex_count{0};
  uint32_t triangle_count{0};
  bool allow_update{false};
  std::atomic<bool> owner_live{true};
  std::atomic<std::size_t> instance_ref_count{0};
  DeviceBuffer gas;
  DeviceBuffer scratch;
  DeviceBuffer micromap;
  DeviceBuffer micromap_indices;
  std::vector<OptixOpacityMicromapUsageCount> micromap_usage;
  OptixTraversableHandle gas_handle{0};
};

struct InstanceScene {
  Context *context{nullptr};
  uint32_t instance_count{0};
  bool allow_update{false};
  std::vector<TriangleGas *> gas_refs;
  DeviceBuffer ias;
  DeviceBuffer scratch;
  DeviceBuffer instance;
  DeviceBuffer launch_params;
  OptixTraversableHandle ias_handle{0};
};

void optix_log(unsigned int level,
               const char *tag,
               const char *message,
               void *) {
  if (level <= 3 && message != nullptr) {
    last_optix_log = std::string("OptiX[") + (tag == nullptr ? "" : tag) +
                     "]: " + message;
  }
}

OptixResult init_optix_library(const std::string &library_path, void **handle) {
  if (library_path.empty()) {
    return optixInitWithHandle(handle);
  }
  *handle = nullptr;
#if defined(_WIN32)
  const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                             library_path.c_str(), -1, nullptr, 0);
  if (wide_size <= 0) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  std::wstring wide_path(static_cast<std::size_t>(wide_size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                          library_path.c_str(), -1, wide_path.data(),
                          wide_size) <= 0) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  *handle = LoadLibraryW(wide_path.c_str());
  if (*handle == nullptr) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  void *symbol = reinterpret_cast<void *>(
      GetProcAddress(static_cast<HMODULE>(*handle), "optixQueryFunctionTable"));
#else
  *handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle == nullptr) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  void *symbol = dlsym(*handle, "optixQueryFunctionTable");
#endif
  if (symbol == nullptr) {
    optixUninitWithHandle(*handle);
    *handle = nullptr;
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
  auto *query = reinterpret_cast<OptixQueryFunctionTable_t *>(symbol);
  const auto result = query(OPTIX_ABI_VERSION, 0, nullptr, nullptr,
                            &OPTIX_FUNCTION_TABLE_SYMBOL,
                            sizeof(OPTIX_FUNCTION_TABLE_SYMBOL));
  if (result != OPTIX_SUCCESS) {
    optixUninitWithHandle(*handle);
    *handle = nullptr;
  }
  return result;
}

TiForgeOptixResult retain_optix_loader(const char *library_path) {
  std::lock_guard<std::mutex> lock(optix_loader_mutex);
  const std::string requested = library_path == nullptr ? "" : library_path;
  if (optix_context_count == 0) {
    const auto result = init_optix_library(requested, &optix_library_handle);
    if (result != OPTIX_SUCCESS) {
      optix_library_handle = nullptr;
      return fail(TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE,
                  std::string("OptiX runtime initialization failed: ") +
                      (optixGetErrorName(result) == nullptr
                           ? std::to_string(static_cast<int>(result))
                           : optixGetErrorName(result)));
    }
    active_optix_runtime_library_path = requested;
  } else if (requested != active_optix_runtime_library_path) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "all live contexts in one adapter must use the same OptiX runtime library");
  }
  ++optix_context_count;
  return TI_FORGE_OPTIX_SUCCESS;
}

void release_optix_loader() {
  std::lock_guard<std::mutex> lock(optix_loader_mutex);
  if (optix_context_count == 0) {
    return;
  }
  --optix_context_count;
  if (optix_context_count == 0 && optix_library_handle != nullptr) {
    optixUninitWithHandle(optix_library_handle);
    optix_library_handle = nullptr;
    active_optix_runtime_library_path.clear();
  }
}

TiForgeOptixResult probe_runtime(const char *library_path) {
  clear_error_state();
  std::lock_guard<std::mutex> lock(optix_loader_mutex);
  const std::string requested = library_path == nullptr ? "" : library_path;
  if (optix_context_count != 0) {
    if (requested == active_optix_runtime_library_path) {
      return TI_FORGE_OPTIX_SUCCESS;
    }
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "cannot probe another OptiX runtime while contexts are live");
  }
  void *handle = nullptr;
  const auto result = init_optix_library(requested, &handle);
  if (result != OPTIX_SUCCESS) {
    return fail(TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE,
                std::string("OptiX runtime probe failed: ") +
                    (optixGetErrorName(result) == nullptr
                         ? std::to_string(static_cast<int>(result))
                         : optixGetErrorName(result)));
  }
  optixUninitWithHandle(handle);
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult copy_sbt_record(DeviceBuffer &destination,
                                   OptixProgramGroup program_group) {
  SbtRecord<EmptySbtData> record{};
  auto result = optix_check(optixSbtRecordPackHeader(program_group, &record),
                            "optixSbtRecordPackHeader");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = destination.allocate(sizeof(record));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return cuda_check(cuMemcpyHtoD(destination.pointer, &record, sizeof(record)),
                    "cuMemcpyHtoD(SBT)");
}

TiForgeOptixResult create_pipeline(Context *owner,
                                   RayPipeline *context,
                                   int variant) {
  const bool typed = variant != 0;
  const bool alpha = variant >= 2;
  OptixModuleCompileOptions module_options{};
  module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  OptixPipelineCompileOptions pipeline_options{};
  pipeline_options.usesMotionBlur = false;
  pipeline_options.allowOpacityMicromaps = variant == 3;
  pipeline_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipeline_options.numPayloadValues = typed ? 7 : 4;
  pipeline_options.numAttributeValues = 2;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
#if OPTIX_ABI_VERSION == 118
  pipeline_options.pipelineLaunchParamsSizeInBytes =
      alpha ? sizeof(AlphaLaunchParams) : sizeof(LaunchParams);
#endif

  char log[8192]{};
  std::size_t log_size = sizeof(log);
  const auto *ptx = alpha ? ti_forge_optix_device_2_ptx :
      typed ? ti_forge_optix_device_1_ptx : ti_forge_optix_device_0_ptx;
  auto result =
      optix_check(optixModuleCreate(owner->optix_context, &module_options,
                                    &pipeline_options, ptx, std::strlen(ptx),
                                    log, &log_size, &context->module),
                  "optixModuleCreate");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    if (log_size > 1) {
      last_error += std::string("; module log: ") + log;
    }
    return result;
  }

  OptixProgramGroupOptions group_options{};
  OptixProgramGroupDesc raygen_desc{};
  raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_desc.raygen.module = context->module;
  raygen_desc.raygen.entryFunctionName =
      typed ? "__raygen__forge_batch_ray_typed" : "__raygen__forge_batch_ray";
  log_size = sizeof(log);
  result = optix_check(
      optixProgramGroupCreate(owner->optix_context, &raygen_desc, 1,
                              &group_options, log, &log_size, &context->raygen),
      "optixProgramGroupCreate(raygen)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  OptixProgramGroupDesc miss_desc{};
  miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_desc.miss.module = context->module;
  miss_desc.miss.entryFunctionName = "__miss__forge_batch_ray";
  log_size = sizeof(log);
  result = optix_check(
      optixProgramGroupCreate(owner->optix_context, &miss_desc, 1,
                              &group_options, log, &log_size, &context->miss),
      "optixProgramGroupCreate(miss)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  OptixProgramGroupDesc hitgroup_desc{};
  hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_desc.hitgroup.moduleCH = context->module;
  hitgroup_desc.hitgroup.entryFunctionNameCH =
      typed ? "__closesthit__forge_batch_ray_typed"
            : "__closesthit__forge_batch_ray";
  if (alpha) {
    hitgroup_desc.hitgroup.moduleAH = context->module;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__forge_alpha_mask";
  }
  log_size = sizeof(log);
  result = optix_check(optixProgramGroupCreate(
                           owner->optix_context, &hitgroup_desc, 1,
                           &group_options, log, &log_size, &context->hitgroup),
                       "optixProgramGroupCreate(hitgroup)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  const OptixProgramGroup groups[] = {context->raygen, context->miss,
                                      context->hitgroup};
  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;
  log_size = sizeof(log);
  result =
      optix_check(optixPipelineCreate(owner->optix_context, &pipeline_options,
                                      &link_options, groups, 3, log, &log_size,
                                      &context->pipeline),
                  "optixPipelineCreate");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result =
      optix_check(optixPipelineSetStackSize(context->pipeline, 0, 0, 8192, 2),
                  "optixPipelineSetStackSize");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  result = copy_sbt_record(context->raygen_record, context->raygen);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = copy_sbt_record(context->miss_record, context->miss);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = copy_sbt_record(context->hitgroup_record, context->hitgroup);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  context->sbt.raygenRecord = context->raygen_record.pointer;
  context->sbt.missRecordBase = context->miss_record.pointer;
  context->sbt.missRecordStrideInBytes = sizeof(SbtRecord<EmptySbtData>);
  context->sbt.missRecordCount = 1;
  context->sbt.hitgroupRecordBase = context->hitgroup_record.pointer;
  context->sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<EmptySbtData>);
  context->sbt.hitgroupRecordCount = 1;
  return TI_FORGE_OPTIX_SUCCESS;
}

void destroy_pipeline(RayPipeline *context) {
  if (context->pipeline != nullptr) {
    optixPipelineDestroy(context->pipeline);
  }
  if (context->hitgroup != nullptr) {
    optixProgramGroupDestroy(context->hitgroup);
  }
  if (context->miss != nullptr) {
    optixProgramGroupDestroy(context->miss);
  }
  if (context->raygen != nullptr) {
    optixProgramGroupDestroy(context->raygen);
  }
  if (context->module != nullptr) {
    optixModuleDestroy(context->module);
  }
  context->pipeline = nullptr;
  context->hitgroup = nullptr;
  context->miss = nullptr;
  context->raygen = nullptr;
  context->module = nullptr;
  context->raygen_record.reset();
  context->miss_record.reset();
  context->hitgroup_record.reset();
}

TiForgeOptixResult create_transform_module(Context *context) {
  auto result = cuda_check(
      cuModuleLoadData(&context->transform_module,
                       ti_forge_optix_instance_transform_pack_ptx),
      "cuModuleLoadData(instance transform pack)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    context->transform_module = nullptr;
    return result;
  }
  result = cuda_check(
      cuModuleGetFunction(&context->transform_pack, context->transform_module,
                          "forge_pack_instance_transforms"),
      "cuModuleGetFunction(forge_pack_instance_transforms)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    cuModuleUnload(context->transform_module);
    context->transform_module = nullptr;
    context->transform_pack = nullptr;
  }
  return result;
}

void destroy_transform_module(Context *context) {
  context->transform_pack = nullptr;
  if (context->transform_module != nullptr) {
    cuModuleUnload(context->transform_module);
    context->transform_module = nullptr;
  }
}

TiForgeOptixResult prepare_typed(TiForgeOptixContext raw_context) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT, "OptiX context is null");
  }
  std::lock_guard<std::mutex> lock(context->prepare_mutex);
  if (context->typed_ready.load(std::memory_order_acquire)) {
    return TI_FORGE_OPTIX_SUCCESS;
  }
  const auto result = create_pipeline(context, &context->typed, true);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    destroy_pipeline(&context->typed);
  } else {
    context->typed_ready.store(true, std::memory_order_release);
  }
  return result;
}

TiForgeOptixResult prepare_alpha(TiForgeOptixContext raw_context) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT, "OptiX context is null");
  }
  std::lock_guard<std::mutex> lock(context->prepare_mutex);
  if (context->alpha_ready.load(std::memory_order_acquire)) {
    return TI_FORGE_OPTIX_SUCCESS;
  }
  const auto result = create_pipeline(context, &context->alpha, 2);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    destroy_pipeline(&context->alpha);
  } else {
    context->alpha_ready.store(true, std::memory_order_release);
  }
  return result;
}

OptixBuildInput triangle_build_input(
    const TiForgeOptixTriangleSceneDesc &desc,
    CUdeviceptr *vertex_buffer,
    unsigned int *geometry_flags) {
  *vertex_buffer = static_cast<CUdeviceptr>(desc.vertices);
  *geometry_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
  input.triangleArray.numVertices = desc.vertex_count;
  input.triangleArray.vertexBuffers = vertex_buffer;
  input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
  input.triangleArray.numIndexTriplets = desc.triangle_count;
  input.triangleArray.indexBuffer = static_cast<CUdeviceptr>(desc.indices);
  input.triangleArray.flags = geometry_flags;
  input.triangleArray.numSbtRecords = 1;
  return input;
}

OptixAccelBuildOptions build_options(bool allow_update,
                                     OptixBuildOperation operation) {
  OptixAccelBuildOptions options{};
  options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                       (allow_update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0u);
  options.operation = operation;
  return options;
}

TiForgeOptixResult create_gas(Scene *scene,
                              const TiForgeOptixTriangleSceneDesc &desc) {
  CUdeviceptr vertex_buffer = 0;
  unsigned int geometry_flags = 0;
  auto input = triangle_build_input(desc, &vertex_buffer, &geometry_flags);
  auto options = build_options(scene->allow_update, OPTIX_BUILD_OPERATION_BUILD);
  OptixAccelBufferSizes sizes{};
  auto result = optix_check(
      optixAccelComputeMemoryUsage(scene->context->optix_context, &options,
                                   &input, 1, &sizes),
      "optixAccelComputeMemoryUsage(GAS)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->gas.allocate(sizes.outputSizeInBytes);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->scratch.allocate(std::max(sizes.tempSizeInBytes,
                                            sizes.tempUpdateSizeInBytes));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixAccelBuild(scene->context->optix_context,
                      reinterpret_cast<CUstream>(desc.cuda_stream), &options,
                      &input, 1, scene->scratch.pointer, scene->scratch.bytes,
                      scene->gas.pointer, scene->gas.bytes, &scene->gas_handle,
                      nullptr, 0),
      "optixAccelBuild(GAS)");
}

TiForgeOptixResult create_ias(Scene *scene, CUstream stream) {
  OptixInstance host_instance{};
  const float identity[12] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::memcpy(host_instance.transform, identity, sizeof(identity));
  host_instance.instanceId = 0;
  host_instance.sbtOffset = 0;
  host_instance.visibilityMask = 0xff;
  // Preserve query-owned alpha eligibility without changing the GAS layout.
  // Opaque queries use RAY_FLAG_DISABLE_ANYHIT, which overrides this flag.
  host_instance.flags = OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT;
  host_instance.traversableHandle = scene->gas_handle;

  auto result = scene->instance.allocate(sizeof(host_instance));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = cuda_check(cuMemcpyHtoDAsync(scene->instance.pointer, &host_instance,
                                       sizeof(host_instance), stream),
                      "cuMemcpyHtoDAsync(OptixInstance)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  input.instanceArray.instances = scene->instance.pointer;
  input.instanceArray.numInstances = 1;
  const auto options =
      build_options(scene->allow_update, OPTIX_BUILD_OPERATION_BUILD);
  OptixAccelBufferSizes sizes{};
  result =
      optix_check(optixAccelComputeMemoryUsage(scene->context->optix_context,
                                               &options, &input, 1, &sizes),
                  "optixAccelComputeMemoryUsage(IAS)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->ias_scratch.allocate(
      std::max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->ias.allocate(sizes.outputSizeInBytes);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = optix_check(
      optixAccelBuild(scene->context->optix_context, stream, &options, &input,
                      1, scene->ias_scratch.pointer, scene->ias_scratch.bytes,
                      scene->ias.pointer, scene->ias.bytes, &scene->ias_handle,
                      nullptr, 0),
      "optixAccelBuild(IAS)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = cuda_check(cuStreamSynchronize(stream),
                      "cuStreamSynchronize(scene build)");
  if (result == TI_FORGE_OPTIX_SUCCESS && !scene->allow_update) {
    scene->ias_scratch.reset();
  }
  return result;
}

std::size_t shared_pipeline_sbt_bytes(Context *context) {
  std::lock_guard<std::mutex> lock(context->prepare_mutex);
  return context->legacy.raygen_record.bytes + context->legacy.miss_record.bytes +
         context->legacy.hitgroup_record.bytes +
         context->typed.raygen_record.bytes + context->typed.miss_record.bytes +
         context->typed.hitgroup_record.bytes +
         context->alpha.raygen_record.bytes + context->alpha.miss_record.bytes +
         context->alpha.hitgroup_record.bytes +
         context->micromap.raygen_record.bytes + context->micromap.miss_record.bytes +
         context->micromap.hitgroup_record.bytes;
}

void initialize_memory_result(TiForgeOptixSceneMemory *memory) {
  memory->reserved = 0;
  memory->gas_bytes = 0;
  memory->ias_bytes = 0;
  memory->build_update_scratch_bytes = 0;
  memory->instance_bytes = 0;
  memory->launch_params_bytes = 0;
  memory->shared_pipeline_sbt_bytes = 0;
}

bool valid_affine_transform(const float transform[12]) {
  for (unsigned int i = 0; i < 12; ++i) {
    if (!std::isfinite(transform[i])) {
      return false;
    }
  }
  const double a = transform[0], b = transform[1], c = transform[2];
  const double d = transform[4], e = transform[5], f = transform[6];
  const double g = transform[8], h = transform[9], i = transform[10];
  const double determinant =
      a * (e * i - f * h) - b * (d * i - f * g) +
      c * (d * h - e * g);
  return std::isfinite(determinant) && determinant != 0.0;
}

void delete_triangle_gas(TriangleGas *gas) {
  gas->gas_handle = 0;
  gas->scratch.reset();
  gas->gas.reset();
  gas->context->gas_count.fetch_sub(1);
  delete gas;
}

void release_triangle_gas_reference(TriangleGas *gas) {
  const auto previous = gas->instance_ref_count.fetch_sub(1);
  if (previous == 1 && !gas->owner_live.load(std::memory_order_acquire)) {
    delete_triangle_gas(gas);
  }
}

void release_instance_references(InstanceScene *scene) {
  for (auto *gas : scene->gas_refs) {
    release_triangle_gas_reference(gas);
  }
  scene->gas_refs.clear();
}

// Cold import only. Host descriptors and classification data are borrowed for
// this call; the GAS owns the resulting array and optional index buffer.
TiForgeOptixResult import_micromap(
    TriangleGas *gas,
    const TiForgeOptixMicromapDesc *desc,
    CUstream stream,
    TiForgeOptixMicromapMemory *memory) {
  // Bakers can emit only predefined triangle states and no array entries.
  // Supply one unreferenced native entry: OptiX requires an array for indexed
  // attachment even though no triangle needs to reference an actual micromap.
  TiForgeOptixMicromapDesc normalized{};
  const uint8_t unused_state = 0;
  const TiForgeOptixMicromapEntry unused_entry{0, 0, 1};
  if (desc->struct_size >= sizeof(*desc) && !desc->micromap_count &&
      !desc->data_size && desc->triangle_indices) {
    for (uint32_t i = 0; i < desc->triangle_count; ++i) {
      if (desc->triangle_indices[i] < -4 || desc->triangle_indices[i] >= 0) {
        return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                    "invalid empty OMM triangle mapping");
      }
    }
    normalized = *desc;
    normalized.data_size = 1;
    normalized.data = &unused_state;
    normalized.micromap_count = 1;
    normalized.entries = &unused_entry;
    desc = &normalized;
  }
  if (desc->struct_size < sizeof(*desc) || !desc->micromap_count ||
      !desc->data || !desc->data_size || !desc->entries || desc->reserved ||
      desc->triangle_count != gas->triangle_count ||
      (!desc->triangle_indices &&
       desc->micromap_count != gas->triangle_count)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid baked OMM descriptor");
  }
  try {
    using Key = std::pair<unsigned int, unsigned int>;
    std::map<Key, unsigned int> histogram, usage;
    for (uint32_t i = 0; i < desc->micromap_count; ++i) {
      const auto &entry = desc->entries[i];
      if (entry.subdivision_level >
              OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL ||
          (entry.format != 1 && entry.format != 2)) {
        return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                    "invalid OMM level or format");
      }
      const uint64_t bytes =
          ((1ull << (2 * entry.subdivision_level)) * entry.format + 7) / 8;
      if (uint64_t(entry.byte_offset) + bytes > desc->data_size) {
        return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                    "OMM entry exceeds baked data");
      }
      ++histogram[{entry.subdivision_level, entry.format}];
    }
    if (desc->triangle_indices) {
      for (uint32_t i = 0; i < desc->triangle_count; ++i) {
        const int32_t index = desc->triangle_indices[i];
        if (index < -4 ||
            (index >= 0 && uint32_t(index) >= desc->micromap_count)) {
          return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                      "invalid OMM triangle mapping");
        }
        if (index >= 0) {
          const auto &entry = desc->entries[index];
          ++usage[{entry.subdivision_level, entry.format}];
        }
      }
    } else {
      usage = histogram;
    }
    std::vector<OptixOpacityMicromapHistogramEntry> native_histogram;
    for (const auto &item : histogram) {
      native_histogram.push_back(
          {item.second, item.first.first,
           static_cast<OptixOpacityMicromapFormat>(item.first.second)});
    }
    for (const auto &item : usage) {
      gas->micromap_usage.push_back(
          {item.second, item.first.first,
           static_cast<OptixOpacityMicromapFormat>(item.first.second)});
    }
    static_assert(sizeof(TiForgeOptixMicromapEntry) ==
                  sizeof(OptixOpacityMicromapDesc));
    DeviceBuffer data, entries, scratch;
    auto result = data.allocate(desc->data_size);
    if (result == TI_FORGE_OPTIX_SUCCESS) {
      result = entries.allocate(size_t(desc->micromap_count) *
                                sizeof(OptixOpacityMicromapDesc));
    }
    if (result == TI_FORGE_OPTIX_SUCCESS) {
      result = cuda_check(cuMemcpyHtoD(data.pointer, desc->data, data.bytes),
                          "OMM data upload");
    }
    if (result == TI_FORGE_OPTIX_SUCCESS) {
      result = cuda_check(
          cuMemcpyHtoD(entries.pointer, desc->entries, entries.bytes),
          "OMM entries upload");
    }
    if (result == TI_FORGE_OPTIX_SUCCESS && desc->triangle_indices) {
      result = gas->micromap_indices.allocate(size_t(desc->triangle_count) *
                                              sizeof(int32_t));
      if (result == TI_FORGE_OPTIX_SUCCESS) {
        result = cuda_check(
            cuMemcpyHtoD(gas->micromap_indices.pointer, desc->triangle_indices,
                         gas->micromap_indices.bytes),
            "OMM indices upload");
      }
    }
    OptixOpacityMicromapArrayBuildInput input{};
    input.flags = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
    input.inputBuffer = data.pointer;
    input.perMicromapDescBuffer = entries.pointer;
    input.numMicromapHistogramEntries =
        static_cast<unsigned int>(native_histogram.size());
    input.micromapHistogramEntries = native_histogram.data();
    OptixMicromapBufferSizes sizes{};
    if (result == TI_FORGE_OPTIX_SUCCESS) {
      result = optix_check(optixOpacityMicromapArrayComputeMemoryUsage(
                               gas->context->optix_context, &input, &sizes),
                           "OMM memory usage");
    }
    if (result == TI_FORGE_OPTIX_SUCCESS)
      result = gas->micromap.allocate(sizes.outputSizeInBytes);
    if (result == TI_FORGE_OPTIX_SUCCESS)
      result = scratch.allocate(sizes.tempSizeInBytes);
    if (result == TI_FORGE_OPTIX_SUCCESS) {
      OptixMicromapBuffers buffers{};
      buffers.output = gas->micromap.pointer;
      buffers.outputSizeInBytes = gas->micromap.bytes;
      buffers.temp = scratch.pointer;
      buffers.tempSizeInBytes = scratch.bytes;
      result = optix_check(
          optixOpacityMicromapArrayBuild(gas->context->optix_context, stream,
                                         &input, &buffers),
          "OMM build");
      // Includes failure rollback: borrowed host input / temporary device
      // buffers cannot retire until work submitted by the build has completed.
      auto completed =
          cuda_check(cuStreamSynchronize(stream), "OMM build completion");
      if (completed != TI_FORGE_OPTIX_SUCCESS)
        result = completed;
    }
    if (result == TI_FORGE_OPTIX_SUCCESS) {
      memory->reserved = 0;
      memory->array_bytes = gas->micromap.bytes;
      memory->index_bytes = gas->micromap_indices.bytes;
      memory->build_temporary_bytes =
          data.bytes + entries.bytes + scratch.bytes;
    }
    return result;
  } catch (const std::bad_alloc &) {
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "OMM import metadata allocation failed");
  }
}

void attach_micromap(TriangleGas *gas,
                     OptixBuildInput &input,
                     unsigned int &flags) {
  if (!gas->micromap.pointer)
    return;
  flags = OPTIX_GEOMETRY_FLAG_NONE;
  auto &omm = input.triangleArray.opacityMicromap;
  omm.indexingMode = gas->micromap_indices.pointer
                         ? OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED
                         : OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
  omm.opacityMicromapArray = gas->micromap.pointer;
  omm.indexBuffer = gas->micromap_indices.pointer;
  omm.indexSizeInBytes = omm.indexBuffer ? sizeof(int32_t) : 0;
  omm.numMicromapUsageCounts =
      static_cast<unsigned int>(gas->micromap_usage.size());
  omm.micromapUsageCounts = gas->micromap_usage.data();
}

TiForgeOptixResult create_triangle_gas_impl(
    TiForgeOptixContext raw_context,
    const TiForgeOptixTriangleSceneDesc *desc,
    TiForgeOptixTriangleGas *out_gas,
    const TiForgeOptixMicromapDesc *micromap = nullptr,
    TiForgeOptixMicromapMemory *micromap_memory = nullptr) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr || desc == nullptr || out_gas == nullptr ||
      desc->struct_size < sizeof(*desc) || desc->vertex_count == 0 ||
      desc->triangle_count == 0 || desc->vertices == 0 || desc->indices == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX triangle GAS descriptor");
  }
  *out_gas = nullptr;
  auto *gas = new (std::nothrow) TriangleGas;
  if (gas == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate the Forge OptiX triangle GAS");
  }
  gas->context = context;
  gas->vertex_count = desc->vertex_count;
  gas->triangle_count = desc->triangle_count;
  gas->allow_update = desc->allow_update != 0;

  if (micromap) {
    auto imported = import_micromap(
        gas, micromap, reinterpret_cast<CUstream>(desc->cuda_stream),
        micromap_memory);
    if (imported != TI_FORGE_OPTIX_SUCCESS) {
      delete gas;
      return imported;
    }
  }

  CUdeviceptr vertex_buffer = 0;
  unsigned int geometry_flags = 0;
  auto input = triangle_build_input(*desc, &vertex_buffer, &geometry_flags);
  attach_micromap(gas, input, geometry_flags);
  auto options = build_options(gas->allow_update, OPTIX_BUILD_OPERATION_BUILD);
  OptixAccelBufferSizes sizes{};
  auto result =
      optix_check(optixAccelComputeMemoryUsage(context->optix_context, &options,
                                               &input, 1, &sizes),
                  "optixAccelComputeMemoryUsage(shared GAS)");
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = gas->gas.allocate(sizes.outputSizeInBytes);
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = gas->scratch.allocate(
        std::max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  bool build_may_be_enqueued = false;
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    build_may_be_enqueued = true;
    result = optix_check(
        optixAccelBuild(context->optix_context, stream, &options, &input, 1,
                        gas->scratch.pointer, gas->scratch.bytes,
                        gas->gas.pointer, gas->gas.bytes, &gas->gas_handle,
                        nullptr, 0),
        "optixAccelBuild(shared GAS)");
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = cuda_check(cuStreamSynchronize(stream),
                        "cuStreamSynchronize(shared GAS build)");
  }
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    if (build_may_be_enqueued) {
      const auto rollback_result =
          cuda_check(cuStreamSynchronize(stream),
                     "cuStreamSynchronize(shared GAS build rollback)");
      if (rollback_result != TI_FORGE_OPTIX_SUCCESS) {
        result = rollback_result;
      }
    }
    delete gas;
    return result;
  }
  if (!gas->allow_update) {
    gas->scratch.reset();
  }
  context->gas_count.fetch_add(1);
  *out_gas = gas;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult create_triangle_gas(
    TiForgeOptixContext context,
    const TiForgeOptixTriangleSceneDesc *desc,
    TiForgeOptixTriangleGas *out_gas) {
  return create_triangle_gas_impl(context, desc, out_gas);
}

TiForgeOptixResult create_triangle_gas_micromap(
    TiForgeOptixContext raw_context,
    const TiForgeOptixTriangleSceneDesc *desc,
    const TiForgeOptixMicromapDesc *micromap,
    TiForgeOptixTriangleGas *out_gas,
    TiForgeOptixMicromapMemory *memory) {
  clear_error_state();
  if (!raw_context || !micromap || !memory ||
      memory->struct_size < sizeof(*memory)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OMM import request");
  }
  auto *context = static_cast<Context *>(raw_context);
  {
    std::lock_guard<std::mutex> lock(context->prepare_mutex);
    if (!context->micromap_ready.load(std::memory_order_acquire)) {
      auto result = create_pipeline(context, &context->micromap, 3);
      if (result != TI_FORGE_OPTIX_SUCCESS) {
        destroy_pipeline(&context->micromap);
        return result;
      }
      context->micromap_ready.store(true, std::memory_order_release);
    }
  }
  return create_triangle_gas_impl(raw_context, desc, out_gas, micromap, memory);
}

TiForgeOptixResult update_triangle_gas(
    TiForgeOptixTriangleGas raw_gas,
    const TiForgeOptixTriangleSceneDesc *desc) {
  clear_error_state();
  auto *gas = static_cast<TriangleGas *>(raw_gas);
  if (gas == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->vertex_count != gas->vertex_count ||
      desc->triangle_count != gas->triangle_count || desc->vertices == 0 ||
      desc->indices == 0 || !gas->owner_live.load(std::memory_order_acquire)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX shared GAS update must preserve geometry shape");
  }
  if (!gas->allow_update) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX shared GAS was not created with update support");
  }
  CUdeviceptr vertex_buffer = 0;
  unsigned int geometry_flags = 0;
  auto input = triangle_build_input(*desc, &vertex_buffer, &geometry_flags);
  attach_micromap(gas, input, geometry_flags);
  auto options = build_options(true, OPTIX_BUILD_OPERATION_UPDATE);
  return optix_check(
      optixAccelBuild(gas->context->optix_context,
                      reinterpret_cast<CUstream>(desc->cuda_stream), &options,
                      &input, 1, gas->scratch.pointer, gas->scratch.bytes,
                      gas->gas.pointer, gas->gas.bytes, &gas->gas_handle,
                      nullptr, 0),
      "optixAccelBuild(shared GAS update)");
}

TiForgeOptixResult get_triangle_gas_memory(
    TiForgeOptixTriangleGas raw_gas,
    TiForgeOptixSceneMemory *out_memory) {
  clear_error_state();
  auto *gas = static_cast<TriangleGas *>(raw_gas);
  if (gas == nullptr || out_memory == nullptr ||
      out_memory->struct_size < sizeof(*out_memory) ||
      !gas->owner_live.load(std::memory_order_acquire)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX shared GAS memory query");
  }
  initialize_memory_result(out_memory);
  out_memory->gas_bytes = gas->gas.bytes;
  out_memory->build_update_scratch_bytes = gas->scratch.bytes;
  out_memory->shared_pipeline_sbt_bytes =
      shared_pipeline_sbt_bytes(gas->context);
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult destroy_triangle_gas(TiForgeOptixTriangleGas raw_gas) {
  clear_error_state();
  auto *gas = static_cast<TriangleGas *>(raw_gas);
  if (gas == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX triangle GAS is null");
  }
  if (!gas->owner_live.exchange(false, std::memory_order_acq_rel)) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX triangle GAS owner was already released");
  }
  if (gas->instance_ref_count.load(std::memory_order_acquire) == 0) {
    const auto result =
        cuda_check(cuCtxSynchronize(), "cuCtxSynchronize(shared GAS destroy)");
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      gas->owner_live.store(true, std::memory_order_release);
      return result;
    }
    delete_triangle_gas(gas);
  }
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult create_instance_scene(
    TiForgeOptixContext raw_context,
    const TiForgeOptixInstanceSceneDesc *desc,
    TiForgeOptixInstanceScene *out_scene) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr || desc == nullptr || out_scene == nullptr ||
      desc->struct_size < sizeof(*desc) || desc->instance_count == 0 ||
      desc->instances == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX instance scene descriptor");
  }
  *out_scene = nullptr;
  auto *scene = new (std::nothrow) InstanceScene;
  if (scene == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate the Forge OptiX instance scene");
  }
  scene->context = context;
  scene->instance_count = desc->instance_count;
  scene->allow_update = desc->allow_update != 0;
  std::vector<OptixInstance> host_instances;
  try {
    host_instances.resize(desc->instance_count);
    scene->gas_refs.reserve(desc->instance_count);
  } catch (const std::bad_alloc &) {
    delete scene;
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate OptiX instance metadata");
  }
  for (uint32_t index = 0; index < desc->instance_count; ++index) {
    const auto &source = desc->instances[index];
    if (source.struct_size < sizeof(source)) {
      delete scene;
      return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                  "OptiX instance metadata is truncated");
    }
    auto *gas = static_cast<TriangleGas *>(source.gas);
    if (gas == nullptr || gas->context != context ||
        !gas->owner_live.load(std::memory_order_acquire) ||
        source.custom_index > 0xffffffu || source.visibility_mask > 0xffu ||
        (source.reserved & ~1u) != 0 ||
        (gas->micromap.pointer && (source.reserved & 1u)) ||
        !valid_affine_transform(source.transform)) {
      delete scene;
      return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                  "OptiX instances require live same-context GAS handles, "
                  "finite invertible transforms, 24-bit custom indices and "
                  "8-bit visibility masks");
    }
    auto &destination = host_instances[index];
    std::memcpy(destination.transform, source.transform,
                sizeof(destination.transform));
    destination.instanceId = source.custom_index;
    destination.sbtOffset = 0;
    destination.visibilityMask = source.visibility_mask;
    destination.flags = (source.reserved & 1u)
                            ? OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
                            : (gas->micromap.pointer ? OPTIX_INSTANCE_FLAG_NONE
                                                     : OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT);
    destination.traversableHandle = gas->gas_handle;
    scene->gas_refs.push_back(gas);
  }
  for (auto *gas : scene->gas_refs) {
    gas->instance_ref_count.fetch_add(1);
  }

  auto result = scene->launch_params.allocate(sizeof(LaunchParams));
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = scene->instance.allocate(host_instances.size() * sizeof(OptixInstance));
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  bool copy_may_be_enqueued = false;
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    copy_may_be_enqueued = true;
    result = cuda_check(
        cuMemcpyHtoDAsync(scene->instance.pointer, host_instances.data(),
                          scene->instance.bytes, stream),
        "cuMemcpyHtoDAsync(shared GAS instances)");
  }
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  input.instanceArray.instances = scene->instance.pointer;
  input.instanceArray.numInstances = scene->instance_count;
  const auto options =
      build_options(scene->allow_update, OPTIX_BUILD_OPERATION_BUILD);
  OptixAccelBufferSizes sizes{};
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = optix_check(
        optixAccelComputeMemoryUsage(context->optix_context, &options, &input,
                                     1, &sizes),
        "optixAccelComputeMemoryUsage(multi-instance IAS)");
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = scene->scratch.allocate(
        std::max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = scene->ias.allocate(sizes.outputSizeInBytes);
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = optix_check(
        optixAccelBuild(context->optix_context, stream, &options, &input, 1,
                        scene->scratch.pointer, scene->scratch.bytes,
                        scene->ias.pointer, scene->ias.bytes,
                        &scene->ias_handle, nullptr, 0),
        "optixAccelBuild(multi-instance IAS)");
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = cuda_check(cuStreamSynchronize(stream),
                        "cuStreamSynchronize(instance scene build)");
  }
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    if (copy_may_be_enqueued) {
      const auto rollback_result = cuda_check(
          cuStreamSynchronize(stream),
          "cuStreamSynchronize(instance scene build rollback)");
      if (rollback_result != TI_FORGE_OPTIX_SUCCESS) {
        result = rollback_result;
      }
    }
    release_instance_references(scene);
    delete scene;
    return result;
  }
  if (!scene->allow_update) {
    scene->scratch.reset();
  }
  context->instance_scene_count.fetch_add(1);
  *out_scene = scene;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult update_instance_scene(
    TiForgeOptixInstanceScene raw_scene,
    const TiForgeOptixInstanceUpdateDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  if (scene == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->instance_count != scene->instance_count) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX instance update must preserve the fixed instance count");
  }
  if (!scene->allow_update) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX instance scene was not created with update support");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  if (desc->transforms != 0) {
    CUdeviceptr transforms = static_cast<CUdeviceptr>(desc->transforms);
    CUdeviceptr instances = scene->instance.pointer;
    uint32_t count = scene->instance_count;
    void *arguments[] = {&transforms, &count, &instances};
    const uint32_t block_size = 256;
    const uint32_t grid_size = (count - 1) / block_size + 1;
    auto result = cuda_check(
        cuLaunchKernel(scene->context->transform_pack, grid_size, 1, 1,
                       block_size, 1, 1, 0, stream, arguments, nullptr),
        "cuLaunchKernel(instance transform pack)");
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return result;
    }
  }
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  input.instanceArray.instances = scene->instance.pointer;
  input.instanceArray.numInstances = scene->instance_count;
  const auto options = build_options(true, OPTIX_BUILD_OPERATION_UPDATE);
  return optix_check(
      optixAccelBuild(scene->context->optix_context, stream, &options, &input, 1,
                      scene->scratch.pointer, scene->scratch.bytes,
                      scene->ias.pointer, scene->ias.bytes, &scene->ias_handle,
                      nullptr, 0),
      "optixAccelBuild(multi-instance IAS update)");
}

TiForgeOptixResult create_context(const TiForgeOptixContextDesc *desc,
                                  TiForgeOptixContext *out_context) {
  clear_error_state();
  if (desc == nullptr || out_context == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixContextDesc)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX context descriptor");
  }
  *out_context = nullptr;
  CUcontext cuda_context = reinterpret_cast<CUcontext>(desc->cuda_context);
  if (cuda_context == nullptr) {
    auto result = cuda_check(cuCtxGetCurrent(&cuda_context), "cuCtxGetCurrent");
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return result;
    }
  }
  if (cuda_context == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_CUDA_CONTEXT,
                "Forge OptiX provider requires the active Taichi CUDA context");
  }
  auto result = retain_optix_loader(desc->runtime_library_path);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  auto *context = new (std::nothrow) Context;
  if (context == nullptr) {
    release_optix_loader();
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate the Forge OptiX context");
  }
  context->cuda_context = cuda_context;
  OptixDeviceContextOptions options{};
  options.logCallbackFunction = optix_log;
  options.logCallbackLevel = desc->validation_mode ? 4 : 1;
  result = optix_check(
      optixDeviceContextCreate(cuda_context, &options, &context->optix_context),
      "optixDeviceContextCreate");
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_pipeline(context, &context->legacy, false);
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_transform_module(context);
  }
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    destroy_transform_module(context);
    destroy_pipeline(&context->legacy);
    if (context->optix_context != nullptr) {
      optixDeviceContextDestroy(context->optix_context);
    }
    delete context;
    release_optix_loader();
    return result;
  }
  *out_context = context;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult destroy_context(TiForgeOptixContext raw_context) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX context is null");
  }
  if (context->scene_count.load() != 0 || context->gas_count.load() != 0 ||
      context->instance_scene_count.load() != 0) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX context still owns live acceleration structures");
  }
  auto result = cuda_check(cuCtxSynchronize(), "cuCtxSynchronize(context destroy)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  destroy_transform_module(context);
  destroy_pipeline(&context->micromap);
  destroy_pipeline(&context->alpha);
  destroy_pipeline(&context->typed);
  destroy_pipeline(&context->legacy);
  if (context->optix_context != nullptr) {
    optixDeviceContextDestroy(context->optix_context);
  }
  delete context;
  release_optix_loader();
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult create_triangle_scene(
    TiForgeOptixContext raw_context,
    const TiForgeOptixTriangleSceneDesc *desc,
    TiForgeOptixTriangleScene *out_scene) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr || desc == nullptr || out_scene == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixTriangleSceneDesc) ||
      desc->vertex_count == 0 || desc->triangle_count == 0 ||
      desc->vertices == 0 || desc->indices == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX triangle scene descriptor");
  }
  *out_scene = nullptr;
  auto *scene = new (std::nothrow) Scene;
  if (scene == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate the Forge OptiX scene");
  }
  scene->context = context;
  scene->vertex_count = desc->vertex_count;
  scene->triangle_count = desc->triangle_count;
  scene->allow_update = desc->allow_update != 0;
  auto result = scene->launch_params.allocate(sizeof(LaunchParams));
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_gas(scene, *desc);
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_ias(scene, reinterpret_cast<CUstream>(desc->cuda_stream));
  }
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    delete scene;
    return result;
  }
  context->scene_count.fetch_add(1);
  *out_scene = scene;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult update_triangle_scene(
    TiForgeOptixTriangleScene raw_scene,
    const TiForgeOptixTriangleSceneDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || desc == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixTriangleSceneDesc) ||
      desc->vertex_count != scene->vertex_count ||
      desc->triangle_count != scene->triangle_count || desc->vertices == 0 ||
      desc->indices == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX scene update must preserve geometry shape");
  }
  if (!scene->allow_update) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX scene was not created with update support");
  }
  CUdeviceptr vertex_buffer = 0;
  unsigned int geometry_flags = 0;
  auto input = triangle_build_input(*desc, &vertex_buffer, &geometry_flags);
  auto options = build_options(true, OPTIX_BUILD_OPERATION_UPDATE);
  auto result = optix_check(
      optixAccelBuild(scene->context->optix_context,
                      reinterpret_cast<CUstream>(desc->cuda_stream), &options,
                      &input, 1, scene->scratch.pointer, scene->scratch.bytes,
                      scene->gas.pointer, scene->gas.bytes, &scene->gas_handle,
                      nullptr, 0),
      "optixAccelBuild(GAS update)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  // A changed GAS bound does not refresh its parent IAS. The topology and GAS
  // handle are fixed, so reuse the instance description and retained scratch.
  // Both updates are ordered on the caller's stream; there is no readback or
  // host synchronization between the producer, these updates and later rays.
  OptixBuildInput instance_input{};
  instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instance_input.instanceArray.instances = scene->instance.pointer;
  instance_input.instanceArray.numInstances = 1;
  return optix_check(
      optixAccelBuild(scene->context->optix_context,
                      reinterpret_cast<CUstream>(desc->cuda_stream), &options,
                      &instance_input, 1, scene->ias_scratch.pointer,
                      scene->ias_scratch.bytes, scene->ias.pointer,
                      scene->ias.bytes, &scene->ias_handle, nullptr, 0),
      "optixAccelBuild(IAS update)");
}

TiForgeOptixResult trace(TiForgeOptixTriangleScene raw_scene,
                         const TiForgeOptixTraceDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || desc == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixTraceDesc) ||
      desc->ray_count == 0 || desc->rays == 0 || desc->hits == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX trace descriptor");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  const LaunchParams params{static_cast<CUdeviceptr>(desc->rays),
                            static_cast<CUdeviceptr>(desc->hits),
                            scene->ias_handle, 0};
  auto result = cuda_check(cuMemcpyHtoDAsync(scene->launch_params.pointer,
                                             &params, sizeof(params), stream),
                           "cuMemcpyHtoDAsync(launch params)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixLaunch(scene->context->legacy.pipeline, stream,
                  scene->launch_params.pointer, sizeof(params),
                  &scene->context->legacy.sbt, desc->ray_count, 1, 1),
      "optixLaunch");
}

TiForgeOptixResult trace_typed(TiForgeOptixTriangleScene raw_scene,
                               const TiForgeOptixTypedTraceDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || desc == nullptr ||
      desc->struct_size < sizeof(*desc) || !desc->ray_count || !desc->rays ||
      !desc->hits || !desc->hit_indices ||
      !scene->context->typed_ready.load(std::memory_order_acquire)) {
    return fail(
        TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
        "OptiX typed trace requires a prepared pipeline and valid storage");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  const LaunchParams params{desc->rays, desc->hits, scene->ias_handle,
                            desc->hit_indices};
  auto result = cuda_check(cuMemcpyHtoDAsync(scene->launch_params.pointer,
                                             &params, sizeof(params), stream),
                           "cuMemcpyHtoDAsync(typed launch params)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixLaunch(scene->context->typed.pipeline, stream,
                  scene->launch_params.pointer, sizeof(params),
                  &scene->context->typed.sbt, desc->ray_count, 1, 1),
      "optixLaunch(typed)");
}

TiForgeOptixResult trace_instance_scene(
    TiForgeOptixInstanceScene raw_scene,
    const TiForgeOptixTraceDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  if (scene == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->ray_count == 0 || desc->rays == 0 || desc->hits == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX instance trace descriptor");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  const LaunchParams params{static_cast<CUdeviceptr>(desc->rays),
                            static_cast<CUdeviceptr>(desc->hits),
                            scene->ias_handle, 0};
  auto result = cuda_check(cuMemcpyHtoDAsync(scene->launch_params.pointer,
                                             &params, sizeof(params), stream),
                           "cuMemcpyHtoDAsync(instance launch params)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixLaunch(scene->context->legacy.pipeline, stream,
                  scene->launch_params.pointer, sizeof(params),
                  &scene->context->legacy.sbt, desc->ray_count, 1, 1),
      "optixLaunch(instance scene)");
}

TiForgeOptixResult trace_instance_scene_typed(
    TiForgeOptixInstanceScene raw_scene,
    const TiForgeOptixTypedTraceDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  if (scene == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      desc->ray_count == 0 || desc->rays == 0 || desc->hits == 0 ||
      desc->hit_indices == 0 ||
      !scene->context->typed_ready.load(std::memory_order_acquire)) {
    return fail(
        TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
        "OptiX typed instance trace requires a prepared pipeline and valid storage");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  const LaunchParams params{desc->rays, desc->hits, scene->ias_handle,
                            desc->hit_indices};
  auto result = cuda_check(cuMemcpyHtoDAsync(scene->launch_params.pointer,
                                             &params, sizeof(params), stream),
                           "cuMemcpyHtoDAsync(typed instance launch params)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixLaunch(scene->context->typed.pipeline, stream,
                  scene->launch_params.pointer, sizeof(params),
                  &scene->context->typed.sbt, desc->ray_count, 1, 1),
      "optixLaunch(typed instance scene)");
}

template <typename SceneType, bool Micromap = false>
TiForgeOptixResult launch_alpha(SceneType *scene,
                                 uint32_t instance_count,
                                 const TiForgeOptixAlphaTraceDesc *desc) {
  clear_error_state();
  if (scene == nullptr || desc == nullptr || desc->struct_size < sizeof(*desc) ||
      !desc->ray_count || !desc->rays || !desc->hits || !desc->hit_indices ||
      !desc->masks || !desc->launch_params || desc->mask_count != instance_count ||
      desc->any_hit > 1 ||
      !(Micromap ? scene->context->micromap_ready : scene->context->alpha_ready)
           .load(std::memory_order_acquire)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX alpha trace requires prepared masks, pipeline and workspace");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  auto &pipeline = Micromap ? scene->context->micromap : scene->context->alpha;
  const AlphaLaunchParams params{
      {desc->rays, desc->hits, scene->ias_handle, desc->hit_indices},
      desc->masks, desc->any_hit, 0};
  auto result = cuda_check(cuMemcpyHtoDAsync(desc->launch_params, &params,
                                           sizeof(params), stream),
                           "cuMemcpyHtoDAsync(alpha launch params)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixLaunch(pipeline.pipeline, stream, desc->launch_params,
                  sizeof(params), &pipeline.sbt,
                  desc->ray_count, 1, 1),
      "optixLaunch(alpha mask)");
}

TiForgeOptixResult trace_alpha(TiForgeOptixTriangleScene raw_scene,
                                const TiForgeOptixAlphaTraceDesc *desc) {
  return launch_alpha(static_cast<Scene *>(raw_scene), 1, desc);
}

TiForgeOptixResult trace_instance_alpha(
    TiForgeOptixInstanceScene raw_scene,
    const TiForgeOptixAlphaTraceDesc *desc) {
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  return launch_alpha(scene, scene ? scene->instance_count : 0, desc);
}

TiForgeOptixResult trace_instance_micromap(
    TiForgeOptixInstanceScene raw_scene,
    const TiForgeOptixAlphaTraceDesc *desc) {
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  return launch_alpha<InstanceScene, true>(scene, scene ? scene->instance_count : 0, desc);
}

TiForgeOptixResult get_instance_scene_memory(
    TiForgeOptixInstanceScene raw_scene,
    TiForgeOptixSceneMemory *out_memory) {
  clear_error_state();
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  if (scene == nullptr || out_memory == nullptr ||
      out_memory->struct_size < sizeof(*out_memory)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX instance scene memory query");
  }
  initialize_memory_result(out_memory);
  out_memory->ias_bytes = scene->ias.bytes;
  out_memory->build_update_scratch_bytes = scene->scratch.bytes;
  out_memory->instance_bytes = scene->instance.bytes;
  out_memory->launch_params_bytes = scene->launch_params.bytes;
  out_memory->shared_pipeline_sbt_bytes =
      shared_pipeline_sbt_bytes(scene->context);
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult destroy_instance_scene(
    TiForgeOptixInstanceScene raw_scene) {
  clear_error_state();
  auto *scene = static_cast<InstanceScene *>(raw_scene);
  if (scene == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX instance scene is null");
  }
  auto result =
      cuda_check(cuCtxSynchronize(), "cuCtxSynchronize(instance scene destroy)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  scene->ias_handle = 0;
  scene->launch_params.reset();
  scene->ias.reset();
  scene->scratch.reset();
  scene->instance.reset();
  release_instance_references(scene);
  scene->context->instance_scene_count.fetch_sub(1);
  delete scene;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult get_scene_memory(TiForgeOptixTriangleScene raw_scene,
                                    TiForgeOptixSceneMemory *out_memory) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || out_memory == nullptr ||
      out_memory->struct_size < sizeof(TiForgeOptixSceneMemory)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX scene memory query");
  }
  initialize_memory_result(out_memory);
  out_memory->gas_bytes = scene->gas.bytes;
  out_memory->ias_bytes = scene->ias.bytes;
  out_memory->build_update_scratch_bytes =
      scene->scratch.bytes + scene->ias_scratch.bytes;
  out_memory->instance_bytes = scene->instance.bytes;
  out_memory->launch_params_bytes = scene->launch_params.bytes;
  out_memory->shared_pipeline_sbt_bytes =
      shared_pipeline_sbt_bytes(scene->context);
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult destroy_triangle_scene(
    TiForgeOptixTriangleScene raw_scene) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX triangle scene is null");
  }
  auto result =
      cuda_check(cuCtxSynchronize(), "cuCtxSynchronize(triangle scene destroy)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  scene->context->scene_count.fetch_sub(1);
  delete scene;
  return TI_FORGE_OPTIX_SUCCESS;
}

std::size_t get_last_error(char *destination, std::size_t destination_size) {
  const std::size_t required = last_error.size() + 1;
  if (destination != nullptr && destination_size != 0) {
    const std::size_t copied = std::min(last_error.size(), destination_size - 1);
    std::memcpy(destination, last_error.data(), copied);
    destination[copied] = '\0';
  }
  return required;
}

}  // namespace

extern "C" TI_FORGE_OPTIX_EXPORT TiForgeOptixResult
taichi_forge_optix_provider_query(uint32_t requested_abi_version,
                                  size_t api_size,
                                  TiForgeOptixProviderApi *out_api) {
  clear_error_state();
  if (requested_abi_version != TI_FORGE_OPTIX_PROVIDER_ABI_VERSION) {
    return fail(TI_FORGE_OPTIX_ERROR_ABI_MISMATCH,
                "unsupported Forge OptiX provider ABI");
  }
  if (out_api == nullptr ||
      api_size < offsetof(TiForgeOptixProviderApi, prepare_typed)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "Forge OptiX provider API table is too small");
  }
  TiForgeOptixProviderApi negotiated{};
  auto *destination = out_api;
  out_api = &negotiated;
  out_api->struct_size =
      static_cast<uint32_t>(std::min(api_size, sizeof(negotiated)));
  out_api->provider_abi_version = TI_FORGE_OPTIX_PROVIDER_ABI_VERSION;
  out_api->info.struct_size = sizeof(TiForgeOptixProviderInfo);
  out_api->info.provider_abi_version = TI_FORGE_OPTIX_PROVIDER_ABI_VERSION;
  out_api->info.optix_abi_version = OPTIX_ABI_VERSION;
  out_api->info.optix_version = OPTIX_VERSION;
  out_api->info.features = kFeatures;
  out_api->info.provider_name = kProviderName;
  out_api->info.build_identity = kBuildIdentity;
  out_api->probe_runtime = probe_runtime;
  out_api->create_context = create_context;
  out_api->destroy_context = destroy_context;
  out_api->create_triangle_scene = create_triangle_scene;
  out_api->update_triangle_scene = update_triangle_scene;
  out_api->trace = trace;
  out_api->get_scene_memory = get_scene_memory;
  out_api->destroy_triangle_scene = destroy_triangle_scene;
  out_api->get_last_error = get_last_error;
  out_api->prepare_typed = prepare_typed;
  out_api->trace_typed = trace_typed;
  out_api->create_triangle_gas = create_triangle_gas;
  out_api->update_triangle_gas = update_triangle_gas;
  out_api->get_triangle_gas_memory = get_triangle_gas_memory;
  out_api->destroy_triangle_gas = destroy_triangle_gas;
  out_api->create_instance_scene = create_instance_scene;
  out_api->update_instance_scene = update_instance_scene;
  out_api->trace_instance_scene = trace_instance_scene;
  out_api->trace_instance_scene_typed = trace_instance_scene_typed;
  out_api->get_instance_scene_memory = get_instance_scene_memory;
  out_api->destroy_instance_scene = destroy_instance_scene;
  out_api->prepare_alpha = prepare_alpha;
  out_api->trace_alpha = trace_alpha;
  out_api->trace_instance_alpha = trace_instance_alpha;
  out_api->create_triangle_gas_micromap = create_triangle_gas_micromap;
  out_api->trace_instance_micromap = trace_instance_micromap;
  std::memcpy(destination, out_api, out_api->struct_size);
  return TI_FORGE_OPTIX_SUCCESS;
}
