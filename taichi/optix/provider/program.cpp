#include "taichi/optix/provider/program_internal.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <optix_stubs.h>
#include <optix_stack_size.h>

namespace taichi::optix_provider {
namespace {

struct Program {
  TiForgeOptixContext owner{};
  ProgramContextView context;
  std::vector<OptixModule> modules;
  std::vector<OptixProgramGroup> groups;
  std::vector<uint32_t> kinds;
  OptixPipeline pipeline{};
  uint32_t parameter_size{};
  uint32_t continuation_stack{};
  bool allow_micromaps{};
  std::atomic<uint32_t> launches{0};

  ~Program() {
    if (pipeline)
      optixPipelineDestroy(pipeline);
    for (auto group : groups)
      if (group)
        optixProgramGroupDestroy(group);
    for (auto module : modules)
      if (module)
        optixModuleDestroy(module);
    if (owner)
      release_program_context(owner);
  }
};

struct PreparedLaunch {
  Program *program{};
  std::vector<TiForgeOptixProgramScene> scenes;
  CUdeviceptr device{};
  void *host{};
  size_t bytes{};
  OptixShaderBindingTable sbt{};
  uint32_t width{}, height{}, depth{};
  CUstream stream{};
  bool initialized{};
  bool initialization_attempted{};

  ~PreparedLaunch() {
    if (device)
      cuMemFree(device);
    if (host)
      cuMemFreeHost(host);
    for (const auto &scene : scenes)
      release_program_scene(scene);
    if (program)
      program->launches.fetch_sub(1);
  }
};

template <typename F>
TiForgeOptixResult boundary(F &&call) noexcept {
  clear_error_state();
  try {
    return call();
  } catch (const std::bad_alloc &) {
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "programmable OptiX allocation failed");
  } catch (const std::exception &error) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT, error.what());
  }
}

void require(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(message);
}

TiForgeOptixResult logged(OptixResult result,
                          const char *operation,
                          const char *log,
                          size_t log_size) {
  auto status = optix_check(result, operation);
  if (status != TI_FORGE_OPTIX_SUCCESS && log_size > 1 && log[0]) {
    last_error += "; program log: ";
    const auto *end =
        std::find(log, log + std::min(log_size, size_t(8192)), '\0');
    last_error.append(log, end);
  }
  return status;
}

TiForgeOptixResult create(TiForgeOptixContext context,
                          const TiForgeOptixProgramDesc *desc,
                          TiForgeOptixProgram *out) {
  return boundary([&] {
    require(out != nullptr, "program output is null");
    *out = nullptr;
    require(desc && desc->struct_size >= sizeof(*desc) && desc->module_count &&
                desc->modules && desc->group_count && desc->groups,
            "program requires complete module and group descriptors");
    require(desc->payload_count <= 32 && desc->attribute_count >= 2 &&
                desc->attribute_count <= 8 &&
                desc->allow_opacity_micromaps <= 1,
            "invalid program payload/attribute/micromap contract");
    require(!desc->parameter_size ||
                (desc->parameter_name && desc->parameter_name[0]),
            "nonempty launch parameters require a variable name");
    auto program = std::make_unique<Program>();
    auto result = retain_program_context(context, &program->context);
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    program->owner = context;
    program->parameter_size = desc->parameter_size;
    program->allow_micromaps = desc->allow_opacity_micromaps != 0;
    unsigned int max_depth{};
    result = optix_check(optixDeviceContextGetProperty(
                             program->context.optix_context,
                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
                             &max_depth, sizeof(max_depth)),
                         "optixDeviceContextGetProperty(trace depth)");
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    require(desc->max_trace_depth <= max_depth,
            "program trace depth exceeds the device limit");
    OptixModuleCompileOptions module_options{};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions options{};
    options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    options.numPayloadValues = desc->payload_count;
    options.numAttributeValues = desc->attribute_count;
    options.pipelineLaunchParamsVariableName =
        desc->parameter_size ? desc->parameter_name : nullptr;
    options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    options.allowOpacityMicromaps = program->allow_micromaps;
#if OPTIX_ABI_VERSION == 118
    options.pipelineLaunchParamsSizeInBytes = desc->parameter_size;
#endif
    program->modules.resize(desc->module_count);
    char log[8192]{};
    for (uint32_t i = 0; i < desc->module_count; ++i) {
      const auto &module = desc->modules[i];
      require(module.ptx && module.ptx_size, "empty program PTX module");
      size_t log_size = sizeof(log);
      log[0] = 0;
      auto status = optixModuleCreate(
          program->context.optix_context, &module_options, &options, module.ptx,
          module.ptx_size, log, &log_size, &program->modules[i]);
      result = logged(status, "optixModuleCreate(user program)", log, log_size);
      if (result != TI_FORGE_OPTIX_SUCCESS)
        return result;
    }
    auto module_for =
        [&](const TiForgeOptixProgramEntry &entry) -> OptixModule {
      if (!entry.name)
        return nullptr;
      require(entry.name[0] && entry.module_index < program->modules.size(),
              "invalid program entry/module index");
      return program->modules[entry.module_index];
    };
    program->groups.resize(desc->group_count);
    program->kinds.reserve(desc->group_count);
    bool has_raygen = false;
    for (uint32_t i = 0; i < desc->group_count; ++i) {
      const auto &group = desc->groups[i];
      OptixProgramGroupDesc native{};
      require(group.kind <= 2,
              "only raygen, miss and triangle hit groups are supported");
      if (group.kind < 2) {
        require(group.entry.name && !group.any_hit.name,
                "raygen/miss require exactly one entry");
        native.kind = group.kind == 0 ? OPTIX_PROGRAM_GROUP_KIND_RAYGEN
                                      : OPTIX_PROGRAM_GROUP_KIND_MISS;
        auto &single = group.kind == 0 ? native.raygen : native.miss;
        single.module = module_for(group.entry);
        single.entryFunctionName = group.entry.name;
        has_raygen |= group.kind == 0;
      } else {
        require(group.entry.name || group.any_hit.name,
                "triangle hit group requires an entry");
        native.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        native.hitgroup.moduleCH = module_for(group.entry);
        native.hitgroup.entryFunctionNameCH = group.entry.name;
        native.hitgroup.moduleAH = module_for(group.any_hit);
        native.hitgroup.entryFunctionNameAH = group.any_hit.name;
      }
      OptixProgramGroupOptions group_options{};
      size_t log_size = sizeof(log);
      log[0] = 0;
      auto status = optixProgramGroupCreate(program->context.optix_context,
                                            &native, 1, &group_options, log,
                                            &log_size, &program->groups[i]);
      result = logged(status, "optixProgramGroupCreate(user program)", log,
                      log_size);
      if (result != TI_FORGE_OPTIX_SUCCESS)
        return result;
      program->kinds.push_back(group.kind);
    }
    require(has_raygen, "program requires a raygen group");
    OptixPipelineLinkOptions link{};
    link.maxTraceDepth = desc->max_trace_depth;
    size_t log_size = sizeof(log);
    log[0] = 0;
    auto status = optixPipelineCreate(
        program->context.optix_context, &options, &link, program->groups.data(),
        desc->group_count, log, &log_size, &program->pipeline);
    result = logged(status, "optixPipelineCreate(user program)", log, log_size);
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    OptixStackSizes stack{};
    for (auto group : program->groups) {
      result = optix_check(
          optixUtilAccumulateStackSizes(group, &stack, program->pipeline),
          "optixUtilAccumulateStackSizes");
      if (result != TI_FORGE_OPTIX_SUCCESS)
        return result;
    }
    unsigned int from_traversal{}, from_state{}, continuation{};
    result = optix_check(
        optixUtilComputeStackSizes(&stack, desc->max_trace_depth, 0, 0,
                                   &from_traversal, &from_state, &continuation),
        "optixUtilComputeStackSizes");
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    result =
        optix_check(optixPipelineSetStackSize(program->pipeline, from_traversal,
                                              from_state, continuation, 2),
                    "optixPipelineSetStackSize(user program)");
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    program->continuation_stack = continuation;
    *out = program.release();
    return TI_FORGE_OPTIX_SUCCESS;
  });
}

TiForgeOptixResult destroy(TiForgeOptixProgram raw) {
  return boundary([&] {
    auto *program = static_cast<Program *>(raw);
    require(program != nullptr, "program is null");
    if (program->launches.load() != 0) {
      return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                  "program still has prepared launches");
    }
    delete program;
    return TI_FORGE_OPTIX_SUCCESS;
  });
}

size_t align_record(size_t size) {
  constexpr size_t alignment = OPTIX_SBT_RECORD_ALIGNMENT;
  require(size <= std::numeric_limits<size_t>::max() - alignment,
          "SBT size overflow");
  return (size + alignment - 1) & ~(alignment - 1);
}

uint32_t record_stride(const Program &program,
                       const TiForgeOptixSbtRecord *records,
                       uint32_t count,
                       uint32_t kind) {
  require(!count || records, "SBT records are null");
  size_t payload = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const auto &record = records[i];
    require(record.group_index < program.groups.size() &&
                program.kinds[record.group_index] == kind,
            "SBT record references the wrong group kind/index");
    require(!record.data_size || record.data, "SBT payload is null");
    payload = std::max(payload, static_cast<size_t>(record.data_size));
  }
  size_t stride =
      count ? align_record(OPTIX_SBT_RECORD_HEADER_SIZE + payload) : 0;
  require(stride <= UINT32_MAX, "SBT stride exceeds the native ABI");
  return static_cast<uint32_t>(stride);
}

TiForgeOptixResult prepare(TiForgeOptixProgram raw,
                           const TiForgeOptixPrepareLaunchDesc *desc,
                           TiForgeOptixPreparedLaunch *out) {
  return boundary([&] {
    require(out != nullptr, "launch output is null");
    *out = nullptr;
    auto *program = static_cast<Program *>(raw);
    require(program && desc && desc->struct_size >= sizeof(*desc),
            "invalid launch descriptor");
    require(desc->width && desc->height && desc->depth &&
                uint64_t(desc->width) * desc->height <=
                    (uint64_t(1) << 30) / desc->depth,
            "OptiX launch dimensions must be nonzero and fit the launch limit");
    require(desc->parameter_size == program->parameter_size &&
                (!desc->parameter_size || desc->parameters),
            "launch parameter size does not match the program");
    require(!desc->scene_count || desc->scenes,
            "launch scene bindings are null");
    const auto raygen_stride = record_stride(*program, &desc->raygen, 1, 0);
    const auto miss_stride =
        record_stride(*program, desc->miss, desc->miss_count, 1);
    const auto hit_stride =
        record_stride(*program, desc->hit, desc->hit_count, 2);
    auto launch = std::make_unique<PreparedLaunch>();
    launch->program = program;
    program->launches.fetch_add(1);
    launch->scenes.reserve(desc->scene_count);
    for (uint32_t i = 0; i < desc->scene_count; ++i) {
      TiForgeOptixProgramSceneInfo info{};
      info.struct_size = sizeof(info);
      auto result =
          retain_program_scene(program->owner, desc->scenes[i], &info);
      if (result != TI_FORGE_OPTIX_SUCCESS)
        return result;
      launch->scenes.push_back(desc->scenes[i]);
      require(info.max_sbt_offset < desc->hit_count,
              "SBT does not cover the scene's instance record offsets");
      require(!info.has_opacity_micromaps || program->allow_micromaps,
              "program must enable opacity micromaps for this scene");
    }
    const auto advance = [](size_t offset, size_t stride, size_t count) {
      require(
          !stride ||
              count <= (std::numeric_limits<size_t>::max() - offset) / stride,
          "launch data size overflow");
      return offset + stride * count;
    };
    const size_t raygen_offset = align_record(program->parameter_size);
    const size_t miss_offset = advance(raygen_offset, raygen_stride, 1);
    const size_t hit_offset =
        advance(miss_offset, miss_stride, desc->miss_count);
    launch->bytes = advance(hit_offset, hit_stride, desc->hit_count);
    auto result = cuda_check(
        cuMemHostAlloc(&launch->host, launch->bytes, CU_MEMHOSTALLOC_PORTABLE),
        "cuMemHostAlloc(program initialization)");
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    std::memset(launch->host, 0, launch->bytes);
    if (program->parameter_size)
      std::memcpy(launch->host, desc->parameters, program->parameter_size);
    auto pack = [&](const TiForgeOptixSbtRecord *records, uint32_t count,
                    uint32_t stride, size_t offset) {
      for (uint32_t i = 0; i < count; ++i) {
        auto *record =
            static_cast<char *>(launch->host) + offset + size_t(i) * stride;
        auto status =
            optix_check(optixSbtRecordPackHeader(
                            program->groups[records[i].group_index], record),
                        "optixSbtRecordPackHeader(user program)");
        if (status != TI_FORGE_OPTIX_SUCCESS)
          return status;
        if (records[i].data_size)
          std::memcpy(record + OPTIX_SBT_RECORD_HEADER_SIZE, records[i].data,
                      records[i].data_size);
      }
      return TI_FORGE_OPTIX_SUCCESS;
    };
    result = pack(&desc->raygen, 1, raygen_stride, raygen_offset);
    if (result == TI_FORGE_OPTIX_SUCCESS)
      result = pack(desc->miss, desc->miss_count, miss_stride, miss_offset);
    if (result == TI_FORGE_OPTIX_SUCCESS)
      result = pack(desc->hit, desc->hit_count, hit_stride, size_t(hit_offset));
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    result = cuda_check(cuMemAlloc(&launch->device, launch->bytes),
                        "cuMemAlloc(program data)");
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    launch->sbt.raygenRecord = launch->device + raygen_offset;
    launch->sbt.missRecordBase =
        desc->miss_count ? launch->device + miss_offset : 0;
    launch->sbt.missRecordStrideInBytes = miss_stride;
    launch->sbt.missRecordCount = desc->miss_count;
    launch->sbt.hitgroupRecordBase =
        desc->hit_count ? launch->device + hit_offset : 0;
    launch->sbt.hitgroupRecordStrideInBytes = hit_stride;
    launch->sbt.hitgroupRecordCount = desc->hit_count;
    launch->width = desc->width;
    launch->height = desc->height;
    launch->depth = desc->depth;
    *out = launch.release();
    return TI_FORGE_OPTIX_SUCCESS;
  });
}

TiForgeOptixResult initialize(TiForgeOptixPreparedLaunch raw,
                              uint64_t cuda_stream) {
  return boundary([&] {
    auto *launch = static_cast<PreparedLaunch *>(raw);
    require(launch != nullptr, "prepared launch is null");
    auto stream = reinterpret_cast<CUstream>(cuda_stream);
    if (launch->initialized) {
      require(launch->stream == stream,
              "prepared launch must use its initialization stream");
      return TI_FORGE_OPTIX_SUCCESS;
    }
    launch->stream = stream;
    launch->initialization_attempted = true;
    auto result = cuda_check(
        cuMemcpyHtoDAsync(launch->device, launch->host, launch->bytes, stream),
        "cuMemcpyHtoDAsync(program initialization)");
    if (result != TI_FORGE_OPTIX_SUCCESS)
      return result;
    launch->stream = stream;
    launch->initialized = true;
    return TI_FORGE_OPTIX_SUCCESS;
  });
}

TiForgeOptixResult run(TiForgeOptixPreparedLaunch raw, uint64_t cuda_stream) {
  auto *launch = static_cast<PreparedLaunch *>(raw);
  if (!launch || !launch->initialized ||
      launch->stream != reinterpret_cast<CUstream>(cuda_stream)) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "launch requires initialization on the same stream");
  }
  // No pointer resolution, SBT rebuild, parameter copy or scene scan in replay.
  return optix_check(
      optixLaunch(launch->program->pipeline, launch->stream,
                  launch->program->parameter_size ? launch->device : 0,
                  launch->program->parameter_size, &launch->sbt, launch->width,
                  launch->height, launch->depth),
      "optixLaunch(user program)");
}

TiForgeOptixResult destroy_launch(TiForgeOptixPreparedLaunch raw) {
  return boundary([&] {
    auto *launch = static_cast<PreparedLaunch *>(raw);
    require(launch != nullptr, "prepared launch is null");
    if (launch->initialization_attempted) {
      auto result = cuda_check(cuStreamSynchronize(launch->stream),
                               "cuStreamSynchronize(program launch close)");
      if (result != TI_FORGE_OPTIX_SUCCESS)
        return result;
    }
    delete launch;
    return TI_FORGE_OPTIX_SUCCESS;
  });
}

TiForgeOptixResult memory(TiForgeOptixPreparedLaunch raw,
                          TiForgeOptixProgramMemory *out) {
  return boundary([&] {
    auto *launch = static_cast<PreparedLaunch *>(raw);
    require(launch && out && out->struct_size >= sizeof(*out),
            "invalid program memory query");
    out->continuation_stack_bytes = launch->program->continuation_stack;
    out->device_data_bytes = launch->bytes;
    out->pinned_host_bytes = launch->bytes;
    out->parameter_bytes = launch->program->parameter_size;
    out->miss_stride = launch->sbt.missRecordStrideInBytes;
    out->hit_stride = launch->sbt.hitgroupRecordStrideInBytes;
    return TI_FORGE_OPTIX_SUCCESS;
  });
}

}  // namespace

TiForgeOptixResult get_program_api(size_t bytes, TiForgeOptixProgramApi *out) {
  if (!out || bytes < sizeof(TiForgeOptixProgramApi)) {
    return fail(TI_FORGE_OPTIX_ERROR_ABI_MISMATCH,
                "program API table is truncated");
  }
  const TiForgeOptixProgramApi api{sizeof(TiForgeOptixProgramApi),
                                   1,
                                   create,
                                   destroy,
                                   prepare,
                                   initialize,
                                   run,
                                   destroy_launch,
                                   memory,
                                   program_scene_info};
  std::memcpy(out, &api, sizeof(api));
  return TI_FORGE_OPTIX_SUCCESS;
}

}  // namespace taichi::optix_provider
