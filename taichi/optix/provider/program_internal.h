#pragma once

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <string>
#include <cuda.h>
#include <optix.h>
#include "taichi/optix/forge_optix_program.h"

namespace taichi::optix_provider {

// Narrow views of existing owners; the programmable implementation does not
// gain access to the fixed pipeline variants or create a second AS/context.
struct ProgramContextView {
  CUcontext cuda_context{};
  OptixDeviceContext optix_context{};
};

void clear_error_state();
TiForgeOptixResult fail(TiForgeOptixResult result, std::string message);
TiForgeOptixResult cuda_check(CUresult result, const char *operation);
TiForgeOptixResult optix_check(OptixResult result, const char *operation);
extern thread_local std::string last_error;

TiForgeOptixResult retain_program_context(TiForgeOptixContext,
                                          ProgramContextView *);
void release_program_context(TiForgeOptixContext);
TiForgeOptixResult program_scene_info(TiForgeOptixContext,
                                      const TiForgeOptixProgramScene *,
                                      TiForgeOptixProgramSceneInfo *);
TiForgeOptixResult retain_program_scene(TiForgeOptixContext,
                                        const TiForgeOptixProgramScene &,
                                        TiForgeOptixProgramSceneInfo *);
void release_program_scene(const TiForgeOptixProgramScene &);
TiForgeOptixResult get_program_api(size_t, TiForgeOptixProgramApi *);

}  // namespace taichi::optix_provider
