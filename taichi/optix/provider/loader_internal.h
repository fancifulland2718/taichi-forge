#pragma once

#include <string>

#include "taichi/optix/forge_optix_provider.h"

namespace taichi::optix_provider {

void clear_error_state();
TiForgeOptixResult fail(TiForgeOptixResult result, std::string message);
extern thread_local std::string last_error;
std::size_t get_last_error(char *destination, std::size_t destination_size);

// Shared by legacy contexts and common sessions. All live leases in one
// adapter must use the same requested library path.
TiForgeOptixResult retain_optix_loader(const char *library_path);
void release_optix_loader();
TiForgeOptixResult probe_runtime(const char *library_path);

}  // namespace taichi::optix_provider
