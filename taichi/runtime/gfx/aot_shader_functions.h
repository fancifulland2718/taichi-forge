#pragma once

#include <string_view>
#include <unordered_map>

#include "taichi/runtime/gfx/aot_utils.h"

namespace taichi::lang::gfx {

// Input spans borrow the builder's immutable word vectors through dump().
using StoredShaders = std::unordered_map<std::string_view, std::string>;
using ShaderFunctionRefs = std::map<std::string, std::vector<AotShaderFunction>>;

ShaderFunctionRefs write_aot_shaders(const std::string &directory,
                                     const StoredShaders &shaders,
                                     bool share_functions);

}  // namespace taichi::lang::gfx
