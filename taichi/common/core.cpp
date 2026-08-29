/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/core.h"
#include "taichi/common/version.h"
#include "taichi/common/commit_hash.h"
#include "taichi/common/runtime_contract.h"

#include <spdlog/fmt/fmt.h>
#include <cstdlib>
#include <cstring>
#include "taichi/common/logging.h"

#if defined(TI_PLATFORM_WINDOWS)
#include "taichi/platform/windows/windows.h"
#else
// Mac and Linux
#include <unistd.h>
#endif

namespace taichi {

std::string python_package_dir;

std::string get_python_package_dir() {
  return python_package_dir;
}

void set_python_package_dir(const std::string &dir) {
  python_package_dir = dir;
}

std::string get_repo_dir() {
#if defined(TI_PLATFORM_WINDOWS)
  return "C:/taichi_cache/";
#elif defined(TI_PLATFORM_ANDROID)
  // @FIXME: Not supported on Android. A possibility would be to return the
  // application cache directory. This feature is not used yet on this OS so
  // it should not break anything (yet!)
  return "";
#else
  auto xdg_cache = std::getenv("XDG_CACHE_HOME");

  std::string xdg_cache_str;
  if (xdg_cache != nullptr) {
    xdg_cache_str = xdg_cache;
  } else {
    // XDG_CACHE_HOME is not defined, defaulting to ~/.cache
    auto home = std::getenv("HOME");
    TI_ASSERT(home != nullptr);
    xdg_cache_str = home;
    xdg_cache_str += "/.cache";
  }

  return xdg_cache_str + "/taichi/";
#endif
}

CoreState &CoreState::get_instance() {
  static CoreState state;
  return state;
}

int __trash__;

std::string get_version_string() {
  return fmt::format("{}.{}.{}", get_version_major(), get_version_minor(),
                     get_version_patch());
}

int get_version_major() {
  return TI_VERSION_MAJOR;
}

int get_version_minor() {
  return TI_VERSION_MINOR;
}

int get_version_patch() {
  return TI_VERSION_PATCH;
}

std::string get_commit_hash() {
  return TI_COMMIT_HASH;
}

int get_forge_native_abi_revision() {
  return kForgeNativeAbiRevision;
}

int get_forge_contract_manifest_schema_version() {
  return kForgeContractManifestSchemaVersion;
}

std::uint32_t get_forge_runtime_statistics_schema_version() {
  return kForgeRuntimeStatisticsSchemaVersion;
}

std::uint64_t get_forge_native_feature_bitmap() {
  std::uint64_t result = kForgeFeatureCpu;
#if defined(TI_WITH_LLVM)
  result |= kForgeFeatureLlvm;
#endif
#if defined(TI_WITH_CUDA)
  result |= kForgeFeatureCuda;
#endif
#if defined(TI_WITH_VULKAN)
  result |= kForgeFeatureVulkan;
#endif
#if defined(TI_WITH_GGUI)
  result |= kForgeFeatureGgui;
#endif
#if defined(TI_WITH_OPENGL)
  result |= kForgeFeatureOpenGl;
#endif
  return result;
}

std::string get_forge_native_compiler_abi() {
  return forge_compiler_abi_identity();
}

extern "C" int taichi_forge_runtime_bootstrap_v1(
    ForgeRuntimeBootstrapV1 *output,
    std::uint32_t output_size) {
  if (output == nullptr || output_size < sizeof(ForgeRuntimeBootstrapV1)) {
    return 1;
  }
  const auto compiler_abi = forge_compiler_abi_identity();
  if (compiler_abi.size() >= sizeof(output->compiler_abi)) {
    return 2;
  }
  *output = {};
  output->struct_size = sizeof(ForgeRuntimeBootstrapV1);
  output->manifest_schema_version = kForgeContractManifestSchemaVersion;
  output->native_abi_revision = kForgeNativeAbiRevision;
  output->runtime_statistics_schema_version =
      kForgeRuntimeStatisticsSchemaVersion;
  output->feature_bitmap = get_forge_native_feature_bitmap();
  std::memcpy(output->compiler_abi, compiler_abi.c_str(),
              compiler_abi.size() + 1);
  return 0;
}

std::string get_cuda_version_string() {
  return TI_CUDA_LIBDEVICE_VERSION;
}

int PID::get_pid() {
#if defined(TI_PLATFORM_WINDOWS)
  return (int)GetCurrentProcessId();
#else
  return (int)getpid();
#endif
}

int PID::get_parent_pid() {
#if defined(TI_PLATFORM_WINDOWS)
  TI_NOT_IMPLEMENTED
  return -1;
#else
  return (int)getppid();
#endif
}

}  // namespace taichi
