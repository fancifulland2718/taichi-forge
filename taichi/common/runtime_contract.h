#pragma once

#include <cstdint>
#include <string>

namespace taichi {

// This header is compiled into both the Python shim and the native runtime.
// Keep the values POD-like: they are the bootstrap contract used before any
// backend or Program instance exists.
inline constexpr int kForgeContractManifestSchemaVersion = 1;
inline constexpr int kForgeNativeAbiRevision = 2;
inline constexpr std::uint32_t kForgeRuntimeStatisticsSchemaVersion = 3;

inline constexpr std::uint64_t kForgeFeatureCpu = 1ull << 0;
inline constexpr std::uint64_t kForgeFeatureLlvm = 1ull << 1;
inline constexpr std::uint64_t kForgeFeatureCuda = 1ull << 2;
inline constexpr std::uint64_t kForgeFeatureVulkan = 1ull << 3;
inline constexpr std::uint64_t kForgeFeatureGgui = 1ull << 4;
inline constexpr std::uint64_t kForgeFeatureOpenGl = 1ull << 5;
// Additive native build facts, not new bootstrap fields or ABI requirements.
// The presence bit distinguishes old runtimes from a known portable build.
inline constexpr std::uint64_t kForgeBuildProfileKnown = 1ull << 6;
inline constexpr std::uint64_t kForgeBuildToolkitReference = 1ull << 7;
inline constexpr std::uint64_t kForgeBuildCupti = 1ull << 8;

// Stable C bootstrap used before the pybind module touches any C++ object
// crossing the split-runtime boundary.  Never extend this v1 struct in place;
// add a new bootstrap function if the bootstrap representation itself changes.
struct ForgeRuntimeBootstrapV1 {
  std::uint32_t struct_size{0};
  std::uint32_t manifest_schema_version{0};
  std::int32_t native_abi_revision{0};
  std::uint32_t runtime_statistics_schema_version{0};
  std::uint64_t feature_bitmap{0};
  char compiler_abi[96]{};
};

extern "C" int taichi_forge_runtime_bootstrap_v1(
    ForgeRuntimeBootstrapV1 *output,
    std::uint32_t output_size);

#define TI_FORGE_STRINGIFY_IMPL(x) #x
#define TI_FORGE_STRINGIFY(x) TI_FORGE_STRINGIFY_IMPL(x)

inline std::string forge_compiler_abi_identity() {
#if defined(_MSC_VER)
#if defined(_ITERATOR_DEBUG_LEVEL)
  return "msvc-" TI_FORGE_STRINGIFY(_MSC_VER) "-iterator-" TI_FORGE_STRINGIFY(
      _ITERATOR_DEBUG_LEVEL);
#else
  return "msvc-" TI_FORGE_STRINGIFY(_MSC_VER) "-iterator-default";
#endif
#elif defined(__clang__)
#if defined(_GLIBCXX_USE_CXX11_ABI)
  return "clang-" TI_FORGE_STRINGIFY(
      __clang_major__) "-libstdcxx-cxx11abi-" TI_FORGE_STRINGIFY(_GLIBCXX_USE_CXX11_ABI);
#else
  return "clang-" TI_FORGE_STRINGIFY(__clang_major__) "-default-cxxabi";
#endif
#elif defined(__GNUC__)
#if defined(_GLIBCXX_USE_CXX11_ABI)
  return "gcc-" TI_FORGE_STRINGIFY(
      __GNUC__) "-libstdcxx-cxx11abi-" TI_FORGE_STRINGIFY(_GLIBCXX_USE_CXX11_ABI);
#else
  return "gcc-" TI_FORGE_STRINGIFY(__GNUC__) "-default-cxxabi";
#endif
#else
  return "unknown-cxxabi";
#endif
}

#undef TI_FORGE_STRINGIFY
#undef TI_FORGE_STRINGIFY_IMPL

}  // namespace taichi
