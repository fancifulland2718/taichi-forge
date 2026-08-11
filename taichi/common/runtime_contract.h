#pragma once

#include <cstdint>
#include <string>

namespace taichi {

// This header is compiled into both the Python shim and the native runtime.
// Keep the values POD-like: they are the bootstrap contract used before any
// backend or Program instance exists.
inline constexpr int kForgeContractManifestSchemaVersion = 1;
inline constexpr int kForgeNativeAbiRevision = 1;
inline constexpr std::uint32_t kForgeRuntimeStatisticsSchemaVersion = 3;

inline constexpr std::uint64_t kForgeFeatureCpu = 1ull << 0;
inline constexpr std::uint64_t kForgeFeatureLlvm = 1ull << 1;
inline constexpr std::uint64_t kForgeFeatureCuda = 1ull << 2;
inline constexpr std::uint64_t kForgeFeatureVulkan = 1ull << 3;
inline constexpr std::uint64_t kForgeFeatureGgui = 1ull << 4;
inline constexpr std::uint64_t kForgeFeatureOpenGl = 1ull << 5;

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
