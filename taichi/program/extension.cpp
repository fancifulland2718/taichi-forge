#include "extension.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace taichi::lang {

namespace {
// Phase 1c-D (taichi-forge 0.3.x): experimental opt-in for sparse SNode on
// Vulkan. Primary source of truth is
// CompileConfig::vulkan_sparse_experimental, propagated here by Program ctor
// via set_vulkan_sparse_experimental() before any extension query runs. The
// legacy env var TI_VULKAN_SPARSE=1 is kept as a compatibility fallback so
// existing scripts and CI invocations continue to work unchanged.
//
// is_extension_supported() is also called from paths without direct Program
// config access, so the effective value is process-global while a Program is
// alive. Each Program construction must nevertheless replace the previous
// config value: otherwise ti.reset(); ti.init(..., False) cannot honor the
// documented opt-out after a default-enabled Vulkan Program. The legacy env
// var remains a process-level force-enable compatibility path.
bool legacy_vulkan_sparse_enabled() {
  static const bool enabled = []() {
    const char *v = std::getenv("TI_VULKAN_SPARSE");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

bool &vulkan_sparse_flag() {
  static bool flag = legacy_vulkan_sparse_enabled();
  return flag;
}

bool vulkan_sparse_experimental_enabled() {
  return vulkan_sparse_flag();
}

// G9.1 (taichi-forge 0.3.0): mirrors the sparse flag for quant_array /
// bit_struct on Vulkan. The frontend extension gate gates only whether
// is_extension_supported(Arch::vulkan, Extension::quant{,_basic}) returns
// true. Whether codegen actually succeeds is a separate question handled
// inside spirv_codegen.cpp (incremental TI_NOT_IMPLEMENTED).
bool &vulkan_quant_flag() {
  static bool flag = []() {
    const char *v = std::getenv("TI_VULKAN_QUANT");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return flag;
}

bool vulkan_quant_experimental_enabled() {
  return vulkan_quant_flag();
}
}  // namespace

void set_vulkan_sparse_experimental(bool enabled) {
  // Only one Program may exist at a time. Replacing the prior Program's value
  // is therefore safe and makes the explicit opt-out reliable across reset.
  // TI_VULKAN_SPARSE=1 intentionally remains a force-enable compatibility
  // path for existing scripts.
  vulkan_sparse_flag() = enabled || legacy_vulkan_sparse_enabled();
}

void set_vulkan_quant_experimental(bool enabled) {
  // OR-set, same rationale as the sparse flag.
  if (enabled) {
    vulkan_quant_flag() = true;
  }
}

bool is_extension_supported(Arch arch, Extension ext) {
  static std::unordered_map<Arch, std::unordered_set<Extension>> arch2ext = {
      {Arch::x64,
       {Extension::sparse, Extension::quant, Extension::quant_basic,
        Extension::data64, Extension::adstack, Extension::assertion,
        Extension::extfunc, Extension::mesh}},
      {Arch::arm64,
       {Extension::sparse, Extension::quant, Extension::quant_basic,
        Extension::data64, Extension::adstack, Extension::assertion,
        Extension::mesh}},
      {Arch::cuda,
       {Extension::sparse, Extension::quant, Extension::quant_basic,
        Extension::data64, Extension::adstack, Extension::bls,
        Extension::assertion, Extension::mesh}},
      {Arch::amdgpu, {Extension::assertion}},
      {Arch::metal, {}},
      {Arch::opengl, {Extension::extfunc}},
      {Arch::gles, {}},
      {Arch::vulkan, {}},
      {Arch::dx11, {}},
  };
  // if (with_opengl_extension_data64())
  // arch2ext[Arch::opengl].insert(Extension::data64); // TODO: singleton
  if (arch == Arch::vulkan && ext == Extension::sparse &&
      vulkan_sparse_experimental_enabled()) {
    return true;
  }
  if (arch == Arch::vulkan &&
      (ext == Extension::quant || ext == Extension::quant_basic) &&
      vulkan_quant_experimental_enabled()) {
    return true;
  }
  const auto &exts = arch2ext[arch];
  return exts.find(ext) != exts.end();
}

}  // namespace taichi::lang
