#pragma once

#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "taichi/common/operation_json.h"

namespace taichi::lang {

// Cold, backend-independent metadata shared by source-addon factories. No
// library loads, device calls or compilation occur while decoding an artifact.
struct NativeSourceAddonIdentity {
  std::string binary_sha256;
  uint32_t minimum_driver{0}, ptx_driver{0};
  std::vector<std::pair<bool, uint32_t>> targets;  // SASS or PTX, capability.
};

inline uint32_t source_addon_u32(const liong::json::JsonValue &p,
                                 const char *key) {
  const auto value = operation_json::integer(p[key]);
  if (value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Source addon version exceeds its ABI");
  return uint32_t(value);
}

inline NativeSourceAddonIdentity decode_source_addon_identity(
    const liong::json::JsonValue &p) {
  using operation_json::name;
  NativeSourceAddonIdentity d;
  d.minimum_driver = source_addon_u32(p, "minimum_driver_api_version");
  d.ptx_driver = source_addon_u32(p, "ptx_driver_api_version");
  d.binary_sha256 = name(p["binary_sha256"]);
  if (d.binary_sha256.size() != 64 ||
      d.binary_sha256.find_first_not_of("0123456789abcdef") !=
          std::string::npos)
    throw std::invalid_argument("Source addon frozen identity is invalid");
  const auto &targets = p["target_code"];
  if (!targets.is_arr() || targets.arr.inner.empty())
    throw std::invalid_argument("Source addon requires target code");
  for (const auto &entry : targets.arr.inner) {
    const auto code = name(entry);
    const bool sass = code.rfind("sm_", 0) == 0;
    if (!sass && code.rfind("compute_", 0) != 0)
      throw std::invalid_argument("Invalid source addon target code");
    uint64_t capability = 0;
    for (char c : code.substr(sass ? 3 : 8)) {
      if (c < '0' || c > '9')
        throw std::invalid_argument("Invalid source addon target capability");
      capability = capability * 10 + c - '0';
      if (capability > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Source addon target capability overflows");
    }
    if (!capability)
      throw std::invalid_argument(
          "Source addon target capability is empty or zero");
    d.targets.emplace_back(sass, uint32_t(capability));
  }
  return d;
}

inline void validate_source_addon_target(const NativeSourceAddonIdentity &d,
                                         uint32_t capability,
                                         uint32_t driver_version) {
  bool sass = false, ptx = false;
  for (auto [is_sass, target] : d.targets) {
    sass |= is_sass && target / 10 == capability / 10 && target <= capability;
    ptx |= !is_sass && target <= capability;
  }
  if ((!sass && !ptx) ||
      driver_version < (sass ? d.minimum_driver : d.ptx_driver))
    throw std::invalid_argument(
        "Source addon target code or driver is incompatible");
}

}  // namespace taichi::lang
