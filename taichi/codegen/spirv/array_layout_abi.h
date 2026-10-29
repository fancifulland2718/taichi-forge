#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "taichi/common/json.h"
#include "spirv/unified1/spirv.hpp"

namespace taichi::lang::spirv {
inline constexpr char kRuntimeAffineArray[] =
    "taichi.ndarray.runtime_affine.v1:";

// Compiler declarations live in the actual shader as well as its metadata.
// OpSourceExtension is a non-semantic string in every supported SPIR-V version.
// Insert after optimization so ordinary debug stripping cannot lose the ABI.
inline void declare_runtime_affine_arrays(
    std::vector<uint32_t> &code,
    const std::vector<std::vector<int>> &arguments) {
  if (arguments.empty())
    return;
  std::size_t insertion = 5;
  while (insertion < code.size()) {
    const auto opcode = static_cast<spv::Op>(code[insertion] & 0xffff);
    if (opcode != spv::OpCapability && opcode != spv::OpExtension &&
        opcode != spv::OpExtInstImport && opcode != spv::OpMemoryModel &&
        opcode != spv::OpEntryPoint && opcode != spv::OpExecutionMode &&
        opcode != spv::OpExecutionModeId)
      break;
    const auto words = code[insertion] >> 16;
    if (!words || words > code.size() - insertion)
      throw std::invalid_argument("Invalid SPIR-V instruction");
    insertion += words;
  }
  std::vector<uint32_t> declarations;
  for (const auto &argument : arguments) {
    const auto text = std::string(kRuntimeAffineArray) +
                      liong::json::print(liong::json::serialize(argument));
    const auto words = 1 + (text.size() + 1 + 3) / 4;
    if (words > std::numeric_limits<uint16_t>::max())
      throw std::invalid_argument(
          "Affine argument path exceeds SPIR-V instruction encoding");
    const auto begin = declarations.size();
    declarations.resize(begin + words, 0);
    declarations[begin] =
        (static_cast<uint32_t>(words) << 16) | spv::OpSourceExtension;
    std::memcpy(declarations.data() + begin + 1, text.c_str(), text.size());
  }
  code.insert(code.begin() + insertion, declarations.begin(),
              declarations.end());
}

inline std::vector<std::vector<int>> runtime_affine_arrays(
    const std::vector<uint32_t> &code) {
  std::vector<std::vector<int>> result;
  for (std::size_t pos = 5; pos < code.size();) {
    const auto words = code[pos] >> 16;
    if (!words || words > code.size() - pos)
      throw std::invalid_argument("Invalid SPIR-V instruction");
    if ((code[pos] & 0xffff) == spv::OpSourceExtension) {
      const auto *text = reinterpret_cast<const char *>(code.data() + pos + 1);
      const auto *end = static_cast<const char *>(
          std::memchr(text, 0, (words - 1) * sizeof(uint32_t)));
      if (!end)
        throw std::invalid_argument("Invalid SPIR-V source annotation");
      const std::string value(text, end);
      if (value.compare(0, sizeof(kRuntimeAffineArray) - 1,
                        kRuntimeAffineArray) == 0) {
        std::vector<int> argument;
        liong::json::deserialize(liong::json::parse_complete(value.substr(
                                     sizeof(kRuntimeAffineArray) - 1)),
                                 argument);
        if (argument.empty() ||
            std::any_of(argument.begin(), argument.end(),
                        [](int index) { return index < 0; }))
          throw std::invalid_argument("Invalid compiled affine argument path");
        result.push_back(std::move(argument));
      }
    }
    pos += words;
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
}  // namespace taichi::lang::spirv
