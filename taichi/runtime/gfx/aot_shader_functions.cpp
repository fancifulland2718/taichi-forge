#include "taichi/runtime/gfx/aot_shader_functions.h"

#include <filesystem>
#include <fstream>

#include "external/PicoSHA2/picosha2.h"
#include "spirv/unified1/spirv.hpp"

namespace taichi::lang::gfx {
namespace {
struct SharedBody {
  size_t uses{0};
  std::string digest;
};
struct Function {
  size_t offset;
  std::string_view bytes;
  SharedBody *body;
};

void write_bytes(const std::filesystem::path &path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.write(bytes.data(), bytes.size());
  out.close();
}
}  // namespace

ShaderFunctionRefs write_aot_shaders(const std::string &directory,
                                     const StoredShaders &shaders,
                                     bool share_functions) {
  std::filesystem::path base(directory);
  std::unordered_map<std::string_view, SharedBody> bodies;
  std::unordered_map<std::string_view, std::vector<Function>> functions;
  if (!share_functions) {
    for (const auto &[bytes, name] : shaders)
      write_bytes(base / (name + ".spv"), bytes);
    return {};
  }
  for (const auto &[bytes, name] : shaders) {
    auto &regions = functions[name];
    const auto *words = reinterpret_cast<const uint32_t *>(bytes.data());
    const size_t size = bytes.size() / sizeof(uint32_t);
    size_t start = 0;
    for (size_t i = 5; i < size;) {
      const size_t count = words[i] >> 16;
      const uint32_t op = words[i] & 0xffff;
      TI_ERROR_IF(!count || count > size - i,
                  "Invalid SPIR-V instruction in {}", name);
      if (op == spv::OpFunction) {
        start = i;
      } else if (op == spv::OpFunctionEnd) {
        TI_ERROR_IF(!start, "Unmatched SPIR-V function end in {}", name);
        const auto body = bytes.substr(start * 4, (i + count - start) * 4);
        // A separate ZIP entry, digest and references outweigh tiny bodies.
        if (body.size() >= 256) {
          auto &shared = bodies[body];
          ++shared.uses;
          regions.push_back({start * 4, body, &shared});
        }
        start = 0;
      }
      i += count;
    }
  }

  ShaderFunctionRefs result;
  bool created_data_dir = false;
  for (const auto &[bytes, name] : shaders) {
    size_t last = 0;
    std::vector<AotShaderFunction> refs;
    std::ofstream root;
    for (const auto &func : functions[name]) {
      auto &shared = *func.body;
      // For short functions a ZIP member and digest references can cost
      // more than the second compressed copy. Keep those bytes in the root
      // unless enough occurrences share the cost.
      const size_t min_duplicates = 1 + (1024 - 1) / func.bytes.size();
      if (shared.uses - 1 < min_duplicates)
        continue;
      if (refs.empty()) {
        root.exceptions(std::ios::failbit | std::ios::badbit);
        root.open(base / (name + ".spvroot"), std::ios::binary);
      }
      if (shared.digest.empty()) {
        shared.digest =
            picosha2::hash256_hex_string(func.bytes.begin(), func.bytes.end());
        if (!created_data_dir) {
          std::filesystem::create_directories(base / "data");
          created_data_dir = true;
        }
        write_bytes(base / "data" / (shared.digest + ".bin"), func.bytes);
      }
      root.write(bytes.data() + last, func.offset - last);
      last = func.offset + func.bytes.size();
      refs.push_back({func.offset / 4, func.bytes.size() / 4, shared.digest});
    }
    if (refs.empty()) {
      write_bytes(base / (name + ".spv"), bytes);
    } else {
      root.write(bytes.data() + last, bytes.size() - last);
      root.close();
      result.emplace(name, std::move(refs));
    }
  }
  return result;
}
}  // namespace taichi::lang::gfx
