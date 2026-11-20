#include "taichi/runtime/gfx/aot_shader_functions.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>

#include "external/PicoSHA2/picosha2.h"
#include "spirv/unified1/spirv.hpp"
#include "taichi/common/miniz.h"

namespace taichi::lang::gfx {
namespace {
struct SharedBody {
  size_t uses{0};
  std::string digest;
  std::string anchor_digest;
  std::string_view base;
};
struct Function {
  size_t offset;
  std::string_view bytes;
  SharedBody *body;
};

std::string_view as_bytes(const std::vector<uint32_t> &words) {
  return {reinterpret_cast<const char *>(words.data()), words.size() * 4};
}

std::vector<uint32_t> word_differences(std::string_view bytes,
                                       std::string_view base) {
  const auto *words = reinterpret_cast<const uint32_t *>(bytes.data());
  const auto *anchor = reinterpret_cast<const uint32_t *>(base.data());
  std::vector<uint32_t> result(bytes.size() / 4);
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = words[i] - anchor[i];
  return result;
}

size_t deflated_size(std::string_view bytes, std::vector<uint8_t> &scratch) {
  auto capacity = mz_compressBound(static_cast<mz_ulong>(bytes.size()));
  scratch.resize(capacity);
  const auto *data = reinterpret_cast<const unsigned char *>(bytes.data());
  TI_ERROR_IF(
      mz_compress2(scratch.data(), &capacity, data, bytes.size(), 1) != MZ_OK,
      "Cannot size Vulkan shader function");
  return capacity;
}

// Functions emitted into different SPIR-V modules often have identical
// instructions with different local IDs. The root still owns its ABI and
// declarations. Unsigned subtraction also preserves differing literal words.
void select_function_bases(
    std::unordered_map<std::string_view, SharedBody> &bodies) {
  using Candidate = std::pair<const std::string_view, SharedBody>;
  std::map<std::pair<size_t, uint64_t>, std::vector<Candidate *>> groups;
  for (auto &entry : bodies) {
    const auto bytes = entry.first;
    if (bytes.size() < 4096 ||
        bytes.size() > std::numeric_limits<mz_ulong>::max() / 2)
      continue;
    const auto *words = reinterpret_cast<const uint32_t *>(bytes.data());
    uint64_t shape = 14695981039346656037ull;
    for (size_t i = 0; i < bytes.size() / 4; i += words[i] >> 16)
      shape = (shape ^ words[i]) * 1099511628211ull;
    groups[{bytes.size(), shape}].push_back(&entry);
  }
  std::vector<uint8_t> scratch;
  for (auto &[_, group] : groups) {
    if (group.size() < 2)
      continue;
    auto *anchor = *std::min_element(group.begin(), group.end(),
                                     [](const auto *a, const auto *b) {
                                       if (a->second.uses != b->second.uses)
                                         return a->second.uses > b->second.uses;
                                       return a->first < b->first;
                                     });
    for (auto *candidate : group) {
      if (candidate == anchor)
        continue;
      const auto differences =
          word_differences(candidate->first, anchor->first);
      const auto residual = deflated_size(as_bytes(differences), scratch);
      // Include both member headers, extra references and the base reference.
      // When exact duplicates already share one body, count its bytes once.
      if (deflated_size(candidate->first, scratch) <=
          residual + 768 + 128 * candidate->second.uses)
        continue;
      candidate->second.base = anchor->first;
      if (anchor->second.anchor_digest.empty())
        anchor->second.anchor_digest = picosha2::hash256_hex_string(
            anchor->first.begin(), anchor->first.end());
    }
  }
}

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

  select_function_bases(bodies);
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
      if (shared.base.empty() && shared.anchor_digest.empty() &&
          shared.uses - 1 < min_duplicates)
        continue;
      if (refs.empty()) {
        root.exceptions(std::ios::failbit | std::ios::badbit);
        root.open(base / (name + ".spvroot"), std::ios::binary);
      }
      if (shared.digest.empty()) {
        std::vector<uint32_t> differences;
        auto payload = func.bytes;
        if (!shared.base.empty()) {
          differences = word_differences(func.bytes, shared.base);
          payload = as_bytes(differences);
        }
        shared.digest =
            shared.anchor_digest.empty()
                ? picosha2::hash256_hex_string(payload.begin(), payload.end())
                : shared.anchor_digest;
        if (!created_data_dir) {
          std::filesystem::create_directories(base / "data");
          created_data_dir = true;
        }
        write_bytes(base / "data" / (shared.digest + ".bin"), payload);
      }
      root.write(bytes.data() + last, func.offset - last);
      last = func.offset + func.bytes.size();
      refs.push_back(
          {func.offset / 4, func.bytes.size() / 4, shared.digest,
           shared.base.empty() ? "" : bodies.at(shared.base).anchor_digest});
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
