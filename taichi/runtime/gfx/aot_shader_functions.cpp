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
  size_t relocation_bytes{0};
};
using BodyMap = std::unordered_map<std::string_view, SharedBody>;

struct Region {
  size_t offset;
  std::string_view bytes;
  SharedBody *body;
  bool fragment{false};

  bool shared() const {
    const size_t duplicates = 1 + 1023 / bytes.size();
    return !body->base.empty() || !body->anchor_digest.empty() ||
           body->uses - 1 >= duplicates;
  }
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

std::string sparse_differences(const std::vector<uint32_t> &words) {
  std::string result((words.size() + 7) / 8, '\0');
  for (size_t i = 0; i < words.size(); ++i) {
    if (words[i]) {
      result[i / 8] |= static_cast<char>(1u << (i % 8));
      result.append(reinterpret_cast<const char *>(&words[i]), sizeof(uint32_t));
    }
  }
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
void select_function_bases(BodyMap &bodies) {
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
      auto residual = deflated_size(as_bytes(differences), scratch);
      size_t relocation_bytes = 0;
      const auto sparse = sparse_differences(differences);
      if (sparse.size() < candidate->first.size()) {
        const size_t sparse_cost = deflated_size(sparse, scratch) + 32;
        if (sparse_cost < residual) {
          residual = sparse_cost;
          relocation_bytes = sparse.size();
        }
      }
      // Include both member headers, extra references and the base reference.
      // When exact duplicates already share one body, count its bytes once.
      if (deflated_size(candidate->first, scratch) <=
          residual + 768 + 128 * candidate->second.uses)
        continue;
      candidate->second.base = anchor->first;
      candidate->second.relocation_bytes = relocation_bytes;
      if (anchor->second.anchor_digest.empty())
        anchor->second.anchor_digest = picosha2::hash256_hex_string(
            anchor->first.begin(), anchor->first.end());
    }
  }
}

// Cut only the gaps between already shared whole functions. Boundaries use
// recent opcodes and instruction lengths, so a changed local ID or an edit
// earlier in a shader does not shift every later fragment.
void collect_fragments(std::string_view shader,
                       size_t begin,
                       size_t end,
                       BodyMap &bodies,
                       std::vector<Region> &regions) {
  constexpr size_t min_bytes = 4096;
  if (end - begin < min_bytes)
    return;
  auto emit = [&](size_t stop) {
    if (stop - begin >= min_bytes) {
      const auto bytes = shader.substr(begin, stop - begin);
      auto &body = bodies[bytes];
      ++body.uses;
      regions.push_back({begin, bytes, &body, true});
    }
    begin = stop;
  };
  const auto *words = reinterpret_cast<const uint32_t *>(shader.data());
  uint64_t rolling = 0;
  for (size_t pos = begin; pos < end;) {
    const uint32_t header = words[pos / 4];
    const uint32_t opcode = header & 0xffff;
    if (opcode == spv::OpFunction)
      emit(pos);
    uint64_t mixed = header * 0x9e3779b97f4a7c15ull;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ull;
    rolling = (rolling << 1) + (mixed ^ (mixed >> 27));
    pos += size_t(header >> 16) * 4;
    if (opcode == spv::OpFunctionEnd || pos - begin >= 32768 ||
        (pos - begin >= min_bytes && (rolling & 511) == 0))
      emit(pos);
  }
  emit(end);
}

void write_bytes(const std::filesystem::path &path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.write(bytes.data(), bytes.size());
  out.close();
}
}  // namespace

uint32_t write_aot_shaders(const std::string &directory,
                           const StoredShaders &shaders,
                           bool share_functions,
                           ShaderFunctionRefs &result) {
  std::filesystem::path base(directory);
  BodyMap bodies;
  std::unordered_map<std::string_view, std::vector<Region>> functions;
  if (!share_functions) {
    for (const auto &[bytes, name] : shaders)
      write_bytes(base / (name + ".spv"), bytes);
    return 0;
  }
  for (const auto &[bytes, name] : shaders) {
    auto &regions = functions[name];
    TI_ERROR_IF(bytes.size() < 5 * sizeof(uint32_t) ||
                    bytes.size() % sizeof(uint32_t),
                "Invalid SPIR-V module length in {}", name);
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
  BodyMap fragments;
  for (const auto &[bytes, name] : shaders) {
    auto &regions = functions.at(name);
    std::vector<Region> retained;
    size_t cursor = 5 * sizeof(uint32_t);
    for (const auto &region : regions) {
      if (region.shared()) {
        collect_fragments(bytes, cursor, region.offset, fragments, retained);
        retained.push_back(region);
        cursor = region.offset + region.bytes.size();
      }
    }
    collect_fragments(bytes, cursor, bytes.size(), fragments, retained);
    regions.swap(retained);
  }
  select_function_bases(fragments);
  uint32_t version = 0;
  bool created_data_dir = false;
  for (const auto &[bytes, name] : shaders) {
    size_t last = 0;
    std::vector<AotShaderFunction> refs;
    std::ofstream root;
    for (const auto &func : functions[name]) {
      auto &shared = *func.body;
      if (!func.shared())
        continue;
      const uint32_t function_version = shared.base.empty() ? 21 : 22;
      version = std::max(version, func.fragment ? 23u : function_version);
      if (refs.empty()) {
        root.exceptions(std::ios::failbit | std::ios::badbit);
        root.open(base / (name + ".spvroot"), std::ios::binary);
      }
      root.write(bytes.data() + last, func.offset - last);
      const bool local_relocation = shared.uses == 1 && !shared.base.empty();
      if (shared.digest.empty()) {
        std::vector<uint32_t> differences;
        std::string sparse;
        auto payload = func.bytes;
        if (!shared.base.empty()) {
          differences = word_differences(func.bytes, shared.base);
          payload = as_bytes(differences);
          if (shared.relocation_bytes) {
            sparse = sparse_differences(differences);
            payload = sparse;
          }
        }
        shared.digest =
            shared.anchor_digest.empty()
                ? picosha2::hash256_hex_string(payload.begin(), payload.end())
                : shared.anchor_digest;
        if (local_relocation) {
          root.write(payload.data(), payload.size());
          const uint32_t zero = 0;
          root.write(reinterpret_cast<const char *>(&zero), (4 - payload.size() % 4) % 4);
        } else {
          if (!created_data_dir) {
            std::filesystem::create_directories(base / "data");
            created_data_dir = true;
          }
          write_bytes(base / "data" / (shared.digest + ".bin"), payload);
        }
      }
      last = func.offset + func.bytes.size();
      if (local_relocation || shared.relocation_bytes)
        version = std::max(version, 24u);
      refs.push_back({func.offset / 4, func.bytes.size() / 4, shared.digest,
                      shared.base.empty() ? ""
                                          : (func.fragment ? fragments : bodies)
                                                .at(shared.base)
                                                .anchor_digest,
                      shared.relocation_bytes, local_relocation});
    }
    if (refs.empty()) {
      write_bytes(base / (name + ".spv"), bytes);
    } else {
      root.write(bytes.data() + last, bytes.size() - last);
      root.close();
      result.emplace(name, std::move(refs));
    }
  }
  return version;
}
}  // namespace taichi::lang::gfx
