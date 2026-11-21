#pragma once

#include "taichi/common/virtual_dir.h"

namespace taichi::lang::aot {

// A .b4 member stores byte lanes 0..3 of successive four-byte words. Its
// logical length is unchanged, including trailing bytes of sparse relocations.
struct WordPlaneFile {
  std::string name;
  size_t size{0};
  bool packed{false};

  bool locate(const io::VirtualDir &dir) {
    if (dir.get_file_size(name, size))
      return true;
    name += ".b4";
    packed = true;
    return dir.get_file_size(name, size);
  }

  template <typename T>
  bool read(const io::VirtualDir &dir, std::vector<T> &output) const {
    if (size % sizeof(T))
      return false;
    output.resize(size / sizeof(T));
    if (!packed)
      return dir.load_file(name, output.data(), size) == size;
    std::vector<uint8_t> lanes(size);
    if (dir.load_file(name, lanes.data(), size) != size)
      return false;
    auto *dst = reinterpret_cast<uint8_t *>(output.data());
    const size_t words = size / 4;
    const size_t tail = size % 4;
    const size_t lane1 = words + (tail > 0);
    const size_t lane2 = lane1 + words + (tail > 1);
    const size_t lane3 = lane2 + words + (tail > 2);
    for (size_t i = 0; i < words; ++i) {
      dst[4 * i] = lanes[i];
      dst[4 * i + 1] = lanes[lane1 + i];
      dst[4 * i + 2] = lanes[lane2 + i];
      dst[4 * i + 3] = lanes[lane3 + i];
    }
    const size_t starts[] = {0, lane1, lane2};
    for (size_t i = 0; i < tail; ++i)
      dst[4 * words + i] = lanes[starts[i] + words];
    return true;
  }
};

inline bool load_shader_words(const io::VirtualDir &dir,
                              const std::string &name,
                              std::vector<uint32_t> &words) {
  WordPlaneFile file{name};
  return file.locate(dir) && file.read(dir, words);
}

}  // namespace taichi::lang::aot
