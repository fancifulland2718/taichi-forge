#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "taichi/ir/type_utils.h"

namespace taichi::lang::aot {

enum class ValueStorage { Host, Gfx };

// A leaf in a compiled value buffer. Padding comes exclusively from the
// compiled type; lowered tensors can have explicitly spaced component slots.
struct ValueLayoutMember {
  std::string path;
  std::vector<int> indices;
  DataType scalar_type;
  std::vector<int> shape;
  std::size_t offset;
  std::size_t size;
  std::size_t scalar_size;
  std::size_t scalar_alignment;
  // Empty for contiguous scalar slots; otherwise one byte offset per component.
  std::vector<std::size_t> component_offsets;
};

// Cold reflection of an already compiled argument/return buffer. This does not
// calculate a new ABI or own a runtime allocation. Explicit member offsets are
// authoritative; natural structures use the existing native layout helpers.
// buffer_size is the compiler's size, including target-specific trailing padding.
struct ValueLayout {
  DataType type;
  std::size_t buffer_size;
  std::vector<ValueLayoutMember> members;

  ValueLayout(DataType type, std::size_t buffer_size, ValueStorage storage);
  ValueLayout(DataType logical, DataType physical, std::size_t buffer_size,
              ValueStorage storage);
};

}  // namespace taichi::lang::aot
