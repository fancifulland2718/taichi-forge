#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "taichi/common/core.h"

namespace taichi::lang {

// Public member identity is independent of generated argument names and of
// resource members lifted out of a pack into root launch parameters.
struct ArgumentOrigin {
  std::vector<int> logical;
  std::vector<int> physical;
  std::string name;
  std::string kind;
  TI_IO_DEF(logical, physical, name, kind);

  auto identity() const {
    return std::tie(logical, physical, name, kind);
  }
  bool operator==(const ArgumentOrigin &other) const {
    return identity() == other.identity();
  }
  bool operator<(const ArgumentOrigin &other) const {
    return identity() < other.identity();
  }
};
using ArgumentOrigins = std::vector<ArgumentOrigin>;

}  // namespace taichi::lang
