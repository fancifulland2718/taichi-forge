#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "taichi/ir/type.h"

namespace taichi::lang {
class Ndarray;
class Program;
namespace cuda {

// Cold, private Graph materialization resource. It never replaces the device's
// default pool and has no replay hook. Ndarrays keep ordinary Program
// ownership.
class TI_DLL_EXPORT GraphMemoryPool {
 public:
  GraphMemoryPool(Program &program, std::uint64_t retained_bytes);
  ~GraphMemoryPool();
  GraphMemoryPool(const GraphMemoryPool &) = delete;
  GraphMemoryPool &operator=(const GraphMemoryPool &) = delete;

  static bool available();
  Ndarray *create_ndarray(DataType type, const std::vector<int> &shape);
  // Best-effort release of currently unused backing pages, without waiting.
  void trim();
  void close();
  // Explicit cold observation; closed pools omit unavailable numeric fields.
  std::unordered_map<std::string, std::uint64_t> snapshot();

 private:
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace cuda
}  // namespace taichi::lang
