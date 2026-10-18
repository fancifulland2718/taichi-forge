#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "taichi/ir/type.h"

namespace taichi::lang {
class Ndarray;
class Program;
struct DeviceAllocation;
namespace cuda {
class CudaDevice;

// Cold, private Graph materialization resource. It never replaces the device's
// default pool and has no replay hook. Ndarrays keep ordinary Program
// ownership.
class TI_DLL_EXPORT GraphMemoryPool {
 public:
  GraphMemoryPool(Program &program, std::uint64_t retained_bytes);
  // Standalone consumers use the same device-owned retirement contract,
  // without constructing a Python/JIT Program. The device must outlive calls.
  GraphMemoryPool(CudaDevice &device, std::uint64_t retained_bytes);
  ~GraphMemoryPool();
  GraphMemoryPool(const GraphMemoryPool &) = delete;
  GraphMemoryPool &operator=(const GraphMemoryPool &) = delete;

  static bool available();
  Ndarray *create_ndarray(DataType type, const std::vector<int> &shape);
  // Caller owns the ordinary device allocation. Its allocation record retains
  // the pool even after this factory closes, including queued asynchronous frees.
  DeviceAllocation allocate_memory(std::size_t bytes);
  // Best-effort release of currently unused backing pages, without waiting.
  void trim();
  void close();
  // Explicit cold observation; closed pools omit unavailable numeric fields.
  std::unordered_map<std::string, std::uint64_t> snapshot();

 private:
  void initialize(CudaDevice &device, std::uint64_t retained_bytes);
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace cuda
}  // namespace taichi::lang
