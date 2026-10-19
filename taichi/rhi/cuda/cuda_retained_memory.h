#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "taichi/common/core.h"

namespace taichi::lang::cuda {

// CiG does not qualify stream-ordered CUDA memory on all supported drivers.
// Keep ordinary device blocks instead. Release records a stream-prefix event;
// allocation only reuses completed blocks. Neither operation waits or frees.
// Owners must bound in-flight storage, exactly as for a native CUDA pool.
class TI_DLL_EXPORT RetainedMemoryPool {
 public:
  RetainedMemoryPool();
  ~RetainedMemoryPool();
  RetainedMemoryPool(const RetainedMemoryPool &) = delete;
  RetainedMemoryPool &operator=(const RetainedMemoryPool &) = delete;

  void *allocate(std::size_t bytes);
  void release(void *pointer, void *stream);
  // Explicit cold boundary: cuMemFree may synchronize unrelated device work.
  // Never invoked by allocation, release, replay, or snapshot.
  void trim();
  std::unordered_map<std::string, std::uint64_t> snapshot();
  static RetainedMemoryPool &process_pool();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace taichi::lang::cuda
