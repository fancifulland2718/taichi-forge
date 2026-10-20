#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "taichi/program/ndarray.h"

namespace taichi::lang {

class PreparedNativeStorage;
class NativeOperation;
class Program;

// Fixed storage bindings, not a second device recording.
// The Program arena owns GPU radix scratch/pipelines. CPU uses the shared
// native sort operation's retained workspace. Borrowed ndarray wrappers never
// free the underlying allocation.
class PreparedPrimitiveSort {
 private:
  friend class Program;
  std::shared_ptr<PreparedNativeStorage> storage_;
  // The existing AOT CPU operation retains its own stable-sort workspace.
  // Populate it on first execution, under the Program submission guard.
  mutable std::shared_ptr<NativeOperation> cpu_operation_;
  std::array<std::unique_ptr<Ndarray>, 2> views_;
  std::array<std::uintptr_t, 2> cuda_pointers_{};
  int count_{0};
  int key_type_{0};
  int value_type_{0};
  int nan_policy_{0};
  std::size_t key_bytes_{0};
  std::size_t value_bytes_{0};
};

// Columns share one predicate/prefix and a device count. All bindings are
// resolved together; no other submission may interleave the column scatters.
class PreparedPrimitiveCompact {
 private:
  friend class Program;
  std::shared_ptr<PreparedNativeStorage> storage_;
  std::vector<std::uintptr_t> cuda_pointers_;
  std::vector<std::size_t> column_bytes_;
  int count_{0};
};

}  // namespace taichi::lang
