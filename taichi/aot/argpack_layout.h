#pragma once

#include <functional>
#include <utility>
#include "taichi/aot/value_layout.h"
#include "taichi/program/argument_origin.h"

namespace taichi::lang {
class CallableBase;
}

namespace taichi::lang::aot {

// Cold, resolved storage for one public pack, including its nested packs.
// The resolver is the backend's existing compiler layout routine. Pointer
// targets in LLVM artifacts can still have layout="none"; their zero offsets
// are not compiled offsets. This object never defines a second packing ABI.
class ArgPackLayout {
 public:
  using Resolver = std::function<std::pair<const StructType *, std::size_t>(
      const StructType *)>;
  struct Value {
    ArgumentOrigin origin;
    std::size_t offset;
    ValueLayout layout;
  };
  struct Child {
    std::size_t storage;
    std::size_t pointer_offset;
  };
  struct Storage {
    ArgumentOrigin origin;
    const StructType *type;
    std::size_t bytes;
    std::vector<Value> values;
    std::vector<Child> children;
  };

  ArgPackLayout(const CallableBase &kernel,
               const std::vector<int> &logical_root,
               ValueStorage storage,
               const Resolver &resolve);
  ValueStorage value_storage() const {
    return value_storage_;
  }
  const std::vector<Storage> &storages() const {
    return storages_;
  }
  const ArgumentOrigins &resources() const {
    return resources_;
  }

 private:
  ValueStorage value_storage_;
  std::vector<Storage> storages_;
  ArgumentOrigins resources_;
};

}  // namespace taichi::lang::aot
