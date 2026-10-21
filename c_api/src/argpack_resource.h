#pragma once

#include <optional>
#include "taichi_core_impl.h"
#include "taichi/runtime/native_argpack_storage.h"

namespace capi {

// C descriptor ownership stays at the API boundary. Core storage and backends
// consume only the already resolved allocation view.
class ArgPackResource final : public taichi::lang::NativeResource {
 public:
  struct Member {
    uint32_t physical;
    TiArgument argument;
    std::optional<TiNdArray> gradient;
    std::optional<taichi::lang::DataType> structure_type;
  };
  ArgPackResource(std::shared_ptr<taichi::lang::NativeArgPackStorage> storage,
                  std::vector<Member> members)
      : storage(std::move(storage)), members(std::move(members)) {
  }
  taichi::Arch arch() const override {
    return storage->arch();
  }
  taichi::lang::Device &device() const override {
    return storage->device();
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  std::size_t device_payload_bytes() const override {
    return storage->device_payload_bytes();
  }
  const std::shared_ptr<taichi::lang::NativeArgPackStorage> storage;
  const std::vector<Member> members;
};

// Empty unless an owned public pack needs logical-root expansion. Gradients
// point into immutable registered owners, whose handles remain in arguments.
struct ExpandedArgPackArguments {
  std::vector<TiArgument> arguments;
  std::vector<TiNdArrayBinding> bindings;
  bool expanded{false};
};
ExpandedArgPackArguments expand_argpack_arguments(
    Runtime &runtime, const Runtime::KernelBinding &kernel,
    uint32_t argument_count, const TiArgument *arguments,
    uint32_t binding_count, const TiNdArrayBinding *bindings);

}  // namespace capi
