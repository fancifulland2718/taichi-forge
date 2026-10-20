#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <variant>
#include <memory>
#include <stdexcept>
#include "taichi/common/json.h"
#include "taichi/ir/type.h"
#include "taichi/rhi/device.h"
#include "taichi/runtime/native_resource.h"

namespace taichi::lang {

// Persistent provider identity and parameters, with symbolic storage slots.
// Neither addresses nor live provider objects belong in this description.
struct NativeOperationDescription {
  using Key = std::tuple<std::string, std::string, std::uint32_t>;
  std::string provider;
  std::string operation;
  std::uint32_t version{1};
  liong::json::JsonValue parameters;
  std::map<std::string, std::string> bindings;

  Key key() const {
    return {provider, operation, version};
  }
  static NativeOperationDescription parse(const liong::json::JsonValue &value);
};

// Validated owner-local storage metadata. JIT and AOT adapters supply the same
// cold view; providers retain prepared commands, not this symbolic binding map.
// The enclosing executor owns storage leases through completion.
struct NativeArrayStrides {
  std::vector<std::int64_t> index_bytes;
  std::vector<std::int64_t> element_bytes;
  // Relative to NativeArrayBinding::pointer, not the allocation's origin.
  std::int64_t byte_offset{0};
};

struct NativeArrayBinding {
  DevicePtr pointer;
  // Accessible storage window beginning at pointer, not logical payload size.
  // Packed adapters supply the exact packed extent; affine views may have gaps.
  std::uint64_t bytes{0};
  DataType scalar_type;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> element_shape;
  // Imported allocation origin, used only for alignment qualification.
  std::uint64_t base_offset{0};
  std::shared_ptr<const NativeArrayStrides> strides;
};
class NativeBinding {
 public:
  NativeBinding(NativeArrayBinding array) : value_(std::move(array)) {
  }
  NativeBinding(std::shared_ptr<NativeResource> resource)
      : value_(std::move(resource)) {
    if (!std::get<std::shared_ptr<NativeResource>>(value_))
      throw std::invalid_argument("Native resource binding is null");
  }
  const NativeArrayBinding &array() const {
    const auto &value = dense_array();
    if (value.strides)
      throw std::invalid_argument(
          "Native operation requires a packed array binding");
    return value;
  }
  // Opt-in only. The operation must validate the affine layout and physical
  // range before preparing commands; this accessor does not certify either.
  const NativeArrayBinding &dense_array() const {
    const auto *value = std::get_if<NativeArrayBinding>(&value_);
    if (!value)
      throw std::invalid_argument("Native binding requires an ndarray");
    return *value;
  }
  template <typename Resource>
  std::shared_ptr<Resource> resource() const {
    const auto *value = std::get_if<std::shared_ptr<NativeResource>>(&value_);
    auto result = value ? std::dynamic_pointer_cast<Resource>(*value) : nullptr;
    if (!result)
      throw std::invalid_argument("Native resource binding type mismatch");
    return result;
  }

 private:
  std::variant<NativeArrayBinding, std::shared_ptr<NativeResource>> value_;
};
using NativeOperationBindings = std::map<std::string, NativeBinding>;

}  // namespace taichi::lang
