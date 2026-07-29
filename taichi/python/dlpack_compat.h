#pragma once

#include <cstdint>

namespace taichi::python::dlpack_abi {

constexpr std::uint32_t kSupportedMajorVersion = 1;
constexpr std::uint32_t kSupportedMinorVersion = 0;

constexpr std::uint64_t kFlagReadOnly = 1ull << 0;
constexpr std::uint64_t kFlagIsCopied = 1ull << 1;

enum DeviceType : std::int32_t {
  kCpu = 1,
  kCuda = 2,
  kCudaHost = 3,
  kVulkan = 7,
  kCudaManaged = 13,
};

enum DataTypeCode : std::uint8_t {
  kInt = 0,
  kUInt = 1,
  kFloat = 2,
  kBfloat = 4,
  kComplex = 5,
  kBool = 6,
};

struct Device {
  std::int32_t device_type;
  std::int32_t device_id;
};

struct DataType {
  std::uint8_t code;
  std::uint8_t bits;
  std::uint16_t lanes;
};

struct Tensor {
  void *data;
  Device device;
  std::int32_t ndim;
  DataType dtype;
  std::int64_t *shape;
  std::int64_t *strides;
  std::uint64_t byte_offset;
};

struct ManagedTensor {
  Tensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(ManagedTensor *self);
};

struct Version {
  std::uint32_t major;
  std::uint32_t minor;
};

struct ManagedTensorVersioned {
  Version version;
  void *manager_ctx;
  void (*deleter)(ManagedTensorVersioned *self);
  std::uint64_t flags;
  Tensor dl_tensor;
};

}  // namespace taichi::python::dlpack_abi
