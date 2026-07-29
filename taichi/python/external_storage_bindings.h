#pragma once

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "taichi/ir/type_utils.h"
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "taichi/python/dlpack_compat.h"
#include "taichi/python/export.h"
#include "taichi/rhi/arch.h"
#if defined(TI_WITH_CUDA)
#include "taichi/rhi/llvm/llvm_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

namespace taichi {
namespace {

using python::dlpack_abi::ManagedTensor;
using python::dlpack_abi::ManagedTensorVersioned;
using python::dlpack_abi::Tensor;

constexpr char kDLPackCapsuleName[] = "dltensor";
constexpr char kUsedDLPackCapsuleName[] = "used_dltensor";
constexpr char kDLPackVersionedCapsuleName[] = "dltensor_versioned";
constexpr char kUsedDLPackVersionedCapsuleName[] =
    "used_dltensor_versioned";

class DLPackOwner final {
 public:
  static std::shared_ptr<DLPackOwner> consume(py::capsule capsule) {
    PyObject *object = capsule.ptr();
    if (PyCapsule_IsValid(object, kDLPackVersionedCapsuleName)) {
      auto *managed = static_cast<ManagedTensorVersioned *>(
          PyCapsule_GetPointer(object, kDLPackVersionedCapsuleName));
      if (managed == nullptr) {
        throw py::error_already_set();
      }
      if (managed->version.major !=
          python::dlpack_abi::kSupportedMajorVersion) {
        PyCapsule_SetName(object, kUsedDLPackVersionedCapsuleName);
        if (managed->deleter != nullptr) {
          managed->deleter(managed);
        }
        throw py::buffer_error("unsupported DLPack major version");
      }
      if (PyCapsule_SetName(object, kUsedDLPackVersionedCapsuleName) != 0) {
        throw py::error_already_set();
      }
      return std::shared_ptr<DLPackOwner>(new DLPackOwner(managed));
    }
    if (PyCapsule_IsValid(object, kDLPackCapsuleName)) {
      auto *managed = static_cast<ManagedTensor *>(
          PyCapsule_GetPointer(object, kDLPackCapsuleName));
      if (managed == nullptr) {
        throw py::error_already_set();
      }
      if (PyCapsule_SetName(object, kUsedDLPackCapsuleName) != 0) {
        throw py::error_already_set();
      }
      return std::shared_ptr<DLPackOwner>(new DLPackOwner(managed));
    }
    throw py::value_error(
        "expected an unconsumed dltensor or dltensor_versioned capsule");
  }

  DLPackOwner(const DLPackOwner &) = delete;
  DLPackOwner &operator=(const DLPackOwner &) = delete;

  ~DLPackOwner() {
    if (versioned_ != nullptr) {
      if (versioned_->deleter != nullptr) {
        versioned_->deleter(versioned_);
      }
      return;
    }
    if (legacy_ != nullptr && legacy_->deleter != nullptr) {
      legacy_->deleter(legacy_);
    }
  }

  const Tensor &tensor() const noexcept {
    return versioned_ != nullptr ? versioned_->dl_tensor
                                 : legacy_->dl_tensor;
  }

  bool read_only() const noexcept {
    return versioned_ != nullptr &&
           (versioned_->flags & python::dlpack_abi::kFlagReadOnly) != 0;
  }

  bool copied() const noexcept {
    return versioned_ != nullptr &&
           (versioned_->flags & python::dlpack_abi::kFlagIsCopied) != 0;
  }

 private:
  explicit DLPackOwner(ManagedTensor *managed) : legacy_(managed) {
  }
  explicit DLPackOwner(ManagedTensorVersioned *managed)
      : versioned_(managed) {
  }

  ManagedTensor *legacy_{nullptr};
  ManagedTensorVersioned *versioned_{nullptr};
};

class DLPackImportLifetime final {
 public:
  DLPackImportLifetime(lang::Device *device,
                       lang::DeviceAllocation allocation,
                       std::shared_ptr<DLPackOwner> managed)
      : device_(device),
        allocation_(allocation),
        managed_(std::move(managed)) {
  }

  void release() noexcept {
    if (released_) {
      return;
    }
    released_ = true;
    if (allocation_ != lang::kDeviceNullAllocation) {
      device_->dealloc_memory(allocation_);
    }
    managed_.reset();
  }

 private:
  lang::Device *device_{nullptr};
  lang::DeviceAllocation allocation_{lang::kDeviceNullAllocation};
  std::shared_ptr<DLPackOwner> managed_;
  bool released_{false};
};

class PythonExternalDenseStorage final {
 public:
  PythonExternalDenseStorage(
      lang::Program *program,
      lang::storage::StorageOwnerRef owner,
      lang::storage::DenseStorageBuildResult description,
      std::string provider,
      std::int32_t device_type,
      std::int32_t device_id,
      std::uint64_t allocation_bytes)
      : program_(program),
        program_lifetime_(program != nullptr ? program->weak_lifetime_token()
                                             : std::weak_ptr<void>{}),
        owner_(std::move(owner)),
        description_(std::move(description)),
        provider_(std::move(provider)),
        device_type_(device_type),
        device_id_(device_id),
        allocation_bytes_(allocation_bytes) {
  }

  PythonExternalDenseStorage(const PythonExternalDenseStorage &) = delete;
  PythonExternalDenseStorage &operator=(const PythonExternalDenseStorage &) =
      delete;

  ~PythonExternalDenseStorage() {
    close_noexcept();
  }

  const lang::storage::DenseStorageBuildResult &description() const noexcept {
    return description_;
  }
  const std::string &provider() const noexcept {
    return provider_;
  }
  std::int32_t device_type() const noexcept {
    return device_type_;
  }
  std::int32_t device_id() const noexcept {
    return device_id_;
  }
  std::uint64_t allocation_bytes() const noexcept {
    return allocation_bytes_;
  }
  bool closed() const noexcept {
    return closed_;
  }

  void close() {
    if (closed_) {
      return;
    }
    if (program_ != nullptr && !program_lifetime_.expired() &&
        program_->validate_external_dense_storage_owner(owner_)) {
      program_->retire_external_dense_storage(owner_);
    }
    closed_ = true;
  }

 private:
  void close_noexcept() noexcept {
    try {
      close();
    } catch (...) {
    }
  }

  lang::Program *program_{nullptr};
  std::weak_ptr<void> program_lifetime_;
  lang::storage::StorageOwnerRef owner_;
  lang::storage::DenseStorageBuildResult description_;
  std::string provider_;
  std::int32_t device_type_{0};
  std::int32_t device_id_{0};
  std::uint64_t allocation_bytes_{0};
  bool closed_{false};
};

lang::DataType dlpack_data_type(
    const python::dlpack_abi::DataType &dtype) {
  using python::dlpack_abi::DataTypeCode;
  using lang::PrimitiveType;
  if (dtype.lanes != 1) {
    throw py::buffer_error("DLPack vector lanes are not supported");
  }
  switch (static_cast<DataTypeCode>(dtype.code)) {
    case DataTypeCode::kInt:
      switch (dtype.bits) {
        case 8:
          return PrimitiveType::i8;
        case 16:
          return PrimitiveType::i16;
        case 32:
          return PrimitiveType::i32;
        case 64:
          return PrimitiveType::i64;
      }
      break;
    case DataTypeCode::kUInt:
      switch (dtype.bits) {
        case 8:
          return PrimitiveType::u8;
        case 16:
          return PrimitiveType::u16;
        case 32:
          return PrimitiveType::u32;
        case 64:
          return PrimitiveType::u64;
      }
      break;
    case DataTypeCode::kFloat:
      switch (dtype.bits) {
        case 16:
          return PrimitiveType::f16;
        case 32:
          return PrimitiveType::f32;
        case 64:
          return PrimitiveType::f64;
      }
      break;
    case DataTypeCode::kBool:
      if (dtype.bits == 8) {
        return PrimitiveType::u1;
      }
      break;
    case DataTypeCode::kBfloat:
    case DataTypeCode::kComplex:
      break;
  }
  throw py::buffer_error("unsupported DLPack dtype");
}

bool dlpack_device_matches(Arch backend, std::int32_t device_type) {
  using python::dlpack_abi::DeviceType;
  if (arch_is_cpu(backend)) {
    return device_type == DeviceType::kCpu ||
           device_type == DeviceType::kCudaHost;
  }
  if (backend == Arch::cuda) {
    return device_type == DeviceType::kCuda ||
           device_type == DeviceType::kCudaManaged;
  }
  return false;
}

std::vector<std::int64_t> compact_element_strides(
    const std::vector<std::int64_t> &shape) {
  std::vector<std::int64_t> result(shape.size(), 1);
  for (std::size_t i = shape.size(); i > 1; --i) {
    const std::int64_t extent = shape[i - 1];
    if (extent < 0 ||
        (extent != 0 &&
         result[i - 1] >
             (std::numeric_limits<std::int64_t>::max)() / extent)) {
      throw py::buffer_error("DLPack compact stride overflow");
    }
    result[i - 2] = result[i - 1] * extent;
  }
  return result;
}

std::shared_ptr<PythonExternalDenseStorage> import_dlpack_capsule(
    lang::Program &program,
    py::capsule capsule,
    const std::vector<std::int64_t> &element_shape,
    const std::string &layout,
    const std::string &access) {
  using lang::DeviceAllocation;
  using lang::storage::DenseStorageLayoutSpec;
  using lang::storage::StorageSourceKind;

  if (access != "readwrite") {
    throw py::value_error(
        "executable DLPack views currently require access='readwrite'");
  }
  if (layout != "aos") {
    throw py::buffer_error(
        "DLPack storage import currently supports scalar or AOS elements");
  }

  auto managed = DLPackOwner::consume(std::move(capsule));
  if (managed->copied()) {
    throw py::buffer_error(
        "DLPack producer returned a copy for a zero-copy import");
  }
  if (managed->read_only()) {
    throw py::buffer_error(
        "read-only DLPack storage cannot satisfy a writable view");
  }

  const Tensor &tensor = managed->tensor();
  if (tensor.ndim < 0 ||
      tensor.ndim >
          static_cast<std::int32_t>(
              lang::storage::kMaxDenseStorageRank)) {
    throw py::buffer_error("unsupported DLPack rank");
  }
  if (tensor.ndim != 0 && tensor.shape == nullptr) {
    throw py::buffer_error("DLPack tensor has no shape");
  }
  if (!dlpack_device_matches(program.compile_config().arch,
                             tensor.device.device_type)) {
    throw py::buffer_error(
        "DLPack device is incompatible with the current Taichi backend");
  }

  const lang::DataType scalar_type = dlpack_data_type(tensor.dtype);
  const std::uint64_t scalar_bytes = lang::data_type_size(scalar_type);
  std::vector<std::int64_t> full_shape;
  full_shape.reserve(tensor.ndim);
  for (std::int32_t i = 0; i < tensor.ndim; ++i) {
    if (tensor.shape[i] < 0) {
      throw py::buffer_error("DLPack shape contains a negative extent");
    }
    full_shape.push_back(tensor.shape[i]);
  }
  if (element_shape.size() > full_shape.size()) {
    throw py::buffer_error("DLPack element rank exceeds tensor rank");
  }
  const std::size_t index_rank = full_shape.size() - element_shape.size();
  for (std::size_t i = 0; i < element_shape.size(); ++i) {
    if (element_shape[i] != full_shape[index_rank + i]) {
      throw py::buffer_error(
          "DLPack trailing dimensions do not match the requested element "
          "shape");
    }
  }

  std::vector<std::int64_t> scalar_strides;
  if (tensor.strides == nullptr) {
    scalar_strides = compact_element_strides(full_shape);
  } else {
    scalar_strides.reserve(full_shape.size());
    for (std::size_t i = 0; i < full_shape.size(); ++i) {
      if (tensor.strides[i] < 0) {
        throw py::buffer_error(
            "negative-stride DLPack storage is not executable");
      }
      scalar_strides.push_back(tensor.strides[i]);
    }
  }

  std::vector<std::int64_t> byte_strides;
  byte_strides.reserve(scalar_strides.size());
  for (const std::int64_t stride : scalar_strides) {
    if (stride >
        (std::numeric_limits<std::int64_t>::max)() /
            static_cast<std::int64_t>(scalar_bytes)) {
      throw py::buffer_error("DLPack byte stride overflow");
    }
    byte_strides.push_back(
        stride * static_cast<std::int64_t>(scalar_bytes));
  }

  bool empty = false;
  std::uint64_t reachable_end = tensor.byte_offset;
  for (std::size_t i = 0; i < full_shape.size(); ++i) {
    if (full_shape[i] == 0) {
      empty = true;
      break;
    }
    const std::uint64_t terms =
        static_cast<std::uint64_t>(full_shape[i] - 1);
    const std::uint64_t stride =
        static_cast<std::uint64_t>(byte_strides[i]);
    if (terms != 0 &&
        stride >
            ((std::numeric_limits<std::uint64_t>::max)() - reachable_end) /
                terms) {
      throw py::buffer_error("DLPack reachable byte range overflow");
    }
    reachable_end += terms * stride;
  }
  if (!empty) {
    if (scalar_bytes >
        (std::numeric_limits<std::uint64_t>::max)() - reachable_end) {
      throw py::buffer_error("DLPack reachable byte range overflow");
    }
    reachable_end += scalar_bytes;
  } else {
    reachable_end = 0;
  }
  if (reachable_end != 0 && tensor.data == nullptr) {
    throw py::buffer_error("non-empty DLPack tensor has a null data pointer");
  }

#if defined(TI_WITH_CUDA)
  if (program.compile_config().arch == Arch::cuda &&
      reachable_end != 0) {
    unsigned int memory_type = 0;
    const std::uint32_t result =
        lang::CUDADriver::get_instance().mem_get_attribute.call(
            &memory_type, lang::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            tensor.data);
    if (result != lang::CUDA_SUCCESS ||
        (memory_type != lang::CU_MEMORYTYPE_DEVICE &&
         tensor.device.device_type !=
             python::dlpack_abi::DeviceType::kCudaManaged)) {
      throw py::buffer_error(
          "DLPack CUDA pointer is not addressable by the current CUDA "
          "context");
    }
  }
#endif

  auto *device =
      dynamic_cast<lang::LlvmDevice *>(program.get_compute_device());
  if (device == nullptr) {
    throw py::buffer_error("DLPack import requires an LLVM compute device");
  }
  DeviceAllocation allocation = lang::kDeviceNullAllocation;
  if (reachable_end != 0) {
    allocation = device->import_memory(tensor.data, reachable_end);
  }
  auto lifetime = std::make_shared<DLPackImportLifetime>(
      device, allocation, std::move(managed));
  auto release = [lifetime]() mutable {
    lifetime->release();
  };

  lang::storage::StorageOwnerRef owner;
  try {
    owner = program.register_external_dense_storage(
        allocation, reachable_end, std::move(release));
  } catch (...) {
    lifetime->release();
    throw;
  }

  DenseStorageLayoutSpec spec;
  spec.scalar_type = scalar_type;
  spec.index_shape.assign(full_shape.begin(),
                          full_shape.begin() + index_rank);
  spec.index_strides_bytes.assign(byte_strides.begin(),
                                  byte_strides.begin() + index_rank);
  spec.element_shape = element_shape;
  spec.element_strides_bytes.assign(byte_strides.begin() + index_rank,
                                    byte_strides.end());
  spec.byte_offset = static_cast<std::int64_t>(tensor.byte_offset);
  spec.access = lang::storage::StorageAccess::kReadWrite;

  auto description = lang::storage::build_dense_storage_descriptor(
      owner, StorageSourceKind::kExternalDense, spec);
  if (!description) {
    try {
      program.retire_external_dense_storage(owner);
    } catch (...) {
    }
    throw py::buffer_error(
        std::string("DLPack storage cannot form a dense view: ") +
        lang::storage::to_string(description.reason));
  }
  return std::make_shared<PythonExternalDenseStorage>(
      &program, owner, std::move(description), "dlpack",
      tensor.device.device_type, tensor.device.device_id, reachable_end);
}

void export_external_storage_bindings(py::module &m) {
  using namespace lang;
  using namespace lang::storage;

  py::class_<PythonExternalDenseStorage,
             std::shared_ptr<PythonExternalDenseStorage>>(
      m, "_ExternalDenseStorage")
      .def_property_readonly(
          "description", &PythonExternalDenseStorage::description,
          py::return_value_policy::reference_internal)
      .def_property_readonly("provider",
                             &PythonExternalDenseStorage::provider)
      .def_property_readonly("device_type",
                             &PythonExternalDenseStorage::device_type)
      .def_property_readonly("device_id",
                             &PythonExternalDenseStorage::device_id)
      .def_property_readonly("allocation_bytes",
                             &PythonExternalDenseStorage::allocation_bytes)
      .def_property_readonly("closed",
                             &PythonExternalDenseStorage::closed)
      .def("close", &PythonExternalDenseStorage::close);

  m.def(
      "_import_dlpack_capsule", &import_dlpack_capsule,
      py::arg("program"), py::arg("capsule"),
      py::arg("element_shape") = std::vector<std::int64_t>{},
      py::arg("layout") = "aos", py::arg("access") = "readwrite",
      py::keep_alive<0, 1>());
}

}  // namespace
}  // namespace taichi
