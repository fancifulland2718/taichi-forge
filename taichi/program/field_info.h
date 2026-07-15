#pragma once
#include "taichi/ir/type_utils.h"
#include "taichi/ir/snode.h"
#include "taichi/rhi/device.h"
#include "taichi/program/program.h"

namespace taichi {

namespace ui {

enum class FieldSource : int {
  TaichiNDarray = 0,
  HostMappedPtr = 1,
};

#define DEFINE_PROPERTY(Type, name)       \
  Type name;                              \
  void set_##name(const Type &new_name) { \
    name = new_name;                      \
  }                                       \
  Type get_##name() {                     \
    return name;                          \
  }

struct FieldInfo {
  DEFINE_PROPERTY(bool, valid)
  DEFINE_PROPERTY(std::vector<int>, shape);
  DEFINE_PROPERTY(uint64_t, num_elements);
  DEFINE_PROPERTY(FieldSource, field_source);
  DEFINE_PROPERTY(taichi::lang::DataType, dtype);
  DEFINE_PROPERTY(taichi::lang::DeviceAllocation, dev_alloc);

  // Optional high-level identity for deferred consumers such as GGUI. The
  // DeviceAllocation alone is not an ownership token: its storage may retire
  // after the Python Ndarray wrapper dies but before show() submits the frame.
  taichi::lang::Program *runtime_resource_program{nullptr};
  taichi::lang::RuntimeResourceHandle runtime_resource_handle;

  void bind_runtime_ndarray(const taichi::lang::Ndarray *ndarray) {
    runtime_resource_program =
        ndarray != nullptr ? ndarray->owning_program() : nullptr;
    runtime_resource_handle =
        ndarray != nullptr ? ndarray->runtime_resource_handle()
                           : taichi::lang::RuntimeResourceHandle{};
  }

  FieldInfo() {
    valid = false;
  }
};

taichi::lang::DevicePtr get_device_ptr(taichi::lang::Program *program,
                                       taichi::lang::SNode *snode);

}  // namespace ui

}  // namespace taichi
