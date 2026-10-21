#pragma once

#include <limits>
#include "taichi/ir/type_utils.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Cold packed-record facts shared by the direct JIT adapters and symbolic
// native binders. No workspace allocation, provider lookup or device work.
inline std::size_t native_primitive_record_bytes(const NativeArrayBinding &array) {
  if (array.strides || !array.scalar_type->is<PrimitiveType>())
    throw std::invalid_argument("Native primitive requires packed scalar records");
  const auto width = data_type_size(array.scalar_type);
  if (width <= 0)
    throw std::invalid_argument("Native primitive scalar storage type is invalid");
  std::size_t bytes = width;
  for (const auto extent : array.element_shape) {
    if (!extent || extent > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::invalid_argument("Native primitive record size overflows");
    bytes *= extent;
  }
  return bytes;
}

// Bindings have already passed their owner's packed-storage/range admission.
// Layout normalization also accepts address-free producer facts; actual device
// and alias validation follows after that adapter resolves its storage leases.
SortOperation bind_sort_operation(const NativeArrayBinding &keys,
                                  const NativeArrayBinding *values = nullptr);
CompactOperation bind_compact_operation(
    const NativeArrayBinding &flags,
    const NativeArrayBinding &count,
    const std::vector<std::pair<NativeArrayBinding, NativeArrayBinding>> &columns);
void validate_sort_operation(Device &device, const SortOperation &operation);
void validate_compact_operation(Device &device, const CompactOperation &operation);

}  // namespace taichi::lang
