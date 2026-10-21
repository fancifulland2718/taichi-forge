#include "taichi/program/prepared_primitive.h"

#include <algorithm>
#include <limits>

#include "taichi/ir/type_factory.h"
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_primitive_contract.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {

NativeArrayBinding primitive_array_layout(const storage::DenseStorageDescriptor &descriptor) {
  NativeArrayBinding result;
  result.scalar_type = descriptor.scalar_type();
  auto shape = [](const auto &source) {
    std::vector<std::uint64_t> result;
    for (const auto extent : source) {
      if (extent < 0)
        throw std::invalid_argument("Prepared primitive shape cannot be negative");
      result.push_back(static_cast<std::uint64_t>(extent));
    }
    return result;
  };
  result.shape = shape(descriptor.index_shape());
  result.element_shape = shape(descriptor.element_shape());
  // Storage admission below proves packed access, lifetime and alignment.
  // Do not resolve an address or allocate scratch to normalize semantic facts.
  return result;
}

int sort_value_type(DataType type) {
  if (type == PrimitiveType::i32) {
    return 0;
  }
  if (type == PrimitiveType::f32) {
    return 1;
  }
  if (type == PrimitiveType::u32) {
    return 2;
  }
  if (type == PrimitiveType::u64) {
    return 3;
  }
  if (type == PrimitiveType::i64) {
    return 4;
  }
  if (type == PrimitiveType::f64) {
    return 5;
  }
  TI_ERROR("Prepared sort payloads require i32/u32/f32/i64/u64/f64");
}

}  // namespace

std::shared_ptr<PreparedPrimitiveSort> Program::prepare_primitive_sort(
    const storage::DenseStorageDescriptor &keys,
    const storage::DenseStorageDescriptor *values,
    int nan_policy) try {
  const auto arch = compile_config().arch;
  TI_ERROR_IF(!arch_is_cpu(arch) && arch != Arch::cuda && arch != Arch::vulkan,
              "Prepared sort requires CPU, CUDA or Vulkan");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1 ||
                  (arch != Arch::cuda && nan_policy != 0),
              "Prepared CPU/Vulkan sort supports only nan_policy='last'");
  const auto key_layout = primitive_array_layout(keys);
  const auto value_layout = values ? primitive_array_layout(*values) : NativeArrayBinding{};
  auto normalized = bind_sort_operation(key_layout, values ? &value_layout : nullptr);
  auto plan = std::make_shared<PreparedPrimitiveSort>();
  plan->count_ = static_cast<int>(normalized.size);
  plan->key_type_ = static_cast<int>(normalized.key_type);
  plan->key_bytes_ = data_type_size(keys.scalar_type());
  plan->nan_policy_ = nan_policy;
  std::vector<const storage::DenseStorageDescriptor *> descriptors{&keys};
  if (values) {
    TI_ERROR_IF(values->element_rank() > 2,
                "Prepared sort requires a scalar/vector/matrix payload range");
    plan->value_type_ = sort_value_type(values->scalar_type());
    plan->value_bytes_ = normalized.value_bytes;
    descriptors.push_back(values);
  }
  if (arch == Arch::cuda) {
    TI_ERROR_IF(!cuda_device_radix_sort_available(),
                "CUDA native stable sort is unavailable");
    auto packet = prepare_external_cuda_storage(
        descriptors, std::vector<bool>(descriptors.size(), true));
    plan->storage_ = std::move(packet.storage);
    std::copy(packet.pointers.begin(), packet.pointers.end(),
              plan->cuda_pointers_.begin());
  } else {
    TI_ERROR_IF(arch_is_cpu(arch) ? !cpu_stable_sort_available()
                                : !vulkan_radix_sort_available(),
                "Native stable sort is unavailable");
    plan->storage_ = prepare_native_storage(
        descriptors, std::vector<bool>(descriptors.size(), true));
    for (std::size_t i = 0; arch == Arch::vulkan && i < descriptors.size(); ++i) {
      const auto &binding = plan->storage_->binding(i);
      DeviceAllocation allocation{binding.pointer.device,
                                  binding.pointer.alloc_id};
      DataType dtype = descriptors[i]->scalar_type();
      const auto element_shape = descriptors[i]->element_shape();
      if (!element_shape.empty()) {
        dtype = TypeFactory::create_tensor_type(
            std::vector<int>(element_shape.begin(), element_shape.end()), dtype);
      }
      plan->views_[i] = std::make_unique<Ndarray>(
          allocation, dtype, std::vector<int>{plan->count_});
    }
  }
  normalized.keys = plan->storage_->binding(0).pointer;
  if (values)
    normalized.values = plan->storage_->binding(1).pointer;
  validate_sort_operation(*normalized.keys.device, normalized);
  // No mathematical operation, command submission, scratch allocation or
  // synchronization. First execution may populate the backend workspace.
  return plan;
} catch (const std::invalid_argument &error) {
  TI_ERROR("{}", error.what());
}

std::size_t Program::execute_primitive_sort(const PreparedPrimitiveSort &plan) {
  std::size_t workspace_bytes = 0;
  with_prepared_native_storage(*plan.storage_, [&] {
    if (plan.count_ <= 1) {
      return;
    }
    const auto arch = compile_config().arch;
    if (arch_is_cpu(arch)) {
      if (!plan.cpu_operation_) {
        const auto keys = plan.storage_->binding(0).pointer;
        const auto values = plan.value_bytes_
                                ? plan.storage_->binding(1).pointer
                                : DevicePtr{};
        SortOperation operation{
            keys, values, static_cast<std::size_t>(plan.count_),
            static_cast<NativeSortKeyType>(plan.key_type_), plan.value_bytes_,
            plan.value_bytes_ ? (plan.value_type_ >= 3 ? 8u : 4u) : 0u};
        plan.cpu_operation_ =
            prepare_sort_operation(arch, *keys.device, operation);
      }
      plan.cpu_operation_->run_cpu();
      workspace_bytes = plan.cpu_operation_->workspace_bytes();
    } else if (arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
      workspace_bytes = cuda::driver_stable_radix_sort_strided(
          reinterpret_cast<void *>(plan.cuda_pointers_[0]),
          reinterpret_cast<void *>(plan.cuda_pointers_[1]), plan.count_,
          static_cast<cuda::CudaDriverSortKeyType>(plan.key_type_),
          static_cast<int>(plan.value_bytes_ / sizeof(std::uint32_t)),
          0, plan.key_bytes_, 0, plan.value_bytes_, plan.value_bytes_ != 0,
          plan.nan_policy_, nullptr, &primitive_workspace_arena_);
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else {
      workspace_bytes = vulkan_radix_sort_u32_ndarray(
          plan.views_[0].get(), plan.views_[1].get(), plan.key_type_,
          plan.value_type_, plan.storage_->binding(0).pointer.offset,
          plan.value_bytes_ ? plan.storage_->binding(1).pointer.offset : 0);
    }
  });
  return workspace_bytes;
}

std::shared_ptr<PreparedPrimitiveCompact> Program::prepare_primitive_compact(
    const std::vector<const storage::DenseStorageDescriptor *> &inputs,
    const storage::DenseStorageDescriptor &flags,
    const std::vector<const storage::DenseStorageDescriptor *> &outputs,
    const storage::DenseStorageDescriptor &count) try {
  const auto arch = compile_config().arch;
  TI_ERROR_IF(arch != Arch::cuda && arch != Arch::vulkan,
              "Prepared compact requires CUDA or Vulkan");
  TI_ERROR_IF(inputs.empty() || inputs.size() != outputs.size(),
              "Prepared compact requires matching input/output columns");
  std::vector<std::pair<NativeArrayBinding, NativeArrayBinding>> columns;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    TI_ERROR_IF(!inputs[i] || !outputs[i], "Prepared compact requires live columns");
    columns.emplace_back(primitive_array_layout(*inputs[i]), primitive_array_layout(*outputs[i]));
  }
  auto normalized = bind_compact_operation(primitive_array_layout(flags), primitive_array_layout(count), columns);
  const auto n = normalized.size;
  TI_ERROR_IF(!n, "Prepared compact capacity must be positive");
  auto plan = std::make_shared<PreparedPrimitiveCompact>();
  plan->count_ = static_cast<int>(n);
  std::vector<const storage::DenseStorageDescriptor *> descriptors{&flags, &count};
  std::vector<bool> writable{false, true};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto *input = inputs[i];
    const auto *output = outputs[i];
    // The existing raw-word scatter supports scalar, vector and matrix records.
    sort_value_type(input->scalar_type());
    TI_ERROR_IF(input->element_rank() > 2,
                "Prepared compact requires scalar/vector/matrix records");
    const auto bytes = normalized.columns[i].item_bytes;
    TI_ERROR_IF(arch == Arch::vulkan &&
                    static_cast<std::size_t>(n) >
                        std::numeric_limits<std::uint32_t>::max() / (bytes / 4),
                "Prepared Vulkan compact word count exceeds UINT32_MAX");
    plan->column_bytes_.push_back(bytes);
    descriptors.push_back(input);
    descriptors.push_back(output);
    writable.push_back(false);
    writable.push_back(true);
  }
  if (arch == Arch::cuda) {
    TI_ERROR_IF(!cuda_device_compact_available(), "CUDA native compact is unavailable");
    auto packet = prepare_external_cuda_storage(descriptors, writable);
    plan->storage_ = std::move(packet.storage);
    plan->cuda_pointers_ = std::move(packet.pointers);
  } else {
    TI_ERROR_IF(!vulkan_compact_available(), "Vulkan native compact is unavailable");
    plan->storage_ = prepare_native_storage(descriptors, writable);
  }
  normalized.flags = plan->storage_->binding(0).pointer;
  normalized.count = plan->storage_->binding(1).pointer;
  for (std::size_t i = 0; i < normalized.columns.size(); ++i) {
    normalized.columns[i].values = plan->storage_->binding(2 + i * 2).pointer;
    normalized.columns[i].output = plan->storage_->binding(3 + i * 2).pointer;
  }
  validate_compact_operation(*normalized.flags.device, normalized);
  return plan;
} catch (const std::invalid_argument &error) {
  TI_ERROR("{}", error.what());
}

std::size_t Program::execute_primitive_compact(const PreparedPrimitiveCompact &plan) {
  std::size_t workspace_bytes = 0;
  // This one existing submission boundary encloses the entire column sequence.
  // Prefix reuse cannot escape the call or cross another producer/arena user.
  with_prepared_native_storage(*plan.storage_, [&] {
    for (std::size_t i = 0; i < plan.column_bytes_.size(); ++i) {
      const auto bytes = plan.column_bytes_[i];
      std::size_t used = 0;
      if (compile_config().arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
        const auto &pointers = plan.cuda_pointers_;
        used = cuda::driver_compact_strided(
            reinterpret_cast<void *>(pointers[2 + i * 2]),
            reinterpret_cast<void *>(pointers[0]),
            reinterpret_cast<void *>(pointers[3 + i * 2]),
            reinterpret_cast<void *>(pointers[1]), plan.count_,
            static_cast<int>(bytes / 4), 0, bytes, 0, sizeof(std::int32_t),
            0, bytes, 0, nullptr, &primitive_workspace_arena_, i != 0);
#else
        TI_NOT_IMPLEMENTED;
#endif
      } else {
        used = vulkan_compact_ranges(
            plan.storage_->binding(2 + i * 2).pointer,
            plan.storage_->binding(0).pointer,
            plan.storage_->binding(3 + i * 2).pointer,
            plan.storage_->binding(1).pointer, bytes, plan.count_, i != 0);
      }
      workspace_bytes = std::max(workspace_bytes, used);
    }
  });
  return workspace_bytes;
}

}  // namespace taichi::lang
