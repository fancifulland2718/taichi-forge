#include "taichi/program/prepared_primitive.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "taichi/ir/type_factory.h"
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_primitive_contract.h"
#include "taichi/runtime/gfx/graph_recording.h"
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

namespace {

// Each executable holds a distinct lease, even when it captures a source
// recording more than once. Releasing one executable never clears another's
// operation. The Program retirement callback clears the shared operation before
// its device is destroyed, including source objects retained by Python.
class PrimitiveCaptureResources final : public CudaGraphCaptureResources {
 public:
  explicit PrimitiveCaptureResources(std::shared_ptr<PreparedPrimitiveRecording> owner)
      : owner_(std::move(owner)) {}
  void finalize() override {}
  std::uint64_t requested_device_bytes() const override { return 0; }
  void release(bool) noexcept override { owner_.reset(); }
 private:
  std::shared_ptr<PreparedPrimitiveRecording> owner_;
};

class PrimitiveCudaCommand final : public aot::CudaGraphCaptureCommand {
 public:
  explicit PrimitiveCudaCommand(std::shared_ptr<PreparedPrimitiveRecording> owner)
      : owner_(std::move(owner)) {}
  const char *kind() const override { return "builtin_fixed_primitive"; }
  Program *program() const override { return owner_->program(); }
  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (owner_->program() != &program)
      return false;
    owner_->validate(program, args);
    return true;
  }
  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override { owner_->validate(program, args); }
  void record(const std::unordered_map<std::string, aot::IValue> &,
              Program &, void *stream) override {
    resources_ = std::make_shared<PrimitiveCaptureResources>(owner_);
    owner_->record_cuda(stream);
  }
  bool supports_conditional_body() const override { return true; }
  std::vector<SNodeTreeDependency> snode_tree_dependencies() const override {
    return owner_->tree_dependencies();
  }
  std::shared_ptr<CudaGraphCaptureResources> take_capture_resources() override {
    return std::exchange(resources_, {});
  }
 private:
  std::shared_ptr<PreparedPrimitiveRecording> owner_;
  std::shared_ptr<CudaGraphCaptureResources> resources_;
};

class PrimitiveVulkanCommand final : public gfx::ExternalGraphCommand {
 public:
  explicit PrimitiveVulkanCommand(std::shared_ptr<PreparedPrimitiveRecording> owner)
      : owner_(std::move(owner)) {}
  std::vector<aot::Arg> arguments() const override { return owner_->arguments(); }
  void validate(Program &program,
                const std::unordered_map<std::string, aot::IValue> &args) const override {
    owner_->validate(program, args);
  }
  void record(Device *device, CommandList *commands) const override {
    owner_->record_vulkan(device, commands);
  }
  bool supports_inline_recording() const override { return true; }
  std::optional<std::vector<DeviceAllocation>> buffer_uses() const override {
    return owner_->buffers();
  }
  std::vector<SNodeTreeDependency> snode_tree_dependencies() const override {
    return owner_->tree_dependencies();
  }
 private:
  std::shared_ptr<PreparedPrimitiveRecording> owner_;
};

void validate_recording_arguments(const std::vector<aot::Arg> &arguments,
                                  const std::vector<std::size_t> &indices,
                                  std::size_t binding_count) {
  TI_ERROR_IF(arguments.size() != indices.size(), "Native recording binding arity mismatch");
  std::unordered_set<std::string> names;
  std::unordered_set<std::size_t> used;
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    TI_ERROR_IF(arguments[i].tag != aot::ArgKind::kNdarray ||
                    arguments[i].name.empty() || indices[i] >= binding_count ||
                    !names.insert(arguments[i].name).second || !used.insert(indices[i]).second,
                "Native recording requires distinct valid array bindings");
  }
}
}  // namespace

void PreparedPrimitiveRecording::retire() {
  std::lock_guard<std::mutex> guard(mutex_);
  operation_.reset();
  program_ = nullptr;
}

Program *PreparedPrimitiveRecording::program() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return program_;
}

std::size_t PreparedPrimitiveRecording::workspace_bytes() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return operation_ ? operation_->workspace_bytes() : 0;
}

const std::vector<aot::Arg> &PreparedPrimitiveRecording::arguments() const {
  return arguments_;
}

std::vector<SNodeTreeDependency> PreparedPrimitiveRecording::tree_dependencies() const {
  return storage_->tree_dependencies();
}

void PreparedPrimitiveRecording::validate(
    Program &program, const std::unordered_map<std::string, aot::IValue> &args) const {
  program.with_prepared_native_storage(*storage_, [&] {
    std::lock_guard<std::mutex> guard(mutex_);
    TI_ERROR_IF(program_ != &program || !operation_, "Native primitive recording is retired");
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
      const auto found = args.find(arguments_[i].name);
      TI_ERROR_IF(found == args.end() || found->second.tag != aot::ArgKind::kNdarray ||
                      !found->second.val, "Native recording array binding is missing");
      const auto &value = found->second;
      const auto *array = reinterpret_cast<const Ndarray *>(value.val);
      program.validate_ndarrays_for_external_submission(&array, 1);
      // These symbolic arguments retain owning allocations. Exact subrange
      // offsets/layouts remain frozen in storage_, validated above; the
      // recorder never substitutes the root for an operation's view pointer.
      const DeviceAllocation allocation = storage_->binding(binding_indices_[i]).pointer;
      TI_ERROR_IF(array->get_device_allocation() != allocation,
                  "Native recording requires its original allocation owner");
    }
  });
}

void PreparedPrimitiveRecording::record_cuda(void *stream) {
  std::lock_guard<std::mutex> guard(mutex_);
  TI_ERROR_IF(!operation_, "Native primitive recording is retired");
  operation_->record_cuda(stream);
}

void PreparedPrimitiveRecording::record_vulkan(Device *device, CommandList *commands) {
  std::lock_guard<std::mutex> guard(mutex_);
  TI_ERROR_IF(!operation_ || &operation_->device() != device,
              "Native primitive recording is retired or on another device");
  operation_->record_vulkan(device, commands);
}

std::vector<DeviceAllocation> PreparedPrimitiveRecording::buffers() const {
  std::lock_guard<std::mutex> guard(mutex_);
  TI_ERROR_IF(!operation_, "Native primitive recording is retired");
  auto requirements = operation_->vulkan_recording_requirements();
  TI_ERROR_IF(!requirements.buffers || !requirements.images.empty() || requirements.graphics_queue,
              "Native primitive requires a complete compute buffer contract");
  return *requirements.buffers;
}

std::shared_ptr<aot::CudaGraphCaptureCommand> PreparedPrimitiveRecording::cuda_command() {
  return std::make_shared<PrimitiveCudaCommand>(shared_from_this());
}

std::shared_ptr<gfx::ExternalGraphCommand> PreparedPrimitiveRecording::vulkan_command() {
  return std::make_shared<PrimitiveVulkanCommand>(shared_from_this());
}

std::shared_ptr<PreparedPrimitiveRecording> Program::record_primitive_sort(
    const PreparedPrimitiveSort &plan, const std::vector<aot::Arg> &arguments,
    const std::vector<std::size_t> &binding_indices) {
  TI_ERROR_IF(plan.nan_policy_ != 0, "Persistent sort recording requires nan_policy='last'");
  validate_recording_arguments(arguments, binding_indices, plan.value_bytes_ ? 2 : 1);
  auto result = std::make_shared<PreparedPrimitiveRecording>();
  with_prepared_native_storage(*plan.storage_, [&] {
    const auto keys = plan.storage_->binding(0).pointer;
    SortOperation operation{keys, plan.value_bytes_ ? plan.storage_->binding(1).pointer : DevicePtr{},
                            static_cast<std::size_t>(plan.count_),
                            static_cast<NativeSortKeyType>(plan.key_type_), plan.value_bytes_,
                            plan.value_bytes_ ? (plan.value_type_ >= 3 ? 8u : 4u) : 0u};
    result->operation_ = prepare_sort_operation(compile_config().arch, *keys.device, operation);
    result->program_ = this;
    result->storage_ = plan.storage_;
    result->arguments_ = arguments;
    result->binding_indices_ = binding_indices;
    attach_runtime_client(result);
  });
  return result;
}

std::shared_ptr<PreparedPrimitiveRecording> Program::record_primitive_compact(
    const PreparedPrimitiveCompact &plan, const std::vector<aot::Arg> &arguments,
    const std::vector<std::size_t> &binding_indices) {
  validate_recording_arguments(arguments, binding_indices, 2 + plan.column_bytes_.size() * 2);
  auto result = std::make_shared<PreparedPrimitiveRecording>();
  with_prepared_native_storage(*plan.storage_, [&] {
    CompactOperation operation{};
    operation.flags = plan.storage_->binding(0).pointer;
    operation.count = plan.storage_->binding(1).pointer;
    operation.size = plan.count_;
    for (std::size_t i = 0; i < plan.column_bytes_.size(); ++i)
      operation.columns.push_back({plan.storage_->binding(2 + i * 2).pointer,
                                   plan.storage_->binding(3 + i * 2).pointer, plan.column_bytes_[i]});
    result->operation_ = prepare_compact_operation(compile_config().arch, *operation.flags.device, operation);
    result->program_ = this;
    result->storage_ = plan.storage_;
    result->arguments_ = arguments;
    result->binding_indices_ = binding_indices;
    attach_runtime_client(result);
  });
  return result;
}

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
