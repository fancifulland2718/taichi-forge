#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/vulkan_parallel_sort_plan.h"
#include "taichi/common/operation_json.h"

#include <cstring>
#include <limits>
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
struct SortDescription {
  static constexpr const char *kind = "vulkan_parallel_sort_plan_u32";
  static constexpr uint32_t version = 1;
  uint64_t key_count{0};
  bool payload{false}, fused_prefix{false};
  std::vector<size_t> shader_sizes;
  size_t code_bytes{0};
  NativeArrayBinding keys;
  std::optional<NativeArrayBinding> values;
  std::shared_ptr<NativeHostBlobResource> code;
};
class SortResource : public NativeResource {
 public:
  virtual void record(CommandList *commands) = 0;
};
struct ParallelSortOperation {
  static constexpr const char *kind = "vulkan_parallel_sort_u32";
  static constexpr uint32_t version = 1;
  std::shared_ptr<SortResource> plan;
};

SortDescription decode(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"key_count", "payload", "fused_prefix", "shader_sizes"});
  SortDescription d;
  d.key_count = integer(p["key_count"]);
  vulkan_parallel_sort_storage_bytes(d.key_count);
  d.payload = optional_bool(p, "payload");
  d.fused_prefix = optional_bool(p, "fused_prefix");
  const auto &sizes = p["shader_sizes"];
  if (!sizes.is_arr() || sizes.arr.inner.size() != (d.fused_prefix ? 3 : 5))
    throw std::invalid_argument(
        "Parallel Sort shader count differs from selected strategy");
  for (const auto &size : sizes.arr.inner) {
    const auto bytes = integer(size);
    if (bytes < 20 || bytes % 4 ||
        bytes > std::numeric_limits<size_t>::max() - d.code_bytes)
      throw std::invalid_argument(
          "Parallel Sort shader size is invalid or overflows");
    d.shader_sizes.push_back(size_t(bytes));
    d.code_bytes += size_t(bytes);
  }
  return d;
}

void validate(Device &device, const SortDescription &d) {
  if (d.payload != bool(d.values))
    throw std::invalid_argument(
        "Parallel Sort payload binding differs from its contract");
  if (!d.code || &d.code->device() != &device ||
      d.code->arch() != Arch::vulkan || d.code->bytes().size() != d.code_bytes)
    throw std::invalid_argument(
        "Parallel Sort shader blob size or device mismatch");
  size_t offset = 0;
  for (auto bytes : d.shader_sizes) {
    uint32_t magic = 0;
    std::memcpy(&magic, d.code->bytes().data() + offset, sizeof(magic));
    if (magic != 0x07230203u)
      throw std::invalid_argument(
          "Parallel Sort shader blob has an invalid SPIR-V header");
    offset += bytes;
  }
  const auto bytes = vulkan_parallel_sort_storage_bytes(d.key_count);
  auto range = [&](const NativeArrayBinding &array) {
    if (array.pointer.device != &device || array.bytes != bytes ||
        array.scalar_type != PrimitiveType::u32)
      throw std::invalid_argument(
          "Parallel Sort requires exact compact scalar u32 arrays");
    validate_native_array_layout(array, {{d.key_count}, {}, 4, 4, true});
    return resolve_native_storage_range(Arch::vulkan, array.pointer, bytes);
  };
  const auto keys = range(d.keys);
  if (d.values && !keys.proven_disjoint(range(*d.values)))
    throw std::invalid_argument(
        "Parallel Sort keys and payload must not overlap");
}

#ifdef TI_WITH_VULKAN
class PreparedSort final : public SortResource {
 public:
  PreparedSort(Device &device, const SortDescription &d) : device_(device) {
    auto &vk = dynamic_cast<vulkan::VulkanDevice &>(device);
    auto leases = std::make_shared<std::vector<vkapi::IVkBuffer>>();
    leases->push_back(vk.get_vkbuffer(d.keys.pointer));
    if (d.values)
      leases->push_back(vk.get_vkbuffer(d.values->pointer));
    std::vector<std::vector<uint32_t>> shaders;
    size_t offset = 0;
    for (auto bytes : d.shader_sizes) {
      auto &words = shaders.emplace_back(bytes / sizeof(uint32_t));
      std::memcpy(words.data(), d.code->bytes().data() + offset, bytes);
      offset += bytes;
    }
    core_ = make_vulkan_parallel_sort_plan(
        vk, d.keys.pointer,
        d.values ? std::optional<DevicePtr>(d.values->pointer) : std::nullopt,
        std::move(leases), d.key_count, shaders, false);
    workspace_ = core_->statistics().at("workspace_bytes");
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return workspace_;
  }
  void record(CommandList *commands) override {
    core_->record(commands, true);
  }

 private:
  Device &device_;
  std::shared_ptr<VulkanParallelSortPlanCore> core_;
  size_t workspace_{0};
};
class PreparedSortExecution final : public NativeOperation {
 public:
  explicit PreparedSortExecution(std::shared_ptr<SortResource> plan)
      : plan_(std::move(plan)) {
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return plan_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    plan_->record(commands);
  }

 private:
  std::shared_ptr<SortResource> plan_;
};
#endif
}  // namespace

void register_parallel_sort_operations(NativeOperationRegistry &registry) {
  registry.define<SortDescription>(validate, NativeObjectKind::resource);
  registry.register_schema<SortDescription>(
      "fidelityfx_parallel_sort", "plan_u32", {"keys", "code"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<SortDescription> {
        auto description = decode(p);
        return [description](const NativeOperationBindings &b) {
          auto d = description;
          d.keys = b.at("keys").array();
          d.code = b.at("code").resource<NativeHostBlobResource>();
          if (b.count("values"))
            d.values = b.at("values").array();
          return d;
        };
      },
      {}, {"values"});
  registry.define<ParallelSortOperation>(
      [](Device &device, const ParallelSortOperation &op) {
        if (!op.plan || &op.plan->device() != &device ||
            op.plan->arch() != Arch::vulkan)
          throw std::invalid_argument(
              "Parallel Sort requires its local Vulkan plan");
      });
  registry.register_schema<ParallelSortOperation>(
      "fidelityfx_parallel_sort", "execute_u32", {"plan"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<ParallelSortOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return ParallelSortOperation{b.at("plan").resource<SortResource>()};
        };
      });
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<SortDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const SortDescription &d) {
        return std::make_shared<PreparedSort>(context.device, d);
      });
  registry.register_context_factory<ParallelSortOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const ParallelSortOperation &op) {
        return std::make_shared<PreparedSortExecution>(op.plan);
      });
#endif
}
}  // namespace taichi::lang
