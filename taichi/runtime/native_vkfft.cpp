#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/vulkan_fft_plan.h"
#include "taichi/common/operation_json.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/vkfft/forge_vkfft_runtime_provider.h"
#endif

namespace taichi::lang {
namespace {
struct VkfftPlanDescription {
  static constexpr const char *kind = "vkfft_plan_f32";
  static constexpr std::uint32_t version = 1;
  VulkanFftPlanConfig config;
  NativeArrayBinding data;
};

class VkfftPlanResource : public NativeResource {
 public:
  virtual void record(CommandList *commands) = 0;
};

struct VkfftExecuteOperation {
  static constexpr const char *kind = "vkfft_execute_f32";
  static constexpr std::uint32_t version = 1;
  std::shared_ptr<VkfftPlanResource> plan;
};

void validate_plan(Device &device, const VkfftPlanDescription &description) {
  const auto bytes = vulkan_fft_storage_bytes(description.config);
  const auto &data = description.data;
  if (data.pointer.device != &device || data.bytes != bytes)
    throw std::invalid_argument(
        "VkFFT requires local storage containing the complete compact "
        "transform");
  // ABI1 binds a whole VkBuffer, with no field for a subview byte offset.
  if (data.pointer.offset)
    throw std::invalid_argument(
        "VkFFT does not support buffer subview offsets");
  auto shape = description.config.dimensions;
  if (description.config.batches > 1)
    shape.insert(shape.begin(), description.config.batches);
  shape.push_back(2);
  validate_native_array_layout(data, {shape, {}, 4, 4, true, "f32"});
  resolve_native_storage_range(Arch::vulkan, data.pointer, bytes);
}

VulkanFftPlanConfig decode_config(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"dimensions", "batch_count", "direction", "normalization",
             "batch_tile"});
  VulkanFftPlanConfig config;
  const auto &dimensions = p["dimensions"];
  if (!dimensions.is_arr())
    throw std::invalid_argument("VkFFT dimensions must be an array");
  for (const auto &extent : dimensions.arr.inner)
    config.dimensions.push_back(integer(extent));
  config.batches = optional_integer(p, "batch_count").value_or(1);
  config.batch_tile = optional_integer(p, "batch_tile").value_or(0);
  const auto direction = p.obj.inner.find("direction");
  if (direction != p.obj.inner.end()) {
    const auto value = name(direction->second);
    if (value != "forward" && value != "inverse")
      throw std::invalid_argument("VkFFT direction must be forward or inverse");
    config.direction = value == "forward" ? -1 : 1;
  }
  const auto normalization = p.obj.inner.find("normalization");
  if (normalization != p.obj.inner.end()) {
    const auto value = name(normalization->second);
    if (value != "none" && value != "inverse")
      throw std::invalid_argument(
          "VkFFT normalization must be none or inverse");
    config.normalize_inverse = value == "inverse";
  }
  vulkan_fft_storage_bytes(config);
  return config;
}

#ifdef TI_WITH_VULKAN
NativeOperationAvailability probe_vkfft(NativeProviderSession &providers) {
  const auto runtime = providers.runtime(vulkan_fft_provider_definition());
  const auto api = runtime->execution_api<TiForgeVkfftExecutionApi>();
  const bool available =
      api.provider.struct_size >= sizeof(TiForgeVkfftApi) &&
      api.provider.abi_version == TI_FORGE_VKFFT_ABI_VERSION &&
      api.provider.vkfft_version == 10304 && api.provider.create &&
      api.provider.destroy && api.provider.append && api.provider.memory &&
      api.provider.last_error && api.record_inline;
  return {available, available ? api.provider.vkfft_version : 0};
}

class PreparedVkfftPlan final : public VkfftPlanResource {
 public:
  PreparedVkfftPlan(NativePreparationContext &context,
                    const VkfftPlanDescription &description)
      : device_(context.device) {
    auto &device = dynamic_cast<vulkan::VulkanDevice &>(device_);
    const auto storage =
        static_cast<DeviceAllocation>(description.data.pointer);
    core_ = make_vulkan_fft_plan(
        device, storage, description.data.bytes, device.get_vkbuffer(storage),
        description.config,
        context.providers.runtime(vulkan_fft_provider_definition()));
    if (!core_->supports_inline_recording())
      throw std::invalid_argument(
          "AOT VkFFT requires the inline recording extension");
    workspace_ = core_->statistics().at("persistent_allocation_bytes");
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return workspace_;
  }
  void record(CommandList *commands) override {
    core_->record(commands);
  }

 private:
  Device &device_;
  std::shared_ptr<VulkanFftPlanCore> core_;
  std::size_t workspace_{0};
};

class PreparedVkfftExecution final : public NativeOperation {
 public:
  explicit PreparedVkfftExecution(std::shared_ptr<VkfftPlanResource> plan)
      : plan_(std::move(plan)) {
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return plan_->device();
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    plan_->record(commands);
  }

 private:
  std::shared_ptr<VkfftPlanResource> plan_;
};
#endif
}  // namespace

void register_vkfft_operations(NativeOperationRegistry &registry) {
  registry.define<VkfftPlanDescription>(validate_plan,
                                        NativeObjectKind::resource);
  registry.register_schema<VkfftPlanDescription>(
      "vkfft", "plan_f32", {"data"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<VkfftPlanDescription> {
        const auto config = decode_config(p);
        return [config](const NativeOperationBindings &bindings) {
          return VkfftPlanDescription{config, bindings.at("data").array()};
        };
      });
  registry.define<VkfftExecuteOperation>(
      [](Device &device, const VkfftExecuteOperation &operation) {
        if (!operation.plan || &operation.plan->device() != &device)
          throw std::invalid_argument("VkFFT plan belongs to another device");
      });
  registry.register_schema<VkfftExecuteOperation>(
      "vkfft", "execute_f32", {"plan"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<VkfftExecuteOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &bindings) {
          return VkfftExecuteOperation{
              bindings.at("plan").resource<VkfftPlanResource>()};
        };
      });
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<VkfftPlanDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context,
         const VkfftPlanDescription &description) {
        return std::make_shared<PreparedVkfftPlan>(context, description);
      },
      probe_vkfft);
  registry.register_context_factory<VkfftExecuteOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const VkfftExecuteOperation &operation) {
        return std::make_shared<PreparedVkfftExecution>(operation.plan);
      });
#endif
}
}  // namespace taichi::lang
