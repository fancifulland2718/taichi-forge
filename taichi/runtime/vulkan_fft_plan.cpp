#include "taichi/runtime/vulkan_fft_plan.h"

#ifdef TI_WITH_VULKAN
#include <limits>
#include "taichi/common/dynamic_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/vkfft/forge_vkfft_provider.h"

namespace taichi::lang {
namespace {

std::uint64_t validate_config(const VulkanFftPlanConfig &config) {
  TI_ERROR_IF(config.dimensions.empty() || config.dimensions.size() > 3 ||
                  !config.batches ||
                  (config.direction != -1 && config.direction != 1) ||
                  config.batch_tile > config.batches,
              "Vulkan FFT requires rank 1--3, positive batches, direction "
              "-1/+1 and a batch tile no greater than the batch count");
  std::uint64_t bytes = 2 * sizeof(float);
  for (auto extent : config.dimensions) {
    auto remaining = extent;
    TI_ERROR_IF(!remaining, "Vulkan FFT dimensions must be positive");
    for (auto radix : {2, 3, 5, 7, 11, 13}) {
      while (remaining % radix == 0)
        remaining /= radix;
    }
    TI_ERROR_IF(remaining != 1,
                "Vulkan FFT dimensions must have prime factors at most 13");
    TI_ERROR_IF(extent > std::numeric_limits<std::uint64_t>::max() / bytes,
                "Vulkan FFT storage size overflow");
    bytes *= extent;
  }
  TI_ERROR_IF(
      config.batches > std::numeric_limits<std::uint64_t>::max() / bytes,
      "Vulkan FFT storage size overflow");
  return bytes * config.batches;
}

class VulkanFftPlanImpl final
    : public VulkanFftPlanCore,
      public vkapi::DeviceObj,
      public std::enable_shared_from_this<VulkanFftPlanImpl> {
 public:
  explicit VulkanFftPlanImpl(const std::string &path) : library_(path) {
    TI_ERROR_IF(!library_.loaded(), "Vulkan FFT adapter is unavailable: {}",
                path);
    TiForgeVkfftQueryFn query{};
    library_.load_function(TI_FORGE_VKFFT_QUERY_SYMBOL, query);
    TI_ERROR_IF(query(TI_FORGE_VKFFT_ABI_VERSION, sizeof(api_), &api_) != 0 ||
                    api_.struct_size != sizeof(api_) ||
                    api_.abi_version != TI_FORGE_VKFFT_ABI_VERSION ||
                    api_.vkfft_version != 10304 || !api_.create ||
                    !api_.append || !api_.memory || !api_.destroy ||
                    !api_.last_error,
                "Vulkan FFT adapter ABI is incompatible: {}", path);
    auto recipe_query = reinterpret_cast<TiForgeVkfftRecipeQueryFn>(
        library_.load_function_optional(TI_FORGE_VKFFT_RECIPE_QUERY_SYMBOL));
    if (recipe_query) {
      TI_ERROR_IF(
          recipe_query(TI_FORGE_VKFFT_RECIPE_ABI_VERSION, sizeof(recipe_api_),
                       &recipe_api_) != 0 ||
              recipe_api_.struct_size != sizeof(recipe_api_) ||
              recipe_api_.abi_version != TI_FORGE_VKFFT_RECIPE_ABI_VERSION ||
              !recipe_api_.create || !recipe_api_.describe,
          "Vulkan FFT recipe extension ABI is incompatible: {}", path);
    }
    record_inline_ = reinterpret_cast<TiForgeVkfftRecordInlineFn>(
        library_.load_function_optional(TI_FORGE_VKFFT_RECORD_INLINE_SYMBOL));
  }

  ~VulkanFftPlanImpl() override {
    if (handle_)
      api_.destroy(handle_);
  }

  void initialize(vulkan::VulkanDevice &owner,
                  DeviceAllocation storage,
                  std::size_t bytes,
                  std::shared_ptr<void> lease,
                  const VulkanFftPlanConfig &description) {
    device = owner.vk_device();
    storage_lease_ = std::move(lease);
    buffer_ = owner.get_vkbuffer(storage);
    const auto &properties = owner.get_vk_physical_device_props();
    vendor_id_ = properties.vendorID;
    device_id_ = properties.deviceID;
    driver_version_ = properties.driverVersion;
    api_version_ = properties.apiVersion;

    TiForgeVkfftConfig config{};
    config.struct_size = sizeof(config);
    config.rank = static_cast<std::uint32_t>(description.dimensions.size());
    for (std::size_t axis = 0; axis < description.dimensions.size(); ++axis)
      config.dimensions[axis] = description.dimensions[axis];
    config.batches = description.batches;
    config.physical_device = owner.vk_physical_device();
    config.device = owner.vk_device();
    config.queue = owner.compute_queue();
    config.queue_family = owner.compute_queue_family_index();
    config.direction = description.direction;
    config.normalize_inverse = description.normalize_inverse;
    config.buffer = buffer_->buffer;
    config.buffer_bytes = bytes;
    auto queue_lock = owner.acquire_external_compute_queue_lock();
    int result;
    if (description.batch_tile) {
      TI_ERROR_IF(!recipe_api_.create,
                  "Vulkan FFT adapter has no batch recipe extension");
      TiForgeVkfftRecipeConfig recipe{sizeof(TiForgeVkfftRecipeConfig), 0,
                                      description.batch_tile};
      result = recipe_api_.create(&config, &recipe, &handle_);
    } else {
      result = api_.create(&config, &handle_);
    }
    TI_ERROR_IF(result != 0, "Vulkan FFT plan creation failed: {}",
                api_.last_error());
  }

  bool supports_inline_recording() const override {
    return record_inline_ != nullptr;
  }

  void record(CommandList *commands, bool prefer_inline) override {
    auto *list = static_cast<vulkan::VulkanCommandList *>(commands);
    const auto command = list->begin_external_compute(shared_from_this());
    const auto result = prefer_inline && record_inline_
                            ? record_inline_(handle_, command)
                            : api_.append(handle_, command);
    TI_ERROR_IF(result != 0, "Vulkan FFT recording failed: {}",
                api_.last_error());
  }

  std::unordered_map<std::string, std::uint64_t> statistics() const override {
    TiForgeVkfftMemory memory{};
    api_.memory(handle_, &memory);
    std::unordered_map<std::string, std::uint64_t> result{
        {"persistent_allocation_bytes", memory.persistent_allocation_bytes},
        {"initialization_peak_allocation_bytes",
         memory.initialization_peak_allocation_bytes},
        {"persistent_allocation_count", memory.persistent_allocation_count},
        {"temporary_buffer_bytes", memory.temporary_buffer_bytes},
        {"adapter_abi", api_.abi_version},
        {"inline_recording_available", record_inline_ ? 1 : 0},
        {"vkfft_version", api_.vkfft_version},
        {"glslang_major", api_.glslang_major},
        {"glslang_minor", api_.glslang_minor},
        {"glslang_patch", api_.glslang_patch},
        {"device_vendor_id", vendor_id_},
        {"device_id", device_id_},
        {"vulkan_driver_version", driver_version_},
        {"vulkan_api_version", api_version_}};
    if (recipe_api_.describe) {
      TiForgeVkfftRecipeFacts facts{};
      recipe_api_.describe(handle_, &facts);
      result.insert({{"recipe_extension_abi", recipe_api_.abi_version},
                     {"batch_tile", facts.batch_tile},
                     {"application_count", facts.application_count},
                     {"shader_module_count", facts.shader_module_count},
                     {"dispatch_count", facts.dispatch_count},
                     {"shader_fingerprint", facts.shader_fingerprint},
                     {"dispatch_fingerprint", facts.dispatch_fingerprint}});
    }
    return result;
  }

 private:
  // Keep the library loaded through plan destruction, then release storage.
  DynamicLoader library_;
  TiForgeVkfftApi api_{};
  TiForgeVkfftRecipeApi recipe_api_{};
  TiForgeVkfftRecordInlineFn record_inline_{nullptr};
  TiForgeVkfftPlan handle_{};
  std::shared_ptr<void> storage_lease_;
  vkapi::IVkBuffer buffer_;
  std::uint32_t vendor_id_{0}, device_id_{0}, driver_version_{0},
      api_version_{0};
};
}  // namespace

std::shared_ptr<VulkanFftPlanCore> make_vulkan_fft_plan(
    vulkan::VulkanDevice &device,
    DeviceAllocation storage,
    std::size_t storage_bytes,
    std::shared_ptr<void> storage_lease,
    const VulkanFftPlanConfig &config,
    const std::string &adapter_path) {
  const auto required = validate_config(config);
  TI_ERROR_IF(storage.device != &device || !storage_lease,
              "Vulkan FFT requires local storage and its lifetime lease");
  TI_ERROR_IF(
      storage_bytes != required || device.get_vkbuffer_size(storage) < required,
      "Vulkan FFT buffer must contain the complete compact transform");
  auto plan = std::make_shared<VulkanFftPlanImpl>(adapter_path);
  plan->initialize(device, storage, storage_bytes, std::move(storage_lease),
                   config);
  return plan;
}

}  // namespace taichi::lang
#endif
