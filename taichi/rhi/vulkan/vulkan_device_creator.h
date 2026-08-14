#pragma once

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif  // VK_NO_PROTOTYPES

#include <taichi/rhi/vulkan/vulkan_common.h>
#include <taichi/rhi/device.h>
#include <taichi/program/kernel_profiler.h>

#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <functional>

namespace taichi::lang {
namespace vulkan {

class VulkanDevice;

namespace detail {

uint32_t select_instance_api_version(uint32_t requested_api_version,
                                     uint32_t loader_api_version);

bool uses_core_physical_device_features2(uint32_t instance_api_version);

bool supports_physical_device_features2(uint32_t instance_api_version,
                                        bool has_extension);

bool supports_8bit_storage(uint32_t device_api_version, bool has_extension);

bool supports_shader_atomic_int64(uint32_t device_api_version,
                                  bool has_extension);

void record_shader_atomic_float2_capabilities(
    DeviceCapabilityConfig &caps,
    const VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT &features);

}  // namespace detail

struct VulkanQueueFamilyIndices {
  std::optional<uint32_t> compute_family;
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  // Compute/graphics families already support transfer operations. A separate
  // family is deliberately not selected without an end-to-end ownership and
  // overlap policy; merely discovering one cannot safely speed up copies.

  bool is_complete() const {
    return compute_family.has_value();
  }

  bool is_complete_for_ui() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

/**
 * This class creates a VulkanDevice instance. The underlying Vk* resources are
 * embedded directly inside the class.
 */
class TI_DLL_EXPORT VulkanDeviceCreator {
 public:
  struct Params {
    // User-provided API version. If assigned, the users MUST list all
    // their desired extensions in `additional_instance_extensions` and
    // `additional_device_extensions`; no extension is enabled by default.
    std::optional<uint32_t> api_version;
    bool is_for_ui{false};
    std::vector<std::string> additional_instance_extensions;
    std::vector<std::string> additional_device_extensions;
    // the VkSurfaceKHR needs to be created after creating the VkInstance, but
    // before creating the VkPhysicalDevice thus, we allow the user to pass in a
    // custom surface creator
    std::function<VkSurfaceKHR(VkInstance)> surface_creator;
    bool enable_validation_layer{false};
  };

  explicit VulkanDeviceCreator(const Params &params);
  ~VulkanDeviceCreator();

  const VulkanDevice *device() const {
    return ti_device_.get();
  }

  VulkanDevice *device() {
    return ti_device_.get();
  }

 private:
  void create_instance(uint32_t vk_api_version, bool manual_create);
  void setup_debug_messenger();
  void pick_physical_device(VkSurfaceKHR test_surface);
  void create_logical_device(bool manual_create);
  void query_physical_device_features2(
      VkPhysicalDeviceFeatures2 *features) const;
  void query_physical_device_properties2(
      VkPhysicalDeviceProperties2 *properties) const;

  VkInstance instance_{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VulkanQueueFamilyIndices queue_family_indices_;
  VkDevice device_{VK_NULL_HANDLE};

  VkQueue compute_queue_{VK_NULL_HANDLE};
  VkQueue graphics_queue_{VK_NULL_HANDLE};

  uint32_t instance_api_version_{VK_API_VERSION_1_0};

  std::unique_ptr<VulkanDevice> ti_device_{nullptr};

  Params params_;
};

}  // namespace vulkan
}  // namespace taichi::lang
