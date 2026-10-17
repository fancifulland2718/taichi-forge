#include "taichi/rhi/vulkan/vulkan_pipeline_diagnostics.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include "external/PicoSHA2/picosha2.h"
#include "taichi/common/filesystem.hpp"
#include "taichi/common/json_serde.h"
#include "taichi/system/profiler.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv_reflect.h"

namespace taichi::lang::vulkan {
namespace {
using liong::json::JsonArray;
using liong::json::JsonObject;
using liong::json::JsonValue;

std::string hex_bytes(const uint8_t *data, size_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (size_t i = 0; i < size; ++i) {
    result += digits[data[i] >> 4];
    result += digits[data[i] & 15];
  }
  return result;
}

void write_file(const std::filesystem::path &path,
                const void *data,
                size_t size) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char *>(data), size);
  out.flush();
  if (!out) {
    throw std::runtime_error("Cannot write Vulkan diagnostic artifact: " +
                             path.string());
  }
}
}  // namespace

std::string vulkan_device_identity_json(const VulkanDevice &device) {
  const auto &props = device.get_vk_physical_device_props();
  VkPhysicalDeviceIDProperties ids{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties subgroup{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 query{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  query.pNext = &ids;
  ids.pNext = &subgroup;
  if (auto get = device.vk_device()
                     .instance_functions()
                     .vkGetPhysicalDeviceProperties2) {
    get(device.vk_physical_device(), &query);
  } else if (auto get = device.vk_device()
                            .instance_functions()
                            .vkGetPhysicalDeviceProperties2KHR) {
    get(device.vk_physical_device(), &query);
  }
  JsonObject result;
  auto &r = result.inner;
  r["device_name"] = JsonValue(props.deviceName);
  r["device_uuid"] = JsonValue(hex_bytes(ids.deviceUUID, VK_UUID_SIZE));
  r["driver_uuid"] = JsonValue(hex_bytes(ids.driverUUID, VK_UUID_SIZE));
  r["pipeline_cache_uuid"] =
      JsonValue(hex_bytes(props.pipelineCacheUUID, VK_UUID_SIZE));
  r["vendor_id"] = JsonValue(props.vendorID);
  r["device_id"] = JsonValue(props.deviceID);
  r["api_version"] = JsonValue(props.apiVersion);
  r["driver_version"] = JsonValue(props.driverVersion);
  r["subgroup_size"] = JsonValue(subgroup.subgroupSize);
  r["subgroup_operations"] = JsonValue(subgroup.supportedOperations);
  r["subgroup_stages"] = JsonValue(subgroup.supportedStages);
  r["max_workgroup_invocations"] =
      JsonValue(props.limits.maxComputeWorkGroupInvocations);
  r["max_workgroup_size_x"] =
      JsonValue(props.limits.maxComputeWorkGroupSize[0]);
  r["max_shared_memory_bytes"] =
      JsonValue(props.limits.maxComputeSharedMemorySize);
  return liong::json::print(JsonValue(std::move(result)));
}

VulkanPipelineDiagnostics::VulkanPipelineDiagnostics(
    const VulkanPipeline::Params &params) {
#ifdef _WIN32
  const auto *root = _wgetenv(L"TI_VULKAN_DIAGNOSTICS_DIR");
#else
  const auto *root = std::getenv("TI_VULKAN_DIAGNOSTICS_DIR");
#endif
  const char *validation = std::getenv("TI_VULKAN_VALIDATE_SPIRV");
  const bool validate = validation && std::string(validation) == "1";
  if ((!root || !*root) && !validate)
    return;
  TI_COMPILE_PROFILER("cpp.vulkan.pipeline_diagnostics");
  if (root && *root) {
    std::filesystem::create_directories(root);
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (uint64_t attempt = 0;; ++attempt) {
      const auto candidate =
          std::filesystem::path(root) /
          (std::to_string(stamp) + "-" + std::to_string(attempt));
      if (std::filesystem::create_directory(candidate)) {
        directory_ = candidate.u8string();
        break;
      }
    }
  }
  JsonObject manifest;
  manifest.inner["schema_version"] = JsonValue(1);
  manifest.inner["name"] = JsonValue(params.name);
  manifest.inner["device"] =
      liong::json::parse(vulkan_device_identity_json(*params.device));
  manifest.inner["pipeline_cache_attached"] = JsonValue(bool(params.cache));
  manifest.inner["specialization_constants"] = JsonValue(JsonArray{});
  manifest.inner["push_constant_bytes"] = JsonValue(128);
  JsonArray shaders;
  std::string validation_error;
  for (size_t i = 0; i < params.code.size(); ++i) {
    const auto &code = params.code[i];
    const auto *bytes = reinterpret_cast<const uint8_t *>(code.data);
    const std::string filename = "shader-" + std::to_string(i) + ".spv";
    if (!directory_.empty())
      write_file(std::filesystem::u8path(directory_) / filename, bytes,
                 code.size);
    JsonObject shader;
    bool validation_passed = true;
    if (validate) {
      TI_COMPILE_PROFILER("cpp.vulkan.validate_spirv");
      const auto api = params.device->get_vk_physical_device_props().apiVersion;
      const auto env = api >= VK_API_VERSION_1_3   ? SPV_ENV_VULKAN_1_3
                       : api >= VK_API_VERSION_1_2 ? SPV_ENV_VULKAN_1_2
                       : api >= VK_API_VERSION_1_1 ? SPV_ENV_VULKAN_1_1
                                                   : SPV_ENV_VULKAN_1_0;
      spvtools::SpirvTools validator(env);
      std::string messages;
      validator.SetMessageConsumer([&](spv_message_level_t, const char *,
                                       const spv_position_t &pos,
                                       const char *message) {
        messages += std::to_string(pos.index) + ": " + message + "\n";
      });
      validation_passed =
          code.size % sizeof(uint32_t) == 0 &&
          validator.Validate(code.data, code.size / sizeof(uint32_t));
      shader.inner["validation_passed"] = JsonValue(validation_passed);
      shader.inner["validation_messages"] = JsonValue(messages);
      if (!validation_passed)
        validation_error += filename + ": " + messages;
    }
    // Validation alone does not consume a manifest or reflection. The normal
    // pipeline preparation reflects the shader after this validation succeeds.
    if (directory_.empty())
      continue;
    shader.inner["file"] = JsonValue(filename);
    shader.inner["sha256"] =
        JsonValue(picosha2::hash256_hex_string(bytes, bytes + code.size));
    shader.inner["bytes"] = JsonValue(code.size);
    // Validate before passing malformed code to reflection or the driver.
    TI_COMPILE_PROFILER("cpp.vulkan.diagnostic_reflection");
    SpvReflectShaderModule reflection{};
    shader.inner["entry_point"] = JsonValue(code.entry_point);
    if (validation_passed &&
        spvReflectCreateShaderModule(code.size, code.data, &reflection) ==
            SPV_REFLECT_RESULT_SUCCESS) {
      JsonArray bindings;
      for (uint32_t j = 0; j < reflection.descriptor_binding_count; ++j) {
        const auto &b = reflection.descriptor_bindings[j];
        JsonObject binding;
        binding.inner["set"] = JsonValue(b.set);
        binding.inner["binding"] = JsonValue(b.binding);
        binding.inner["type"] = JsonValue(int(b.descriptor_type));
        binding.inner["count"] = JsonValue(b.count);
        bindings.inner.emplace_back(std::move(binding));
      }
      shader.inner["descriptor_bindings"] = JsonValue(std::move(bindings));
      JsonArray local_size;
      for (uint32_t j = 0; j < reflection.entry_point_count; ++j) {
        if (code.entry_point == reflection.entry_points[j].name) {
          const auto &s = reflection.entry_points[j].local_size;
          for (auto v : {s.x, s.y, s.z})
            local_size.inner.emplace_back(v);
        }
      }
      shader.inner["local_size"] = JsonValue(std::move(local_size));
      spvReflectDestroyShaderModule(&reflection);
    }
    shaders.inner.emplace_back(std::move(shader));
  }
  manifest.inner["shaders"] = JsonValue(std::move(shaders));
  if (!directory_.empty()) {
    const auto json = liong::json::print(JsonValue(std::move(manifest)));
    write_file(std::filesystem::u8path(directory_) / "pipeline.json",
               json.data(), json.size());
  }
  if (!validation_error.empty()) {
    event("validation_failed", validation_error);
    throw std::invalid_argument(
        "Vulkan SPIR-V validation failed before pipeline creation: " +
        validation_error);
  }
  event(validate ? "validation_passed" : "validation_not_requested");
}

void VulkanPipelineDiagnostics::event(const std::string &phase,
                                      const std::string &message,
                                      std::optional<int64_t> result) {
  if (directory_.empty())
    return;
  JsonObject value;
  value.inner["phase"] = JsonValue(phase);
  value.inner["message"] = JsonValue(message);
  value.inner["vk_result"] = result ? JsonValue(*result) : JsonValue(nullptr);
  std::ofstream out(std::filesystem::u8path(directory_) / "events.jsonl",
                    std::ios::app);
  out << liong::json::print(JsonValue(std::move(value))) << '\n';
  out.flush();
  if (!out)
    throw std::runtime_error("Cannot flush Vulkan pipeline diagnostic event");
}
}  // namespace taichi::lang::vulkan
