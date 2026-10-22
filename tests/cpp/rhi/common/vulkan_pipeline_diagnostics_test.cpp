#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include <cstdlib>
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

namespace taichi::lang::vulkan {
namespace {
// Empty Vulkan 1.0 compute shader with LocalSize 1 1 1, assembled by spirv-as.
const uint32_t kShader[] = {
    0x07230203, 0x00010000, 0x00070000, 0x00000005, 0x00000000, 0x00020011,
    0x00000001, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
    0x00000001, 0x6e69616d, 0x00000000, 0x00060010, 0x00000001, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00050036, 0x00000002, 0x00000001, 0x00000000,
    0x00000003, 0x000200f8, 0x00000004, 0x000100fd, 0x00010038};

int pipeline_calls = 0;
VKAPI_ATTR VkResult VKAPI_CALL
fail_pipeline(VkDevice,
              VkPipelineCache,
              uint32_t,
              const VkComputePipelineCreateInfo *,
              const VkAllocationCallbacks *,
              VkPipeline *) {
  ++pipeline_calls;
  return VK_ERROR_INITIALIZATION_FAILED;
}

class ScopedValidation {
 public:
  ScopedValidation() {
    const char *previous = std::getenv("TI_VULKAN_VALIDATE_SPIRV");
    existed_ = previous != nullptr;
    previous_ = previous ? previous : "";
    set("1");
  }
  ~ScopedValidation() {
#ifdef _WIN32
    set(previous_.c_str());
#else
    if (existed_)
      set(previous_.c_str());
    else
      unsetenv("TI_VULKAN_VALIDATE_SPIRV");
#endif
  }

 private:
  static void set(const char *value) {
#ifdef _WIN32
    _putenv_s("TI_VULKAN_VALIDATE_SPIRV", value);
#else
    setenv("TI_VULKAN_VALIDATE_SPIRV", value, 1);
#endif
  }
  bool existed_;
  std::string previous_;
};

class FailureRecorder : public BackendFaultReporter {
 public:
  int reports{0};
  int64_t code{0};
  std::string operation;
  void report_backend_error(const BackendRuntimeError &error,
                            uint64_t) noexcept override {
    ++reports;
    code = error.backend_code();
    operation = error.operation();
  }
  bool backend_calls_safe() const noexcept override {
    return true;
  }
  void throw_if_submission_disallowed(const char *) const override {
  }
};

TEST(VulkanPipelineDiagnostics,
     ValidationAndDriverFailureStayInsideRhiBoundary) {
  if (!VulkanLoader::instance().init() ||
      !VulkanLoader::instance().check_vulkan_device()) {
    GTEST_SKIP() << "Vulkan device unavailable";
  }
  ScopedValidation validation;
  VulkanDeviceCreator creator(VulkanDeviceCreator::Params{});
  auto *device = creator.device();
  auto reporter = std::make_shared<FailureRecorder>();
  device->set_backend_fault_reporter(reporter);
  auto &table = const_cast<VolkDeviceTable &>(device->vk_device().functions());
  struct Restore {
    VolkDeviceTable &table;
    PFN_vkCreateComputePipelines original;
    std::shared_ptr<BackendFaultReporter> reporter;
    Device *device;
    ~Restore() {
      table.vkCreateComputePipelines = original;
      device->clear_backend_fault_reporter(reporter);
    }
  } restore{table, table.vkCreateComputePipelines, reporter, device};
  table.vkCreateComputePipelines = fail_pipeline;
  pipeline_calls = 0;

  PipelineSourceDesc source{};
  source.type = PipelineSourceType::spirv_binary;
  source.stage = PipelineStageType::compute;
  source.data = kShader;
  source.size = sizeof(kShader);
  Pipeline *pipeline = nullptr;
  EXPECT_EQ(
      device->create_pipeline(&pipeline, source, "injected_failure", nullptr),
      RhiResult::error);
  EXPECT_EQ(pipeline, nullptr);
  EXPECT_EQ(pipeline_calls, 1);
  EXPECT_EQ(reporter->reports, 1);
  EXPECT_EQ(reporter->code, VK_ERROR_INITIALIZATION_FAILED);
  EXPECT_EQ(reporter->operation, "vkCreateComputePipelines");

  // Truncated module must be rejected before reaching the injected driver.
  source.size = 5 * sizeof(uint32_t);
  EXPECT_EQ(
      device->create_pipeline(&pipeline, source, "invalid_spirv", nullptr),
      RhiResult::invalid_usage);
  EXPECT_EQ(pipeline, nullptr);
  EXPECT_EQ(pipeline_calls, 1);
  EXPECT_EQ(reporter->reports, 1);
}
}  // namespace
}  // namespace taichi::lang::vulkan
#endif
