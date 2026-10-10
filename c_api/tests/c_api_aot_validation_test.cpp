#include <chrono>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, AotVulkanCapabilityRequirements) {
  if (!ti::is_arch_available(TI_ARCH_VULKAN)) GTEST_SKIP();
  ti::Runtime runtime(TI_ARCH_VULKAN);
  const auto available = runtime.get_capabilities().get(TI_CAPABILITY_SPIRV_VERSION);
  ASSERT_GE(available, 0x10000);
  auto path = std::filesystem::temp_directory_path() /
      ("forge-aot-caps-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(path);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::filesystem::remove(path / "metadata.json");
      std::filesystem::remove(path / "graphs.json");
      std::filesystem::remove(path);
    }
  } cleanup{path};
  std::ofstream(path / "graphs.json") << "[]";
  for (uint32_t required : {uint32_t(0x10000), available, available + 0x100}) {
    std::ofstream(path / "metadata.json")
        << "{\"metadata_version\":1,\"kernels\":[],\"kernel_metadata\":[],"
           "\"fields\":[],\"required_caps\":[{\"key\":\"spirv_version\",\"value\":"
        << required << "}],\"root_buffer_size\":0,\"root_buffer_sizes\":[]}";
    auto module = ti_load_aot_module(runtime, path.string().c_str());
    if (required <= available) {
      ASSERT_NE(module, nullptr);
      ASSERT_TAICHI_SUCCESS();
      ti_destroy_aot_module(module);
    } else {
      EXPECT_EQ(module, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_INCOMPATIBLE_MODULE, "spirv_version");
      if (module) ti_destroy_aot_module(module);
    }
  }
}
