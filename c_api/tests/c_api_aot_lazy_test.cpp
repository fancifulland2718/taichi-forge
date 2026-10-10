#include <fstream>
#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, AotVulkanLazyDirectoryAndArchive) {
  if (!ti::is_arch_available(TI_ARCH_VULKAN))
    GTEST_SKIP();
  const char *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  const std::string archive_path = std::string(folder) + "/module.tcm";
  for (int source_kind = 0; source_kind < 3; ++source_kind) {
    ti::Runtime runtime(TI_ARCH_VULKAN);
    TiAotModule handle;
    if (source_kind == 2) {
      std::ifstream stream(archive_path, std::ios::binary);
      ASSERT_TRUE(stream.is_open());
      std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
      handle = ti_create_aot_module(runtime, bytes.data(), bytes.size());
      // The caller's compressed buffer dies before any kernel lookup.
    } else {
      handle = ti_load_aot_module(
          runtime, source_kind == 0 ? folder : archive_path.c_str());
    }
    ASSERT_NE(handle, nullptr);
    ASSERT_TAICHI_SUCCESS();
    ti::AotModule module(runtime, handle, true);
    auto graph = module.get_compute_graph("good");
    ASSERT_TAICHI_SUCCESS();
    auto out = runtime.allocate_ndarray<int32_t>({4}, {}, true);
    graph["out"] = out;
    for (int repeat = 0; repeat < 2; ++repeat) {
      graph.launch();
      runtime.wait();
      ASSERT_TAICHI_SUCCESS();
      std::vector<int32_t> actual(4);
      out.read(actual);
      EXPECT_EQ(actual, (std::vector<int32_t>{7, 8, 9, 10}));
      EXPECT_EQ(ti_get_aot_module_compute_graph(handle, "bad"), nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA, ".spv");
      EXPECT_EQ(ti_get_aot_module_kernel(handle, "absent"), nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_NAME_NOT_FOUND, "absent");
    }
  }
}
