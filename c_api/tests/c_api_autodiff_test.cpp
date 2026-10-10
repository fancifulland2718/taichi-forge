#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

class AotAutodiffTest : public CapiTest {
 protected:
  void run_field(TiArch arch) {
    if (!ti::is_arch_available(arch)) GTEST_SKIP();
    const char *path = std::getenv("TAICHI_AOT_FOLDER_PATH");
    ASSERT_NE(path, nullptr);
    ti::Runtime runtime(arch);
    auto module = runtime.load_aot_module(path);
    ASSERT_TAICHI_SUCCESS();
    auto output = runtime.allocate_ndarray<float>({8}, {}, true);
    auto initialize = module.get_kernel("initialize");
    auto forward = module.get_kernel("evaluate");
    auto backward = module.get_kernel("evaluate_grad");
    auto readback = module.get_kernel("readback");
    auto graph = module.get_compute_graph("differentiate");
    ASSERT_TAICHI_SUCCESS();
    readback.push_arg(output);
    graph["out"] = output;
    for (bool use_graph : {false, true}) {
      for (float seed : {1.0f, -0.5f, 0.0f}) {
        if (use_graph) {
          graph["seed"] = seed;
          graph.launch();
        } else {
          initialize.clear_args();
          initialize.push_arg(seed);
          initialize.launch();
          forward.launch();
          backward.launch();
          readback.launch();
        }
        runtime.wait();
        ASSERT_TAICHI_SUCCESS();
        std::vector<float> actual(8);
        output.read(actual);
        ASSERT_TAICHI_SUCCESS();
        for (int i = 0; i < 4; ++i) {
          float x = i + 1;
          EXPECT_FLOAT_EQ(actual[i], x * x * x + 2 * x);
          EXPECT_FLOAT_EQ(actual[i + 4], seed * (3 * x * x + 2));
        }
      }
    }
  }
};

TEST_F(AotAutodiffTest, FieldCpu) { run_field(TI_ARCH_X64); }
TEST_F(AotAutodiffTest, FieldCuda) { run_field(TI_ARCH_CUDA); }
TEST_F(AotAutodiffTest, FieldVulkan) { run_field(TI_ARCH_VULKAN); }
