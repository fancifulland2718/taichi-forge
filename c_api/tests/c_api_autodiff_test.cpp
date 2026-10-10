#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

class AotAutodiffTest : public CapiTest {
 protected:
  void run_ndarray(TiArch arch) {
    if (!ti::is_arch_available(arch))
      GTEST_SKIP();
    const char *path = std::getenv("TAICHI_AOT_FOLDER_PATH");
    ASSERT_NE(path, nullptr);
    ti::Runtime runtime(arch);
    auto module = runtime.load_aot_module(path);
    ASSERT_TAICHI_SUCCESS();
    for (const std::string name : {"scalar", "vector", "matrix"}) {
      std::vector<uint32_t> element_shape;
      if (name != "scalar")
        element_shape.push_back(2);
      if (name == "matrix")
        element_shape.push_back(2);
      auto x = runtime.allocate_ndarray<float>({4}, element_shape, true);
      auto y = runtime.allocate_ndarray<float>({4}, element_shape, true);
      auto dx = runtime.allocate_ndarray<float>({4}, element_shape, true);
      auto dy = runtime.allocate_ndarray<float>({4}, element_shape, true);
      auto marker = runtime.allocate_ndarray<float>({1}, {}, true);
      auto forward = module.get_kernel(("forward_" + name).c_str());
      auto backward = module.get_kernel(("backward_" + name).c_str());
      auto graph = module.get_compute_graph(("differentiate_" + name).c_str());
      ASSERT_TAICHI_SUCCESS();
      ASSERT_TRUE(forward.is_valid());
      ASSERT_TRUE(backward.is_valid());
      ASSERT_TRUE(graph.is_valid());
      forward.push_arg(x);
      forward.push_arg(y);
      backward.push_arg(x);
      backward.push_arg(y);
      graph["x"] = x;
      graph["y"] = y;
      graph["marker"] = marker;
      std::vector<TiNdArrayGradientBinding> bindings{{0, dx.ndarray()},
                                                     {1, dy.ndarray()}};
      std::vector<TiNamedNdArrayGradientBinding> named{{"x", dx.ndarray()},
                                                       {"y", dy.ndarray()}};
      std::vector<float> inputs(x.scalar_count());
      for (size_t i = 0; i < inputs.size(); ++i)
        inputs[i] = (i + 1) * 0.25f;
      x.write(inputs);
      marker.write(std::vector<float>{0});
      // A valid first Graph dispatch must not run if a later one lacks
      // gradients.
      graph.launch();
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Missing");
      backward.launch();
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Missing");
      runtime.wait();
      std::vector<float> mark(1);
      marker.read(mark);
      EXPECT_EQ(mark[0], 0);
      if (arch == TI_ARCH_X64 && name == "scalar") {
        auto bad = bindings;
        bad[1].argument_index = 0;
        backward.launch_with_gradients(bad);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Duplicate");
        bad = bindings;
        bad[0].gradient.shape.dims[0] = 3;
        backward.launch_with_gradients(bad);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "shape");
        bad = bindings;
        bad[0].gradient.elem_type = TI_DATA_TYPE_I32;
        backward.launch_with_gradients(bad);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "dtype");
        bad = bindings;
        bad[0].gradient.memory = TI_NULL_HANDLE;
        backward.launch_with_gradients(bad);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "non-null");
        bad = bindings;
        bad[0].argument_index = 2;
        backward.launch_with_gradients(bad);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "out-of-range");
        auto bad_named = named;
        bad_named[0].name = "unknown";
        graph.launch_with_gradients(bad_named);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Unknown");
        bad_named = named;
        bad_named[1].name = "x";
        graph.launch_with_gradients(bad_named);
        EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "duplicate");
      }
      for (bool use_graph : {false, true}) {
        for (float seed : {1.0f, -0.5f, 0.0f}) {
          // Rebinding exercises descriptor lifetime and stale-output handling.
          dx = runtime.allocate_ndarray<float>({4}, element_shape, true);
          dx.write(std::vector<float>(inputs.size(), 0));
          dy.write(std::vector<float>(inputs.size(), seed));
          y.write(std::vector<float>(inputs.size(), -100));
          bindings[0].gradient = dx.ndarray();
          named[0].gradient = dx.ndarray();
          if (use_graph) {
            graph.launch_with_gradients(named);
          } else {
            forward.launch_with_gradients(bindings);
            backward.launch_with_gradients(bindings);
          }
          runtime.wait();
          ASSERT_TAICHI_SUCCESS();
          std::vector<float> values(inputs.size()), grads(inputs.size());
          y.read(values);
          dx.read(grads);
          ASSERT_TAICHI_SUCCESS();
          for (size_t i = 0; i < inputs.size(); ++i) {
            EXPECT_FLOAT_EQ(values[i], inputs[i] * inputs[i] + 2 * inputs[i]);
            EXPECT_FLOAT_EQ(grads[i], seed * (2 * inputs[i] + 2));
          }
        }
      }
    }
    ASSERT_TAICHI_SUCCESS();
  }

  void run_field(TiArch arch) {
    if (!ti::is_arch_available(arch))
      GTEST_SKIP();
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

TEST_F(AotAutodiffTest, FieldCpu) {
  run_field(TI_ARCH_X64);
}
TEST_F(AotAutodiffTest, FieldCuda) {
  run_field(TI_ARCH_CUDA);
}
TEST_F(AotAutodiffTest, FieldVulkan) {
  run_field(TI_ARCH_VULKAN);
}
TEST_F(AotAutodiffTest, NdarrayCpu) {
  run_ndarray(TI_ARCH_X64);
}
TEST_F(AotAutodiffTest, NdarrayCuda) {
  run_ndarray(TI_ARCH_CUDA);
}
TEST_F(AotAutodiffTest, NdarrayVulkan) {
  run_ndarray(TI_ARCH_VULKAN);
}

TEST_F(AotAutodiffTest, CpuMemoryRetirement) {
  if (!ti::is_arch_available(TI_ARCH_X64))
    GTEST_SKIP();
  ti::Runtime runtime(TI_ARCH_X64);
  for (int i = 0; i < 3; ++i) {
    auto array = runtime.allocate_ndarray<float>({4}, {}, true);
    ASSERT_TAICHI_SUCCESS();
    std::vector<float> values(4, static_cast<float>(i));
    array.write(values);
    runtime.wait();
    ASSERT_TAICHI_SUCCESS();
    array.destroy();
    ASSERT_TAICHI_SUCCESS();
  }
}
