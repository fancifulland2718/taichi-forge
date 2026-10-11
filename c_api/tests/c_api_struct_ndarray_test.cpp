#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "taichi/taichi_cpu.h"
#include "c_api/tests/gtest_fixture.h"

#include <cstring>
#include <unordered_map>

class AotStructNdarrayTest : public CapiTest {
 protected:
  void run(TiArch arch) {
    if (!ti::is_arch_available(arch))
      GTEST_SKIP();
    const char *path = std::getenv("TAICHI_AOT_FOLDER_PATH");
    ASSERT_NE(path, nullptr);
    ti::Runtime runtime(arch);
    auto module = runtime.load_aot_module(path);
    ASSERT_TRUE(module.is_valid());
    auto kernel = module.get_kernel("update");
    auto graph = module.get_compute_graph("step");
    ASSERT_TRUE(kernel.is_valid());
    ASSERT_TRUE(graph.is_valid());
    auto layout = ti_get_kernel_ndarray_layout_ext(runtime, kernel, 0);
    auto graph_layout =
        ti_get_compute_graph_ndarray_layout_ext(runtime, graph, "records");
    ASSERT_NE(layout, nullptr);
    EXPECT_EQ(layout, graph_layout);
    TiNdArrayLayoutInfo info{};
    ti_get_ndarray_layout_info_ext(runtime, layout, &info);
    ASSERT_EQ(info.version, 1);
    EXPECT_EQ(info.element_stride, 40);
    EXPECT_EQ(info.alignment, 4);
    EXPECT_EQ(info.member_count, 5);
    std::unordered_map<std::string, TiNdArrayLayoutMember> members;
    for (uint32_t i = 0; i < info.member_count; ++i) {
      TiNdArrayLayoutMember member{};
      ti_get_ndarray_layout_member_ext(runtime, layout, i, &member);
      ASSERT_NE(member.path, nullptr);
      members.emplace(member.path, member);
    }
    EXPECT_EQ(members.at("inner.code").offset, 4);
    EXPECT_EQ(members.at("inner.value").offset, 8);
    EXPECT_EQ(members.at("basis").element_shape.dim_count, 2);
    TiMemoryAllocateInfo allocation{};
    allocation.size = 4 * info.element_stride;
    allocation.host_write = allocation.host_read = TI_TRUE;
    allocation.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    auto records = runtime.allocate_memory(allocation);
    auto result = runtime.allocate_ndarray<float>({4}, {}, true);
    std::vector<uint8_t> bytes(allocation.size, 0);
    auto put = [&](int i, const char *path, const auto &value,
                   size_t component = 0) {
      std::memcpy(bytes.data() + i * info.element_stride +
                      members.at(path).offset + component,
                  &value, sizeof(value));
    };
    for (int i = 0; i < 4; ++i) {
      put(i, "tag", int8_t(i + 1));
      put(i, "inner.code", uint16_t(10 + i));
      put(i, "inner.value", float(2 * i));
      put(i, "velocity", float(3 * i), sizeof(float));
      put(i, "basis", float(4 * i), 2 * sizeof(float));
    }
    records.write(bytes.data(), bytes.size());
    result.write(std::vector<float>(4, 0));
    TiArgument args[2]{};
    args[0].type = args[1].type = TI_ARGUMENT_TYPE_NDARRAY;
    args[0].value.ndarray.memory = records;
    args[0].value.ndarray.shape = {1, {4}};
    args[0].value.ndarray.elem_type = TI_DATA_TYPE_UNKNOWN;
    args[1].value.ndarray = result.ndarray();
    TiNamedArgument named[2]{{"records", args[0]}, {"result", args[1]}};
    TiNdArrayBinding binding{0, info.version, layout, nullptr};
    TiNamedNdArrayBinding named_binding{"records", info.version, layout,
                                        nullptr};
    auto check_marker = [&] {
      runtime.wait();
      std::vector<float> values(4);
      result.read(values);
      EXPECT_EQ(values, std::vector<float>(4, 0));
    };
    ti_launch_kernel(runtime, kernel, 2, args);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Missing");
    ti_launch_compute_graph(runtime, graph, 2, named);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Missing");
    check_marker();
    named_binding.layout_version = 2;
    ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                              &named_binding);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "version");
    named_binding.layout_version = info.version;
    auto other = module.get_kernel("other");
    named_binding.layout = ti_get_kernel_ndarray_layout_ext(runtime, other, 0);
    ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                              &named_binding);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "layout");
    named_binding.layout = layout;
    allocation.size -= 1;
    auto short_buffer = runtime.allocate_memory(allocation);
    named[0].argument.value.ndarray.memory = short_buffer;
    ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                              &named_binding);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "smaller");
    named[0] = {"records", args[0]};
    check_marker();
    {
      ti::Runtime peer(TI_ARCH_X64);
      ti_launch_kernel_with_bindings_ext(peer, kernel, 2, args, 1, &binding);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "runtime");
      ti_launch_compute_graph_with_bindings_ext(peer, graph, 2, named, 1,
                                                &named_binding);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "runtime");
      TiNdArrayLayoutInfo bad{};
      ti_get_ndarray_layout_info_ext(peer, layout, &bad);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "runtime");
      auto foreign = peer.allocate_memory(allocation);
      named[0].argument.value.ndarray.memory = foreign;
      ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                                &named_binding);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "runtime");
    }
    named[0] = {"records", args[0]};
    check_marker();
    if (arch == TI_ARCH_X64) {
      std::vector<uint8_t> unaligned(bytes.size() + 1);
      TiMemory imported =
          ti_import_cpu_memory(runtime, unaligned.data() + 1, bytes.size());
      ASSERT_NE(imported, nullptr);
      named[0].argument.value.ndarray.memory = imported;
      ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                                &named_binding);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "alignment");
      ti_free_memory(runtime, imported);
      named[0] = {"records", args[0]};
      check_marker();
    }
    for (int step = 0; step < 3; ++step) {
      if (step == 0)
        ti_launch_kernel_with_bindings_ext(runtime, kernel, 2, args, 1,
                                           &binding);
      else
        ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                                  &named_binding);
      ASSERT_TAICHI_SUCCESS();
      runtime.wait();
      std::vector<float> values(4);
      result.read(values);
      for (int i = 0; i < 4; ++i)
        EXPECT_EQ(values[i], 11 + 11 * i + step);
    }
    // Rebind the same immutable layout to new device storage.
    {
      auto mixed = module.get_compute_graph("differentiate");
      ASSERT_TRUE(mixed.is_valid());
      auto x = runtime.allocate_ndarray<float>({4}, {}, true);
      auto y = runtime.allocate_ndarray<float>({4}, {}, true);
      auto dx = runtime.allocate_ndarray<float>({4}, {}, true);
      auto dy = runtime.allocate_ndarray<float>({4}, {}, true);
      x.write(std::vector<float>{1, 2, 3, 4});
      dx.write(std::vector<float>(4, 0));
      dy.write(std::vector<float>(4, 1));
      TiNamedArgument combined[4]{named[0], named[1], {}, {}};
      combined[2].name = "x";
      combined[3].name = "y";
      combined[2].argument.type = combined[3].argument.type =
          TI_ARGUMENT_TYPE_NDARRAY;
      combined[2].argument.value.ndarray = x.ndarray();
      combined[3].argument.value.ndarray = y.ndarray();
      TiNamedNdArrayBinding bindings[3]{named_binding,
                                        {"x", 0, nullptr, &dx.ndarray()},
                                        {"y", 0, nullptr, &dy.ndarray()}};
      ti_launch_compute_graph_with_bindings_ext(runtime, mixed, 4, combined, 2,
                                                bindings);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "Missing");
      runtime.wait();
      records.read(bytes.data(), bytes.size());
      for (int i = 0; i < 4; ++i) {
        float value;
        std::memcpy(&value,
                    bytes.data() + i * info.element_stride +
                        members.at("inner.value").offset,
                    sizeof(value));
        EXPECT_EQ(value,
                  2 * i + 3);  // No earlier structure dispatch was submitted.
      }
      ti_launch_compute_graph_with_bindings_ext(runtime, mixed, 4, combined, 3,
                                                bindings);
      runtime.wait();
      ASSERT_TAICHI_SUCCESS();
      std::vector<float> values(4);
      y.read(values);
      EXPECT_EQ(values, (std::vector<float>{1, 4, 9, 16}));
      dx.read(values);
      EXPECT_EQ(values, (std::vector<float>{2, 4, 6, 8}));
      // Restore the original fixture for the independent rebinding check.
      for (int i = 0; i < 4; ++i)
        put(i, "inner.value", float(2 * i));
    }
    allocation.size = bytes.size();
    auto rebound = runtime.allocate_memory(allocation);
    rebound.write(bytes.data(), bytes.size());
    args[0].value.ndarray.memory = rebound;
    ti_launch_kernel_with_bindings_ext(runtime, kernel, 2, args, 1, &binding);
    ASSERT_TAICHI_SUCCESS();
    // Module close retires in-flight work and all public handles.
    module.destroy();
    ti_get_ndarray_layout_info_ext(runtime, layout, &info);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live");
    ti_launch_kernel_with_bindings_ext(runtime, kernel, 2, args, 1, &binding);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live");
    ti_launch_compute_graph_with_bindings_ext(runtime, graph, 2, named, 1,
                                              &named_binding);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live");
    runtime.wait();
    std::vector<float> values(4);
    result.read(values);
    for (int i = 0; i < 4; ++i)
      EXPECT_EQ(values[i], 11 + 11 * i);
    auto fresh = runtime.load_aot_module(path);
    auto fresh_kernel = fresh.get_kernel("update");
    EXPECT_NE(TiKernel(fresh_kernel), TiKernel(kernel));
    binding.layout = ti_get_kernel_ndarray_layout_ext(runtime, fresh_kernel, 0);
    EXPECT_NE(binding.layout, layout);
    ti_launch_kernel_with_bindings_ext(runtime, fresh_kernel, 2, args, 1,
                                       &binding);
    runtime.wait();
    ASSERT_TAICHI_SUCCESS();
  }
};

TEST_F(AotStructNdarrayTest, Cpu) {
  run(TI_ARCH_X64);
}
TEST_F(AotStructNdarrayTest, Cuda) {
  run(TI_ARCH_CUDA);
}
TEST_F(AotStructNdarrayTest, Vulkan) {
  run(TI_ARCH_VULKAN);
}
