#include "gtest/gtest.h"

#ifdef TI_TEST_RUNTIME_C_API
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include "c_api/src/taichi_core_impl.h"
#include "c_api/src/native_resource_registry.h"
#include "taichi/runtime/native_argpack_storage.h"

namespace taichi::lang {
namespace {
void check_error() {
  char message[1024]{};
  uint64_t size = sizeof(message);
  ASSERT_EQ(ti_get_last_error(&size, message), TI_ERROR_SUCCESS) << message;
}

class AotPackSession {
 public:
  TiRuntime handle{nullptr};
  TiAotModule module{nullptr};
  std::vector<TiMemory> memories;
  std::vector<TiNativeResource> resources;
  ~AotPackSession() {
    if (!handle)
      return;
    ti_wait(handle);
    for (auto resource : resources)
      if (resource)
        ti_destroy_native_resource_ext(handle, resource);
    if (module)
      ti_destroy_aot_module(module);
    for (auto memory : memories)
      ti_free_memory(handle, memory);
    ti_destroy_runtime(handle);
  }
  Runtime &runtime() {
    return *reinterpret_cast<Runtime *>(handle);
  }
  TiNdArray array(std::vector<float> values) {
    TiMemoryAllocateInfo info{};
    info.size = values.size() * sizeof(float);
    info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    auto memory = ti_allocate_memory(handle, &info);
    if (!memory)
      throw std::runtime_error("AOT fixture allocation failed");
    memories.push_back(memory);
    TiNdArray result{};
    result.memory = memory;
    result.shape.dim_count = 1;
    result.shape.dims[0] = values.size();
    result.elem_type = TI_DATA_TYPE_F32;
    write(result, values);
    return result;
  }
  void write(const TiNdArray &array, const std::vector<float> &values) {
    auto ptr = runtime().get_memory(array.memory).allocation.get_ptr();
    const void *source = values.data();
    auto bytes = values.size() * sizeof(float);
    ASSERT_EQ(runtime().get().upload_data(&ptr, &source, &bytes),
              RhiResult::success);
  }
  std::vector<float> read(const TiNdArray &array) {
    std::vector<float> result(array.shape.dims[0]);
    auto ptr = runtime().get_memory(array.memory).allocation.get_ptr();
    void *target = result.data();
    auto bytes = result.size() * sizeof(float);
    if (runtime().get().readback_data(&ptr, &target, &bytes) != RhiResult::success)
      throw std::runtime_error("AOT fixture readback failed");
    return result;
  }
  TiNativeResource pack(TiKernel kernel, float scale) {
    auto &rt = runtime();
    std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
    auto layout = rt.get_kernel_argpack_layout(kernel, 0);
    EXPECT_EQ(layout, rt.get_kernel_argpack_layout(kernel, 0));
    if (layout->storages().size() != 2 || layout->resources().size() != 1)
      throw std::runtime_error("Unexpected exported fixture pack");
    const auto &value = layout->storages().front().values.at(0);
    NativeArgPackStorage::Values values;
    auto &buffer = values[value.origin.logical];
    buffer.resize(value.layout.buffer_size);
    std::memcpy(buffer.data() + value.layout.members.at(0).offset, &scale,
                sizeof(scale));
    auto storage = std::make_shared<NativeArgPackStorage>(rt.arch, rt.get(),
                                                         layout, values);
    // The test exercises the existing resource registry and common C binder.
    // The public logical-tree creation API is a separate pending integration.
    auto result = rt.native_resource_registry().publish(storage, {}, {});
    resources.push_back(result);
    return result;
  }
};

void check_aot_pack(TiArch arch, const char *backend) {
  const char *directory = std::getenv("TI_TEST_ARGPACK_AOT_ROOT");
  if (!directory)
    GTEST_SKIP() << "Set TI_TEST_ARGPACK_AOT_ROOT to the exported fixture directory";
  AotPackSession session;
  uint32_t device_index = 0;
  if (arch == TI_ARCH_VULKAN)
    if (const auto *value = std::getenv("TI_TEST_VULKAN_DEVICE_INDEX"))
      device_index = std::stoul(value);
  session.handle = ti_create_runtime(arch, device_index);
  ASSERT_NE(session.handle, nullptr);
  check_error();
  const auto path = std::string(directory) + "/" + backend + "-cold/artifact";
  session.module = ti_load_aot_module(session.handle, path.c_str());
  ASSERT_NE(session.module, nullptr);
  auto forward = ti_get_aot_module_kernel(session.module, "forward");
  auto backward = ti_get_aot_module_kernel(session.module, "backward");
  ASSERT_NE(forward, nullptr);
  ASSERT_NE(backward, nullptr);
  check_error();
  const auto x = session.array({1, -2, 3});
  const auto dx = session.array({0, 0, 0});
  const auto total = session.array({0});
  const auto dtotal = session.array({1});
  TiArgument arguments[3]{};
  arguments[0].type = TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT;
  arguments[0].value.native_resource = session.pack(forward, 2);
  arguments[1].type = arguments[2].type = TI_ARGUMENT_TYPE_NDARRAY;
  arguments[1].value.ndarray = total;
  arguments[2].value.ndarray = x;
  TiNdArrayBinding bindings[2]{};
  bindings[0].argument_index = 1;
  bindings[0].gradient = &dtotal;
  bindings[1].argument_index = 2;
  bindings[1].gradient = &dx;
  auto launch = [&](TiKernel kernel) {
    ti_launch_kernel_with_bindings_ext(session.handle, kernel, 3, arguments,
                                       2, bindings);
    check_error();
  };
  launch(forward);
  launch(backward);
  // Destroying a submitted resource uses the registry's completion boundary.
  ti_destroy_native_resource_ext(session.handle, session.resources.back());
  session.resources.back() = nullptr;
  check_error();
  EXPECT_EQ(session.read(total), (std::vector<float>{84}));
  EXPECT_EQ(session.read(dx), (std::vector<float>{12, -24, 36}));

  // A stale resource must fail before submitting or changing either result.
  ti_launch_kernel_with_bindings_ext(session.handle, forward, 3, arguments,
                                     2, bindings);
  EXPECT_NE(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_EQ(session.read(total), (std::vector<float>{84}));

  arguments[0].value.native_resource = session.pack(forward, 3);
  session.write(x, {-1, 4, -2});
  session.write(dx, {0, 0, 0});
  session.write(total, {0});
  launch(forward);
  launch(backward);
  // The resource has no producer/module pointers; module retirement waits for
  // its submitted kernels, while this immutable pack can be destroyed later.
  ti_destroy_aot_module(session.module);
  session.module = nullptr;
  check_error();
  EXPECT_EQ(session.read(total), (std::vector<float>{189}));
  EXPECT_EQ(session.read(dx), (std::vector<float>{-18, 72, -36}));
}

#ifdef TI_WITH_LLVM
TEST(ArgPackAotBinding, CpuForwardAdjointRebindAndRetirement) {
  check_aot_pack(TI_ARCH_X64, "cpu");
}
#endif
#ifdef TI_WITH_CUDA
TEST(ArgPackAotBinding, CudaForwardAdjointRebindAndRetirement) {
  check_aot_pack(TI_ARCH_CUDA, "cuda");
}
#endif
#ifdef TI_WITH_VULKAN
TEST(ArgPackAotBinding, VulkanForwardAdjointRebindAndRetirement) {
  check_aot_pack(TI_ARCH_VULKAN, "vulkan");
}
#endif
}  // namespace
}  // namespace taichi::lang
#endif
