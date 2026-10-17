// NOTE: This test intends to fixate error code reporting of all 'unexpected'
// usage of the C-APIs. Normal usages should be covered in other test files like
// `c_api_interface_test.cpp`.
#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#ifdef TI_WITH_LLVM
#include "taichi/taichi_cpu.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/taichi_cuda.h"
#endif
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, ImmutableHostBlobsUseRuntimeResourceOwnership) {
  EXPECT_EQ(ti_create_host_blob_ext(nullptr, 0, nullptr), nullptr);
  EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL, "runtime");
  for (TiArch arch :
       {TI_ARCH_X64, TI_ARCH_ARM64, TI_ARCH_CUDA, TI_ARCH_VULKAN}) {
    if (!ti::is_arch_available(arch))
      continue;
    SCOPED_TRACE(int(arch));
    ti::Runtime left(arch), right(arch);
    EXPECT_EQ(ti_create_host_blob_ext(left, 1, nullptr), nullptr);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "host blob");
    unsigned char data[]{0, 255, 17, 0};
    auto first = ti_create_host_blob_ext(left, sizeof(data), data);
    auto empty = ti_create_host_blob_ext(left, 0, nullptr);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(empty, nullptr);
    ASSERT_NE(first, empty);
    ASSERT_TAICHI_SUCCESS();
    data[1] = 0;
    ti_destroy_native_resource_ext(right, first);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    ti_destroy_native_resource_ext(left, first);
    ASSERT_TAICHI_SUCCESS();
    ti_destroy_native_resource_ext(left, first);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    // The empty blob deliberately retires with its runtime.
  }
}

TEST_F(CapiTest, MemoryHandlesBelongToTheirRuntime) {
  for (TiArch arch : {TI_ARCH_X64, TI_ARCH_ARM64, TI_ARCH_CUDA, TI_ARCH_VULKAN}) {
    if (!ti::is_arch_available(arch))
      continue;
    SCOPED_TRACE(int(arch));
    ti::Runtime left(arch), right(arch);
    TiMemoryAllocateInfo info{};
    info.size = 16;
    info.host_read = info.host_write = TI_TRUE;
    info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    TiMemory a = ti_allocate_memory(left, &info);
    TiMemory b = ti_allocate_memory(right, &info);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b);
    ASSERT_TAICHI_SUCCESS();
    auto *initial = static_cast<int *>(ti_map_memory(right, b));
    ASSERT_NE(initial, nullptr);
    initial[0] = 29;
    ti_unmap_memory(right, b);
    ASSERT_TAICHI_SUCCESS();

    // Reject before invoking a backend with a foreign RHI allocation slot.
    EXPECT_EQ(ti_map_memory(right, a), nullptr);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    ti_unmap_memory(right, a);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    ti_free_memory(right, a);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    TiMemorySlice source{a, 0, 16}, target{b, 0, 16};
    ti_copy_memory_device_to_device(right, &target, &source);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    int host[4]{};
    ti_copy_memory_from_host_ext(right, &source, host);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
    ti_copy_memory_to_host_ext(right, host, &source);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");

#ifdef TI_WITH_LLVM
    if (arch == TI_ARCH_X64 || arch == TI_ARCH_ARM64) {
      auto *values = static_cast<int *>(ti_map_memory(right, b));
      ASSERT_NE(values, nullptr);
      values[0] = 29;
      ti_unmap_memory(right, b);
      TiCpuMemoryInteropInfo exported{};
      ti_export_cpu_memory(right, a, &exported);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
      EXPECT_EQ(exported.ptr, nullptr);
      ti_export_cpu_memory(right, b, &exported);
      ASSERT_TAICHI_SUCCESS();
      EXPECT_EQ(static_cast<int *>(exported.ptr)[0], 29);
    }
#endif
#ifdef TI_WITH_CUDA
    if (arch == TI_ARCH_CUDA) {
      TiCudaMemoryInteropInfo exported{};
      ti_export_cuda_memory(right, a, &exported);
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "this runtime");
      EXPECT_EQ(exported.ptr, nullptr);
      ti_export_cuda_memory(right, b, &exported);
      ASSERT_TAICHI_SUCCESS();
      EXPECT_GE(exported.size, 16);  // CUDA may round physical allocations.
    }
#endif

    ti_free_memory(left, a);
    ASSERT_TAICHI_SUCCESS();
    TiMemory replacement = ti_allocate_memory(left, &info);
    ASSERT_NE(replacement, nullptr);
    EXPECT_NE(replacement, a);
    ti_free_memory(left, a);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live allocation");
    ti_free_memory(left, replacement);
    ASSERT_TAICHI_SUCCESS();
    left.destroy();
    // Retiring one runtime must leave both the other runtime's internal
    // allocations and its public buffers usable.
    right.wait();
    ASSERT_TAICHI_SUCCESS();
    auto *retained = static_cast<int *>(ti_map_memory(right, b));
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(retained[0], 29);
    ti_unmap_memory(right, b);
    ti_free_memory(right, b);
    ASSERT_TAICHI_SUCCESS();
    right.destroy();
    ti::Runtime later(arch);
    TiMemory c = ti_allocate_memory(later, &info);
    EXPECT_NE(c, a);
    EXPECT_EQ(ti_map_memory(later, a), nullptr);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live allocation");
    ti_free_memory(later, c);
    ASSERT_TAICHI_SUCCESS();
  }
}

TEST_F(CapiTest, ImportedMemoryCapacityAndRetirement) {
#ifdef TI_WITH_LLVM
  if (!ti::is_arch_available(TI_ARCH_X64))
    GTEST_SKIP();
  ti::Runtime runtime(TI_ARCH_X64);
  int source[4] = {1, 2, 3, 4};
  int target[4] = {9, 9, 9, 9};
  TiMemory a = ti_import_cpu_memory(runtime, source, sizeof(source));
  TiMemory b = ti_import_cpu_memory(runtime, target, sizeof(target));
  ASSERT_TAICHI_SUCCESS();
  TiMemorySlice src{a, 0, sizeof(source)}, dst{b, 1, sizeof(target)};
  ti_copy_memory_device_to_device(runtime, &dst, &src);
  EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "capacity");
  dst.offset = 0;
  src.offset = UINT64_MAX;  // Offset+size must not wrap past the bounds check.
  ti_copy_memory_device_to_device(runtime, &dst, &src);
  EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "capacity");
  EXPECT_EQ(target[0], 9);
  // Invalid slices leave the original bytes intact; valid imports can copy.
  for (int value : target)
    EXPECT_EQ(value, 9);
  src.offset = 0;
  ti_copy_memory_device_to_device(runtime, &dst, &src);
  ASSERT_TAICHI_SUCCESS();
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(target[i], source[i]);
  ti_free_memory(runtime, a);
  ti_free_memory(runtime, b);
  ASSERT_TAICHI_SUCCESS();
  // Releasing import wrappers leaves external allocations under caller control.
  source[0] = 17;
  EXPECT_EQ(source[0], 17);
  TiCpuMemoryInteropInfo exported{};
  ti_export_cpu_memory(runtime, a, &exported);
  EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live allocation");
#else
  GTEST_SKIP() << "CPU interop requires LLVM support";
#endif
}

TEST_F(CapiTest, BulkMemoryTransfers) {
  for (TiArch arch :
       {TI_ARCH_X64, TI_ARCH_ARM64, TI_ARCH_CUDA, TI_ARCH_VULKAN}) {
    if (!ti::is_arch_available(arch))
      continue;
    SCOPED_TRACE(int(arch));
    ti::Runtime runtime(arch);
    // Device allocations need no host mapping flags. Exercise byte offsets as
    // well as ordinary typed-array ranges, with unchanged sentinel bytes.
    ti::Memory a = runtime.allocate_memory(32);
    ti::Memory b = runtime.allocate_memory(32);
    uint8_t input[32], output[32]{};
    for (int i = 0; i < 32; ++i)
      input[i] = uint8_t(i + 17);
    TiMemorySlice whole_a = a.slice(), whole_b = b.slice();
    ti_copy_memory_from_host_ext(runtime, &whole_a, input);
    ti_copy_memory_from_host_ext(runtime, &whole_b, output);
    ASSERT_TAICHI_SUCCESS();
    TiMemorySlice src{a, 3, 13}, dst{b, 9, 13};
    ti_copy_memory_device_to_device(runtime, &dst, &src);
    ASSERT_TAICHI_SUCCESS();
    // No explicit wait: readback orders after the earlier device command.
    ti_copy_memory_to_host_ext(runtime, output, &whole_b);
    ASSERT_TAICHI_SUCCESS();
    for (int i = 0; i < 32; ++i)
      EXPECT_EQ(output[i], i >= 9 && i < 22 ? input[i - 6] : 0);
    ti_copy_memory_from_host_ext(runtime, &dst, input);
    ti_copy_memory_to_host_ext(runtime, output, &dst);
    ASSERT_TAICHI_SUCCESS();
    for (int i = 0; i < 13; ++i)
      EXPECT_EQ(output[i], input[i]);

    TiMemorySlice overlap{a, 4, 13};
    ti_copy_memory_device_to_device(runtime, &overlap, &src);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "overlap");
    ti_copy_memory_device_to_device(runtime, &src, &src);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "overlap");
    TiMemorySlice empty{a, 32, 0};
    ti_copy_memory_from_host_ext(runtime, &empty, nullptr);
    ti_copy_memory_to_host_ext(runtime, nullptr, &empty);
    ti_copy_memory_device_to_device(runtime, &empty, &empty);
    ASSERT_TAICHI_SUCCESS();
    ti_copy_memory_from_host_ext(runtime, &src, nullptr);
    EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    ti_copy_memory_to_host_ext(runtime, nullptr, &src);
    EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    TiMemorySlice bad{a, UINT64_MAX, 13};
    ti_copy_memory_from_host_ext(runtime, &bad, input);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "capacity");
    ti_copy_memory_to_host_ext(runtime, output, &bad);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "capacity");
    a.destroy();
    ti_copy_memory_from_host_ext(runtime, &src, input);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live allocation");
    ti_copy_memory_to_host_ext(runtime, output, &src);
    EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT, "live allocation");
  }
}

TEST_F(CapiTest, TestBehaviorCreateRuntime) {
  auto inner = [this](TiArch arch) {
    TiRuntime runtime = ti_create_runtime(arch, 0);
    TI_ASSERT(runtime == TI_NULL_HANDLE);
    EXPECT_TAICHI_ERROR(TI_ERROR_NOT_SUPPORTED);
  };

  // Attempt to create runtime for unknown arch.
  inner(TI_ARCH_MAX_ENUM);
}

TEST_F(CapiTest, TestBehaviorDestroyRuntime) {
  // Attempt to destroy null handles.
  ti_destroy_runtime(TI_NULL_HANDLE);
  EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
}

TEST_F(CapiTest, TestBehaviorGetRuntimeCapabilities) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);
    {
      // Two nulls, considerred nop.
      ti_get_runtime_capabilities(runtime, nullptr, nullptr);
      ASSERT_TAICHI_SUCCESS();
    }

    {
      // Count is not null, buffer is null.
      // Usually the case the user look up the number of capabilities will be
      // returned.
      uint32_t capability_count = 0;
      ti_get_runtime_capabilities(runtime, &capability_count, nullptr);
      ASSERT_TAICHI_SUCCESS();
      switch (arch) {
        case TI_ARCH_VULKAN:
        case TI_ARCH_OPENGL:
          // Always have `TI_CAPABILITY_SPIRV_VERSION`.
          TI_ASSERT(capability_count > 0);
          break;
        default:
          TI_NOT_IMPLEMENTED;
      }
    }

    {
      // Count is null, buffer non-null.
      // Expect nop.
      std::vector<TiCapabilityLevelInfo> capabilities{
          TiCapabilityLevelInfo{
              (TiCapability)0xcbcbcbcb,
              0xcbcbcbcb,
          },
      };
      ti_get_runtime_capabilities(runtime, nullptr, capabilities.data());
      ASSERT_TAICHI_SUCCESS();
      EXPECT_EQ(capabilities.at(0).capability, (TiCapability)0xcbcbcbcb);
      EXPECT_EQ(capabilities.at(0).level, 0xcbcbcbcb);
    }

    {
      // Both non-null.
      // Normal usage.
      uint32_t capability_count = 0;
      ti_get_runtime_capabilities(runtime, &capability_count, nullptr);
      ASSERT_TAICHI_SUCCESS();
      std::vector<TiCapabilityLevelInfo> capabilities(capability_count);
      ti_get_runtime_capabilities(runtime, &capability_count,
                                  capabilities.data());
      ASSERT_TAICHI_SUCCESS();
      for (size_t i = 0; i < capability_count; ++i) {
        TI_ASSERT(capabilities.at(i).capability !=
                  (TiCapability)TI_CAPABILITY_RESERVED);
        TI_ASSERT(capabilities.at(i).level != 0);
      }
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorAllocateMemory) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so this test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);

    // Out of memory
    {
      TiMemoryAllocateInfo allocate_info{};
      allocate_info.size = 1000000000000000000;
      allocate_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
      TiMemory memory = ti_allocate_memory(runtime, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_OUT_OF_MEMORY);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }

    // runtime and allocate_info are both null
    {
      TiMemory memory = ti_allocate_memory(TI_NULL_HANDLE, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }

    // runtime is not null, allocate_info is null
    {
      TiMemory memory = ti_allocate_memory(runtime, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }

    // runtime is null, allocate is not null;
    {
      TiMemoryAllocateInfo allocate_info{};
      allocate_info.size = 1024;
      TiMemory memory = ti_allocate_memory(TI_NULL_HANDLE, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(memory == TI_NULL_HANDLE);
    }
  };

  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorFreeMemory) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);

    // runtime & allocate_info are both null
    {
      ti_free_memory(TI_NULL_HANDLE, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is null and allocate_info is valid
    {
      TiMemory memory = runtime.allocate_memory(1024);
      ti_free_memory(TI_NULL_HANDLE, memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is not null and allocate_info is null
    {
      ti_free_memory(runtime, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorMapMemory) {
  auto inner = [this](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(1024);

    // runtime & memory are both null
    {
      void *ptr = ti_map_memory(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(ptr == nullptr);
    }

    // runtime is null, memory is valid
    {
      void *ptr = ti_map_memory(TI_NULL_HANDLE, memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(ptr == nullptr);
    }

    // runtime is valid, memory is null
    {
      void *ptr = ti_map_memory(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(ptr == nullptr);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorUnmapMemory) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is nor supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);
    ti::Memory memory = runtime.allocate_memory(1024);

    // runtime & memory are both null
    {
      ti_unmap_memory(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // runtime is null, memory is valid
    {
      ti_map_memory(TI_NULL_HANDLE, memory);
      ti_unmap_memory(TI_NULL_HANDLE, memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      ti_unmap_memory(runtime, memory);
    }

    // runtime is valid, memory is null
    {
      ti_unmap_memory(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

inline TiImageAllocateInfo make_image_allocate_info() {
  TiImageAllocateInfo allocate_info{};
  allocate_info.dimension = TI_IMAGE_DIMENSION_2D;
  allocate_info.format = TI_FORMAT_RGBA8;
  allocate_info.extent.height = 16;
  allocate_info.extent.width = 16;
  allocate_info.extent.depth = 1;
  allocate_info.extent.array_layer_count = 1;
  allocate_info.usage = TI_IMAGE_USAGE_STORAGE_BIT | TI_IMAGE_USAGE_SAMPLED_BIT;
  allocate_info.mip_level_count = 1;
  return allocate_info;
}

TEST_F(CapiTest, TestBehaviorAllocateImage) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);

    // Attemp to allocate a 2D image with invalid demension
    {
      TiImageAllocateInfo allocate_info = make_image_allocate_info();
      allocate_info.dimension = TI_IMAGE_DIMENSION_MAX_ENUM;
      TiImage image = ti_allocate_image(runtime, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_OUT_OF_RANGE);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // Attemp to allocate a 2D image with a invalid format
    {
      TiImageAllocateInfo allocate_info = make_image_allocate_info();
      allocate_info.format = TI_FORMAT_MAX_ENUM;
      TiImage image = ti_allocate_image(runtime, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_OUT_OF_RANGE);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // runtime & allocate_info are both null
    {
      TiImage image = ti_allocate_image(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // runtime is null, allocate_info is valid
    {
      TiImageAllocateInfo allocate_info = make_image_allocate_info();
      TiImage image = ti_allocate_image(TI_NULL_HANDLE, &allocate_info);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }

    // runtime is valid, allocate_info is null;
    {
      TiImage image = ti_allocate_image(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(image == TI_NULL_HANDLE);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorFreeImage) {
  auto inner = [&](TiArch arch) {
    // Attemp to free a normal 2D image
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);

    // Runtime & image are both invalid
    {
      ti_free_image(TI_NULL_HANDLE, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // Runtime is null, Image is valid
    {
      TiImage image = runtime.allocate_image(make_image_allocate_info());
      ti_free_image(TI_NULL_HANDLE, image);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // Runtime is valid, image is null
    {
      ti_free_image(runtime, TI_NULL_HANDLE);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorCopyMemoryDTD) {
  auto inner = [&](TiArch arch) {
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);
    ti::Memory src = runtime.allocate_memory(2048);
    ti::Memory dst = runtime.allocate_memory(2048);

    {
      src.slice(128, 64).copy_to(dst.slice(1024, 64));
      ASSERT_TAICHI_SUCCESS();
    }

    // Attempt copy memory from the big one to the small one
    {
      src.slice(0, 256).copy_to(dst.slice(0, 64));
      EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_ARGUMENT);
    }

    // runtime is null;
    {
      TiMemorySlice src_memory = src.slice();
      TiMemorySlice dst_memory = dst.slice();
      ti_copy_memory_device_to_device(TI_NULL_HANDLE, &dst_memory, &src_memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // dst memory is null;
    {
      TiMemorySlice src_memory = src.slice();
      TiMemorySlice dst_memory = dst.slice();
      dst_memory.memory = TI_NULL_HANDLE;
      ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }

    // src memory is null;
    {
      TiMemorySlice src_memory = src.slice();
      TiMemorySlice dst_memory = dst.slice();
      src_memory.memory = TI_NULL_HANDLE;
      ti_copy_memory_device_to_device(runtime, &dst_memory, &src_memory);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
    }
  };
  inner(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorLoadAOTModuleVulkan) {
  auto test_behavior_load_aot_module_impl = [this](TiArch arch) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
    const std::string module_path = folder_dir + std::string("/module.tcm");

    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }
    ti::Runtime runtime(arch);

    // AOT module from tcm file, normal usage.
    {
      TiAotModule module = ti_load_aot_module(runtime, module_path.c_str());
      ASSERT_TAICHI_SUCCESS();
      TI_ASSERT(module != TI_NULL_HANDLE);
    }

    // AOT module from filesystem directory, normal usage.
    {
      TiAotModule module = ti_load_aot_module(runtime, folder_dir);
      ASSERT_TAICHI_SUCCESS();
      TI_ASSERT(module != TI_NULL_HANDLE);
    }

    // Attempt to load aot module from tcm file, while runtime is null.
    {
      TiAotModule module =
          ti_load_aot_module(TI_NULL_HANDLE, module_path.c_str());
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }

    // Attempt to load aot module from tcm file, while runtime is null.
    {
      TiAotModule module = ti_load_aot_module(TI_NULL_HANDLE, folder_dir);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }

    // Attempt to load aot module without path.
    {
      TiAotModule module = ti_load_aot_module(runtime, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }
    // Attempt to load aot module with a invalid path.
    {
      TiAotModule module = ti_load_aot_module(runtime, "ssssss///");
      EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA);
      TI_ASSERT(module == TI_NULL_HANDLE);
    }
  };

  test_behavior_load_aot_module_impl(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorDestroyAotModuleVulkan) {
  auto test_behavior_destroy_aot_module_impl = [this](TiArch arch) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
    const std::string module_path = folder_dir + std::string("/module.tcm");
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    // Attempt to destroy a null handle.
    ti_destroy_aot_module(TI_NULL_HANDLE);
    EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
  };

  test_behavior_destroy_aot_module_impl(TI_ARCH_VULKAN);
}

TEST_F(CapiTest, TestBehaviorGetCgraphVulkan) {
  auto test_behavior_get_cgraph_impl = [this](TiArch arch) {
    const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
    const std::string module_path = folder_dir;
    if (!ti::is_arch_available(arch)) {
      TI_WARN("arch {} is not supported, so the test is skipped", int(arch));
      return;
    }

    ti::Runtime runtime(arch);
    ti::AotModule module = runtime.load_aot_module(module_path.c_str());

    {
      TiComputeGraph cgraph =
          ti_get_aot_module_compute_graph(module, "run_graph");
      ASSERT_TAICHI_SUCCESS();
      TI_ASSERT(cgraph != TI_NULL_HANDLE);
    }

    // Attemp to get compute graph with null module.
    {
      TiComputeGraph cgraph =
          ti_get_aot_module_compute_graph(TI_NULL_HANDLE, "run_graph");
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(cgraph == TI_NULL_HANDLE);
    }

    // Attemp to get compute graph without graph name.
    {
      TiComputeGraph cgraph = ti_get_aot_module_compute_graph(module, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL);
      TI_ASSERT(cgraph == TI_NULL_HANDLE);
    }

    // Attemp to get compute graph with invalid name.
    {
      TiComputeGraph cgraph = ti_get_aot_module_compute_graph(module, "#$#%*(");
      EXPECT_TAICHI_ERROR(TI_ERROR_NAME_NOT_FOUND);
      TI_ASSERT(cgraph == TI_NULL_HANDLE);
    }
  };

  test_behavior_get_cgraph_impl(TI_ARCH_VULKAN);
}
