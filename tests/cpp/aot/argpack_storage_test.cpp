#include "gtest/gtest.h"
#include "taichi/aot/argpack_layout.h"
#include "taichi/program/callable.h"
#include "taichi/runtime/native_argpack_storage.h"

#include <cstring>
#include <stdexcept>
#include <tuple>
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/gfx/runtime.h"
#endif

namespace taichi::lang {
namespace {
using Layout = aot::ArgPackLayout;
using Values = NativeArgPackStorage::Values;
const StructType *pointer(const Type *body) {
  auto &factory = TypeFactory::get_instance();
  return factory.get_struct_type(
      {{factory.get_pointer_type(const_cast<Type *>(body)), "data_ptr"}})
      ->as<StructType>();
}
CallableBase example() {
  auto &factory = TypeFactory::get_instance();
  const auto vector = factory.get_struct_type(
      {{PrimitiveType::f32, "0"}, {PrimitiveType::f32, "1"}});
  const auto child = pointer(factory.get_struct_type(
      {{vector, "vector"}, {PrimitiveType::u1, "flag"}}));
  const auto empty = pointer(factory.get_struct_type({}));
  const auto root = pointer(factory.get_struct_type(
      {{PrimitiveType::f32, "gain"}, {child, "child"}, {empty, "empty"}}));
  const auto array = factory.get_ndarray_struct_type(PrimitiveType::f32, 1, true);
  CallableBase kernel;
  kernel.args_type = factory.get_struct_type(
      {{root, "pack"}, {array, "lifted0"}, {array, "lifted1"}})->as<StructType>();
  for (auto item : std::vector<std::pair<std::vector<int>, const Type *>>{
           {{0}, root}, {{0, 1}, child}, {{0, 2}, empty}})
    kernel.nested_parameters[item.first] =
        CallableBase::Parameter(item.second, false, true);
  for (auto item : std::vector<std::pair<std::vector<int>, const Type *>>{
           {{0, 0}, PrimitiveType::f32}, {{0, 1, 0}, vector},
           {{0, 1, 1}, PrimitiveType::u1}}) {
    auto parameter = CallableBase::Parameter(item.second);
    parameter.ptype = ParameterType::kScalar;
    kernel.nested_parameters[item.first] = parameter;
  }
  for (int i : {1, 2}) {
    auto parameter = CallableBase::Parameter(array, true);
    parameter.ptype = ParameterType::kNdarray;
    kernel.nested_parameters[{i}] = parameter;
  }
  kernel.value_parameter_types.emplace();
  (*kernel.value_parameter_types)[{0, 1, 0}] =
      factory.get_tensor_type({2}, PrimitiveType::f32);
  kernel.argument_origins = ArgumentOrigins{
      {{0}, {0}, "pack", "argpack"},
      {{0, 0}, {0, 0}, "gain", "value"},
      {{0, 1}, {0, 1}, "child", "argpack"},
      {{0, 1, 0}, {0, 1, 0}, "vector", "value"},
      {{0, 1, 1}, {0, 1, 1}, "flag", "value"},
      {{0, 1, 2}, {1}, "values", "ndarray"},
      {{0, 2}, {0, 2}, "empty", "argpack"},
      {{0, 2, 0}, {2}, "values", "ndarray"}};
  return kernel;
}

// A known native test ABI. Production callers supply the existing target
// compiler resolver; never this test oracle or raw all-zero member offsets.
std::tuple<const StructType *, std::size_t, std::size_t> host_layout_impl(
    const StructType *type) {
  auto members = type->elements();
  std::size_t bytes = 0, alignment = 1;
  for (auto &member : members) {
    std::size_t align, size;
    if (auto child = member.type->cast<StructType>()) {
      const auto resolved = host_layout_impl(child);
      member.type = std::get<0>(resolved);
      size = std::get<1>(resolved);
      align = std::get<2>(resolved);
    } else if (member.type->is<PointerType>()) {
      align = size = 8;
    } else {
      align = data_type_alignment(member.type);
      size = data_type_size(member.type);
    }
    bytes = align_up(bytes, align);
    member.offset = bytes;
    bytes += size;
    alignment = std::max(alignment, align);
  }
  return {TypeFactory::get_instance().get_struct_type(members, "host-test")
              ->as<StructType>(), align_up(bytes, alignment), alignment};
}
std::pair<const StructType *, std::size_t> host_layout(const StructType *type) {
  const auto resolved = host_layout_impl(type);
  return {std::get<0>(resolved), std::get<1>(resolved)};
}
Values payload(const Layout &layout) {
  Values values;
  for (const auto &pack : layout.storages())
    for (const auto &value : pack.values) {
      auto &data = values[value.origin.logical];
      data.assign(value.layout.buffer_size, 0xff);  // Padding must be discarded.
      for (const auto &leaf : value.layout.members) {
        const auto count = leaf.component_offsets.empty()
                               ? leaf.size / leaf.scalar_size
                               : leaf.component_offsets.size();
        for (std::size_t i = 0; i < count; ++i) {
          const auto offset = leaf.component_offsets.empty()
                                  ? leaf.offset + i * leaf.scalar_size
                                  : leaf.component_offsets[i];
          if (leaf.scalar_type == PrimitiveType::u1) {
            std::uint32_t flag = 1;
            std::memcpy(data.data() + offset, &flag, leaf.scalar_size);
          } else {
            const float number = 2.5f + float(i);
            std::memcpy(data.data() + offset, &number, sizeof(number));
          }
        }
      }
    }
  return values;
}
void check_storage(Arch arch, Device &device, aot::ValueStorage kind,
                   const Layout::Resolver &resolver) {
  auto layout = std::make_shared<const Layout>(example(), std::vector<int>{0},
                                               kind, resolver);
  ASSERT_EQ(layout->storages().size(), 3);
  ASSERT_EQ(layout->resources().size(), 2);
  EXPECT_EQ(layout->resources()[0].physical, (std::vector<int>{1}));
  EXPECT_EQ(layout->resources()[1].physical, (std::vector<int>{2}));
  auto input = payload(*layout);
  auto owner = std::make_shared<NativeArgPackStorage>(arch, device, layout, input);
  input.clear();  // No caller host pointers may be retained after cold upload.
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < layout->storages().size(); ++i) {
    const auto &pack = layout->storages()[i];
    std::vector<std::uint8_t> readback(std::max<std::size_t>(pack.bytes, 1));
    auto ptr = owner->allocation(i).get_ptr();
    void *target = readback.data();
    auto size = readback.size();
    ASSERT_EQ(device.readback_data(&ptr, &target, &size), RhiResult::success);
    bytes += size;
    if (i == 0) {
      float gain = 0;
      std::memcpy(&gain, readback.data(), sizeof(gain));
      EXPECT_EQ(gain, 2.5f);
      EXPECT_EQ(readback[4], 0);
    }
    if (i == 1) {
      float vector[2];
      std::memcpy(vector, readback.data(), sizeof(vector));
      EXPECT_EQ(vector[0], 2.5f);
      EXPECT_EQ(vector[1], 3.5f);
      const auto offset = pack.values[1].offset;
      EXPECT_EQ(readback[offset], 1);
      EXPECT_EQ(readback.back(), 0);  // Trailing padding was zero initialized.
    }
    if (i == 2)
      EXPECT_EQ(readback, (std::vector<std::uint8_t>{0}));
    for (const auto &child : pack.children) {
      std::uint64_t address = 0;
      std::memcpy(&address, readback.data() + child.pointer_offset, sizeof(address));
      EXPECT_EQ(address, owner->address(child.storage));
    }
  }
  EXPECT_EQ(owner->device_payload_bytes(), bytes);
  EXPECT_EQ(bytes, kind == aot::ValueStorage::Host ? 37u : 81u);
  EXPECT_EQ(owner->workspace_bytes(), 0);
  std::weak_ptr<NativeArgPackStorage> lifetime = owner;
  std::shared_ptr<NativeResource> retained = owner;
  owner.reset();
  EXPECT_FALSE(lifetime.expired());
  retained.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(ArgPackStorage, RejectsUnresolvedAndIncompleteCompiledLayouts) {
  auto kernel = example();
  const Layout layout(kernel, {0}, aot::ValueStorage::Host, host_layout);
  EXPECT_TRUE(layout.matches(kernel, {0}));
  EXPECT_FALSE(layout.matches(kernel, {1}));
  auto other = kernel;
  (*other.value_parameter_types)[{0, 1, 0}] =
      TypeFactory::get_instance().get_tensor_type({1, 2}, PrimitiveType::f32);
  EXPECT_FALSE(layout.matches(other, {0}));
  EXPECT_THROW(Layout(kernel, {0}, aot::ValueStorage::Host,
                      [](const StructType *type) { return std::make_pair(type, 64u); }),
               std::invalid_argument);
  EXPECT_THROW(Layout(kernel, {1}, aot::ValueStorage::Host, host_layout),
               std::invalid_argument);
  EXPECT_THROW(Layout(kernel, {0}, aot::ValueStorage::Host,
                      [](const StructType *type) {
                        auto result = host_layout(type);
                        result.second = 0;
                        return result;
                      }), std::invalid_argument);
  kernel.argument_origins.reset();
  EXPECT_THROW(Layout(kernel, {0}, aot::ValueStorage::Host, host_layout),
               std::invalid_argument);
}

TEST(ArgPackStorage, RepeatedPackTypesShareColdResolutionWithoutMemberCaps) {
  auto &factory = TypeFactory::get_instance();
  const auto empty = pointer(factory.get_struct_type({}));
  std::vector<AbstractDictionaryMember> members;
  CallableBase kernel;
  kernel.argument_origins = ArgumentOrigins{{{0}, {0}, "root", "argpack"}};
  for (int i = 0; i < 65; ++i) {
    const auto name = "child" + std::to_string(i);
    members.push_back({empty, name});
    kernel.nested_parameters[{0, i}] = CallableBase::Parameter(empty, false, true);
    kernel.argument_origins->push_back({{0, i}, {0, i}, name, "argpack"});
  }
  const auto root = pointer(factory.get_struct_type(members));
  kernel.args_type = factory.get_struct_type({{root, "root"}})->as<StructType>();
  kernel.nested_parameters[{0}] = CallableBase::Parameter(root, false, true);
  int resolutions = 0;
  const Layout layout(kernel, {0}, aot::ValueStorage::Host,
                      [&](const StructType *type) {
                        ++resolutions;
                        return host_layout(type);
                      });
  EXPECT_EQ(layout.storages().size(), 66);
  EXPECT_EQ(layout.storages()[0].children.size(), 65);
  EXPECT_EQ(resolutions, 2);
}

#ifdef TI_WITH_LLVM
TEST(ArgPackStorage, CpuUploadsNestedPointersAndValuesOnce) {
  cpu::CpuDevice device;
  check_storage(host_arch(), device, aot::ValueStorage::Host, host_layout);
}
class FailingDevice : public cpu::CpuDevice {
 public:
  int allocations{0}, live{0}, fail_at{-1};
  bool fail_upload{false};
  RhiResult allocate_memory(const AllocParams &params, DeviceAllocation *out) override {
    if (allocations++ == fail_at)
      return RhiResult::out_of_memory;
    auto status = cpu::CpuDevice::allocate_memory(params, out);
    if (status == RhiResult::success)
      ++live;
    return status;
  }
  void dealloc_memory(DeviceAllocation allocation) override {
    --live;
    cpu::CpuDevice::dealloc_memory(allocation);
  }
  RhiResult upload_data(DevicePtr *ptr, const void **data, size_t *size,
                        int count = 1) noexcept override {
    if (fail_upload) {
      // Complete an actual first upload before failing: partial preparation
      // must still release every allocation.
      cpu::CpuDevice::upload_data(ptr, data, size, 1);
      return RhiResult::error;
    }
    return cpu::CpuDevice::upload_data(ptr, data, size, count);
  }
};
TEST(ArgPackStorage, ColdFailuresReleasePartialStorageAndAllowRetry) {
  auto layout = std::make_shared<const Layout>(example(), std::vector<int>{0},
                                               aot::ValueStorage::Host, host_layout);
  auto values = payload(*layout);
  FailingDevice device;
  auto invalid = values;
  invalid.begin()->second.pop_back();
  EXPECT_THROW(NativeArgPackStorage(host_arch(), device, layout, invalid),
               std::invalid_argument);
  invalid = values;
  invalid[{99}] = {};
  EXPECT_THROW(NativeArgPackStorage(host_arch(), device, layout, invalid),
               std::invalid_argument);
  EXPECT_EQ(device.allocations, 0);
  device.fail_at = 1;
  EXPECT_THROW(NativeArgPackStorage(host_arch(), device, layout, values),
               std::runtime_error);
  EXPECT_EQ(device.live, 0);
  device.fail_at = -1;
  device.fail_upload = true;
  EXPECT_THROW(NativeArgPackStorage(host_arch(), device, layout, values),
               std::runtime_error);
  EXPECT_EQ(device.live, 0);
  device.fail_upload = false;
  {
    NativeArgPackStorage valid(host_arch(), device, layout, values);
    EXPECT_EQ(device.live, 3);
    device.fail_upload = true;
    EXPECT_THROW(NativeArgPackStorage(host_arch(), device, layout, values),
                 std::runtime_error);
    EXPECT_EQ(device.live, 3);
    float gain = 0;
    auto ptr = valid.allocation(0).get_ptr();
    void *output = &gain;
    std::size_t size = sizeof(gain);
    ASSERT_EQ(device.readback_data(&ptr, &output, &size), RhiResult::success);
    EXPECT_EQ(gain, 2.5f);
    device.fail_upload = false;
    NativeArgPackStorage retry(host_arch(), device, layout, values);
    EXPECT_EQ(device.live, 6);
    EXPECT_NE(valid.address(0), retry.address(0));
  }
  EXPECT_EQ(device.live, 0);
}
#endif

#ifdef TI_WITH_CUDA
TEST(ArgPackStorage, CudaUploadsNestedPointersAndValuesOnce) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  check_storage(Arch::cuda, device, aot::ValueStorage::Host, host_layout);
}
#endif
#ifdef TI_WITH_VULKAN
TEST(ArgPackStorage, VulkanUsesSeparateUniformsAndCompiledBoolSlots) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  check_storage(Arch::vulkan, *creator.device(), aot::ValueStorage::Gfx,
                 [](const StructType *type) {
                   return gfx::GfxRuntime::get_struct_type_with_data_layout(type, "1b");
                 });
}
#endif
}  // namespace
}  // namespace taichi::lang
