#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

#include <array>
#include <cstdlib>

using namespace taichi;
using namespace lang;

TEST(GfxAotTest, VulkanDenseField) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  aot_test_utils::run_dense_field_kernel(Arch::vulkan,
                                         embedded_device->device());
}

TEST(GfxAotTest, VulkanKernelTest2) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  aot_test_utils::run_kernel_test2(Arch::vulkan, embedded_device->device());
}

TEST(GfxAotTest, VulkanKernelTest1) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  aot_test_utils::run_kernel_test1(Arch::vulkan, embedded_device->device());
}

TEST(GfxAotTest, VulkanSNodeInstancesShareCodeAndIsolateState) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const char *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  if (!folder)
    GTEST_SKIP() << "Generate vulkan_snode_instances.py first";
  vulkan::VulkanDeviceCreator creator({});
  gfx::GfxRuntime::Params params;
  params.device = creator.device();
  gfx::GfxRuntime runtime(params);
  gfx::AotModuleParams module_params;
  module_params.module_path = folder;
  module_params.runtime = &runtime;
  std::shared_ptr<aot::Module> code = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_TRUE(code && !code->is_corrupted());
  gfx::allocate_aot_root_buffers(code.get());
  const std::vector<std::string> names{"activate", "activate_alias", "update", "read", "drop"};
  for (const auto &name : names)
    ASSERT_NE(code->get_kernel(name), nullptr);
  const auto pipelines = runtime.debug_unique_pipeline_count();
  const auto registrations = runtime.debug_registered_kernel_count();
  auto *original = dynamic_cast<gfx::KernelImpl *>(code->get_kernel("activate"));
  ASSERT_NE(original, nullptr);
  const auto original_bindings = original->params().root_buffer_bindings;
  EXPECT_THROW(gfx::make_aot_snode_instance(code, {"activate", "missing"}),
               std::invalid_argument);
  EXPECT_EQ(runtime.debug_registered_kernel_count(), registrations);

  auto left = gfx::make_aot_snode_instance(code, names);
  auto right = gfx::make_aot_snode_instance(code, names);
  EXPECT_EQ(runtime.debug_unique_pipeline_count(), pipelines);
  // Five names include an alias of activate, so each state has four binding
  // registrations, while all executable pipeline objects remain shared.
  EXPECT_EQ(runtime.debug_registered_kernel_count(), registrations + 8);
  EXPECT_EQ(left->get_kernel("activate")->handle().get_launch_id(),
            left->get_kernel("activate_alias")->handle().get_launch_id());
  EXPECT_THROW(left->get_kernel("unrelated"), std::invalid_argument);
  ASSERT_EQ(left->root_bindings().size(), original_bindings.size());
  int selected = 0;
  for (std::size_t i = 0; i < original_bindings.size(); ++i) {
    if (left->root_bindings()[i] < 0) {
      EXPECT_EQ(right->root_bindings()[i], -1);
      continue;
    }
    ++selected;
    EXPECT_NE(left->root_bindings()[i], right->root_bindings()[i]);
    EXPECT_NE(left->root_bindings()[i], original_bindings[i]);
  }
  EXPECT_EQ(selected, 2);

  auto launch = [](aot::Kernel *kernel, std::initializer_list<int> args) {
    LaunchContextBuilder context(kernel);
    int index = 0;
    for (int arg : args)
      context.set_arg({index++}, arg);
    kernel->launch(context);
  };
  auto read = [](aot::Kernel *kernel) {
    std::array<int, 17> output;
    output.fill(-999);
    LaunchContextBuilder context(kernel);
    context.set_arg_external_array_with_shape({0},
        reinterpret_cast<uint64>(output.data()), sizeof(output), {17});
    kernel->launch(context);
    return output;
  };
  const std::array<int, 17> zeros{};
  EXPECT_EQ(read(left->get_kernel("read")), zeros);
  EXPECT_EQ(read(right->get_kernel("read")), zeros);
  launch(original, {4, 100});
  launch(code->get_kernel("update"), {1});
  launch(left->get_kernel("activate"), {5, 10});
  launch(right->get_kernel("activate"), {2, 50});
  launch(left->get_kernel("update"), {2});
  launch(right->get_kernel("update"), {3});
  const std::array<int, 17> expected_left{
      12, 13, 14, 15, 16, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 7};
  const std::array<int, 17> expected_right{
      53, 54, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5};
  const std::array<int, 17> expected_original{
      101, 102, 103, 104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5};
  EXPECT_EQ(read(left->get_kernel("read")), expected_left);
  EXPECT_EQ(read(right->get_kernel("read")), expected_right);
  EXPECT_EQ(read(code->get_kernel("read")), expected_original);
  EXPECT_EQ(original->params().root_buffer_bindings, original_bindings);
  launch(left->get_kernel("drop"), {});
  launch(left->get_kernel("update"), {1});
  auto dropped = expected_left;
  for (int i = 0; i < 4; ++i)
    dropped[i] = 0;
  for (int i = 4; i < 8; ++i)
    ++dropped[i];
  ++dropped[16];
  EXPECT_EQ(read(left->get_kernel("read")), dropped);
  EXPECT_EQ(read(right->get_kernel("read")), expected_right);

  // A binding view cannot reuse code under changed task/ABI metadata.
  auto incompatible = original->params();
  ++incompatible.kernel_attribs.tasks_attribs.front().advisory_total_num_threads;
  EXPECT_ANY_THROW(runtime.register_taichi_kernel_resource_view(
      std::move(incompatible), original->handle()));
  EXPECT_EQ(runtime.debug_registered_kernel_count(), registrations + 8);
  launch(left->get_kernel("update"), {1});
  left.reset();  // Accepted work drains before this private state is released.
  EXPECT_EQ(read(right->get_kernel("read")), expected_right);
  EXPECT_EQ(runtime.debug_registered_kernel_count(), registrations + 4);
  std::weak_ptr<aot::Module> retained = code;
  code.reset();
  EXPECT_FALSE(retained.expired());
  launch(right->get_kernel("update"), {1});
  auto remaining = expected_right;
  for (int i = 0; i < 4; ++i)
    ++remaining[i];
  ++remaining[16];
  EXPECT_EQ(read(right->get_kernel("read")), remaining);
  EXPECT_EQ(runtime.debug_unique_pipeline_count(), pipelines);
  right.reset();
  EXPECT_TRUE(retained.expired());
  EXPECT_EQ(runtime.debug_registered_kernel_count(), 0);
  EXPECT_EQ(runtime.debug_unique_pipeline_count(), 0);
}
