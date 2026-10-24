#include "gtest/gtest.h"

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/native_resource_registry.h"
#include "taichi/program/program.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/native_vulkan_ray_access.h"
#include "taichi/runtime/native_ray_admission.h"

namespace taichi::lang {

TEST(VulkanProgramASImport, OrderedAliasesRollbackAndOwnerRetirement) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  auto program = std::make_unique<Program>(Arch::vulkan);
  program->materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program->get_compute_device());
  if (!device.vk_caps().acceleration_structure || !device.vk_caps().ray_query)
    GTEST_SKIP();
  auto first = AotRuntimeClient::create(*program);
  auto second = AotRuntimeClient::create(*program);
  auto &registry = reinterpret_cast<::Runtime *>(first->handle())->native_resource_registry();
  auto &peer = reinterpret_cast<::Runtime *>(second->handle())->native_resource_registry();
  const auto a = program->create_vulkan_triangle_blas_resource(3, 1);
  const auto b = program->create_vulkan_triangle_blas_resource(3, 1);
  const auto tlas = program->create_vulkan_instance_tlas_resource({a, a, b});
  auto source = program->acquire_vulkan_ray_native_resource(tlas);
  auto original_a = program->acquire_vulkan_ray_native_resource(a);
  auto original_b = program->acquire_vulkan_ray_native_resource(b);
  auto instances = std::dynamic_pointer_cast<native_vulkan_ray::InstanceAccess>(source);
  ASSERT_EQ(instances->instance_blases().size(), 3);
  EXPECT_EQ(instances->dependencies().size(), 2);
  auto import = [&](AotRuntimeClient &client) {
    return reinterpret_cast<TiNativeResource>(
        client.import_acceleration_structure(program.get(), tlas, false));
  };
  EXPECT_THROW(first->import_acceleration_structure(nullptr, tlas, false),
               std::invalid_argument);
  const auto one = import(*first);
  const auto two = import(*first);
  const auto other = import(*second);
  EXPECT_NE(one, two);
  EXPECT_EQ(registry.get(one), source);
  EXPECT_EQ(registry.get(two), source);
  EXPECT_EQ(peer.get(other), source);
  const auto a0 = registry.resolve(one, {"blas_0"});
  const auto a1 = registry.resolve(one, {"blas_1"});
  const auto b2 = registry.resolve(one, {"blas_2"});
  EXPECT_EQ(a0, a1);
  EXPECT_EQ(registry.get(a0), original_a);
  EXPECT_EQ(registry.get(b2), original_b);
  EXPECT_NE(a0, registry.resolve(two, {"blas_0"}));
  EXPECT_EQ(registry.get(a0), registry.get(registry.resolve(two, {"blas_1"})));
  EXPECT_ANY_THROW(registry.resolve(one, {"blas_3"}));

  // Failing after publishing an input must release it without a queue submit.
  const auto rollback = program->create_vulkan_triangle_blas_resource(3, 1);
  auto rollback_owner = program->acquire_vulkan_ray_native_resource(rollback);
  std::weak_ptr<NativeResource> weak_rollback = rollback_owner;
  const auto submissions = device.queue_submission_snapshot().queue_submit_calls;
  EXPECT_ANY_THROW(registry.publish_borrowed(nullptr, {{"blas_0", rollback_owner}}));
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, submissions);
  program->destroy_vulkan_ray_resource(rollback);
  rollback_owner.reset();
  program->synchronize();
  EXPECT_TRUE(weak_rollback.expired());

  program->destroy_vulkan_ray_resource(a);
  program->destroy_vulkan_ray_resource(b);
  const auto after_blas_close = import(*first);
  EXPECT_EQ(registry.get(registry.resolve(after_blas_close, {"blas_2"})), original_b);
  program->destroy_vulkan_ray_resource(tlas);
  EXPECT_ANY_THROW(import(*first));
  EXPECT_EQ(registry.get(one), source);
  registry.destroy(one);
  EXPECT_ANY_THROW(registry.get(one));
  EXPECT_ANY_THROW(registry.get(a0));
  EXPECT_ANY_THROW(registry.get(b2));
  EXPECT_EQ(registry.get(two), source);
  first->close();
  EXPECT_EQ(peer.get(other), source);
  std::weak_ptr<NativeResource> weak_scene = source, weak_a = original_a, weak_b = original_b;
  source.reset();
  instances.reset();
  original_a.reset();
  original_b.reset();
  program.reset();
  EXPECT_TRUE(second->closed());
  EXPECT_TRUE(weak_scene.expired());
  EXPECT_TRUE(weak_a.expired());
  EXPECT_TRUE(weak_b.expired());
  EXPECT_NO_THROW(second->close());
}

}  // namespace taichi::lang
#endif
