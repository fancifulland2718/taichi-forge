#include <gtest/gtest.h>

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/taichi_vulkan_impl.h"
#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/aot_runtime_client.h"

namespace taichi::lang {

// Integration companion for the import rollback patch. This requires the
// shared-sampled consumer patch: its active loan supplies the track rejection.
// No shader fixture is required; pixel sampling is covered by the Python tests.
TEST(VulkanSharedImageImport, RejectedAliasLeavesReaderAndLaterImportUsable) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device =
      static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!device.vk_caps().timeline_semaphore)
    GTEST_SKIP();
  auto borrower = AotRuntimeClient::create(program);
  auto runtime = reinterpret_cast<TiRuntime>(borrower->handle());
  auto &runtime_impl = *reinterpret_cast<::Runtime *>(runtime);
  auto *image = program.create_texture(BufferFormat::r32f, {2, 2});
  auto lease = std::make_shared<Program::TextureResourceLease>(
      program.acquire_texture_external_lease(image));
  auto *gfx = program.get_gfx_runtime_for_borrowing();
  TiVulkanImageInteropInfo info{};
  ti_set_last_error(TI_ERROR_SUCCESS, "");
  ti_export_vulkan_image(
      runtime, devalloc2devimg(runtime_impl, image->get_device_allocation()),
      &info);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  // Prime one free RHI image wrapper. SyncedPtrStableObjectList reuses its
  // last released slot. Rejected imports must keep returning that slot to the
  // pool; without rollback, the next successful import has a different id.
  auto free_slot = ti_import_vulkan_image(runtime, &info, VK_IMAGE_VIEW_TYPE_2D,
                                          VK_IMAGE_LAYOUT_UNDEFINED);
  ASSERT_NE(free_slot, nullptr);
  ti_free_image(runtime, free_slot);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto workspace = gfx->create_fixed_graph_workspace();
  {
    auto submission = program.acquire_runtime_resource_submission_guard();
    gfx->prepare_fixed_shared_images(
        workspace,
        {{image->runtime_resource_handle(), image->get_device_allocation(),
          lease}},
        program.runtime_fault_domain());
  }
  for (int i = 0; i < 32; ++i) {
    ti_set_last_error(TI_ERROR_SUCCESS, "");
    EXPECT_EQ(ti_import_vulkan_image(runtime, &info, VK_IMAGE_VIEW_TYPE_2D,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
              nullptr);
    char message[512]{};
    std::uint64_t size = sizeof(message);
    EXPECT_EQ(ti_get_last_error(&size, message), TI_ERROR_INVALID_STATE);
    EXPECT_NE(std::string(message).find("image loan"), std::string::npos);
  }
  const auto prefixes = workspace->shared_resource_dependencies();
  ASSERT_EQ(prefixes.size(), 1);
  prefixes.front().wait();
  EXPECT_THROW(gfx->track_image(image->get_device_allocation(),
                                ImageLayout::shader_read),
               std::invalid_argument);
  EXPECT_THROW(gfx->untrack_image(image->get_device_allocation()),
               std::invalid_argument);
  workspace.reset();
  ti_set_last_error(TI_ERROR_SUCCESS, "");
  auto imported =
      ti_import_vulkan_image(runtime, &info, VK_IMAGE_VIEW_TYPE_2D,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  ASSERT_NE(imported, nullptr);
  EXPECT_EQ(imported, free_slot)
      << "rejected import leaked its RHI wrapper slot";
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto rejected = gfx->create_fixed_graph_workspace();
  {
    auto submission = program.acquire_runtime_resource_submission_guard();
    EXPECT_ANY_THROW(gfx->prepare_fixed_shared_images(
        rejected,
        {{image->runtime_resource_handle(), image->get_device_allocation(),
          lease}},
        program.runtime_fault_domain()));
  }
  ti_free_image(runtime, imported);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  {
    auto submission = program.acquire_runtime_resource_submission_guard();
    EXPECT_NO_THROW(gfx->prepare_fixed_shared_images(
        rejected,
        {{image->runtime_resource_handle(), image->get_device_allocation(),
          lease}},
        program.runtime_fault_domain()));
  }
  rejected.reset();
  borrower->close();
  lease.reset();
  program.delete_texture(image);
  program.finalize();
}
}  // namespace taichi::lang
#endif
