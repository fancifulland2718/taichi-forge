# Build-local edits to ImGui f5befd2d29e66809cd1110a152e375a7f1981f06.
# Never modify the dependency checkout or require a private submodule commit.
function(_ti_imgui_replace old new)
  string(REPLACE "${old}" "" _without "${_ti_imgui_source}")
  string(LENGTH "${_ti_imgui_source}" _before_length)
  string(LENGTH "${_without}" _after_length)
  string(LENGTH "${old}" _old_length)
  math(EXPR _removed "${_before_length} - ${_after_length}")
  if(NOT _removed EQUAL _old_length)
    message(FATAL_ERROR "ImGui Vulkan patch anchor is missing or ambiguous")
  endif()
  string(REPLACE "${old}" "${new}" _ti_imgui_source "${_ti_imgui_source}")
  set(_ti_imgui_source "${_ti_imgui_source}" PARENT_SCOPE)
endfunction()

function(ti_prepare_imgui_vulkan output)
  set(_input "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp")
  set(_header "${PROJECT_SOURCE_DIR}/taichi/ui/ggui/imgui_vulkan_forge.h")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_input}" "${_header}")
  file(READ "${_input}" _ti_imgui_source)
  string(REPLACE "\r\n" "\n" _ti_imgui_source "${_ti_imgui_source}")
  string(SHA256 _preimage "${_ti_imgui_source}")
  if(NOT _preimage STREQUAL "1af35121a8cc8ff4d16d4d9da89c5724bbc2cb930920247eeeac0faa6b1ff006")
    message(FATAL_ERROR "Pinned ImGui Vulkan source changed; review the Forge backend patch")
  endif()
  _ti_imgui_replace([==[#include "imgui_impl_vulkan.h"
]==] [==[#include "imgui_impl_vulkan.h"
#include "taichi/ui/ggui/imgui_vulkan_forge.h"
]==])
  _ti_imgui_replace([==[    VkCommandBuffer             TexCommandBuffer;
]==] [==[    VkCommandBuffer             TexCommandBuffer;
    // Forge: keep upload allocations reachable until a checked queue proof.
    VkBuffer                    TexUploadBuffer;
    VkDeviceMemory              TexUploadMemory;
    bool                        TexUploadMapped;
    bool                        TexUploadUnsealed;
    bool                        TexUploadReady;
]==])
  _ti_imgui_replace([==[void ImGui_ImplVulkan_RenderDrawData(ImDrawData* draw_data, VkCommandBuffer command_buffer, VkPipeline pipeline)
]==] [==[static void ImGui_ImplVulkan_RenderDrawDataImpl(ImDrawData* draw_data, VkCommandBuffer command_buffer, VkPipeline pipeline, uint32_t frame_slot, bool explicit_slot)
]==])
  _ti_imgui_replace([==[    wrb->Index = (wrb->Index + 1) % wrb->Count;
]==] [==[    if (explicit_slot)
    {
        IM_ASSERT(frame_slot < wrb->Count);
        if (frame_slot >= wrb->Count)
            return;
        wrb->Index = frame_slot;
    }
    else
        wrb->Index = (wrb->Index + 1) % wrb->Count;
]==])
  # Limit result-check edits to the known font-upload function.
  string(FIND "${_ti_imgui_source}" "bool ImGui_ImplVulkan_CreateFontsTexture()\n" _font_start)
  string(FIND "${_ti_imgui_source}" "// You probably never need to call this" _font_end)
  math(EXPR _font_length "${_font_end} - ${_font_start}")
  if(_font_start LESS 0 OR _font_length LESS_EQUAL 0)
    message(FATAL_ERROR "ImGui Vulkan font function boundary changed")
  endif()
  string(SUBSTRING "${_ti_imgui_source}" 0 ${_font_start} _prefix)
  string(SUBSTRING "${_ti_imgui_source}" ${_font_end} -1 _suffix)
  string(SUBSTRING "${_ti_imgui_source}" ${_font_start} ${_font_length} _ti_imgui_source)
  _ti_imgui_replace([==[    VkResult err;
]==] [==[    VkResult err;
    if (bd->TexUploadUnsealed)
        return false;
    bd->TexUploadReady = false;
]==])
  _ti_imgui_replace([==[        vkQueueWaitIdle(v->Queue);
]==] [==[        bd->TexUploadUnsealed = true;
        err = vkQueueWaitIdle(v->Queue);
        check_vk_result(err);
        bd->TexUploadUnsealed = false;
]==])
  _ti_imgui_replace([==[        vkCreateCommandPool(v->Device, &info, v->Allocator, &bd->TexCommandPool);
]==] [==[        err = vkCreateCommandPool(v->Device, &info, v->Allocator, &bd->TexCommandPool);
        check_vk_result(err);
]==])
  _ti_imgui_replace([==[    VkDeviceMemory upload_buffer_memory;
    VkBuffer upload_buffer;
]==] [==[    // Recorded-only failures retain these fields for checked backend cleanup.
    VkDeviceMemory& upload_buffer_memory = bd->TexUploadMemory;
    VkBuffer& upload_buffer = bd->TexUploadBuffer;
]==])
  _ti_imgui_replace([==[        memcpy(map, pixels, upload_size);
]==] [==[        bd->TexUploadMapped = true;
        memcpy(map, pixels, upload_size);
]==])
  _ti_imgui_replace([==[        vkUnmapMemory(v->Device, upload_buffer_memory);
]==] [==[        vkUnmapMemory(v->Device, upload_buffer_memory);
        bd->TexUploadMapped = false;
]==])
  _ti_imgui_replace([==[    // Store our identifier
    io.Fonts->SetTexID((ImTextureID)backend_tex->DescriptorSet);

]==] [==[]==])
  _ti_imgui_replace([==[    err = vkQueueSubmit(v->Queue, 1, &end_info, VK_NULL_HANDLE);
]==] [==[    bd->TexUploadUnsealed = true;
    err = vkQueueSubmit(v->Queue, 1, &end_info, VK_NULL_HANDLE);
]==])
  _ti_imgui_replace([==[    vkDestroyBuffer(v->Device, upload_buffer, v->Allocator);
    vkFreeMemory(v->Device, upload_buffer_memory, v->Allocator);
]==] [==[    bd->TexUploadUnsealed = false;
    ImGui_ImplVulkan_ForgeReleaseUpload(bd);
    bd->TexUploadReady = true;
    // Publish the atlas only after the checked upload retirement proof.
    io.Fonts->SetTexID((ImTextureID)backend_tex->DescriptorSet);
]==])
  string(REPLACE "check_vk_result(err);" "" _without_checks "${_ti_imgui_source}")
  string(LENGTH "${_ti_imgui_source}" _checked_length)
  string(LENGTH "${_without_checks}" _unchecked_length)
  string(LENGTH "check_vk_result(err);" _check_length)
  math(EXPR _font_check_count "(${_checked_length} - ${_unchecked_length}) / ${_check_length}")
  if(NOT _font_check_count EQUAL 17)
    message(FATAL_ERROR "ImGui Vulkan font result-check count changed")
  endif()
  string(REPLACE "check_vk_result(err);" "if (err != VK_SUCCESS) { check_vk_result(err); return false; }" _ti_imgui_source "${_ti_imgui_source}")
  set(_ti_imgui_source "${_prefix}${_ti_imgui_source}${_suffix}")
  _ti_imgui_replace([==[bool ImGui_ImplVulkan_CreateFontsTexture()
]==] [==[void ImGui_ImplVulkan_RenderDrawData(ImDrawData* draw_data, VkCommandBuffer command_buffer, VkPipeline pipeline)
{
    ImGui_ImplVulkan_RenderDrawDataImpl(draw_data, command_buffer, pipeline, 0, false);
}

void ImGui_ImplVulkan_ForgeRenderDrawData(ImDrawData* draw_data, VkCommandBuffer command_buffer, uint32_t frame_slot)
{
    ImGui_ImplVulkan_RenderDrawDataImpl(draw_data, command_buffer, VK_NULL_HANDLE, frame_slot, true);
}

ImGui_ImplVulkan_ForgeFontState ImGui_ImplVulkan_ForgeGetFontState()
{
    ImGui_ImplVulkan_Data* bd = ImGui_ImplVulkan_GetBackendData();
    if (!bd)
        return ImGui_ImplVulkan_ForgeFontState::empty;
    if (bd->TexUploadUnsealed)
        return ImGui_ImplVulkan_ForgeFontState::unsealed;
    if (bd->TexUploadReady && bd->FontTexture.DescriptorSet &&
        ImGui::GetIO().Fonts->TexID == (ImTextureID)bd->FontTexture.DescriptorSet)
        return ImGui_ImplVulkan_ForgeFontState::ready;
    return ImGui_ImplVulkan_ForgeFontState::empty;
}

static void ImGui_ImplVulkan_ForgeReleaseUpload(ImGui_ImplVulkan_Data* bd)
{
    if (bd->TexUploadUnsealed)
        return;
    ImGui_ImplVulkan_InitInfo* v = &bd->VulkanInitInfo;
    if (bd->TexUploadMapped) { vkUnmapMemory(v->Device, bd->TexUploadMemory); bd->TexUploadMapped = false; }
    if (bd->TexUploadBuffer) { vkDestroyBuffer(v->Device, bd->TexUploadBuffer, v->Allocator); bd->TexUploadBuffer = VK_NULL_HANDLE; }
    if (bd->TexUploadMemory) { vkFreeMemory(v->Device, bd->TexUploadMemory, v->Allocator); bd->TexUploadMemory = VK_NULL_HANDLE; }
}

bool ImGui_ImplVulkan_CreateFontsTexture()
]==])
  _ti_imgui_replace([==[void ImGui_ImplVulkan_DestroyFontsTexture()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplVulkan_Data* bd = ImGui_ImplVulkan_GetBackendData();
]==] [==[void ImGui_ImplVulkan_DestroyFontsTexture()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplVulkan_Data* bd = ImGui_ImplVulkan_GetBackendData();
    if (bd->TexUploadUnsealed)
        return;
    bd->TexUploadReady = false;
]==])
  _ti_imgui_replace([==[void    ImGui_ImplVulkan_DestroyDeviceObjects()
{
    ImGui_ImplVulkan_Data* bd = ImGui_ImplVulkan_GetBackendData();
]==] [==[void    ImGui_ImplVulkan_DestroyDeviceObjects()
{
    ImGui_ImplVulkan_Data* bd = ImGui_ImplVulkan_GetBackendData();
    if (bd->TexUploadUnsealed)
        return;
    ImGui_ImplVulkan_ForgeReleaseUpload(bd);
]==])
  _ti_imgui_replace([==[    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();
]==] [==[    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    // Never delete the owner of an upload that lacks a completion proof.
    if (bd->TexUploadUnsealed)
        return;
    ImGuiIO& io = ImGui::GetIO();
]==])
  string(SHA256 _postimage "${_ti_imgui_source}")
  if(NOT _postimage STREQUAL "f736009d842ddd23aaf5ee92b33e9b31cc50f728386f5af7ea2ac01e9d059ef7")
    message(FATAL_ERROR "Unexpected generated ImGui Vulkan source")
  endif()
  set(_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/imgui")
  file(MAKE_DIRECTORY "${_directory}")
  set(_generated "${_directory}/imgui_impl_vulkan.cpp")
  file(WRITE "${_generated}.in" "${_ti_imgui_source}")
  configure_file("${_generated}.in" "${_generated}" COPYONLY)
  file(SHA256 "${_generated}" _generated_raw)
  file(SHA256 "${_input}" _raw_input)
  file(SHA256 "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" _transform)
  file(SHA256 "${_header}" _private_header)
  file(WRITE "${_directory}/source-provenance.json"
    "{\n  \"scheme\": \"forge-imgui-vulkan-v1\",\n  \"upstream_commit\": \"f5befd2d29e66809cd1110a152e375a7f1981f06\",\n  \"input_raw_sha256\": \"${_raw_input}\",\n  \"input_lf_sha256\": \"${_preimage}\",\n  \"transform_sha256\": \"${_transform}\",\n  \"private_header_sha256\": \"${_private_header}\",\n  \"generated_lf_sha256\": \"${_postimage}\",\n  \"generated_raw_sha256\": \"${_generated_raw}\"\n}\n")
  set(${output} "${_generated}" PARENT_SCOPE)
endfunction()
