#pragma once

// Forge-private extensions to pinned ImGui 1.91.9b. The original public
// backend header and InitInfo layout remain unchanged.
#include <imgui_impl_vulkan.h>

enum class ImGui_ImplVulkan_ForgeFontState { empty, ready, unsealed };

ImGui_ImplVulkan_ForgeFontState ImGui_ImplVulkan_ForgeGetFontState();

// The caller owns the slot schedule and must prove its previous GPU reader
// complete before calling. This function does not rotate or wait for a slot.
void ImGui_ImplVulkan_ForgeRenderDrawData(ImDrawData* draw_data,
                                        VkCommandBuffer command_buffer,
                                        uint32_t frame_slot);
