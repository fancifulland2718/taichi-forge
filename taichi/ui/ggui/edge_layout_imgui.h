#pragma once

#include <algorithm>
#include <string>

#include <imgui.h>

#include "taichi/ui/common/window_layout.h"

namespace taichi::ui::vulkan {

inline ImGuiMouseCursor edge_resize_cursor(WindowEdge edge) {
  return edge == WindowEdge::left || edge == WindowEdge::right
             ? ImGuiMouseCursor_ResizeEW
             : ImGuiMouseCursor_ResizeNS;
}

inline void edge_splitter_geometry(const WindowLayoutState &layout,
                                   WindowEdge edge,
                                   ImVec2 *position,
                                   ImVec2 *size) {
  constexpr float kExpandedHitSize = 8.0f;
  constexpr float kCollapsedThickness = 16.0f;
  constexpr float kCollapsedLength = 40.0f;
  const auto &snapshot = layout.snapshot();
  const auto &config = layout.region(edge);
  const auto &rect = snapshot.edge_regions[window_edge_index(edge)];

  if (!config.collapsed) {
    if (edge == WindowEdge::left) {
      *position = ImVec2(rect.x1 - kExpandedHitSize * 0.5f, rect.y0);
      *size = ImVec2(kExpandedHitSize, rect.height());
    } else if (edge == WindowEdge::right) {
      *position = ImVec2(rect.x0 - kExpandedHitSize * 0.5f, rect.y0);
      *size = ImVec2(kExpandedHitSize, rect.height());
    } else if (edge == WindowEdge::top) {
      *position = ImVec2(rect.x0, rect.y1 - kExpandedHitSize * 0.5f);
      *size = ImVec2(rect.width(), kExpandedHitSize);
    } else {
      *position = ImVec2(rect.x0, rect.y0 - kExpandedHitSize * 0.5f);
      *size = ImVec2(rect.width(), kExpandedHitSize);
    }
    return;
  }

  if (edge == WindowEdge::left || edge == WindowEdge::right) {
    *position = ImVec2(
        edge == WindowEdge::left
            ? 0.0f
            : std::max(0.0f, snapshot.logical_width - kCollapsedThickness),
        std::max(0.0f,
                 (snapshot.logical_height - kCollapsedLength) * 0.5f));
    *size = ImVec2(kCollapsedThickness, kCollapsedLength);
  } else {
    *position = ImVec2(
        std::max(0.0f,
                 (snapshot.logical_width - kCollapsedLength) * 0.5f),
        edge == WindowEdge::top
            ? 0.0f
            : std::max(0.0f,
                       snapshot.logical_height - kCollapsedThickness));
    *size = ImVec2(kCollapsedLength, kCollapsedThickness);
  }
}

inline void render_edge_splitter(WindowLayoutState *layout, WindowEdge edge) {
  if (layout == nullptr) {
    return;
  }
  const auto config = layout->region(edge);
  if (!config.enabled || (!config.resizable && !config.collapsible)) {
    return;
  }

  ImVec2 position;
  ImVec2 size;
  edge_splitter_geometry(*layout, edge, &position, &size);
  if (size.x <= 0.0f || size.y <= 0.0f) {
    return;
  }

  const std::string suffix =
      std::string("##taichi_edge_splitter_") + window_edge_name(edge);
  ImGui::SetNextWindowPos(position, ImGuiCond_Always);
  ImGui::SetNextWindowSize(size, ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(config.collapsed ? 0.72f : 0.0f);
  constexpr ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
      ImGuiWindowFlags_NoFocusOnAppearing;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::Begin(suffix.c_str(), nullptr, window_flags);
  ImGui::InvisibleButton(suffix.c_str(), size);

  const bool hovered = ImGui::IsItemHovered();
  const bool active = ImGui::IsItemActive();
  if (hovered || active) {
    ImGui::SetMouseCursor(edge_resize_cursor(edge));
  }

  if (config.collapsed) {
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
      layout->request_region_collapsed(edge, false);
    }
  } else {
    if (active && config.resizable) {
      const ImVec2 delta = ImGui::GetIO().MouseDelta;
      float extent_delta = 0.0f;
      if (edge == WindowEdge::left) {
        extent_delta = delta.x;
      } else if (edge == WindowEdge::right) {
        extent_delta = -delta.x;
      } else if (edge == WindowEdge::top) {
        extent_delta = delta.y;
      } else {
        extent_delta = -delta.y;
      }
      if (extent_delta != 0.0f) {
        layout->request_region_resize(edge, extent_delta);
      }
    }
    if (hovered && config.collapsible &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      layout->request_region_collapsed(edge, true);
    }
  }

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImU32 color =
      ImGui::GetColorU32(hovered || active ? ImGuiCol_SeparatorActive
                                          : ImGuiCol_Separator);
  const ImVec2 minimum = ImGui::GetWindowPos();
  const ImVec2 maximum =
      ImVec2(minimum.x + size.x, minimum.y + size.y);
  if (edge == WindowEdge::left || edge == WindowEdge::right) {
    const float x = (minimum.x + maximum.x) * 0.5f;
    draw_list->AddLine(ImVec2(x, minimum.y), ImVec2(x, maximum.y), color,
                       config.collapsed ? 2.0f : 1.0f);
  } else {
    const float y = (minimum.y + maximum.y) * 0.5f;
    draw_list->AddLine(ImVec2(minimum.x, y), ImVec2(maximum.x, y), color,
                       config.collapsed ? 2.0f : 1.0f);
  }

  ImGui::End();
  ImGui::PopStyleVar(2);
}

inline bool begin_edge_region(WindowLayoutState *layout,
                              const std::string &name,
                              WindowEdge edge) {
  if (layout == nullptr) {
    return false;
  }
  const auto config = layout->region(edge);
  if (!config.enabled) {
    return false;
  }
  if (config.collapsed) {
    render_edge_splitter(layout, edge);
    return false;
  }

  const auto &rect =
      layout->snapshot().edge_regions[window_edge_index(edge)];
  if (rect.width() <= 0.0f || rect.height() <= 0.0f) {
    return false;
  }
  const std::string window_name =
      name + "##taichi_edge_region_" + window_edge_name(edge);
  ImGui::SetNextWindowPos(ImVec2(rect.x0, rect.y0), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(rect.width(), rect.height()),
                           ImGuiCond_Always);
  constexpr ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoSavedSettings;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::Begin(window_name.c_str(), nullptr, window_flags);
  return true;
}

inline void end_edge_region(WindowLayoutState *layout, WindowEdge edge) {
  ImGui::End();
  ImGui::PopStyleVar();
  render_edge_splitter(layout, edge);
}

}  // namespace taichi::ui::vulkan
