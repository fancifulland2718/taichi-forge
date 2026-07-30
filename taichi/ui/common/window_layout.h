#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace taichi::ui {

enum class WindowEdge : std::uint8_t {
  top = 0,
  bottom = 1,
  left = 2,
  right = 3,
};

inline constexpr std::size_t kWindowEdgeCount = 4;

inline constexpr std::size_t window_edge_index(WindowEdge edge) {
  return static_cast<std::size_t>(edge);
}

inline const char *window_edge_name(WindowEdge edge) {
  switch (edge) {
    case WindowEdge::top:
      return "top";
    case WindowEdge::bottom:
      return "bottom";
    case WindowEdge::left:
      return "left";
    case WindowEdge::right:
      return "right";
  }
  return "unknown";
}

inline WindowEdge parse_window_edge(const std::string &edge) {
  if (edge == "top") {
    return WindowEdge::top;
  }
  if (edge == "bottom") {
    return WindowEdge::bottom;
  }
  if (edge == "left") {
    return WindowEdge::left;
  }
  if (edge == "right") {
    return WindowEdge::right;
  }
  throw std::invalid_argument(
      "edge must be 'top', 'bottom', 'left', or 'right'");
}

struct WindowRect {
  float x0{0.0f};
  float y0{0.0f};
  float x1{0.0f};
  float y1{0.0f};

  float width() const noexcept {
    return std::max(0.0f, x1 - x0);
  }

  float height() const noexcept {
    return std::max(0.0f, y1 - y0);
  }

  bool contains(float x, float y) const noexcept {
    return x >= x0 && x < x1 && y >= y0 && y < y1;
  }
};

struct WindowFramebufferRect {
  std::int32_t x0{0};
  std::int32_t y0{0};
  std::int32_t x1{0};
  std::int32_t y1{0};

  std::int32_t width() const noexcept {
    return std::max<std::int32_t>(0, x1 - x0);
  }

  std::int32_t height() const noexcept {
    return std::max<std::int32_t>(0, y1 - y0);
  }
};

struct WindowEdgeRegionConfig {
  bool enabled{false};
  float size{240.0f};
  float minimum_size{120.0f};
  float maximum_fraction{0.5f};
  bool resizable{true};
  bool collapsible{true};
  bool collapsed{false};
};

struct WindowLayoutSnapshot {
  float logical_width{0.0f};
  float logical_height{0.0f};
  std::int32_t framebuffer_width{0};
  std::int32_t framebuffer_height{0};
  std::array<WindowRect, kWindowEdgeCount> edge_regions{};
  WindowRect render_viewport;
  std::array<WindowFramebufferRect, kWindowEdgeCount>
      framebuffer_edge_regions{};
  WindowFramebufferRect framebuffer_render_viewport;
};

class WindowLayoutState {
 public:
  const WindowLayoutSnapshot &snapshot() const noexcept {
    return snapshot_;
  }

  const WindowEdgeRegionConfig &region(WindowEdge edge) const noexcept {
    return regions_[window_edge_index(edge)];
  }

  void configure_region(WindowEdge edge,
                        const WindowEdgeRegionConfig &config) {
    validate_region_config(config);
    auto &stored = regions_[window_edge_index(edge)];
    stored = config;
    if (!stored.collapsible) {
      stored.collapsed = false;
    }
    if (!stored.collapsed) {
      restore_sizes_[window_edge_index(edge)] = stored.size;
    }
    recompute();
  }

  void disable_region(WindowEdge edge) {
    regions_[window_edge_index(edge)].enabled = false;
    recompute();
  }

  void set_region_size(WindowEdge edge, float size) {
    validate_non_negative(size, "size");
    auto &config = regions_[window_edge_index(edge)];
    config.size = size;
    if (!config.collapsed) {
      restore_sizes_[window_edge_index(edge)] = size;
    }
    recompute();
  }

  void resize_region(WindowEdge edge, float delta) {
    validate_finite(delta, "delta");
    auto &config = regions_[window_edge_index(edge)];
    if (!config.enabled || config.collapsed || !config.resizable) {
      return;
    }
    set_region_size(edge, std::max(0.0f, config.size + delta));
  }

  void set_region_collapsed(WindowEdge edge, bool collapsed) {
    auto &config = regions_[window_edge_index(edge)];
    if (!config.enabled || !config.collapsible) {
      return;
    }
    const auto index = window_edge_index(edge);
    if (collapsed && !config.collapsed) {
      restore_sizes_[index] = std::max(config.size, config.minimum_size);
    } else if (!collapsed && config.collapsed) {
      config.size = std::max(restore_sizes_[index], config.minimum_size);
    }
    config.collapsed = collapsed;
    recompute();
  }

  void toggle_region(WindowEdge edge) {
    set_region_collapsed(edge, !region(edge).collapsed);
  }

  void request_region_size(WindowEdge edge, float size) {
    validate_non_negative(size, "size");
    pending_sizes_[window_edge_index(edge)] = size;
  }

  void request_region_resize(WindowEdge edge, float delta) {
    validate_finite(delta, "delta");
    const auto &config = region(edge);
    if (!config.enabled || config.collapsed || !config.resizable) {
      return;
    }
    const auto index = window_edge_index(edge);
    const float base =
        std::isfinite(pending_sizes_[index]) ? pending_sizes_[index]
                                            : config.size;
    pending_sizes_[index] = std::max(0.0f, base + delta);
  }

  void request_region_collapsed(WindowEdge edge, bool collapsed) {
    const auto &config = region(edge);
    if (!config.enabled || !config.collapsible) {
      return;
    }
    pending_collapsed_[window_edge_index(edge)] = collapsed ? 1 : 0;
  }

  void apply_pending_updates() {
    bool changed = false;
    for (std::size_t index = 0; index < kWindowEdgeCount; ++index) {
      auto &config = regions_[index];
      if (std::isfinite(pending_sizes_[index])) {
        config.size = pending_sizes_[index];
        if (!config.collapsed) {
          restore_sizes_[index] = config.size;
        }
        pending_sizes_[index] =
            (std::numeric_limits<float>::quiet_NaN)();
        changed = true;
      }
      if (pending_collapsed_[index] >= 0) {
        const bool collapsed = pending_collapsed_[index] != 0;
        if (config.enabled && config.collapsible &&
            collapsed != config.collapsed) {
          if (collapsed) {
            restore_sizes_[index] =
                std::max(config.size, config.minimum_size);
          } else {
            config.size =
                std::max(restore_sizes_[index], config.minimum_size);
          }
          config.collapsed = collapsed;
          changed = true;
        }
        pending_collapsed_[index] = -1;
      }
    }
    if (changed) {
      recompute();
    }
  }

  void set_minimum_render_size(float width, float height) {
    validate_non_negative(width, "minimum render width");
    validate_non_negative(height, "minimum render height");
    minimum_render_width_ = width;
    minimum_render_height_ = height;
    recompute();
  }

  float minimum_render_width() const noexcept {
    return minimum_render_width_;
  }

  float minimum_render_height() const noexcept {
    return minimum_render_height_;
  }

  void update_dimensions(float logical_width,
                         float logical_height,
                         std::int32_t framebuffer_width,
                         std::int32_t framebuffer_height) {
    validate_non_negative(logical_width, "logical width");
    validate_non_negative(logical_height, "logical height");
    if (framebuffer_width < 0 || framebuffer_height < 0) {
      throw std::invalid_argument(
          "framebuffer dimensions must be non-negative");
    }
    if (snapshot_.logical_width == logical_width &&
        snapshot_.logical_height == logical_height &&
        snapshot_.framebuffer_width == framebuffer_width &&
        snapshot_.framebuffer_height == framebuffer_height) {
      return;
    }
    snapshot_.logical_width = logical_width;
    snapshot_.logical_height = logical_height;
    snapshot_.framebuffer_width = framebuffer_width;
    snapshot_.framebuffer_height = framebuffer_height;
    recompute();
  }

  bool cursor_in_render_viewport(float normalized_x,
                                 float normalized_y_from_bottom) const noexcept {
    if (snapshot_.logical_width <= 0.0f ||
        snapshot_.logical_height <= 0.0f) {
      return false;
    }
    const float x = normalized_x * snapshot_.logical_width;
    const float y =
        (1.0f - normalized_y_from_bottom) * snapshot_.logical_height;
    return snapshot_.render_viewport.contains(x, y);
  }

  std::pair<float, float> render_cursor_position(
      float normalized_x,
      float normalized_y_from_bottom,
      bool clamp) const noexcept {
    const auto &viewport = snapshot_.render_viewport;
    if (viewport.width() <= 0.0f || viewport.height() <= 0.0f ||
        snapshot_.logical_width <= 0.0f ||
        snapshot_.logical_height <= 0.0f) {
      return {0.0f, 0.0f};
    }
    const float x = normalized_x * snapshot_.logical_width;
    const float y_from_top =
        (1.0f - normalized_y_from_bottom) * snapshot_.logical_height;
    float local_x = (x - viewport.x0) / viewport.width();
    float local_y_from_top = (y_from_top - viewport.y0) / viewport.height();
    if (clamp) {
      local_x = std::clamp(local_x, 0.0f, 1.0f);
      local_y_from_top = std::clamp(local_y_from_top, 0.0f, 1.0f);
    }
    return {local_x, 1.0f - local_y_from_top};
  }

 private:
  static void validate_finite(float value, const char *name) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(name) + " must be finite");
    }
  }

  static void validate_non_negative(float value, const char *name) {
    validate_finite(value, name);
    if (value < 0.0f) {
      throw std::invalid_argument(std::string(name) +
                                  " must be non-negative");
    }
  }

  static void validate_region_config(const WindowEdgeRegionConfig &config) {
    validate_non_negative(config.size, "size");
    validate_non_negative(config.minimum_size, "minimum_size");
    validate_finite(config.maximum_fraction, "maximum_fraction");
    if (config.maximum_fraction <= 0.0f ||
        config.maximum_fraction > 1.0f) {
      throw std::invalid_argument(
          "maximum_fraction must be greater than zero and at most one");
    }
  }

  static float desired_extent(const WindowEdgeRegionConfig &config,
                              float available_extent) {
    if (!config.enabled || config.collapsed || available_extent <= 0.0f) {
      return 0.0f;
    }
    const float maximum = config.maximum_fraction * available_extent;
    const float minimum = std::min(config.minimum_size, maximum);
    return std::clamp(config.size, minimum, maximum);
  }

  static std::pair<float, float> fit_pair(float first,
                                         float second,
                                         float first_minimum,
                                         float second_minimum,
                                         float available) {
    available = std::max(0.0f, available);
    if (first + second <= available) {
      return {first, second};
    }

    first_minimum = std::min(first, std::max(0.0f, first_minimum));
    second_minimum = std::min(second, std::max(0.0f, second_minimum));
    const float shrinkable =
        (first - first_minimum) + (second - second_minimum);
    const float excess = first + second - available;
    if (shrinkable > 0.0f && excess <= shrinkable) {
      const float ratio = excess / shrinkable;
      return {
          first - (first - first_minimum) * ratio,
          second - (second - second_minimum) * ratio,
      };
    }

    const float minimum_sum = first_minimum + second_minimum;
    if (minimum_sum <= 0.0f) {
      return {0.0f, 0.0f};
    }
    const float ratio = available / minimum_sum;
    return {first_minimum * ratio, second_minimum * ratio};
  }

  WindowFramebufferRect to_framebuffer(const WindowRect &rect) const {
    const float sx =
        snapshot_.logical_width > 0.0f
            ? static_cast<float>(snapshot_.framebuffer_width) /
                  snapshot_.logical_width
            : 0.0f;
    const float sy =
        snapshot_.logical_height > 0.0f
            ? static_cast<float>(snapshot_.framebuffer_height) /
                  snapshot_.logical_height
            : 0.0f;
    const auto scale_and_clamp = [](float value, float scale,
                                    std::int32_t maximum) {
      const auto rounded = static_cast<std::int32_t>(
          std::lround(std::max(0.0f, value) * scale));
      return std::clamp<std::int32_t>(rounded, 0, maximum);
    };
    return {
        scale_and_clamp(rect.x0, sx, snapshot_.framebuffer_width),
        scale_and_clamp(rect.y0, sy, snapshot_.framebuffer_height),
        scale_and_clamp(rect.x1, sx, snapshot_.framebuffer_width),
        scale_and_clamp(rect.y1, sy, snapshot_.framebuffer_height),
    };
  }

  void recompute() {
    const float width = snapshot_.logical_width;
    const float height = snapshot_.logical_height;

    const auto &top_config = region(WindowEdge::top);
    const auto &bottom_config = region(WindowEdge::bottom);
    float top = desired_extent(top_config, height);
    float bottom = desired_extent(bottom_config, height);
    const float vertical_available =
        std::max(0.0f, height - std::min(height, minimum_render_height_));
    std::tie(top, bottom) =
        fit_pair(top, bottom,
                 top_config.enabled && !top_config.collapsed
                     ? top_config.minimum_size
                     : 0.0f,
                 bottom_config.enabled && !bottom_config.collapsed
                     ? bottom_config.minimum_size
                     : 0.0f,
                 vertical_available);

    const auto &left_config = region(WindowEdge::left);
    const auto &right_config = region(WindowEdge::right);
    float left = desired_extent(left_config, width);
    float right = desired_extent(right_config, width);
    const float horizontal_available =
        std::max(0.0f, width - std::min(width, minimum_render_width_));
    std::tie(left, right) =
        fit_pair(left, right,
                 left_config.enabled && !left_config.collapsed
                     ? left_config.minimum_size
                     : 0.0f,
                 right_config.enabled && !right_config.collapsed
                     ? right_config.minimum_size
                     : 0.0f,
                 horizontal_available);

    snapshot_.edge_regions[window_edge_index(WindowEdge::top)] =
        WindowRect{0.0f, 0.0f, width, top};
    snapshot_.edge_regions[window_edge_index(WindowEdge::bottom)] =
        WindowRect{0.0f, height - bottom, width, height};
    snapshot_.edge_regions[window_edge_index(WindowEdge::left)] =
        WindowRect{0.0f, top, left, height - bottom};
    snapshot_.edge_regions[window_edge_index(WindowEdge::right)] =
        WindowRect{width - right, top, width, height - bottom};
    snapshot_.render_viewport =
        WindowRect{left, top, width - right, height - bottom};

    for (std::size_t i = 0; i < kWindowEdgeCount; ++i) {
      snapshot_.framebuffer_edge_regions[i] =
          to_framebuffer(snapshot_.edge_regions[i]);
    }
    snapshot_.framebuffer_render_viewport =
        to_framebuffer(snapshot_.render_viewport);
  }

  std::array<WindowEdgeRegionConfig, kWindowEdgeCount> regions_{};
  std::array<float, kWindowEdgeCount> restore_sizes_{
      240.0f,
      240.0f,
      240.0f,
      240.0f,
  };
  std::array<float, kWindowEdgeCount> pending_sizes_{
      (std::numeric_limits<float>::quiet_NaN)(),
      (std::numeric_limits<float>::quiet_NaN)(),
      (std::numeric_limits<float>::quiet_NaN)(),
      (std::numeric_limits<float>::quiet_NaN)(),
  };
  std::array<std::int8_t, kWindowEdgeCount> pending_collapsed_{
      -1,
      -1,
      -1,
      -1,
  };
  float minimum_render_width_{1.0f};
  float minimum_render_height_{1.0f};
  WindowLayoutSnapshot snapshot_;
};

}  // namespace taichi::ui
