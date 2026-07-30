#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "taichi/ui/common/window_layout.h"
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

class GuiBase {
 public:
  void set_font_scale(float scale) {
    validate_positive(scale, "scale");
    font_scale_mode_ = FontScaleMode::fixed_scale;
    unzoomed_font_scale_ = scale;
    unzoomed_font_size_ = base_font_size_ * unzoomed_font_scale_;
    apply_font_zoom();
  }

  void set_font_scale_from_window_height(float reference_height,
                                         float reference_scale) {
    validate_positive(reference_height, "reference_height");
    validate_positive(reference_scale, "reference_scale");
    font_scale_mode_ = FontScaleMode::window_height_scale;
    reference_height_ = reference_height;
    reference_scale_ = reference_scale;
    update_font_scale(last_logical_height_, base_font_size_);
  }

  float get_font_scale() const {
    return font_scale_;
  }

  void set_font_size(float size) {
    validate_positive(size, "size");
    font_scale_mode_ = FontScaleMode::fixed_size;
    unzoomed_font_size_ = size;
    unzoomed_font_scale_ = unzoomed_font_size_ / base_font_size_;
    apply_font_zoom();
  }

  void set_font_size_from_window_height(float reference_height,
                                        float reference_size,
                                        float minimum_size,
                                        float maximum_size) {
    validate_positive(reference_height, "reference_height");
    validate_positive(reference_size, "reference_size");
    validate_positive(minimum_size, "minimum_size");
    validate_positive(maximum_size, "maximum_size");
    if (minimum_size > reference_size || reference_size > maximum_size) {
      throw std::invalid_argument(
          "reference_size must be between minimum_size and maximum_size");
    }
    font_scale_mode_ = FontScaleMode::window_height_size;
    reference_height_ = reference_height;
    reference_size_ = reference_size;
    minimum_size_ = minimum_size;
    maximum_size_ = maximum_size;
    update_font_scale(last_logical_height_, base_font_size_);
  }

  float get_font_size() const {
    return font_size_;
  }

  void set_font_zoom(float zoom) {
    validate_positive(zoom, "zoom");
    font_zoom_ = zoom;
    apply_font_zoom();
  }

  void adjust_font_zoom(float delta) {
    if (!std::isfinite(delta)) {
      throw std::invalid_argument("delta must be finite");
    }
    set_font_zoom(std::clamp(font_zoom_ + delta, 0.5f, 3.0f));
  }

  void reset_font_zoom() {
    set_font_zoom(1.0f);
  }

  float get_font_zoom() const {
    return font_zoom_;
  }

  void set_font_shortcuts_enabled(bool enabled) {
    font_shortcuts_enabled_ = enabled;
  }

  bool font_shortcuts_enabled() const {
    return font_shortcuts_enabled_;
  }

  virtual void begin(const std::string &name,
                     float x,
                     float y,
                     float width,
                     float height) = 0;
  virtual void begin_auto(const std::string &name,
                          float x,
                          float y,
                          float width) = 0;
  virtual bool begin_collapsible_section(const std::string &name,
                                         bool default_open) = 0;
  virtual void end_collapsible_section() = 0;
  virtual bool begin_edge_region(const std::string &name,
                                 WindowEdge edge) = 0;
  virtual void end_edge_region(WindowEdge edge) = 0;
  virtual void end() = 0;
  virtual void text(const std::string &text) = 0;
  virtual void text(const std::string &text, glm::vec3 color) = 0;
  virtual bool checkbox(const std::string &name, bool old_value) = 0;
  virtual int slider_int(const std::string &name,
                         int old_value,
                         int minimum,
                         int maximum) = 0;
  virtual float slider_float(const std::string &name,
                             float old_value,
                             float minimum,
                             float maximum) = 0;
  virtual glm::vec3 color_edit_3(const std::string &name,
                                 glm::vec3 old_value) = 0;
  virtual bool button(const std::string &text) = 0;
  virtual void prepare_for_next_frame() = 0;
  virtual bool has_widgets() const = 0;
  virtual bool wants_capture_mouse() const = 0;
  virtual bool wants_capture_keyboard() const = 0;
  virtual void end_frame() = 0;
  virtual ~GuiBase() = default;

 protected:
  float update_font_scale(float logical_height, float base_font_size) {
    if (std::isfinite(base_font_size) && base_font_size > 0.0f) {
      base_font_size_ = base_font_size;
    }
    if (std::isfinite(logical_height) && logical_height > 0.0f) {
      last_logical_height_ = logical_height;
      if (font_scale_mode_ == FontScaleMode::window_height_scale) {
        const float scale =
            reference_scale_ * logical_height / reference_height_;
        if (std::isfinite(scale) && scale > 0.0f) {
          unzoomed_font_scale_ = scale;
        }
      } else if (font_scale_mode_ == FontScaleMode::window_height_size) {
        const float size = reference_size_ * logical_height / reference_height_;
        if (std::isfinite(size) && size > 0.0f) {
          unzoomed_font_size_ =
              std::clamp(size, minimum_size_, maximum_size_);
        }
      }
    }
    if (font_scale_mode_ == FontScaleMode::fixed_scale ||
        font_scale_mode_ == FontScaleMode::window_height_scale) {
      unzoomed_font_size_ = base_font_size_ * unzoomed_font_scale_;
    } else {
      unzoomed_font_scale_ = unzoomed_font_size_ / base_font_size_;
    }
    apply_font_zoom();
    return font_scale_;
  }

 private:
  enum class FontScaleMode {
    fixed_scale,
    window_height_scale,
    fixed_size,
    window_height_size,
  };

  static void validate_positive(float value, const char *name) {
    if (!std::isfinite(value) || value <= 0.0f) {
      throw std::invalid_argument(std::string(name) +
                                  " must be finite and greater than zero");
    }
  }

  void apply_font_zoom() {
    font_scale_ = unzoomed_font_scale_ * font_zoom_;
    font_size_ = unzoomed_font_size_ * font_zoom_;
  }

  FontScaleMode font_scale_mode_{FontScaleMode::fixed_scale};
  float font_scale_{1.0f};
  float font_size_{13.0f};
  float unzoomed_font_scale_{1.0f};
  float unzoomed_font_size_{13.0f};
  float font_zoom_{1.0f};
  float base_font_size_{13.0f};
  float reference_height_{1.0f};
  float reference_scale_{1.0f};
  float reference_size_{16.0f};
  float minimum_size_{12.0f};
  float maximum_size_{24.0f};
  float last_logical_height_{0.0f};
  bool font_shortcuts_enabled_{true};
};

}  // namespace taichi::ui
