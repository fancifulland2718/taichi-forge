#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

class GuiBase {
 public:
  void set_font_scale(float scale) {
    validate_font_scale(scale, "scale");
    font_scale_mode_ = FontScaleMode::fixed;
    font_scale_ = scale;
  }

  void set_font_scale_from_window_height(float reference_height,
                                         float reference_scale) {
    validate_font_scale(reference_height, "reference_height");
    validate_font_scale(reference_scale, "reference_scale");
    font_scale_mode_ = FontScaleMode::window_height;
    reference_height_ = reference_height;
    reference_scale_ = reference_scale;
    update_font_scale(last_logical_height_);
  }

  float get_font_scale() const {
    return font_scale_;
  }

  virtual void begin(const std::string &name,
                     float x,
                     float y,
                     float width,
                     float height) = 0;
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
  virtual void end_frame() = 0;
  virtual ~GuiBase() = default;

 protected:
  float update_font_scale(float logical_height) {
    if (std::isfinite(logical_height) && logical_height > 0.0f) {
      last_logical_height_ = logical_height;
      if (font_scale_mode_ == FontScaleMode::window_height) {
        const float scale =
            reference_scale_ * logical_height / reference_height_;
        if (std::isfinite(scale) && scale > 0.0f) {
          font_scale_ = scale;
        }
      }
    }
    return font_scale_;
  }

 private:
  enum class FontScaleMode {
    fixed,
    window_height,
  };

  static void validate_font_scale(float value, const char *name) {
    if (!std::isfinite(value) || value <= 0.0f) {
      throw std::invalid_argument(std::string(name) +
                                  " must be finite and greater than zero");
    }
  }

  FontScaleMode font_scale_mode_{FontScaleMode::fixed};
  float font_scale_{1.0f};
  float reference_height_{1.0f};
  float reference_scale_{1.0f};
  float last_logical_height_{0.0f};
};

}  // namespace taichi::ui
