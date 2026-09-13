#include "canvas.h"
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/sceneV2.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;

Canvas::Canvas(Renderer *renderer) : renderer_(renderer) {
}

void Canvas::set_background_color(const glm::vec3 &color) {
  renderer_->set_background_color(color);
}

void Canvas::set_image(const SetImageInfo &info) {
  renderer_->set_image(info);
}

void Canvas::set_image(const DisplayFrameInfo &info) {
  renderer_->set_image(info);
}

void Canvas::set_image(Texture *tex) {
  renderer_->set_image(tex);
}

void Canvas::set_image_transpose(bool transpose) {
  renderer_->set_image_transpose(transpose);
}

bool Canvas::set_image_shared_cuda(const SetImageInfo &info) {
  return renderer_->set_image_shared_cuda(info);
}

std::shared_ptr<DisplayCompletion> Canvas::track_display_frame(bool writable) {
  return renderer_->track_display_frame(writable);
}

void Canvas::finish_display_write() {
  renderer_->finish_display_write();
}
RuntimeCompletion Canvas::record_display_source_completion() {
  return renderer_->record_display_source_completion();
}
void Canvas::cancel_display_frame() {
  renderer_->cancel_display_frame();
}
bool Canvas::repeat_display_frame() {
  return renderer_->repeat_display_frame();
}

std::shared_ptr<SharedCudaVulkanImage>
Canvas::acquire_shared_cuda_vulkan_image(int width, int height) {
  return renderer_->acquire_shared_cuda_vulkan_image(width, height);
}

void Canvas::triangles(const TrianglesInfo &info) {
  renderer_->triangles(info);
}

void Canvas::lines(const LinesInfo &info) {
  renderer_->lines(info);
}

void Canvas::circles(const CirclesInfo &info) {
  renderer_->circles(info);
}

void Canvas::scene(SceneBase *scene_base) {
  if (SceneV2 *scene = dynamic_cast<SceneV2 *>(scene_base)) {
    renderer_->scene_v2(scene);
  } else {
    renderer_->scene(scene_base);
  }
}

}  // namespace vulkan

}  // namespace taichi::ui
