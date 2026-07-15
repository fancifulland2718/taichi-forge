#include "renderer.h"

#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;
#ifdef TI_WITH_METAL
using namespace taichi::lang::metal;
#endif

void Renderer::init(Program *prog,
                    TaichiWindow *window,
                    const AppConfig &config) {
  switch (config.ggui_arch) {
    case Arch::vulkan:
      app_context_.init_with_vulkan(prog, window, config);
      break;
    case Arch::metal:
      app_context_.init_with_metal(prog, window, config);
      break;
    default:
      throw std::runtime_error("Incorrect arch for GGUI");
  }

  swap_chain_.init(&app_context_);

#ifdef TI_WITH_METAL
  if (config.ggui_arch == Arch::metal) {
    MetalSurface *mtl_surf =
        dynamic_cast<MetalSurface *>(&(swap_chain_.surface()));

    NSWindowAdapter nswin_adapter;
    nswin_adapter.set_content_view(window, mtl_surf);
  }
#endif
}

template <typename T>
T *Renderer::get_renderable_of_type(VertexAttributes vbo_attrs) {
  std::unique_ptr<T> r = std::make_unique<T>(&app_context_, vbo_attrs);
  T *ret = r.get();
  renderables_.push_back(std::move(r));

  return ret;
}

void Renderer::set_background_color(const glm::vec3 &color) {
  background_color_ = color;
}

void Renderer::set_image(const SetImageInfo &info) {
  retain_field_info(info.img);
  SetImage *s = pending_set_image_;
  if (!s) {
    s = get_set_image_renderable();
    render_queue_.push_back(s);
    pending_set_image_ = s;
  }
  s->update_data(info);
}

void Renderer::set_image(const DisplayFrameInfo &info) {
  SetImage *s = pending_set_image_;
  if (!s) {
    s = get_set_image_renderable();
    render_queue_.push_back(s);
    pending_set_image_ = s;
  }
  s->update_data(info);
}

void Renderer::set_image(Texture *tex) {
  retain_texture(tex);
  SetImage *s = pending_set_image_;
  if (!s) {
    s = get_set_image_renderable();
    render_queue_.push_back(s);
    pending_set_image_ = s;
  }
  s->update_data(tex);
}

void Renderer::triangles(const TrianglesInfo &info) {
  retain_renderable_info(info.renderable_info);
  Triangles *triangles =
      get_renderable_of_type<Triangles>(info.renderable_info.vbo_attrs);
  triangles->update_data(info);
  render_queue_.push_back(triangles);
}

void Renderer::lines(const LinesInfo &info) {
  retain_renderable_info(info.renderable_info);
  Lines *lines = get_renderable_of_type<Lines>(info.renderable_info.vbo_attrs);
  lines->update_data(info);
  render_queue_.push_back(lines);
}

void Renderer::circles(const CirclesInfo &info) {
  retain_renderable_info(info.renderable_info);
  Circles *circles =
      get_renderable_of_type<Circles>(info.renderable_info.vbo_attrs);
  circles->update_data(info);
  render_queue_.push_back(circles);
}

void Renderer::scene_lines(const SceneLinesInfo &info) {
  retain_renderable_info(info.renderable_info);
  SceneLines *scene_lines =
      get_renderable_of_type<SceneLines>(info.renderable_info.vbo_attrs);
  scene_lines->update_data(info);
  render_queue_.push_back(scene_lines);
}

void Renderer::mesh(const MeshInfo &info) {
  retain_renderable_info(info.renderable_info);
  if (info.mesh_attribute_info.has_attribute) {
    retain_field_info(info.mesh_attribute_info.mesh_attribute);
  }
  Mesh *mesh = get_renderable_of_type<Mesh>(info.renderable_info.vbo_attrs);
  mesh->update_data(info);
  render_queue_.push_back(mesh);
}

void Renderer::particles(const ParticlesInfo &info) {
  retain_renderable_info(info.renderable_info);
  Particles *particles =
      get_renderable_of_type<Particles>(info.renderable_info.vbo_attrs);
  particles->update_data(info);
  render_queue_.push_back(particles);
}

void Renderer::resize_lights_ssbo(int new_ssbo_size) {
  if (lights_ssbo_ != nullptr && new_ssbo_size == lights_ssbo_size) {
    return;
  }
  lights_ssbo_.reset();
  lights_ssbo_size = new_ssbo_size;
  if (lights_ssbo_size) {
    auto [buf, res] = app_context_.device().allocate_memory_unique(
        {lights_ssbo_size, /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage});
    TI_ASSERT(res == RhiResult::success);
    lights_ssbo_ = std::move(buf);
  }
}

void Renderer::init_scene_ubo() {
  scene_ubo_.reset();
  auto [buf, res] = app_context_.device().allocate_memory_unique(
      {sizeof(SceneBase::UBOScene), /*host_write=*/true, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Uniform});
  TI_ASSERT(res == RhiResult::success);
  scene_ubo_ = std::move(buf);
}

void Renderer::update_scene_data(SceneBase *scene) {
  // Update SSBO
  {
    size_t new_ssbo_size = scene->point_lights_.size() * sizeof(PointLight);
    resize_lights_ssbo(new_ssbo_size);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_.device().map(lights_ssbo_->get_ptr(), &mapped));
    memcpy(mapped, scene->point_lights_.data(), new_ssbo_size);
    app_context_.device().unmap(*lights_ssbo_);
  }

  // Update UBO
  {
    init_scene_ubo();

    SceneBase::UBOScene ubo;
    ubo.scene = scene->current_scene_data_;
    ubo.window_width = app_context_.config.width;
    ubo.window_height = app_context_.config.height;
    ubo.tan_half_fov = tanf(glm::radians(scene->camera_.fov) / 2);
    ubo.aspect_ratio =
        float(app_context_.config.width) / float(app_context_.config.height);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_.device().map(scene_ubo_->get_ptr(0), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_.device().unmap(*scene_ubo_);
  }
}

void Renderer::scene_v2(SceneBase *scene) {
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.width() / (float)swap_chain_.height();
  scene->update_ubo(aspect_ratio);
  update_scene_data(scene);

  for (auto renderable_ : render_queue_) {
    if (renderable_->is_3d_renderable) {
      renderable_->update_scene_data(lights_ssbo_->get_ptr(0),
                                     scene_ubo_->get_ptr(0));
    }
  }

  scene->point_lights_.clear();
}

void Renderer::scene(SceneBase *scene) {
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.width() / (float)swap_chain_.height();
  scene->update_ubo(aspect_ratio);
  update_scene_data(scene);

  int object_count = scene->mesh_infos_.size() +
                     scene->particles_infos_.size() +
                     scene->scene_lines_infos_.size();
  int mesh_id = 0;
  int particles_id = 0;
  int scene_lines_id = 0;
  for (int i = 0; i < object_count; ++i) {
    if (mesh_id < scene->mesh_infos_.size() &&
        scene->mesh_infos_[mesh_id].object_id == i) {
      mesh(scene->mesh_infos_[mesh_id]);
      ++mesh_id;
    }
    if (particles_id < scene->particles_infos_.size() &&
        scene->particles_infos_[particles_id].object_id == i) {
      particles(scene->particles_infos_[particles_id]);
      ++particles_id;
    }
    // Scene Lines
    if (scene_lines_id < scene->scene_lines_infos_.size() &&
        scene->scene_lines_infos_[scene_lines_id].object_id == i) {
      scene_lines(scene->scene_lines_infos_[scene_lines_id]);
      ++scene_lines_id;
    }
  }
  scene->next_object_id_ = 0;
  scene->mesh_infos_.clear();
  scene->particles_infos_.clear();
  scene->scene_lines_infos_.clear();
  scene->point_lights_.clear();

  for (auto renderable_ : render_queue_) {
    if (renderable_->is_3d_renderable) {
      renderable_->update_scene_data(lights_ssbo_->get_ptr(0),
                                     scene_ubo_->get_ptr(0));
    }
  }
}

Renderer::~Renderer() {
  discard_pending_frame();
  auto &device = app_context_.device();
  if (device.backend_calls_safe()) {
    try {
      wait_for_in_flight_frames();
      device.wait_idle();
    } catch (const taichi::lang::BackendRuntimeError &error) {
      device.report_backend_error(error);
    } catch (const std::exception &error) {
      TI_WARN("GGUI renderer teardown wait failed: {}", error.what());
    } catch (...) {
      TI_WARN("GGUI renderer teardown wait failed");
    }
  }
  for (const auto &set_image : reusable_set_images_) {
    erase_direct_set_image_state(set_image.get());
  }
}

void Renderer::prepare_for_next_frame() {
  retire_completed_frames();
}

bool Renderer::can_accept_frame() {
  retire_completed_frames();
  return in_flight_frames_.size() < max_frames_in_flight();
}

void Renderer::wait_for_frame_slot() {
  while (!can_accept_frame()) {
    wait_oldest_frame();
  }
}

bool Renderer::has_render_work() const {
  return !render_queue_.empty();
}

void Renderer::discard_pending_frame() {
  recycle_renderable_list(renderables_);
  render_queue_.clear();
  pending_set_image_ = nullptr;
  pending_runtime_resource_handles_.clear();
  pending_ndarray_resource_leases_.clear();
  pending_texture_resource_leases_.clear();
}

size_t Renderer::max_frames_in_flight() {
  return std::max<size_t>(
      2, static_cast<size_t>(swap_chain_.surface().get_image_count()));
}

SetImage *Renderer::get_set_image_renderable() {
  std::unique_ptr<SetImage> r;
  if (!reusable_set_images_.empty()) {
    r = std::move(reusable_set_images_.back());
    reusable_set_images_.pop_back();
  }
  if (!r) {
    r = std::make_unique<SetImage>(&app_context_, VboHelpers::all());
  }
  SetImage *ret = r.get();
  renderables_.push_back(std::move(r));
  return ret;
}

bool Renderer::remember_runtime_resource_handle(
    RuntimeResourceHandle handle) {
  if (!handle) {
    return false;
  }
  if (std::find(pending_runtime_resource_handles_.begin(),
                pending_runtime_resource_handles_.end(),
                handle) != pending_runtime_resource_handles_.end()) {
    return false;
  }
  pending_runtime_resource_handles_.push_back(handle);
  return true;
}

void Renderer::retain_field_info(const FieldInfo &field) {
  if (!field.valid || !field.runtime_resource_handle) {
    return;
  }
  Program *owner = field.runtime_resource_program;
  TI_ERROR_IF(owner == nullptr,
              "GGUI received an Ndarray identity without an owning Program");
  TI_ERROR_IF(owner != app_context_.prog(),
              "GGUI cannot consume an Ndarray owned by another Program");
  if (!remember_runtime_resource_handle(field.runtime_resource_handle)) {
    return;
  }
  try {
    SharedNdarrayResourceLease lease;
    const auto cached =
        shared_ndarray_frame_leases_.find(field.runtime_resource_handle.index);
    if (cached != shared_ndarray_frame_leases_.end()) {
      lease = cached->second.lock();
      if (lease && (*lease).handle() != field.runtime_resource_handle) {
        lease.reset();
      }
    }
    if (!lease) {
      lease = std::make_shared<Program::NdarrayResourceLease>(
          owner->acquire_ndarray_external_lease(
              field.runtime_resource_handle));
      shared_ndarray_frame_leases_[field.runtime_resource_handle.index] =
          lease;
    }
    TI_ERROR_IF(lease->get()->get_device_allocation() != field.dev_alloc,
                "GGUI Ndarray identity does not match its DeviceAllocation");
    pending_ndarray_resource_leases_.push_back(std::move(lease));
  } catch (...) {
    pending_runtime_resource_handles_.pop_back();
    throw;
  }
}

void Renderer::retain_renderable_info(const RenderableInfo &info) {
  retain_field_info(info.vbo);
  if (info.indices.valid) {
    retain_field_info(info.indices);
  }
}

void Renderer::retain_texture(Texture *texture) {
  if (texture == nullptr || texture->owning_program() == nullptr) {
    return;
  }
  Program *owner = texture->owning_program();
  TI_ERROR_IF(owner != app_context_.prog(),
              "GGUI cannot consume a Texture owned by another Program");
  RuntimeResourceHandle handle = texture->runtime_resource_handle();
  if (!remember_runtime_resource_handle(handle)) {
    return;
  }
  try {
    SharedTextureResourceLease lease;
    const auto cached = shared_texture_frame_leases_.find(handle.index);
    if (cached != shared_texture_frame_leases_.end()) {
      lease = cached->second.lock();
      if (lease && (*lease).handle() != handle) {
        lease.reset();
      }
    }
    if (!lease) {
      lease = std::make_shared<Program::TextureResourceLease>(
          owner->acquire_texture_external_lease(texture));
      shared_texture_frame_leases_[handle.index] = lease;
    }
    TI_ERROR_IF(lease->get() != texture,
                "GGUI Texture identity does not match its runtime resource");
    pending_texture_resource_leases_.push_back(std::move(lease));
  } catch (...) {
    pending_runtime_resource_handles_.pop_back();
    throw;
  }
}

void Renderer::recycle_renderable_list(
    std::vector<std::unique_ptr<Renderable>> &list) {
  for (auto &renderable : list) {
    if (auto *set_image = dynamic_cast<SetImage *>(renderable.get())) {
      renderable.release();
      reusable_set_images_.push_back(std::unique_ptr<SetImage>(set_image));
    }
  }
  list.clear();
}

void Renderer::recycle_renderables(InFlightFrame &frame) {
  recycle_renderable_list(frame.renderables);
}

void Renderer::retire_completed_frames() {
  while (!in_flight_frames_.empty()) {
    auto &frame = in_flight_frames_.front();
    if (frame.complete && !frame.complete->is_ready()) {
      return;
    }
    recycle_renderables(frame);
    in_flight_frames_.pop_front();
  }
}

void Renderer::wait_oldest_frame() {
  if (in_flight_frames_.empty()) {
    return;
  }
  auto &frame = in_flight_frames_.front();
  if (!frame.complete || !frame.complete->wait()) {
    app_context_.device().wait_idle();
    while (!in_flight_frames_.empty()) {
      recycle_renderables(in_flight_frames_.front());
      in_flight_frames_.pop_front();
    }
    return;
  }
  recycle_renderables(frame);
  in_flight_frames_.pop_front();
}

void Renderer::wait_for_in_flight_frames() {
  while (!in_flight_frames_.empty()) {
    wait_oldest_frame();
  }
}

bool Renderer::draw_frame(GuiBase *gui_base, bool blocking_acquire) {
  SurfaceImage surface_image;
  if (blocking_acquire) {
    surface_image = swap_chain_.surface().acquire_surface_image();
  } else if (!swap_chain_.surface().try_acquire_surface_image(&surface_image)) {
    return false;
  }
  if (surface_image.image.device == nullptr) {
    return false;
  }
  StreamSemaphore semaphore = surface_image.image_available;
  auto image = surface_image.image;
  std::vector<StreamSemaphore> present_waits_after_acquire;
  if (app_context_.config.ggui_arch == Arch::vulkan) {
    auto *surface = dynamic_cast<VulkanSurface *>(&swap_chain_.surface());
    if (surface) {
      present_waits_after_acquire =
          surface->take_present_waits_after_acquire(surface_image.image_index);
    }
  }

  auto stream = app_context_.device().get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  assert(res == RhiResult::success && "Failed to allocate command list");

  bool color_clear = true;
  std::vector<float> clear_colors = {background_color_[0], background_color_[1],
                                     background_color_[2], 1};
  cmd_list->image_transition(image, ImageLayout::undefined,
                             ImageLayout::color_attachment);
  auto depth_image = swap_chain_.depth_allocation();

  for (auto renderable : render_queue_) {
    renderable->record_prepass_this_frame_commands(cmd_list.get());
  }

  cmd_list->begin_renderpass(
      /*x0=*/0, /*y0=*/0, /*x1=*/swap_chain_.width(),
      /*y1=*/swap_chain_.height(), /*num_color_attachments=*/1, &image,
      &color_clear, &clear_colors, &depth_image,
      /*depth_clear=*/true);

  for (auto renderable : render_queue_) {
    renderable->record_this_frame_commands(cmd_list.get());
  }

  if (app_context_.config.ggui_arch == Arch::vulkan) {
    Gui *gui = static_cast<Gui *>(gui_base);
    if (gui != nullptr && gui->has_widgets()) {
      VkRenderPass pass = static_cast<VulkanCommandList *>(cmd_list.get())
                              ->current_renderpass()
                              ->renderpass;

      if (gui->render_pass() == VK_NULL_HANDLE) {
        gui->init_render_resources(pass);
      } else if (gui->render_pass() != pass) {
        gui->cleanup_render_resources();
        gui->init_render_resources(pass);
      }
      gui->draw(cmd_list.get());
    } else if (gui != nullptr) {
      gui->end_frame();
    }
  }
#ifdef TI_WITH_METAL
  else if (app_context_.config.ggui_arch == Arch::metal) {
    GuiMetal *gui = static_cast<GuiMetal *>(gui_base);
    if (gui != nullptr && gui->has_widgets()) {

      auto mtl_cmd_list = static_cast<MetalCommandList *>(cmd_list.get());

      MTLRenderPassDescriptor *pass = mtl_cmd_list->create_render_pass_desc(
          false, mtl_cmd_list->is_renderpass_active());
      mtl_cmd_list->set_renderpass_active();

      gui->init_render_resources(pass);
      gui->draw(cmd_list.get());
    } else if (gui != nullptr) {
      gui->end_frame();
    }
  }
#endif
  else {
    TI_NOT_IMPLEMENTED;
  }

  cmd_list->end_renderpass();

  std::vector<StreamSemaphore> wait_semaphores;

  if (app_context_.prog()) {
    auto sema = app_context_.prog()->flush();
    if (sema) {
      wait_semaphores.push_back(sema);
    }
  }

  if (semaphore) {
    wait_semaphores.push_back(semaphore);
  }

  render_complete_semaphore_ = stream->submit(cmd_list.get(), wait_semaphores);
  render_surface_image_ = surface_image;

  render_queue_.clear();
  pending_set_image_ = nullptr;
  InFlightFrame frame;
  frame.complete = render_complete_semaphore_;
  frame.present_waits_after_acquire = std::move(present_waits_after_acquire);
  frame.renderables = std::move(renderables_);
  frame.ndarray_resource_leases =
      std::move(pending_ndarray_resource_leases_);
  frame.texture_resource_leases =
      std::move(pending_texture_resource_leases_);
  pending_runtime_resource_handles_.clear();
  in_flight_frames_.push_back(std::move(frame));
  return true;
}

const AppContext &Renderer::app_context() const {
  return app_context_;
}

AppContext &Renderer::app_context() {
  return app_context_;
}

const SwapChain &Renderer::swap_chain() const {
  return swap_chain_;
}

SwapChain &Renderer::swap_chain() {
  return swap_chain_;
}

taichi::lang::StreamSemaphore Renderer::get_render_complete_semaphore() {
  return render_complete_semaphore_;
}

const taichi::lang::SurfaceImage &Renderer::get_render_surface_image() const {
  return render_surface_image_;
}

}  // namespace vulkan

}  // namespace taichi::ui
