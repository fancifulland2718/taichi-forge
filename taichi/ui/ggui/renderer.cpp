#include "renderer.h"

#include "taichi/ui/utils/utils.h"
#include <limits>
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_execution_domain.h"
#include "taichi/ui/ggui/renderables/vulkan_display_buffer.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/execution_domain.h"
#endif

using taichi::lang::Program;

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;
#ifdef TI_WITH_METAL
using namespace taichi::lang::metal;
#endif

namespace {

void bind_cuda_display_domain(
    const std::shared_ptr<DisplayStorage> &storage,
    const std::shared_ptr<cuda::ExecutionDomain> &domain) {
  auto cuda_image = std::dynamic_pointer_cast<SharedCudaVulkanImage>(storage);
  TI_ERROR_IF(!cuda_image,
              "CUDA display domain requires CUDA-Vulkan display storage");
  cuda_image->set_execution_domain(domain);
}

}  // namespace

void Renderer::init(Program *prog,
                    TaichiWindow *window,
                    const AppConfig &config,
                    WindowLayoutState *window_layout) {
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
  app_context_.set_window_layout(window_layout);

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

void Renderer::require_general_display_path() {
  TI_ERROR_IF(display_vulkan_execution_domain_,
              "Selected Vulkan display requires prepared image-only frames; "
              "mixed geometry and runtime targets are unsupported");
  general_display_path_used_ = true;
}

void Renderer::set_background_color(const glm::vec3 &color) {
  background_color_ = color;
}

void Renderer::set_image(const SetImageInfo &info) {
  require_general_display_path();
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
  require_general_display_path();
  SetImage *s = pending_set_image_;
  if (!s) {
    s = get_set_image_renderable();
    render_queue_.push_back(s);
    pending_set_image_ = s;
  }
  s->update_data(info);
}

std::shared_ptr<DisplayStorage>
Renderer::acquire_shared_cuda_vulkan_image(int width, int height) {
#ifdef TI_WITH_VULKAN
  if (display_vulkan_execution_domain_) {
    TI_ERROR_IF(pending_set_image_ || !render_queue_.empty(),
                "Submit or cancel the pending Vulkan display frame first");
    TI_ERROR_IF(!prepared_display_width_ || width != prepared_display_width_ ||
                    height != prepared_display_height_,
                "Prepare the Vulkan writable pool at a cold boundary");
    auto *image = get_set_image_renderable();
    if (!image)
      return nullptr;
    pending_set_image_ = image;
    render_queue_.push_back(image);
    try {
      auto storage = std::dynamic_pointer_cast<VulkanWritableDisplayBuffer>(
          image->shared_image());
      TI_ERROR_IF(!storage, "Vulkan acquire_frame requires prepare_frame_pool");
      image->use_shared_image(storage);
      storage->prepare_write();
      return storage;
    } catch (const taichi::lang::vulkan::ExecutionDomain::Busy &) {
      cancel_display_frame();
      return nullptr;
    } catch (...) {
      cancel_display_frame();
      throw;
    }
  }
#endif
  require_general_display_path();
  TI_ERROR_IF(prepared_display_width_ && (width != prepared_display_width_ ||
                                          height != prepared_display_height_),
              "Reprepare the display pool at a cold resize boundary");
  SetImage *set_image = pending_set_image_;
  if (!set_image) {
    set_image = get_set_image_renderable();
    if (!set_image)
      return nullptr;
    render_queue_.push_back(set_image);
    pending_set_image_ = set_image;
  }
  auto previous = set_image->shared_image();
  TI_ERROR_IF(previous && previous->has_live_execution_domain() &&
                  (previous->width() != width || previous->height() != height),
              "Close the previous display domain before resizing its storage");
  previous.reset();
  auto image = set_image->acquire_shared_cuda_vulkan_image(width, height);
  if (image && display_execution_domain_)
    bind_cuda_display_domain(image, display_execution_domain_);
  return image;
}

void Renderer::set_display_execution_domain(
    std::shared_ptr<cuda::ExecutionDomain> domain) {
#ifdef TI_WITH_CUDA
  require_general_display_path();
  TI_ERROR_IF(pending_set_image_,
              "Submit or cancel the display frame before binding a domain");
  TI_ERROR_IF(
      !domain || !app_context_.prog() ||
          !domain->belongs_to(app_context_.prog()->runtime_fault_domain()) ||
          !domain->snapshot().accepting,
      "Canvas domain requires the current live CUDA runtime");
  TI_ERROR_IF(display_execution_domain_ &&
                  display_execution_domain_ != domain &&
                  !display_execution_domain_->snapshot().closed,
              "Close the previous display domain before replacing it");
  display_execution_domain_ = std::move(domain);
#else
  TI_ERROR("CUDA display domains are unavailable");
#endif
}

std::vector<DisplayStorage *>
Renderer::prepare_display_pool(int width, int height, std::size_t capacity) {
#ifdef TI_WITH_VULKAN
  if (display_vulkan_execution_domain_) {
    validate_display_pool_replacement();
    auto buffers = VulkanWritableDisplayBuffer::create_pool(
        app_context_, display_vulkan_execution_domain_, width, height, capacity,
        max_frames_in_flight() + 1);
    std::vector<std::unique_ptr<SetImage>> prepared;
    std::vector<DisplayStorage *> views;
    prepared.reserve(buffers.size());
    views.reserve(buffers.size());
    for (const auto &buffer : buffers) {
      auto image = std::make_unique<SetImage>(&app_context_, VboHelpers::all());
      image->use_shared_image(buffer);
      views.push_back(buffer.get());
      prepared.push_back(std::move(image));
    }
    render_queue_.reserve(1);
    renderables_.reserve(1);
    install_display_pool(std::move(prepared), width, height);
    prepared_vulkan_display_.reset();
    return views;
  }
#endif
  require_general_display_path();
  TI_ERROR_IF(!display_execution_domain_ || width <= 0 || height <= 0 ||
                  capacity < 2 || capacity > max_frames_in_flight() + 1,
              "Display pool requires a selected domain and bounded positive "
              "dimensions/capacity");
  validate_display_pool_replacement();
  std::vector<std::unique_ptr<SetImage>> prepared;
  std::vector<DisplayStorage *> views;
  const auto repeat_capacity = max_frames_in_flight() + 1;
  prepared.reserve(capacity + repeat_capacity);
  views.reserve(capacity);
  for (std::size_t i = 0; i < capacity; ++i) {
    auto image = std::make_unique<SetImage>(&app_context_, VboHelpers::all());
    auto storage = image->acquire_shared_cuda_vulkan_image(width, height);
    TI_ERROR_IF(!storage, "Unable to prepare CUDA/Vulkan display pool");
    bind_cuda_display_domain(storage, display_execution_domain_);
    views.push_back(storage.get());
    prepared.push_back(std::move(image));
  }
  install_display_pool(std::move(prepared), width, height);
  return views;
}

void Renderer::validate_display_pool_replacement() const {
  TI_ERROR_IF(pending_set_image_ || !renderables_.empty(),
              "Submit or cancel pending work before preparing display storage");
  const auto require_retired = [](const auto &items) {
    for (const auto &item : items) {
      auto *image = dynamic_cast<SetImage *>(item.get());
      TI_ERROR_IF(image && image->shared_image() &&
                      image->shared_image()->has_live_execution_domain(),
                  "Close the previous display domain before preparing its "
                  "replacement pool");
    }
  };
  require_retired(reusable_set_images_);
  for (const auto &frame : in_flight_frames_)
    require_retired(frame.renderables);
  TI_ERROR_IF(last_shared_display_image_ &&
                  last_shared_display_image_->has_live_execution_domain(),
              "Close the previous display domain before replacing its pool");
}

void Renderer::install_display_pool(
    std::vector<std::unique_ptr<SetImage>> prepared, int width, int height) {
  const auto repeat_capacity = max_frames_in_flight() + 1;
  TI_ERROR_IF(prepared.empty() || prepared.size() >
                  std::numeric_limits<std::size_t>::max() - repeat_capacity,
              "Invalid prepared display pool size");
  prepared.reserve(prepared.size() + repeat_capacity);
  // Repeating an image whose earlier Vulkan read is still in flight needs a
  // distinct renderable, but no new pixel storage. Prepare its tiny UBO and
  // pipeline state here so that overlap cannot allocate them in repeat().
  for (size_t i = 0; i < repeat_capacity; ++i) {
    auto repeat = std::make_unique<SetImage>(&app_context_, VboHelpers::all());
    repeat->use_shared_image(prepared.front()->shared_image());
    repeat->use_shared_image(nullptr);
    prepared.push_back(std::move(repeat));
  }
  // Reconfiguration is explicitly cold: retire Vulkan readers and the old
  // external owners before freeing their storage. No simulation-tail wait is
  // introduced into acquire/submit/repeat.
  wait_for_in_flight_frames();
  for (const auto &image : reusable_set_images_) {
    if (image->shared_image())
      image->shared_image()->close();
    erase_direct_set_image_state(image.get());
  }
  if (last_shared_display_image_)
    last_shared_display_image_->close();
  last_shared_display_image_.reset();
  reusable_set_images_ = std::move(prepared);
  prepared_display_width_ = width;
  prepared_display_height_ = height;
}

void Renderer::set_display_vulkan_execution_domain(
    std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain) {
#ifdef TI_WITH_VULKAN
  TI_ERROR_IF(pending_set_image_ || !render_queue_.empty() || has_offscreen_targets(),
              "Bind a Vulkan display domain before recording image-only work");
  auto *program = app_context_.prog();
  TI_ERROR_IF(!program || app_context_.config.ti_arch != Arch::vulkan ||
                  app_context_.config.ggui_arch != Arch::vulkan ||
                  program->get_graphics_device() != &app_context_.device() ||
                  !domain || !domain->belongs_to(program->runtime_fault_domain()) ||
                  !domain->snapshot().accepting || display_execution_domain_,
              "Canvas domain requires the current live Vulkan Program");
  TI_ERROR_IF(display_vulkan_execution_domain_ &&
                  display_vulkan_execution_domain_ != domain &&
                  !display_vulkan_execution_domain_->snapshot().closed,
              "Close the previous display domain before replacing it");
  display_vulkan_execution_domain_ = std::move(domain);
#else
  TI_ERROR("Vulkan execution domains are unavailable");
#endif
}

void Renderer::prepare_vulkan_display_sources(const std::vector<FieldInfo> &sources,
                                               std::size_t capacity) {
#ifdef TI_WITH_VULKAN
  TI_ERROR_IF(!display_vulkan_execution_domain_, "Select a Vulkan display domain first");
  validate_display_pool_replacement();
  auto buffers = VulkanDisplayBuffer::create_pool(
      app_context_, display_vulkan_execution_domain_, sources, capacity,
      max_frames_in_flight() + 1);
  std::vector<std::unique_ptr<SetImage>> prepared;
  prepared.reserve(buffers.size());
  for (const auto &buffer : buffers) {
    auto image = std::make_unique<SetImage>(&app_context_, VboHelpers::all());
    image->use_shared_image(buffer);
    prepared.push_back(std::move(image));
  }
  // All allocation/recording completes before the old pool is replaced.
  render_queue_.reserve(1);
  renderables_.reserve(1);
  install_display_pool(std::move(prepared), buffers.front()->width(),
                        buffers.front()->height());
  prepared_vulkan_display_ = std::dynamic_pointer_cast<VulkanDisplayBuffer>(buffers.front());
#else
  TI_ERROR("Vulkan execution domains are unavailable");
#endif
}

bool Renderer::set_image_vulkan_on_domain(const SetImageInfo &info,
                                          const RuntimeCompletion &producer) {
#ifdef TI_WITH_VULKAN
  TI_ERROR_IF(!display_vulkan_execution_domain_, "Select a Vulkan display domain first");
  TI_ERROR_IF(pending_set_image_ || !render_queue_.empty(),
              "Show or cancel the pending Vulkan display frame first");
  auto prepared = prepared_vulkan_display_.lock();
  TI_ERROR_IF(!prepared, "Call prepare_frame_sources before submitting Vulkan display");
  auto source = prepared->source_lease(info.img);
  prepared->validate_producer(producer);
  // This probe must not pin a facade while the pool checks shared_ptr counts.
  prepared.reset();
  auto *image = get_set_image_renderable();
  if (!image)
    return false;
  pending_set_image_ = image;
  render_queue_.push_back(image);
  try {
    auto storage = image->shared_image();
    TI_ERROR_IF(!storage, "Vulkan display pool lost its storage binding");
    image->use_shared_image(storage);  // Prepared UBO/pipeline, before enqueue.
    if (!storage->try_copy_image(info.img.dev_alloc.get_ptr(), producer, source)) {
      cancel_display_frame();
      return false;
    }
    return true;
  } catch (...) {
    cancel_display_frame();
    throw;
  }
#else
  TI_ERROR("Vulkan execution domains are unavailable");
#endif
}

std::shared_ptr<DisplayCompletion> Renderer::track_display_frame(
    bool writable) {
  TI_ERROR_IF(!pending_set_image_, "No pending display image");
#ifdef TI_WITH_VULKAN
  TI_ERROR_IF(writable && display_vulkan_execution_domain_ &&
                  !dynamic_cast<VulkanWritableDisplayBuffer *>(pending_set_image_->shared_image().get()),
              "Vulkan display write requires its prepared writable pool");
#endif
  auto completion = std::make_shared<DisplayCompletion>();
  pending_set_image_->track_display(completion, writable);
  return completion;
}

bool Renderer::set_image_shared_cuda(const SetImageInfo &info) {
  return set_image_shared_cuda_on_domain(info, {});
}

bool Renderer::set_image_shared_cuda_on_domain(
    const SetImageInfo &info,
    const RuntimeCompletion &producer) {
  require_general_display_path();
  TI_ERROR_IF(
      info.img.shape.size() != 2 || info.img.dtype != PrimitiveType::u32,
      "Shared packed display input must be a 2D u32 image");
  retain_field_info(info.img);
  auto image =
      acquire_shared_cuda_vulkan_image(info.img.shape[0], info.img.shape[1]);
  if (!image) {
    cancel_display_frame();
    return false;
  }
  SharedNdarrayResourceLease source_lease;
  if (display_execution_domain_) {
    const auto found = shared_ndarray_frame_leases_.find(
        info.img.runtime_resource_handle.index);
    if (found != shared_ndarray_frame_leases_.end())
      source_lease = found->second.lock();
    TI_ERROR_IF(!source_lease ||
                    source_lease->handle() != info.img.runtime_resource_handle,
                "Selected display copy requires a retained source ndarray");
  }
  if (!image->try_copy_image(info.img.dev_alloc.get_ptr(), producer,
                             source_lease)) {
    cancel_display_frame();
    return false;
  }
  return true;
}

void Renderer::finish_display_write() {
  TI_ERROR_IF(!pending_set_image_ || !pending_set_image_->shared_image(),
              "No borrowed display image");
  pending_set_image_->shared_image()->finish_write();
  pending_set_image_->seal_display_write();
}

bool Renderer::try_finish_display_write(const RuntimeCompletion &producer,
                                        bool track_source) {
  TI_ERROR_IF((!display_execution_domain_ && !display_vulkan_execution_domain_) || !pending_set_image_ ||
                  !pending_set_image_->shared_image(),
              "No borrowed image on a selected display domain");
  auto image = pending_set_image_->shared_image();
  // Missing observation must reject before sealing an already written frame.
  if (track_source && !producer.valid() && image->ready_for_vulkan_submit())
    (void)image->source_completion();
  if (!image->try_finish_write(producer))
    return false;
  pending_set_image_->seal_display_write();
  return true;
}

RuntimeCompletion Renderer::record_display_source_completion() {
  if (display_execution_domain_ || display_vulkan_execution_domain_) {
    TI_ERROR_IF(!pending_set_image_ || !pending_set_image_->shared_image(),
                "No pending shared display image");
    return pending_set_image_->shared_image()->source_completion();
  }
  // Explicit observation of the producer stream, including work already
  // covered by an earlier Graph ticket. Do not use the no-pending fast token.
  auto *program = app_context_.prog();
  TI_ERROR_IF(!program, "Display source completion requires an active Program");
  program->mark_runtime_submission_pending();
  return program->record_runtime_completion();
}

void Renderer::cancel_display_frame() {
  if (!pending_set_image_)
    return;
  // Cancellation does not consume a ready signal. A completed CUDA producer
  // is rearmed by prepare_cuda_write on reuse; an unwritten image remains
  // ready.
  pending_set_image_->discard_display_write();
  render_queue_.erase(std::remove(render_queue_.begin(), render_queue_.end(),
                                  pending_set_image_),
                      render_queue_.end());
  const auto owner = std::find_if(
      renderables_.begin(), renderables_.end(),
      [this](const auto &item) { return item.get() == pending_set_image_; });
  if (owner != renderables_.end()) {
    reusable_set_images_.reserve(reusable_set_images_.size() + 1);
    reusable_set_images_.emplace_back(
        static_cast<SetImage *>(owner->release()));
    renderables_.erase(owner);
  }
  pending_set_image_ = nullptr;
}

bool Renderer::repeat_display_frame() {
  TI_ERROR_IF(pending_set_image_,
              "Submit or cancel the pending display frame first");
  if (!last_shared_display_image_)
    return false;
  if (display_execution_domain_)
    bind_cuda_display_domain(last_shared_display_image_, display_execution_domain_);
  if (!last_shared_display_image_->try_finish_write())
    return false;
  auto *image = get_set_image_renderable(/*reuse_last_image=*/true);
  image->use_shared_image(last_shared_display_image_);
  render_queue_.push_back(image);
  pending_set_image_ = image;
  return true;
}

void Renderer::set_image(Texture *tex) {
  require_general_display_path();
  retain_texture(tex);
  SetImage *s = pending_set_image_;
  if (!s) {
    s = get_set_image_renderable();
    render_queue_.push_back(s);
    pending_set_image_ = s;
  }
  s->update_data(tex);
}

void Renderer::set_image_transpose(bool transpose) {
  TI_ERROR_IF(!pending_set_image_, "No pending display image");
  pending_set_image_->set_transpose(transpose);
}

void Renderer::triangles(const TrianglesInfo &info) {
  require_general_display_path();
  retain_renderable_info(info.renderable_info);
  Triangles *triangles =
      get_renderable_of_type<Triangles>(info.renderable_info.vbo_attrs);
  triangles->update_data(info);
  render_queue_.push_back(triangles);
}

void Renderer::lines(const LinesInfo &info) {
  require_general_display_path();
  retain_renderable_info(info.renderable_info);
  Lines *lines = get_renderable_of_type<Lines>(info.renderable_info.vbo_attrs);
  lines->update_data(info);
  render_queue_.push_back(lines);
}

void Renderer::circles(const CirclesInfo &info) {
  require_general_display_path();
  retain_renderable_info(info.renderable_info);
  Circles *circles =
      get_renderable_of_type<Circles>(info.renderable_info.vbo_attrs);
  circles->update_data(info);
  render_queue_.push_back(circles);
}

void Renderer::scene_lines(const SceneLinesInfo &info) {
  require_general_display_path();
  retain_renderable_info(info.renderable_info);
  SceneLines *scene_lines =
      get_renderable_of_type<SceneLines>(info.renderable_info.vbo_attrs);
  scene_lines->update_data(info);
  render_queue_.push_back(scene_lines);
}

void Renderer::mesh(const MeshInfo &info) {
  require_general_display_path();
  retain_renderable_info(info.renderable_info);
  if (info.mesh_attribute_info.has_attribute) {
    retain_field_info(info.mesh_attribute_info.mesh_attribute);
  }
  Mesh *mesh = get_renderable_of_type<Mesh>(info.renderable_info.vbo_attrs);
  mesh->update_data(info);
  render_queue_.push_back(mesh);
}

void Renderer::particles(const ParticlesInfo &info) {
  require_general_display_path();
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

    auto ubo = make_raster_scene_uniforms(
        scene->current_scene_data_, app_context_.render_width(),
        app_context_.render_height(), scene->camera_.fov);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_.device().map(scene_ubo_->get_ptr(0), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_.device().unmap(*scene_ubo_);
  }
}

void Renderer::scene_v2(SceneBase *scene) {
  require_general_display_path();
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio =
      app_context_.render_width() / (float)app_context_.render_height();
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
  require_general_display_path();
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio =
      app_context_.render_width() / (float)app_context_.render_height();
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
  if (!selected_vulkan_graphics_retired_ && device.backend_calls_safe()) {
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
  last_shared_display_image_.reset();
  for (const auto &set_image : reusable_set_images_) {
    erase_direct_set_image_state(set_image.get());
  }
}

void Renderer::retire_shared_display_resources() {
#ifdef TI_WITH_CUDA
  if (display_execution_domain_)
    display_execution_domain_->close();
#endif
  const auto close = [](const auto &renderables) {
    for (const auto &item : renderables) {
      auto *image = dynamic_cast<SetImage *>(item.get());
      if (image && image->shared_image())
        image->shared_image()->close();
    }
  };
  close(renderables_);
  close(reusable_set_images_);
  for (const auto &frame : in_flight_frames_)
    close(frame.renderables);
  if (last_shared_display_image_)
    last_shared_display_image_->close();
}

bool Renderer::retire_selected_vulkan_graphics(Gui *gui) {
  validate_frame_retirement();
  if (gui)
    gui->validate_retirement();
  if (!display_vulkan_execution_domain_ || general_display_path_used_ ||
      has_offscreen_targets()) {
    return false;
  }
  if (selected_vulkan_graphics_retired_) {
    return true;
  }
  TI_ERROR_IF(!app_context_.device().backend_calls_safe(),
              "Cannot prove selected Vulkan graphics retirement after a fatal fault");
  // Pool close retires only its instance's exact copy prefix, then all slot
  // readers. It also rejects a graphics submit that returned no sealed proof.
  retire_shared_display_resources();
  while (!in_flight_frames_.empty()) {
    auto &frame = in_flight_frames_.front();
    // Do not route missing proof through wait_oldest_frame's stream-wide
    // fallback. The Window caller keeps this closure if retirement fails.
    TI_ERROR_IF(!frame.complete || !frame.complete->wait(),
                "Selected Vulkan frame has no graphics retirement proof");
    recycle_renderables(frame);
    in_flight_frames_.pop_front();
  }
  if (gui)
    gui->retire_draw_readers();
  selected_vulkan_graphics_retired_ = true;
  return true;
}


void Renderer::validate_frame_retirement() {
  for (const auto &frame : in_flight_frames_)
    TI_ERROR_IF(frame.phase != FramePhase::submitted ||
                    (display_vulkan_execution_domain_ && !frame.complete),
                "GGUI frame lacks a sealed graphics retirement proof");
}

std::pair<VkRenderPass, uint32_t> Renderer::gui_renderpass(bool prepare) {
  auto &device = static_cast<VulkanDevice &>(app_context_.device());
  VulkanRenderPassDesc desc;
  uint32_t count = 2;
  if (has_offscreen_targets()) {
    desc.color_attachments.emplace_back(
        std::get<2>(device.get_vk_image(
            offscreen_color_->get()->get_device_allocation())), true);
    desc.depth_attachment = std::get<2>(device.get_vk_image(
        offscreen_depth_->get()->get_device_allocation()));
    desc.color_final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  } else {
    const auto [result, format] =
        buffer_format_ti_to_vk(swap_chain_.surface().image_format());
    TI_ERROR_IF(result != RhiResult::success, "ImGui surface format is unsupported");
    desc.color_attachments.emplace_back(format, true);
    desc.depth_attachment =
        std::get<2>(device.get_vk_image(swap_chain_.depth_allocation()));
    count = swap_chain_.surface().get_image_count();
  }
  desc.clear_depth = true;
  if (prepare) {
    const auto pass = device.get_renderpass(desc)->renderpass;
    gui_renderpass_desc_ = std::move(desc);
    gui_renderpass_ = pass;
  } else {
    TI_ERROR_IF(!gui_renderpass_desc_ || !(*gui_renderpass_desc_ == desc),
                "ImGui renderpass changed; cold preparation is required");
  }
  return {gui_renderpass_, count};
}

void Renderer::prepare_gui(Gui &gui) {
  validate_frame_retirement();
  Program::RuntimeResourceSubmissionGuard submission_guard;
  if (auto *program = app_context_.prog();
      program && program->get_graphics_device() == &app_context_.device())
    submission_guard = program->acquire_runtime_resource_submission_guard();
  const auto [pass, count] = gui_renderpass(/*prepare=*/true);
  gui.prepare_render_resources(pass, count);
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

bool Renderer::last_frame_used_shared_cuda_vulkan() const noexcept {
  return last_frame_used_shared_cuda_vulkan_;
}

void Renderer::discard_pending_frame() {
  validate_frame_retirement();
  for (const auto &item : renderables_) {
    if (auto *image = dynamic_cast<SetImage *>(item.get()))
      image->discard_display_write();
  }
  recycle_renderable_list(renderables_);
  render_queue_.clear();
  pending_set_image_ = nullptr;
  pending_runtime_resource_handles_.clear();
  pending_ndarray_resource_leases_.clear();
  pending_texture_resource_leases_.clear();
}

size_t Renderer::max_frames_in_flight() {
  if (has_offscreen_targets()) {
    return 2;
  }
  return std::max<size_t>(
      2, static_cast<size_t>(swap_chain_.surface().get_image_count()));
}

SetImage *Renderer::get_set_image_renderable() {
  return get_set_image_renderable(/*reuse_last_image=*/false);
}

SetImage *Renderer::get_set_image_renderable(bool reuse_last_image) {
  retire_completed_frames();
  std::unique_ptr<SetImage> r;
  auto reusable = reusable_set_images_.end();
  if (prepared_display_width_ && !reuse_last_image) {
    reusable = std::find_if(
        reusable_set_images_.begin(), reusable_set_images_.end(),
        [this](const auto &item) {
          const auto image = item->shared_image();
          if (!image || image == last_shared_display_image_)
            return false;
#ifdef TI_WITH_VULKAN
          if (display_vulkan_execution_domain_) {
            auto *buffer = dynamic_cast<VulkanDisplayBuffer *>(image.get());
            auto *writable = dynamic_cast<VulkanWritableDisplayBuffer *>(image.get());
            if ((!buffer || !buffer->copy_ready()) &&
                (!writable || !writable->write_ready()))
              return false;
          }
#endif
          // Idle repeat renderables may alias this storage. Every other
          // reference (in particular an in-flight Vulkan read) prevents a
          // write. Count only the bounded pool, then detach idle aliases below.
          const auto idle_owners = std::count_if(
              reusable_set_images_.begin(), reusable_set_images_.end(),
              [&image](const auto &other) {
                return other->shared_image() == image;
              });
          return image.use_count() == idle_owners + 1;
        });
    // A prepared pool is a hard storage bound. Busy must neither allocate a
    // replacement image nor destroy any prepared Graph/AOT binding.
    if (reusable == reusable_set_images_.end())
      return nullptr;
  } else if (last_shared_display_image_) {
    reusable = std::find_if(
        reusable_set_images_.begin(), reusable_set_images_.end(),
        [this, reuse_last_image](const auto &item) {
          auto image = item->shared_image();
          return reuse_last_image ? image == last_shared_display_image_
                                  : image && image != last_shared_display_image_;
        });
    if (reusable == reusable_set_images_.end()) {
      // Replacing a different image can synchronize unrelated CUDA work when
      // its imported memory is freed. Retain that storage for future writes.
      reusable = std::find_if(
          reusable_set_images_.begin(), reusable_set_images_.end(),
          [](const auto &item) { return !item->shared_image(); });
    }
  } else if (!reusable_set_images_.empty()) {
    reusable = std::prev(reusable_set_images_.end());
  }
  if (reusable != reusable_set_images_.end()) {
    r = std::move(*reusable);
    reusable_set_images_.erase(reusable);
  }
  if (!r) {
    r = std::make_unique<SetImage>(&app_context_, VboHelpers::all());
  }
  if (!reuse_last_image) {
    auto image = r->shared_image();
    if (image && image != last_shared_display_image_) {
      // Completed repeats can alias one image. Detach idle aliases before a
      // new write; in-flight renderables and the last display remain pinned.
      // Empty renderables are then reused by subsequent repeats.
      for (const auto &item : reusable_set_images_) {
        if (item->shared_image() == image)
          item->use_shared_image(nullptr);
      }
    }
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
  for (const auto &item : frame.renderables) {
    if (auto *image = dynamic_cast<SetImage *>(item.get()))
      image->retire_display();
  }
  recycle_renderable_list(frame.renderables);
}

void Renderer::retire_completed_frames() {
  validate_frame_retirement();
  while (!in_flight_frames_.empty()) {
    auto &frame = in_flight_frames_.front();
    TI_ERROR_IF(display_vulkan_execution_domain_ && !frame.complete,
                "Selected Vulkan frame has no graphics retirement proof");
    if (frame.complete && !frame.complete->is_ready()) {
      return;
    }
    recycle_renderables(frame);
    in_flight_frames_.pop_front();
  }
}

void Renderer::wait_oldest_frame() {
  validate_frame_retirement();
  if (in_flight_frames_.empty()) {
    return;
  }
  auto &frame = in_flight_frames_.front();
  TI_ERROR_IF(display_vulkan_execution_domain_ && !frame.complete,
              "Selected Vulkan frame has no graphics retirement proof");
  if (!frame.complete || !frame.complete->wait()) {
    TI_ERROR_IF(display_vulkan_execution_domain_,
                "Selected Vulkan frame has no graphics retirement proof");
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

void Renderer::record_renderpass(CommandList *commands,
                                DeviceAllocation image,
                                DeviceAllocation depth_image,
                                GuiBase *gui_base,
                                std::optional<Gui::DrawReservation> *gui_draw) {
  bool color_clear = true;
  std::vector<float> clear_colors = {background_color_[0], background_color_[1],
                                     background_color_[2], 1};

  for (auto renderable : render_queue_) {
    renderable->record_prepass_this_frame_commands(commands);
  }

  if (app_context_.config.ggui_arch == Arch::vulkan) {
    auto *vk = static_cast<VulkanCommandList *>(commands);
    // Include vertex/index reads after compute expansion and transfer writes.
    // The ordinary compute buffer barrier does not cover graphics consumers.
    vk->graphics_memory_barrier(true);
    if (has_offscreen_targets()) {
      // The Program tracks managed attachments, not a presentation surface.
      vk->set_next_renderpass_color_final_layout(ImageLayout::color_attachment);
    }
  }

  commands->begin_renderpass(
      /*x0=*/0, /*y0=*/0, /*x1=*/swap_chain_.width(),
      /*y1=*/swap_chain_.height(), /*num_color_attachments=*/1, &image,
      &color_clear, &clear_colors, &depth_image,
      /*depth_clear=*/true);

  const auto &render_viewport =
      app_context_.window_layout().framebuffer_render_viewport;
  if (render_viewport.width() > 0 && render_viewport.height() > 0) {
    commands->set_raster_viewport_and_scissor(
        render_viewport.x0, render_viewport.y0, render_viewport.x1,
        render_viewport.y1);
    for (auto renderable : render_queue_) {
      renderable->record_this_frame_commands(commands);
    }
  }

  if (app_context_.config.ggui_arch == Arch::vulkan) {
    Gui *gui = static_cast<Gui *>(gui_base);
    if (gui != nullptr && gui->has_widgets()) {
      VkRenderPass pass = static_cast<VulkanCommandList *>(commands)
                              ->current_renderpass()
                              ->renderpass;

      // No upload, cleanup, wait or ring advance is allowed in recording.
      const uint32_t count = has_offscreen_targets()
                                 ? 2 : swap_chain_.surface().get_image_count();
      gui->validate_draw_resources(pass, count);
      TI_ERROR_IF(!gui_draw || !*gui_draw, "ImGui recording lacks its reserved slot");
      if (!gui->draw(commands, **gui_draw))
        gui_draw->reset();
    } else if (gui != nullptr) {
      gui->end_frame();
    }
  }
#ifdef TI_WITH_METAL
  else if (app_context_.config.ggui_arch == Arch::metal) {
    GuiMetal *gui = static_cast<GuiMetal *>(gui_base);
    if (gui != nullptr && gui->has_widgets()) {

      auto mtl_cmd_list = static_cast<MetalCommandList *>(commands);

      MTLRenderPassDescriptor *pass = mtl_cmd_list->create_render_pass_desc(
          false, mtl_cmd_list->is_renderpass_active());
      mtl_cmd_list->set_renderpass_active();

      gui->init_render_resources(pass);
      gui->draw(commands);
    } else if (gui != nullptr) {
      gui->end_frame();
    }
  }
#endif
  else {
    TI_NOT_IMPLEMENTED;
  }

  commands->end_renderpass();
  if (app_context_.config.ggui_arch == Arch::vulkan) {
    static_cast<VulkanCommandList *>(commands)->graphics_memory_barrier(false);
  }
}

bool Renderer::has_offscreen_targets() const noexcept {
  return bool(offscreen_color_);
}

void Renderer::set_offscreen_targets(Texture *color, Texture *depth) {
  require_general_display_path();
  Program *program = app_context_.prog();
  TI_ERROR_IF(!program || app_context_.config.show_window ||
                  program->compile_config().arch != Arch::vulkan,
              "Runtime render targets require a Vulkan-owned hidden renderer");
  TI_ERROR_IF(has_offscreen_targets() || has_render_work() ||
                  !in_flight_frames_.empty(),
              "Runtime render targets must be fixed before recording any frame");
  TI_ERROR_IF(!color || !depth || color->owning_program() != program ||
                  depth->owning_program() != program,
              "Raster attachments must belong to the active Program");
  const std::array<int, 3> shape{static_cast<int>(swap_chain_.width()),
                                static_cast<int>(swap_chain_.height()), 1};
  TI_ERROR_IF(color->get_size() != shape || depth->get_size() != shape ||
                  color->get_dimension() != ImageDimension::d2D ||
                  depth->get_dimension() != ImageDimension::d2D ||
                  color->get_mip_levels() != 1 || depth->get_mip_levels() != 1 ||
                  color->get_buffer_format() != BufferFormat::rgba8 ||
                  depth->get_buffer_format() != BufferFormat::depth32f,
              "Raster targets require matching single-level RGBA8 and D32F images");
  auto color_lease = std::make_shared<Program::TextureResourceLease>(
      program->acquire_texture_external_lease(color));
  auto depth_lease = std::make_shared<Program::TextureResourceLease>(
      program->acquire_texture_external_lease(depth));
  // No frame has used the default attachments. Release their color/depth and
  // readback allocations instead of keeping two resident framebuffer sets.
  swap_chain_.release_default_attachments();
  offscreen_color_ = std::move(color_lease);
  offscreen_depth_ = std::move(depth_lease);
}

bool Renderer::draw_offscreen_targets(GuiBase *gui_base) {
  auto submission_guard =
      app_context_.prog()->acquire_runtime_resource_submission_guard();
  auto *gui = static_cast<Gui *>(gui_base);
  std::optional<Gui::DrawReservation> gui_draw;
  if (gui) {
    gui->validate_retirement();
    if (gui->has_widgets()) {
      const auto [pass, count] = gui_renderpass();
      gui->validate_draw_resources(pass, count);
      gui_draw = gui->reserve_draw_slot(/*blocking=*/true);
    }
  }
  const bool had_gui_draw = bool(gui_draw);
  const auto color = offscreen_color_->get()->get_device_allocation();
  const auto depth = offscreen_depth_->get()->get_device_allocation();
  auto *program = app_context_.prog();
  InFlightFrame *retained = nullptr;
  try {
    pending_texture_resource_leases_.reserve(
        pending_texture_resource_leases_.size() + 2);
    in_flight_frames_.emplace_back();
    auto &frame = in_flight_frames_.back();
    retained = &frame;
    frame.phase = FramePhase::preparing;
    frame.renderables = std::move(renderables_);
    frame.ndarray_resource_leases = std::move(pending_ndarray_resource_leases_);
    frame.texture_resource_leases = std::move(pending_texture_resource_leases_);
    frame.texture_resource_leases.push_back(offscreen_color_);
    frame.texture_resource_leases.push_back(offscreen_depth_);
    frame.scene_uniform = std::move(scene_ubo_);
    frame.scene_lights = std::move(lights_ssbo_);
    frame.complete = program->enqueue_graphics_op_lambda(
        [this, color, depth, gui_base, gui, &gui_draw, &frame]
        (GraphicsDevice *, CommandList *commands) {
          record_renderpass(commands, color, depth, gui_base, &gui_draw);
          if (gui_draw) {
            gui->validate_recorded_draw(*gui_draw);
            gui->mark_draw_submitting(*gui_draw);
          }
          // The callback returns into code that can enqueue graphics before
          // a later compute bridge fails; retain unknown work on that path.
          frame.phase = FramePhase::unsealed;
        },
        {{color, ImageLayout::color_attachment, ImageLayout::shader_read},
         {depth, ImageLayout::depth_attachment, ImageLayout::shader_read}});
    TI_ERROR_IF(!frame.complete, "Runtime graphics submit returned no completion");
    frame.phase = FramePhase::submitted;
    if (gui_draw) {
      gui->seal_draw(*gui_draw, frame.complete);
      gui_draw.reset();
    }
    render_complete_semaphore_ = frame.complete;
  } catch (...) {
    if (gui_draw) {
      if (retained && retained->phase == FramePhase::unsealed)
        gui->quarantine_draw();
      else
        gui->cancel_recorded_draw(*gui_draw);
    }
    // GUI may have partially mutated backend buffers or have an unknown
    // graphics reader. Preserve the complete Window closure, without a drain.
    if (gui && (gui->retirement_failed() ||
                (had_gui_draw && retained && retained->phase != FramePhase::submitted)))
      throw;
    if (retained) {
      // Preserve the previous ordinary offscreen recovery contract when no
      // GUI resource was touched. Selected targets remain forbidden.
      if (app_context_.device().backend_calls_safe())
        app_context_.device().wait_idle();
      render_queue_.clear();
      pending_set_image_ = nullptr;
      pending_runtime_resource_handles_.clear();
      recycle_renderables(*retained);
      in_flight_frames_.pop_back();
    }
    throw;
  }
  render_queue_.clear();
  pending_set_image_ = nullptr;
  pending_runtime_resource_handles_.clear();
  return true;
}

bool Renderer::draw_frame(GuiBase *gui_base, bool blocking_acquire) {
  validate_frame_retirement();
  auto *gui = app_context_.config.ggui_arch == Arch::vulkan
                  ? static_cast<Gui *>(gui_base) : nullptr;
  if (gui)
    gui->validate_retirement();
  if (!display_vulkan_execution_domain_)
    general_display_path_used_ = true;
  if (display_vulkan_execution_domain_) {
    TI_ERROR_IF(has_offscreen_targets() || render_queue_.size() != 1 ||
                    !pending_set_image_ ||
                    !pending_set_image_->has_pending_shared_cuda_vulkan_image(),
                "Selected Vulkan display requires one prepared image without geometry");
  }
  TI_ERROR_IF(pending_set_image_ && pending_set_image_->display_write_open(),
              "Submit or cancel the current display write before rendering");
  last_frame_used_shared_cuda_vulkan_ = false;
  if (has_offscreen_targets())
    return draw_offscreen_targets(gui_base);

  std::optional<Gui::DrawReservation> gui_draw;
  if (gui && gui->has_widgets()) {
    const auto [pass, count] = gui_renderpass();
    gui->validate_draw_resources(pass, count);
    gui_draw = gui->reserve_draw_slot(blocking_acquire);
    if (!gui_draw)
      return false;  // No acquire, map, growth, or image consume on GUI busy.
  }
  const bool retain_transaction = display_vulkan_execution_domain_ || gui_draw;
  InFlightFrame *retained = nullptr;
  bool acquired = false;
  try {
    // Preallocate the C4b owner node before surface acquisition. Gui ordinary
    // frames use the same closure; this does not erase general-path history.
    if (retain_transaction) {
      in_flight_frames_.emplace_back();
      retained = &in_flight_frames_.back();
      retained->phase = FramePhase::preparing;
      retained->submit_waits.reserve(1);
    }
    SurfaceImage surface_image;
    if (blocking_acquire) {
      surface_image = swap_chain_.surface().acquire_surface_image();
    } else if (!swap_chain_.surface().try_acquire_surface_image(&surface_image)) {
      if (retained)
        in_flight_frames_.pop_back();
      if (gui_draw)
        gui->cancel_recorded_draw(*gui_draw);
      return false;
    }
    if (surface_image.image.device == nullptr) {
      if (retained)
        in_flight_frames_.pop_back();
      if (gui_draw)
        gui->cancel_recorded_draw(*gui_draw);
      return false;
    }
    acquired = true;
    if (retained) {
      retained->renderables = std::move(renderables_);
      retained->ndarray_resource_leases = std::move(pending_ndarray_resource_leases_);
      retained->texture_resource_leases = std::move(pending_texture_resource_leases_);
      // Save the returned acquisition owner even if a later host step fails.
      if (surface_image.image_available)
        retained->submit_waits.push_back(surface_image.image_available);
    }
    Program::RuntimeResourceSubmissionGuard submission_guard;
    if (auto *program = app_context_.prog();
        program && program->get_graphics_device() == &app_context_.device())
      submission_guard = program->acquire_runtime_resource_submission_guard();
    auto image = surface_image.image;
    std::vector<StreamSemaphore> present_waits_after_acquire;
    if (app_context_.config.ggui_arch == Arch::vulkan) {
      if (auto *surface = dynamic_cast<VulkanSurface *>(&swap_chain_.surface()))
        present_waits_after_acquire =
            surface->take_present_waits_after_acquire(surface_image.image_index);
    }
    if (retained)
      retained->present_waits_after_acquire = std::move(present_waits_after_acquire);
    auto stream = app_context_.device().get_graphics_stream();
    auto [cmd_list, res] = stream->new_command_list_unique();
    TI_ERROR_IF(res != RhiResult::success || !cmd_list, "Failed to allocate command list");
    auto *commands = cmd_list.get();
    if (retained)
      retained->commands = std::move(cmd_list);
    commands->image_transition(image, ImageLayout::undefined,
                               ImageLayout::color_attachment);
    record_renderpass(commands, image, swap_chain_.depth_allocation(),
                      gui_base, &gui_draw);
    std::vector<StreamSemaphore> wait_semaphores;
    const bool use_shared_cuda_vulkan =
        pending_set_image_ && pending_set_image_->has_pending_shared_cuda_vulkan_image();
    const bool shared_image_only = use_shared_cuda_vulkan && render_queue_.size() == 1;
    // Cold fonts and CPU-written GUI buffers require no Program flush. Other
    // geometry retains the original producer flush contract.
    if (app_context_.prog() && !shared_image_only) {
      auto sema = app_context_.prog()->flush();
      if (sema)
        wait_semaphores.push_back(sema);
    }
    if (surface_image.image_available)
      wait_semaphores.push_back(surface_image.image_available);
    if (retained) {
      retained->submit_waits = std::move(wait_semaphores);
      auto *submitted_image = pending_set_image_;
      auto *vulkan_stream = dynamic_cast<VulkanStream *>(stream);
      TI_ERROR_IF(!vulkan_stream, "GUI/selected display requires its Vulkan stream");
      if (gui_draw)
        gui->validate_recorded_draw(*gui_draw);
      render_queue_.clear();
      pending_set_image_ = nullptr;
      pending_runtime_resource_handles_.clear();
      retained->phase = FramePhase::unsealed;
      if (gui_draw)
        gui->mark_draw_submitting(*gui_draw);
      retained->complete = use_shared_cuda_vulkan
          ? submitted_image->submit_shared_cuda_vulkan_frame(
                *vulkan_stream, commands, retained->submit_waits)
          : stream->submit(commands, retained->submit_waits);
      TI_ERROR_IF(!retained->complete, "Graphics submit returned no completion");
      retained->phase = FramePhase::submitted;
      if (gui_draw) {
        gui->seal_draw(*gui_draw, retained->complete);
        gui_draw.reset();
      }
      // Both proofs are durable before publishing the preallocated ticket.
      render_complete_semaphore_ = retained->complete;
      last_frame_used_shared_cuda_vulkan_ =
          use_shared_cuda_vulkan && !display_vulkan_execution_domain_;
      if (submitted_image) {
        submitted_image->attach_display_completion(retained->complete);
        last_shared_display_image_ = use_shared_cuda_vulkan &&
            (submitted_image->tracks_display() || display_vulkan_execution_domain_)
                ? submitted_image->shared_image() : nullptr;
      }
      render_surface_image_ = surface_image;
      return true;
    }

    // Preserve the existing non-GUI ordinary route, including Metal.
    if (use_shared_cuda_vulkan) {
      auto *vulkan_stream = dynamic_cast<VulkanStream *>(stream);
      TI_ERROR_IF(!vulkan_stream, "Shared CUDA-Vulkan frame requires a Vulkan stream");
      render_complete_semaphore_ = pending_set_image_->submit_shared_cuda_vulkan_frame(
          *vulkan_stream, commands, wait_semaphores);
    } else {
      render_complete_semaphore_ = stream->submit(commands, wait_semaphores);
    }
    last_frame_used_shared_cuda_vulkan_ = use_shared_cuda_vulkan;
    if (pending_set_image_) {
      pending_set_image_->attach_display_completion(render_complete_semaphore_);
      last_shared_display_image_ = use_shared_cuda_vulkan && pending_set_image_->tracks_display()
          ? pending_set_image_->shared_image() : nullptr;
    }
    render_surface_image_ = surface_image;
    render_queue_.clear();
    pending_set_image_ = nullptr;
    InFlightFrame frame;
    frame.complete = render_complete_semaphore_;
    frame.present_waits_after_acquire = std::move(present_waits_after_acquire);
    frame.renderables = std::move(renderables_);
    frame.ndarray_resource_leases = std::move(pending_ndarray_resource_leases_);
    frame.texture_resource_leases = std::move(pending_texture_resource_leases_);
    pending_runtime_resource_handles_.clear();
    in_flight_frames_.push_back(std::move(frame));
    return true;
  } catch (...) {
    if (gui_draw) {
      if (retained && retained->phase == FramePhase::unsealed) {
        gui->quarantine_draw();
      } else {
        if (retained)
          retained->commands.reset();  // Discard recording before known cancel.
        gui->cancel_recorded_draw(*gui_draw);
      }
    }
    if (retained && !acquired)
      in_flight_frames_.pop_back();
    // An acquired WSI image is not returned by cancelling a Gui reservation.
    // Keep preparing/unsealed closure for Window's fail-closed retirement.
    throw;
  }
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
