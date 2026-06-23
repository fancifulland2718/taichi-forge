#include "set_image.h"

#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/rhi/interop/vulkan_cuda_interop.h"
#include "taichi/ui/utils/utils.h"

#include <unordered_map>

using taichi::lang::Program;

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

static_assert(sizeof(SetImage::DirectUniformBufferObject) == 48);

namespace {

struct DirectSetImageState {
  taichi::lang::Pipeline *pipeline{nullptr};
  std::unique_ptr<taichi::lang::ShaderResourceSet> resource_set{nullptr};
  taichi::lang::DeviceAllocationUnique ubo{nullptr};
  taichi::lang::DevicePtr display_buffer{taichi::lang::kDeviceNullPtr};
  int ubo_width{0};
  int ubo_height{0};
  bool ubo_valid{false};
  bool enabled{false};
};

std::unordered_map<const SetImage *, DirectSetImageState>
    direct_set_image_states;

DirectSetImageState *find_direct_state(const SetImage *set_image) {
  auto it = direct_set_image_states.find(set_image);
  if (it == direct_set_image_states.end()) {
    return nullptr;
  }
  return &it->second;
}

DirectSetImageState &get_direct_state(const SetImage *set_image) {
  return direct_set_image_states[set_image];
}

}  // namespace

void SetImage::update_ubo(float x_factor, float y_factor, bool transpose) {
  glm::vec2 pixel_size = glm::vec2(1.0f / width_, 1.0f / height_);
  glm::vec2 lower_bound = pixel_size * 0.5f;
  glm::vec2 upper_bound = glm::vec2(1.0f, 1.0f) - pixel_size * 0.5f;
  UniformBufferObject ubo = {lower_bound, upper_bound, x_factor, y_factor,
                             int(transpose)};
  void *mapped{nullptr};
  RHI_VERIFY(app_context_->device().map(uniform_buffer_renderable_->get_ptr(0),
                                        &mapped));
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(*uniform_buffer_renderable_);
}

void SetImage::update_direct_buffer_ubo() {
  DirectSetImageState &state = get_direct_state(this);
  if (!state.ubo) {
    auto [buf, res] = app_context_->device().allocate_memory_unique(
        {sizeof(DirectUniformBufferObject), /*host_write=*/true,
         /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Uniform});
    TI_ASSERT(res == RhiResult::success);
    state.ubo = std::move(buf);
  }
  if (state.ubo_valid && state.ubo_width == width_ &&
      state.ubo_height == height_) {
    return;
  }

  glm::vec2 pixel_size = glm::vec2(1.0f / width_, 1.0f / height_);
  glm::vec2 lower_bound = pixel_size * 0.5f;
  glm::vec2 upper_bound = glm::vec2(1.0f, 1.0f) - pixel_size * 0.5f;
  DirectUniformBufferObject ubo = {lower_bound,
                                   upper_bound,
                                   1.0f,
                                   1.0f,
                                   1,
                                   width_,
                                   height_};
  void *mapped{nullptr};
  RHI_VERIFY(app_context_->device().map(state.ubo->get_ptr(0), &mapped));
  memcpy(mapped, &ubo, sizeof(ubo));
  app_context_->device().unmap(*state.ubo);
  state.ubo_width = width_;
  state.ubo_height = height_;
  state.ubo_valid = true;
}

bool SetImage::can_use_direct_buffer(DevicePtr ptr) const {
  return app_context_->config.ggui_arch == Arch::vulkan &&
         (ptr.device == &app_context_->device() ||
          is_cuda_to_vulkan_copy(&app_context_->device(), ptr.device));
}

void SetImage::use_direct_buffer_pipeline() {
  DirectSetImageState &state = get_direct_state(this);
  if (!state.pipeline) {
    state.pipeline = app_context_->get_raster_pipeline(
        {app_context_->config.package_path + "/shaders/SetImageBuffer_vk_frag.spv",
         config_.vertex_shader_path,
         config_.topology_type, config_.depth, config_.polygon_mode,
         config_.blending, config_.vertex_input_rate_instance});
  }
  if (!state.resource_set) {
    state.resource_set = app_context_->device().create_resource_set_unique();
  }
  state.enabled = true;
  pending_upload_ = false;
}

void SetImage::use_texture_pipeline() {
  if (auto *state = find_direct_state(this)) {
    state->enabled = false;
    state->display_buffer = kDeviceNullPtr;
  }
}

void SetImage::update_data(const SetImageInfo &info) {
  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = app_context_->prog();
  StreamSemaphore sema = nullptr;

  const FieldInfo &img = info.img;

  // Image is a width x height field of u32 which contains encoded RGBA8
  TI_ASSERT_INFO(
      img.shape.size() == 2 && img.dtype == taichi::lang::PrimitiveType::u32,
      "set_image buffer input must be 2D field of u32");

  int new_width = img.shape[0];
  int new_height = img.shape[1];

  // If data source is not a host mapped pointer, it is a DeviceAllocation
  // from the same backend as GGUI
  DevicePtr img_dev_ptr = info.img.dev_alloc.get_ptr();
  bool uses_host = img.field_source == FieldSource::HostMappedPtr;
  if (uses_host) {
    use_texture_pipeline();
    resize_texture(new_width, new_height, BufferFormat::rgba8);
    update_ubo(1.0f, 1.0f, true);
    void *src_ptr = reinterpret_cast<uint8_t *>(img.dev_alloc.alloc_id);
    img_dev_ptr = upload_host_rgba8(src_ptr, width_, height_,
                                    height_ * sizeof(uint32_t));
    pending_upload_buffer_ = img_dev_ptr;
    pending_upload_ = true;
    return;
  }

  if (can_use_direct_buffer(img_dev_ptr)) {
    width_ = new_width;
    height_ = new_height;
    format_ = BufferFormat::rgba8;
    update_direct_buffer_ubo();
    if (img_dev_ptr.device != &app_context_->device()) {
      img_dev_ptr = stage_device_rgba8(
          img_dev_ptr,
          static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_) *
              sizeof(uint32_t));
    } else {
      reset_upload_staging();
    }
    get_direct_state(this).display_buffer = img_dev_ptr;
    use_direct_buffer_pipeline();
    return;
  }

  use_texture_pipeline();
  resize_texture(new_width, new_height, BufferFormat::rgba8);
  update_ubo(1.0f, 1.0f, true);

  if (img_dev_ptr.device != &app_context_->device()) {
    img_dev_ptr = stage_device_rgba8(
        img_dev_ptr,
        static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_) *
            sizeof(uint32_t));
    pending_upload_buffer_ = img_dev_ptr;
    pending_upload_ = true;
    return;
  }

  reset_upload_staging();
  pending_upload_ = false;

  auto copy_op = [&, img_dev_ptr](Device *, CommandList *cmdlist) {
    BufferImageCopyParams copy_params;
    // these are flipped because taichi is y-major and vulkan is x-major
    copy_params.image_extent.x = height_;
    copy_params.image_extent.y = width_;
    cmdlist->image_transition(*texture_, ImageLayout::undefined,
                              ImageLayout::transfer_dst);
    cmdlist->buffer_barrier(img_dev_ptr);
    cmdlist->buffer_to_image(*texture_, img_dev_ptr, ImageLayout::transfer_dst,
                             copy_params);
    cmdlist->image_transition(*texture_, ImageLayout::transfer_dst,
                              ImageLayout::shader_read);
  };

  if (prog && prog->get_graphics_device() == &app_context_->device()) {
    // If it's the same device, we do not use the staging buffer and directly
    // copy from the src ptr to the image
    prog->enqueue_compute_op_lambda(copy_op, {});
  } else {
    // Create a single time command
    auto stream = app_context_->device().get_graphics_stream();
    auto [cmdlist, res] = stream->new_command_list_unique();
    TI_ASSERT_INFO(res == RhiResult::success,
                   "Failed to allocate command list");
    copy_op(&app_context_->device(), cmdlist.get());
    if (sema) {
      stream->submit(cmdlist.get(), {sema});
    } else {
      stream->submit(cmdlist.get());
    }
  }
}

void SetImage::update_data(const DisplayFrameInfo &info) {
  TI_ASSERT_INFO(info.host_rgba8 != nullptr,
                 "display frame host RGBA8 pointer must not be null");
  TI_ASSERT_INFO(info.width > 0 && info.height > 0,
                 "display frame size must be positive");

  use_texture_pipeline();
  resize_texture(info.width, info.height, BufferFormat::rgba8);
  update_ubo(1.0f, 1.0f, info.transpose);

  int row_stride_bytes = info.row_stride_bytes;
  if (row_stride_bytes == 0) {
    row_stride_bytes = info.height * 4;
  }
  DevicePtr img_dev_ptr =
      upload_host_rgba8(info.host_rgba8, width_, height_, row_stride_bytes);
  pending_upload_buffer_ = img_dev_ptr;
  pending_upload_ = true;
}

void SetImage::update_data(Texture *tex) {
  Program *prog = app_context_->prog();
  use_texture_pipeline();
  reset_upload_staging();
  pending_upload_ = false;

  auto shape = tex->get_size();
  auto new_format = tex->get_buffer_format();

  TI_ASSERT_INFO(shape[2] == 1,
                 "Must be a 2D image! Received image shape: {}x{}x{}", shape[0],
                 shape[1], shape[2]);

  // Reminder: y/x is flipped in Taichi. I would like to use the correct
  // orientation, but we have existing code already using the previous
  // convention
  const int new_width = shape[1];
  const int new_height = shape[0];
  resize_texture(new_width, new_height, new_format);

  update_ubo(1.0f, 1.0f, false);

  ImageCopyParams copy_params;
  copy_params.width = shape[0];
  copy_params.height = shape[1];
  copy_params.depth = shape[2];

  DeviceAllocation src_alloc = tex->get_device_allocation();
  auto copy_op = [&, src_alloc](Device *device, CommandList *cmdlist) {
    cmdlist->image_transition(*this->texture_, ImageLayout::undefined,
                              ImageLayout::transfer_dst);
    cmdlist->copy_image(*this->texture_, src_alloc, ImageLayout::transfer_dst,
                        ImageLayout::transfer_src, copy_params);
    cmdlist->image_transition(*this->texture_, ImageLayout::transfer_dst,
                              ImageLayout::shader_read);
  };

  // In the current state if we called this direct image update data method, we
  // gurantee to have a program.
  // FIXME: However, if we don't have a Program, where does the layout come
  // from?
  if (prog && prog->get_graphics_device() == &app_context_->device()) {
    prog->enqueue_compute_op_lambda(
        copy_op, {ComputeOpImageRef{src_alloc, ImageLayout::transfer_src,
                                    ImageLayout::transfer_src}});
  } else {
    TI_ERROR("`update_data` received Texture from a different device");
  }
}

SetImage::SetImage(AppContext *app_context, VertexAttributes vbo_attrs) {
  RenderableConfig config;
  config.draw_vertex_count = 6;
  config.ubo_size = sizeof(UniformBufferObject);
  config.fragment_shader_path =
      app_context->config.package_path + "/shaders/SetImage_vk_frag.spv";
  config.vertex_shader_path =
      app_context->config.package_path + "/shaders/SetImage_vk_vert.spv";

  Renderable::init(config, app_context);
  create_graphics_pipeline();

  // Create UBO
  {
    auto [buf, res] = app_context_->device().allocate_memory_unique(
        {config_.ubo_size, /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Uniform});
    TI_ASSERT(res == RhiResult::success);
    uniform_buffer_renderable_ = std::move(buf);
  }

  // Create & upload vertex buffer (constant)
  const std::vector<Vertex> vertices = {
      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{-1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},

      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}, {1.f, 1.f, 1.f}},
  };
  {
    auto [buf, res] = app_context_->device().allocate_memory_unique(
        {sizeof(Vertex) * vertices.size(), /*host_write=*/true,
         /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Vertex});
    TI_ASSERT(res == RhiResult::success);
    vertex_buffer_ = std::move(buf);
  }
  void *mapped_vbo{nullptr};
  RHI_VERIFY(
      app_context_->device().map(vertex_buffer_->get_ptr(0), &mapped_vbo));
  memcpy(mapped_vbo, vertices.data(), sizeof(Vertex) * vertices.size());
  app_context_->device().unmap(*vertex_buffer_);
}

void erase_direct_set_image_state(const SetImage *set_image) {
  direct_set_image_states.erase(set_image);
}

void SetImage::record_this_frame_commands(CommandList *command_list) {
  if (auto *state = find_direct_state(this); state && state->enabled) {
    TI_ASSERT_INFO(state->display_buffer != kDeviceNullPtr,
                   "set_image direct display buffer must be valid");
    state->resource_set->rw_buffer(
        0, state->display_buffer,
        static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_) *
            sizeof(uint32_t));
    state->resource_set->buffer(1, state->ubo->get_ptr());

    auto raster_state = app_context_->device().create_raster_resources_unique();
    raster_state->vertex_buffer(vertex_buffer_->get_ptr(), 0);

    command_list->bind_pipeline(state->pipeline);
    command_list->bind_raster_resources(raster_state.get());
    command_list->bind_shader_resources(state->resource_set.get());
    command_list->draw(6);
    return;
  }

  resource_set_->image(0, *texture_, {});
  resource_set_->buffer(1, uniform_buffer_renderable_->get_ptr());

  auto raster_state = app_context_->device().create_raster_resources_unique();
  raster_state->vertex_buffer(vertex_buffer_->get_ptr(), 0);

  command_list->bind_pipeline(pipeline_);
  command_list->bind_raster_resources(raster_state.get());
  command_list->bind_shader_resources(resource_set_.get());
  command_list->draw(6);
}

void SetImage::record_prepass_this_frame_commands(CommandList *command_list) {
  if (auto *state = find_direct_state(this); state && state->enabled) {
    if (state->display_buffer != kDeviceNullPtr) {
      command_list->buffer_barrier(state->display_buffer);
    }
    return;
  }
  if (!pending_upload_) {
    return;
  }
  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = height_;
  copy_params.image_extent.y = width_;
  command_list->image_transition(*texture_, ImageLayout::undefined,
                                 ImageLayout::transfer_dst);
  command_list->buffer_barrier(pending_upload_buffer_);
  command_list->buffer_to_image(*texture_, pending_upload_buffer_,
                                ImageLayout::transfer_dst, copy_params);
  command_list->image_transition(*texture_, ImageLayout::transfer_dst,
                                 ImageLayout::shader_read);
  pending_upload_ = false;
}

void SetImage::resize_texture(int width,
                              int height,
                              taichi::lang::BufferFormat format) {
  if (width_ == width && height_ == height && format_ == format &&
      texture_ != nullptr) {
    return;
  }

  texture_.reset();

  width_ = width;
  height_ = height;
  format_ = format;

  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = format_;
  params.initial_layout = ImageLayout::undefined;
  // these are flipped because taichi is y-major and vulkan is x-major
  params.x = height_;
  params.y = width_;
  params.z = 1;
  params.export_sharing = false;

  texture_ = app_context_->device().create_image_unique(params);
}

DevicePtr SetImage::upload_host_rgba8(const void *host_ptr,
                                      int width,
                                      int height,
                                      int row_stride_bytes) {
  const int packed_row_bytes = height * 4;
  const uint64_t img_size_bytes =
      static_cast<uint64_t>(width) * static_cast<uint64_t>(packed_row_bytes);
  DevicePtr staging_ptr =
      ensure_upload_staging(img_size_bytes, true, false);

  void *dst_ptr{nullptr};
  RHI_VERIFY(app_context_->device().map(host_staging_->get_ptr(0), &dst_ptr));
  if (row_stride_bytes == packed_row_bytes) {
    memcpy(dst_ptr, host_ptr, img_size_bytes);
  } else {
    auto *dst = reinterpret_cast<uint8_t *>(dst_ptr);
    auto *src = reinterpret_cast<const uint8_t *>(host_ptr);
    for (int i = 0; i < width; ++i) {
      memcpy(dst + static_cast<uint64_t>(i) * packed_row_bytes,
             src + static_cast<uint64_t>(i) * row_stride_bytes,
             packed_row_bytes);
    }
  }
  app_context_->device().unmap(*host_staging_);
  return staging_ptr;
}

DevicePtr SetImage::stage_device_rgba8(DevicePtr src, uint64_t size_bytes) {
  if (is_cuda_to_vulkan_copy(&app_context_->device(), src.device)) {
    DevicePtr dst = ensure_upload_staging(size_bytes, false, true);
    memcpy_cuda_to_vulkan_fast(dst, src, size_bytes);
    return dst;
  }

  DevicePtr dst = ensure_upload_staging(size_bytes, true, false);
  auto capability = Device::check_memcpy_capability(dst, src, size_bytes);
  TI_ASSERT_INFO(
      capability != Device::MemcpyCapability::RequiresHost,
      "display frame device RGBA8 source cannot be copied to Vulkan directly");
  Device::memcpy_direct(dst, src, size_bytes);
  return dst;
}

DevicePtr SetImage::ensure_upload_staging(uint64_t size_bytes,
                                          bool host_write,
                                          bool export_sharing) {
  if (host_staging_ && upload_staging_size_ >= size_bytes &&
      upload_staging_host_write_ == host_write &&
      upload_staging_export_sharing_ == export_sharing) {
    return host_staging_->get_ptr(0);
  }

  auto [staging, res] = app_context_->device().allocate_memory_unique(
      {size_bytes, host_write, false, export_sharing,
       export_sharing ? (AllocUsage::Upload | AllocUsage::Storage)
                      : AllocUsage::Upload});
  TI_ASSERT(res == RhiResult::success);
  host_staging_ = std::move(staging);
  upload_staging_size_ = size_bytes;
  upload_staging_host_write_ = host_write;
  upload_staging_export_sharing_ = export_sharing;
  return host_staging_->get_ptr(0);
}

void SetImage::reset_upload_staging() {
  host_staging_.reset();
  upload_staging_size_ = 0;
  upload_staging_host_write_ = false;
  upload_staging_export_sharing_ = false;
  pending_upload_buffer_ = kDeviceNullPtr;
}

}  // namespace vulkan

}  // namespace taichi::ui
