#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include <memory>
#include <mutex>
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/vertex.h"

#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/renderable.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/common/canvas_base.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {
namespace cuda {
class ExecutionDomain;
}
namespace storage {
struct DenseStorageBuildResult;
}  // namespace storage
namespace vulkan {
class VulkanStream;
}  // namespace vulkan
}  // namespace taichi::lang

namespace taichi::ui {

namespace vulkan {

// Optional completion for a Canvas image, attached to the existing render
// submission. This is not a second queue or a promise of on-screen
// presentation.
class TI_DLL_EXPORT DisplayCompletion final {
 public:
  DisplayCompletion();
  bool done();
  void wait();
  std::string status();
  void attach(taichi::lang::StreamSemaphore completion);
  void retire();
  void cancel();

 private:
  std::mutex mutex_;
  std::string status_{"pending"};
  taichi::lang::StreamSemaphore completion_;
};

// Storage and producer handoff used by the existing display pool. This
// interface owns no frame queue or scheduling policy. Prepared producers keep
// their backend owner, not an extra reference to this renderer-facing object.
class DisplayStorage {
 public:
  virtual ~DisplayStorage() = default;

  virtual const taichi::lang::storage::DenseStorageBuildResult &description()
      const = 0;
  virtual std::uint64_t identity() const noexcept = 0;
  virtual taichi::lang::DevicePtr vulkan_ptr() const noexcept = 0;
  virtual int width() const noexcept = 0;
  virtual int height() const noexcept = 0;
  virtual bool ready_for_vulkan_submit() const noexcept = 0;
  virtual void prepare_write() = 0;
  // Explicit discard of an acquired host lease. Completion association (for
  // example track_display) must not call this hook.
  virtual void cancel_write() {}
  virtual void finish_write() = 0;
  virtual bool try_finish_write(
      const taichi::lang::RuntimeCompletion &producer = {}) = 0;
  virtual bool try_copy_image(
      taichi::lang::DevicePtr source,
      const taichi::lang::RuntimeCompletion &producer = {},
      std::shared_ptr<taichi::lang::Program::NdarrayResourceLease>
          source_lease = nullptr) = 0;
  virtual taichi::lang::RuntimeCompletion source_completion() const = 0;
  virtual bool has_live_execution_domain() const = 0;
  virtual void close() = 0;
  virtual taichi::lang::StreamSemaphore submit_vulkan_frame(
      taichi::lang::vulkan::VulkanStream &stream,
      taichi::lang::CommandList *command_list,
      const std::vector<taichi::lang::StreamSemaphore> &additional_waits) = 0;
};

class SharedCudaVulkanImage final : public DisplayStorage {
 public:
  static std::shared_ptr<SharedCudaVulkanImage> create(
      AppContext *app_context,
      int width,
      int height);

  ~SharedCudaVulkanImage() override;
  SharedCudaVulkanImage(const SharedCudaVulkanImage &) = delete;
  SharedCudaVulkanImage &operator=(const SharedCudaVulkanImage &) = delete;

  const taichi::lang::storage::DenseStorageBuildResult &description() const override;
  std::uint64_t identity() const noexcept override;
  taichi::lang::DevicePtr vulkan_ptr() const noexcept override;
  int width() const noexcept override;
  int height() const noexcept override;
  bool ready_for_vulkan_submit() const noexcept override;
  void prepare_write() override;
  void finish_write() override;
  void copy_cuda_image(taichi::lang::DevicePtr source);
  void set_execution_domain(
      std::shared_ptr<taichi::lang::cuda::ExecutionDomain> domain);
  bool try_finish_write(
      const taichi::lang::RuntimeCompletion &producer = {}) override;
  bool try_copy_image(
      taichi::lang::DevicePtr source,
      const taichi::lang::RuntimeCompletion &producer = {},
      std::shared_ptr<taichi::lang::Program::NdarrayResourceLease>
          source_lease = nullptr) override;
  taichi::lang::RuntimeCompletion source_completion() const override;
  bool has_live_execution_domain() const override;
  void close() override;
  taichi::lang::StreamSemaphore submit_vulkan_frame(
      taichi::lang::vulkan::VulkanStream &stream,
      taichi::lang::CommandList *command_list,
      const std::vector<taichi::lang::StreamSemaphore> &additional_waits) override;

 private:
  class Impl;
  explicit SharedCudaVulkanImage(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class SetImage;
void erase_direct_set_image_state(const SetImage *set_image);

class SetImage final : public Renderable {
 public:
  struct UniformBufferObject {
    glm::vec2 lower_bound;
    glm::vec2 upper_bound;
    // in non_packed_mode,
    // the actual image is only a corner of the whole image
    float x_factor{1.0};
    float y_factor{1.0};
    int transpose{0};
  };

  struct DirectUniformBufferObject {
    glm::vec2 lower_bound;
    glm::vec2 upper_bound;
    float x_factor{1.0};
    float y_factor{1.0};
    int transpose{0};
    int width{0};
    int height{0};
    int padding0{0};
    int padding1{0};
    int padding2{0};
  };

  SetImage(AppContext *app_context, VertexAttributes vbo_attrs);

  ~SetImage() override {
    cancel_display();
    erase_direct_set_image_state(this);
  }

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) final;

  void record_prepass_this_frame_commands(
      taichi::lang::CommandList *command_list) final;

  void update_data(const SetImageInfo &info);

  void update_data(const DisplayFrameInfo &info);

  void update_data(taichi::lang::Texture *tex);

  void set_transpose(bool transpose);

  void use_shared_image(std::shared_ptr<DisplayStorage> image);
  std::shared_ptr<DisplayStorage> shared_image() const {
    return shared_cuda_vulkan_image_;
  }
  void track_display(std::shared_ptr<DisplayCompletion> completion,
                     bool writable);
  bool display_write_open() const {
    return display_write_open_;
  }
  void seal_display_write() {
    display_write_open_ = false;
  }
  bool tracks_display() const {
    return bool(display_completion_);
  }
  void attach_display_completion(taichi::lang::StreamSemaphore completion);
  void retire_display();
  void cancel_display();
  void discard_display_write();

  std::shared_ptr<DisplayStorage> acquire_shared_cuda_vulkan_image(
      int width,
      int height);

  bool has_pending_shared_cuda_vulkan_image() const noexcept;

  taichi::lang::StreamSemaphore submit_shared_cuda_vulkan_frame(
      taichi::lang::vulkan::VulkanStream &stream,
      taichi::lang::CommandList *command_list,
      const std::vector<taichi::lang::StreamSemaphore> &additional_waits);

 private:
  int width_{0};
  int height_{0};

  taichi::lang::DeviceImageUnique texture_{nullptr};
  taichi::lang::DeviceAllocationUnique host_staging_{nullptr};
  uint64_t upload_staging_size_{0};
  bool upload_staging_host_write_{false};
  bool upload_staging_export_sharing_{false};
  taichi::lang::DevicePtr pending_upload_buffer_{taichi::lang::kDeviceNullPtr};
  bool pending_upload_{false};
  std::shared_ptr<DisplayStorage> shared_cuda_vulkan_image_;
  bool shared_cuda_vulkan_disabled_{false};
  bool pending_shared_cuda_vulkan_{false};
  std::shared_ptr<DisplayCompletion> display_completion_;
  bool display_write_open_{false};

  taichi::lang::BufferFormat format_;

 private:
  void resize_texture(int width, int height, taichi::lang::BufferFormat format);

  void update_ubo(float x_factor, float y_factor, bool transpose);

  void update_direct_buffer_ubo(bool transpose = true);

  bool can_use_direct_buffer(taichi::lang::DevicePtr ptr) const;

  void use_direct_buffer_pipeline();

  void use_texture_pipeline();

  taichi::lang::DevicePtr upload_host_rgba8(const void *host_ptr,
                                            int width,
                                            int height,
                                            int row_stride_bytes);

  taichi::lang::DevicePtr stage_device_rgba8(taichi::lang::DevicePtr src,
                                             uint64_t size_bytes);

  taichi::lang::DevicePtr ensure_upload_staging(uint64_t size_bytes,
                                                bool host_write,
                                                bool export_sharing);

  void reset_upload_staging();
};

}  // namespace vulkan

}  // namespace taichi::ui
