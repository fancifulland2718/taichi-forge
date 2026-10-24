#pragma once

#include "taichi/ui/ggui/renderables/set_image.h"

namespace taichi::lang::vulkan {
class ExecutionDomain;
}

namespace taichi::ui::vulkan {

// Copy-only storage for the existing Renderer pool. It exposes no writable
// dense view: all inputs are generation-qualified, cold-prepared Ndarrays.
class VulkanDisplayBuffer final : public DisplayStorage {
 public:
  static std::vector<std::shared_ptr<DisplayStorage>> create_pool(
      AppContext &context,
      std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain,
      const std::vector<FieldInfo> &sources,
      std::size_t capacity,
      std::size_t graphics_capacity);

  ~VulkanDisplayBuffer() override;
  const taichi::lang::storage::DenseStorageBuildResult &description()
      const override;
  std::uint64_t identity() const noexcept override;
  taichi::lang::DevicePtr vulkan_ptr() const noexcept override;
  int width() const noexcept override;
  int height() const noexcept override;
  bool ready_for_vulkan_submit() const noexcept override;
  void prepare_write() override;
  void finish_write() override;
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
      taichi::lang::CommandList *commands,
      const std::vector<taichi::lang::StreamSemaphore> &waits) override;

  // Used by the original pool selector before moving a renderable. Copy
  // cancellation leaves this false until the accepted copy actually finishes.
  bool copy_ready() const;
  std::shared_ptr<taichi::lang::Program::NdarrayResourceLease> source_lease(
      const FieldInfo &source) const;
  void validate_producer(
      const taichi::lang::RuntimeCompletion &producer) const;

 private:
  struct Pool;
  VulkanDisplayBuffer(std::shared_ptr<Pool> pool, std::size_t slot);
  std::shared_ptr<Pool> pool_;
  std::size_t slot_;
  std::uint64_t identity_;
  taichi::lang::DevicePtr pointer_;
};

// Writable local storage uses the same Renderer pool and draw path. Its
// generation-qualified native owner, not this facade, is borrowed by writers.
class VulkanWritableDisplayBuffer final : public DisplayStorage {
 public:
  static std::vector<std::shared_ptr<DisplayStorage>> create_pool(
      AppContext &context,
      std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain,
      int width, int height, std::size_t capacity,
      std::size_t graphics_capacity);
  ~VulkanWritableDisplayBuffer() override;
  const taichi::lang::storage::DenseStorageBuildResult &description() const override;
  std::uint64_t identity() const noexcept override;
  taichi::lang::DevicePtr vulkan_ptr() const noexcept override;
  int width() const noexcept override;
  int height() const noexcept override;
  bool ready_for_vulkan_submit() const noexcept override;
  bool write_ready() const;
  void prepare_write() override;
  void cancel_write() override;
  void finish_write() override;
  bool try_finish_write(const taichi::lang::RuntimeCompletion &producer = {}) override;
  bool try_copy_image(
      taichi::lang::DevicePtr source,
      const taichi::lang::RuntimeCompletion &producer = {},
      std::shared_ptr<taichi::lang::Program::NdarrayResourceLease> source_lease = nullptr) override;
  taichi::lang::RuntimeCompletion source_completion() const override;
  bool has_live_execution_domain() const override;
  void close() override;
  taichi::lang::StreamSemaphore submit_vulkan_frame(
      taichi::lang::vulkan::VulkanStream &stream,
      taichi::lang::CommandList *commands,
      const std::vector<taichi::lang::StreamSemaphore> &waits) override;

 private:
  struct Pool;
  VulkanWritableDisplayBuffer(std::shared_ptr<Pool> pool, std::size_t slot);
  std::shared_ptr<Pool> pool_;
  std::size_t slot_;
  taichi::lang::DevicePtr pointer_;
};

}  // namespace taichi::ui::vulkan
