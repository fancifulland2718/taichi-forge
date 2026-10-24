// Real Vulkan recording plus Gui's public reader ledger.
// No driver-failure hook and no allocator override. Run this dedicated process
// normally; never replace process teardown with _Exit/quick_exit/TerminateProcess.
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "taichi/program/program.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/gui.h"
#include "taichi/ui/ggui/swap_chain.h"

namespace taichi::lang {
namespace {
namespace vk = taichi::lang::vulkan;
namespace ui = taichi::ui;
using Gui = ui::vulkan::Gui;

// Same real host timeline mechanism as tests/cpp/aot/vulkan/timeline_gate.h,
// aimed explicitly at the graphics queue under its existing queue lock.
// Does not assume that graphics aliases a numbered compute queue.
class GraphicsTimelineGate final {
 public:
  explicit GraphicsTimelineGate(vk::VulkanDevice &device)
      : device_(device.vk_device()) {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore_ = vkapi::create_semaphore(device_, 0, &type);
    fence_ = vkapi::create_fence(device_, 0);
    const uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo values{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &value;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    wait.pNext = &values;
    wait.waitSemaphoreCount = 1;
    wait.pWaitSemaphores = &semaphore_->semaphore;
    wait.pWaitDstStageMask = &stage;
    auto lock = device.acquire_external_graphics_queue_lock();
    if (device_.functions().vkQueueSubmit(device.graphics_queue(), 1, &wait,
                                        fence_->fence) != VK_SUCCESS)
      throw std::runtime_error("Cannot enqueue graphics test timeline gate");
  }

  ~GraphicsTimelineGate() {
    release();
    EXPECT_EQ(device_.functions().vkWaitForFences(
                  device_, 1, &fence_->fence, VK_TRUE, UINT64_MAX), VK_SUCCESS);
  }

  void release() {
    if (released_)
      return;
    VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signal.semaphore = semaphore_->semaphore;
    signal.value = 1;
    const auto result = device_.functions().vkSignalSemaphore(device_, &signal);
    EXPECT_EQ(result, VK_SUCCESS);
    released_ = result == VK_SUCCESS;
  }

 private:
  vkapi::DeviceHandle device_;
  vkapi::IVkSemaphore semaphore_;
  vkapi::IVkFence fence_;
  bool released_{false};
};

struct CompletionCounts {
  int queries{0};
  int waits{0};
  int destroyed{0};
};

// Instrument only Gui's calls. Both readiness and wait delegate to the actual
// returned Vulkan completion; this is not a synthetic ready/pending result.
class ObservedCompletion final : public StreamSemaphoreObject {
 public:
  ObservedCompletion(StreamSemaphore proof, CompletionCounts &counts)
      : proof_(std::move(proof)), counts_(counts) {}
  ~ObservedCompletion() override { ++counts_.destroyed; }
  bool is_ready() const override {
    ++counts_.queries;
    return proof_->is_ready();
  }
  bool wait() const override {
    ++counts_.waits;
    return proof_->wait();
  }
 private:
  StreamSemaphore proof_;
  CompletionCounts &counts_;
};

struct Recording {
  std::unique_ptr<CommandList> commands;
  Gui::DrawReservation reservation;
};

class GuiHarness final {
 public:
  GuiHarness() : program_(Arch::vulkan) {
    program_.materialize_runtime();
    ui::AppConfig config{};
    config.name = "native Gui reader lifecycle";
    config.width = 320;
    config.height = 240;
    config.show_window = false;
    config.ti_arch = Arch::vulkan;
    config.ggui_arch = Arch::vulkan;
    layout_.update_dimensions(320, 240, 320, 240);
    context_.set_window_layout(&layout_);
    context_.init_with_vulkan(&program_, nullptr, config);
    swap_.init(&context_);
    gui = std::make_unique<Gui>(&context_, &swap_, nullptr, &layout_);

    vk::VulkanRenderPassDesc desc;
    const auto [result, format] =
        vk::buffer_format_ti_to_vk(swap_.surface().image_format());
    if (result != RhiResult::success)
      throw std::runtime_error("Unsupported GUI test surface format");
    desc.color_attachments.emplace_back(format, true);
    desc.depth_attachment = std::get<2>(device().get_vk_image(
        swap_.depth_allocation()));
    desc.clear_depth = true;
    render_pass_ = device().get_renderpass(desc)->renderpass;
    // The direct ledger qualification owns two draw slots; it does not select
    // a Canvas display pool or test WSI acquisition.
    gui->prepare_render_resources(render_pass_, 2);
  }

  vk::VulkanDevice &device() {
    return static_cast<vk::VulkanDevice &>(context_.device());
  }
  Stream *stream() { return device().get_graphics_stream(); }

  void widgets() {
    gui->prepare_for_next_frame();
    gui->begin("exact reader", .05f, .05f, .8f, .8f);
    gui->text("REAL FONT / DRAW / GPU READER", glm::vec3(1, 0, 0));
    gui->end();
  }

  Recording record() {
    widgets();
    const auto reservation = gui->reserve_draw_slot(false);
    if (!reservation)
      throw std::runtime_error("GUI test draw slot unexpectedly busy");
    auto surface = swap_.surface().acquire_surface_image();
    if (surface.image.device == nullptr || surface.image_available)
      throw std::runtime_error("GUI test requires a hidden non-WSI surface");
    auto [commands, result] = stream()->new_command_list_unique();
    if (result != RhiResult::success || !commands)
      throw std::runtime_error("GUI test command allocation failed");
    auto *native = static_cast<vk::VulkanCommandList *>(commands.get());
    native->graphics_memory_barrier(true);
    auto depth = swap_.depth_allocation();
    bool clear = true;
    std::vector<float> color{0, 0, 0, 1};
    commands->begin_renderpass(0, 0, 320, 240, 1, &surface.image,
                              &clear, &color, &depth, true);
    if (native->current_renderpass()->renderpass != render_pass_)
      throw std::runtime_error("GUI test renderpass differs from cold preparation");
    if (!gui->draw(commands.get(), *reservation))
      throw std::runtime_error("GUI test recorded no draw");
    commands->end_renderpass();
    gui->validate_recorded_draw(*reservation);
    return {std::move(commands), *reservation};
  }

  StreamSemaphore submit(Recording &recording) {
    gui->validate_recorded_draw(recording.reservation);
    gui->mark_draw_submitting(recording.reservation);
    auto proof = stream()->submit(recording.commands.get());
    if (!proof)
      throw std::runtime_error("GUI test submit returned no native proof");
    return proof;
  }

  void discard(Recording &recording) {
    // A recorded command must be gone before returning its GUI buffers to idle.
    recording.commands.reset();
    gui->cancel_recorded_draw(recording.reservation);
  }

 private:
  // Reverse destruction keeps Program/device alive through Gui and attachments.
  Program program_;
  ui::WindowLayoutState layout_;
  ui::vulkan::AppContext context_;
  ui::vulkan::SwapChain swap_;
  VkRenderPass render_pass_{VK_NULL_HANDLE};
 public:
  std::unique_ptr<Gui> gui;
};

TEST(GuiReaderLifecycle, KnownUnsubmittedRecordingCancelsAndReusesSlot) {
  if (!vk::is_vulkan_api_available())
    GTEST_SKIP();
  GuiHarness h;
  auto first = h.record();
  const auto old = first.reservation;
  h.discard(first);
  EXPECT_ANY_THROW(h.gui->validate_recorded_draw(old));
  EXPECT_NO_THROW(h.gui->validate_retirement());
  EXPECT_FALSE(h.gui->retirement_failed());

  auto second = h.record();
  EXPECT_EQ(second.reservation.slot, old.slot);
  EXPECT_EQ(second.reservation.generation, old.generation);
  EXPECT_GT(second.reservation.serial, old.serial);
  EXPECT_ANY_THROW(h.gui->validate_recorded_draw(old));
  EXPECT_NO_THROW(h.gui->validate_recorded_draw(second.reservation));
  h.discard(second);
  EXPECT_NO_THROW(h.gui->retire_draw_readers());
}

TEST(GuiReaderLifecycle, ReturnedProofRetainsExactReadersUntilRetirement) {
  if (!vk::is_vulkan_api_available())
    GTEST_SKIP();
  CompletionCounts counts;
  GuiHarness h;
  if (!h.device().vk_caps().timeline_semaphore)
    GTEST_SKIP();
  GraphicsTimelineGate gate(h.device());
  auto first = h.record();
  auto proof_a = h.submit(first);
  auto observed_a = std::make_shared<ObservedCompletion>(proof_a, counts);
  std::weak_ptr<StreamSemaphoreObject> weak_a = observed_a;
  h.gui->seal_draw(first.reservation, observed_a);
  observed_a.reset();

  auto second = h.record();
  auto proof_b = h.submit(second);
  auto observed_b = std::make_shared<ObservedCompletion>(proof_b, counts);
  std::weak_ptr<StreamSemaphoreObject> weak_b = observed_b;
  h.gui->seal_draw(second.reservation, observed_b);
  observed_b.reset();
  EXPECT_NE(first.reservation.slot, second.reservation.slot);
  ASSERT_FALSE(proof_a->is_ready());
  ASSERT_FALSE(proof_b->is_ready());
  EXPECT_FALSE(weak_a.expired());
  EXPECT_FALSE(weak_b.expired());

  h.widgets();
  const int before_queries = counts.queries;
  EXPECT_FALSE(h.gui->reserve_draw_slot(false).has_value());
  EXPECT_EQ(counts.queries - before_queries, 2);
  EXPECT_EQ(counts.waits, 0);
  EXPECT_EQ(counts.destroyed, 0);
  EXPECT_NO_THROW(h.gui->validate_retirement());

  gate.release();
  EXPECT_NO_THROW(h.gui->retire_draw_readers());
  EXPECT_EQ(counts.waits, 2);
  EXPECT_EQ(counts.destroyed, 2);
  EXPECT_TRUE(weak_a.expired());
  EXPECT_TRUE(weak_b.expired());
  EXPECT_TRUE(proof_a->is_ready());
  EXPECT_TRUE(proof_b->is_ready());
  first.commands.reset();
  second.commands.reset();
  auto third = h.record();
  h.discard(third);
  EXPECT_NO_THROW(h.gui->retire_draw_readers());
}

// One intentional process-lifetime root, only for the dedicated unknown-state
// case. It includes Program/device. No pending queue work remains when assigned.
// Normal main return still runs all ordinary CRT/static exit handlers.
GuiHarness *unknown_gui_owner = nullptr;

TEST(GuiReaderLifecycle, GuiWithoutProofRejectsReuseAndCleanup) {
  if (!vk::is_vulkan_api_available())
    GTEST_SKIP();
  ASSERT_EQ(unknown_gui_owner, nullptr);
  auto owner = std::make_unique<GuiHarness>();
  if (!owner->device().vk_caps().timeline_semaphore)
    GTEST_SKIP();
  // Register the bounded root before entering the intentionally unsealed state.
  // Even a failed expectation cannot accidentally tear down its Program.
  unknown_gui_owner = owner.release();
  auto *h = unknown_gui_owner;
  auto gate = std::make_unique<GraphicsTimelineGate>(h->device());
  auto recording = h->record();
  // The harness receives and owns a REAL proof. Intentionally withhold it from
  // Gui: this qualifies missing-proof ledger state, not a driver submit fault.
  auto harness_proof = h->submit(recording);
  EXPECT_FALSE(harness_proof->is_ready());
  EXPECT_ANY_THROW(h->gui->validate_retirement());
  h->gui->cancel_recorded_draw(recording.reservation);
  EXPECT_ANY_THROW(h->gui->validate_retirement());
  h->gui->quarantine_draw();
  EXPECT_TRUE(h->gui->retirement_failed());
  EXPECT_ANY_THROW(h->gui->prepare_for_next_frame());
  EXPECT_ANY_THROW(h->gui->reserve_draw_slot(false));
  EXPECT_ANY_THROW(h->gui->retire_draw_readers());
  EXPECT_ANY_THROW(h->gui->cleanup_render_resources());

  // Never leave the actual queue blocked just to test the conservative ledger.
  gate->release();
  const bool retired = harness_proof->wait();
  EXPECT_TRUE(retired);
  gate.reset();
  recording.commands.reset();
  EXPECT_TRUE(h->gui->retirement_failed());
  EXPECT_ANY_THROW(h->gui->cleanup_render_resources());
  // The bounded raw root stays registered. Do not destroy Program/device under
  // resources whose Gui owner still rejects its own cleanup.
}

}  // namespace
}  // namespace taichi::lang
