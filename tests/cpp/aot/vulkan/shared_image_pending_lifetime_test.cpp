#include <gtest/gtest.h>

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/taichi_vulkan_impl.h"
#include "c_api/src/native_resource_registry.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/native_image.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <future>

namespace taichi::lang {
namespace {
using namespace std::chrono_literals;

void require_success() {
  char message[1024]{};
  uint64_t size = sizeof(message);
  if (ti_get_last_error(&size, message) != TI_ERROR_SUCCESS)
    throw std::runtime_error(message);
}

struct BorrowScope {
  explicit BorrowScope(AotRuntimeClient &client) : client(client) {
    client.enter();
    ti_set_last_error(TI_ERROR_SUCCESS, "");
  }
  ~BorrowScope() {
    client.leave();
  }
  AotRuntimeClient &client;
};

// TimelineGate's DeviceHandle owns function tables, not VkDevice. Retire these
// test-only semaphore/fence objects in Program's existing RuntimeClient phase,
// after domain draining and before backend destruction. release and retire
// share a test mutex so the final signal cannot race gate destruction.
class GateOwner final : public RuntimeClient {
 public:
  std::array<std::unique_ptr<test_detail::TimelineGate>, 2> gates;
  void release(std::size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gates[index])
      gates[index]->release();
  }
  void release_all() {
    release(0);
    release(1);
  }
  void retire() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &gate : gates)
      gate.reset();
  }

 private:
  std::mutex mutex_;
};

// Failed assertions/exceptions must release the controlled GPU dependency
// before std::future joins the cold close/finalize thread.
class PendingAction {
 public:
  PendingAction(std::shared_ptr<GateOwner> gates, std::function<void()> action)
      : gates_(std::move(gates)),
        result_(
            std::async(std::launch::async, [this, action = std::move(action)] {
              started_.set_value();
              action();
            })) {
    started_.get_future().wait();
  }
  ~PendingAction() {
    gates_->release_all();
    if (result_.valid())
      result_.wait();
  }
  std::future_status status(std::chrono::milliseconds timeout) {
    return result_.wait_for(timeout);
  }
  void finish() {
    result_.get();
  }

 private:
  std::shared_ptr<GateOwner> gates_;
  std::promise<void> started_;
  std::future<void> result_;
};

struct BorrowedImageSession {
  Program program{Arch::vulkan};
  std::shared_ptr<AotRuntimeClient> borrower;
  std::shared_ptr<GateOwner> gates = std::make_shared<GateOwner>();
  TiRuntime runtime{nullptr};
  TiAotModule module{nullptr};
  Texture *texture{nullptr};
  Texture *peer_texture{nullptr};
  DeviceAllocation image{kDeviceNullAllocation};
  std::weak_ptr<NativeSampledImage> observed_image;
  std::array<TiVulkanExecutionDomain, 2> domains{};
  std::array<TiPreparedExecution, 2> executions{};
  std::array<TiCompletion, 2> completions{};
  std::array<TiMemory, 2> outputs{};

  BorrowedImageSession() {
    program.materialize_runtime();
  }
  ~BorrowedImageSession() {
    gates->release_all();
    // Program owns the borrowed Runtime and its modules/resources/recordings.
    // No C API runtime pointer is dereferenced after Program finalization.
    program.finalize();
    for (auto completion : completions)
      if (completion)
        ti_release_completion_ext(completion);
    for (auto domain : domains)
      if (domain)
        ti_release_vulkan_execution_domain_ext(domain);
  }
  vulkan::VulkanDevice &device() {
    return static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  }
  gfx::GfxRuntime &gfx() {
    return *program.get_gfx_runtime_for_borrowing();
  }

  void initialize(const char *fixture, bool exclusive = false) {
    borrower = AotRuntimeClient::create(program);
    runtime = reinterpret_cast<TiRuntime>(borrower->handle());
    texture = program.create_texture(BufferFormat::r32f, {2, 2});
    image = texture->get_device_allocation();
    auto [upload, result] = device().allocate_memory_unique(
        {16, false, false, false, AllocUsage::Storage});
    if (result != RhiResult::success)
      throw std::runtime_error("test upload allocation failed");
    BufferImageCopyParams copy;
    copy.buffer_row_length = copy.buffer_image_height = 2;
    copy.image_extent = {2, 2, 1};
    {
      auto submission = program.acquire_runtime_resource_submission_guard();
      gfx().enqueue_compute_op_lambda(
          [target = image, source = upload->get_ptr(), copy](
              Device *, CommandList *commands) {
            commands->buffer_fill(source, 16, 0x40400000u);  // four 3.0f texels
            commands->buffer_barrier(source);
            commands->buffer_to_image(target, source, ImageLayout::transfer_dst,
                                      copy);
          },
          {{image, ImageLayout::transfer_dst, ImageLayout::transfer_dst}});
      gfx().synchronize();
    }
    const auto imported =
        reinterpret_cast<TiNativeResource>(borrower->import_texture(texture));
    auto peer_imported = imported;
    if (exclusive) {
      peer_texture = program.create_texture(BufferFormat::r32f, {2, 2});
      peer_imported = reinterpret_cast<TiNativeResource>(borrower->import_texture(peer_texture));
    }
    BorrowScope scope(*borrower);
    auto &impl = *reinterpret_cast<::Runtime *>(runtime);
    observed_image = std::dynamic_pointer_cast<NativeSampledImage>(
        impl.native_resource_registry().get(imported));
    module = ti_load_aot_module(runtime, fixture);
    require_success();
    if (!module)
      throw std::runtime_error("missing texture AOT fixture");
    // Existing texture_aot_test_.py --arch vulkan (without --graph) exports
    // run2, which samples its two inputs and writes their sum to an ndarray.
    const std::string plan = std::string(R"({"version":26,"operations":[)") +
        (exclusive ? R"({"kind":"kernel","entry":"write2","arguments":["image"]},)" : "") +
        R"({"kind":"kernel","entry":"run2","arguments":["image","image","output"]}]})";
    const TiVulkanExecutionDomainCreateInfo options{2, 1, TI_FALSE};
    for (std::size_t i = 0; i < domains.size(); ++i) {
      domains[i] = ti_create_vulkan_execution_domain_ext(runtime, &options);
      require_success();
      const TiMemoryAllocateInfo memory{16, TI_FALSE, TI_TRUE, TI_FALSE,
                                        TI_MEMORY_USAGE_STORAGE_BIT};
      outputs[i] = ti_allocate_memory(runtime, &memory);
      require_success();
      std::array<TiNamedArgument, 2> args{};
      args[0].name = "image";
      args[0].argument.type = TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT;
      args[0].argument.value.native_resource = i ? peer_imported : imported;
      args[1].name = "output";
      args[1].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
      auto &array = args[1].argument.value.ndarray;
      array.memory = outputs[i];
      array.shape.dim_count = 2;
      array.shape.dims[0] = array.shape.dims[1] = 2;
      array.elem_type = TI_DATA_TYPE_F32;
      executions[i] = ti_prepare_execution_plan_on_vulkan_domain_ext(
          runtime, module, domains[i], plan.size(), plan.data(), args.size(),
          args.data(), 0, nullptr, 0, nullptr);
      require_success();
      ti_initialize_prepared_ext(runtime, executions[i]);
      require_success();
    }
    program.retain_runtime_client(gates);
  }

  void submit_pending(bool release_handles = true) {
    BorrowScope scope(*borrower);
    for (std::size_t i = 0; i < domains.size(); ++i) {
      TiVulkanExecutionDomainInfo info{};
      ti_get_vulkan_execution_domain_info_ext(domains[i], &info);
      require_success();
      gates->gates[i] = std::make_unique<test_detail::TimelineGate>(
          device(), info.queue_index);
      if (!ti_try_submit_prepared_ext(runtime, executions[i], 0, nullptr,
                                      &completions[i])) {
        require_success();
        throw std::runtime_error("test domain unexpectedly full");
      }
      require_success();
      // Host-only release must not release the in-flight read loan.
      if (release_handles) {
        ti_destroy_prepared_ext(runtime, executions[i]);
        executions[i] = nullptr;
        require_success();
      }
    }
  }

  void expect_pixels(std::size_t index) {
    BorrowScope scope(*borrower);
    const auto allocation = reinterpret_cast<::Runtime *>(runtime)
                                ->get_memory(outputs[index])
                                .allocation;
    void *mapped = nullptr;
    ASSERT_EQ(device().map(allocation, &mapped), RhiResult::success);
    for (int i = 0; i < 4; ++i)
      EXPECT_FLOAT_EQ(static_cast<const float *>(mapped)[i], 6.0f);
    device().unmap(allocation);
  }
};

const char *texture_fixture(bool exclusive = false) {
  return std::getenv(exclusive ? "TAICHI_VULKAN_WRITABLE_TEXTURE_AOT_DIR"
                              : "TAICHI_VULKAN_SHARED_TEXTURE_AOT_DIR");
}
}  // namespace

void check_pending_close(bool instance_only, bool exclusive = false) {
  if (!vulkan::is_vulkan_api_available() || !texture_fixture(exclusive))
    GTEST_SKIP()
        << "Requires Vulkan and the existing texture_aot_test_ fixture";
  BorrowedImageSession session;
  if (!session.device().vk_caps().timeline_semaphore ||
      session.device().created_compute_queue_count() < 3)
    GTEST_SKIP();
  session.initialize(texture_fixture(exclusive), exclusive);
  session.submit_pending(!instance_only);
  TiVulkanExecutionInstance instance = nullptr;
  if (instance_only) {
    BorrowScope scope(*session.borrower);
    instance = ti_get_prepared_vulkan_instance_ext(session.runtime,
                                                   session.executions[0]);
    require_success();
    ASSERT_NE(instance, nullptr);
  }
  EXPECT_EQ(ti_query_completion_ext(session.completions[0]), TI_FALSE);
  EXPECT_EQ(ti_query_completion_ext(session.completions[1]), TI_FALSE);
  PendingAction close(session.gates, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, "");
    if (instance_only)
      ti_retire_vulkan_execution_instance_ext(instance);
    else
      ti_close_vulkan_execution_domain_ext(session.domains[0]);
    require_success();
  });
  EXPECT_EQ(close.status(100ms), std::future_status::timeout);
  EXPECT_FALSE(session.observed_image.expired());
  EXPECT_THROW(
      session.gfx().transition_image(session.image, ImageLayout::transfer_dst),
      std::invalid_argument);
  session.gates->release(0);
  const auto first_closed = close.status(2s);
  EXPECT_EQ(first_closed, std::future_status::ready)
      << "closing one reader must not wait for the independently gated peer";
  if (first_closed != std::future_status::ready)
    session.gates->release(1);  // bounded cleanup of a failed isolation check
  EXPECT_NO_THROW(close.finish());
  if (instance_only) {
    BorrowScope scope(*session.borrower);
    ti_finish_prepared_instance_retirement_ext(session.runtime,
                                               session.executions[0]);
    require_success();
    session.executions[0] = nullptr;
    ti_release_vulkan_execution_instance_ext(instance);
  }
  EXPECT_EQ(ti_query_completion_ext(session.completions[0]), TI_TRUE);
  EXPECT_EQ(ti_query_completion_ext(session.completions[1]), TI_FALSE);
  session.expect_pixels(0);
  if (exclusive) {
    EXPECT_NO_THROW(session.gfx().transition_image(session.image, ImageLayout::shader_read));
    EXPECT_THROW(session.gfx().transition_image(session.peer_texture->get_device_allocation(),
                                                ImageLayout::shader_read),
                 std::invalid_argument);
  } else {
    EXPECT_THROW(
        session.gfx().transition_image(session.image, ImageLayout::transfer_dst),
        std::invalid_argument);
  }
  session.gates->release(1);
  ti_wait_completion_ext(session.completions[1]);
  require_success();
  session.expect_pixels(1);
  ti_close_vulkan_execution_domain_ext(session.domains[1]);
  require_success();
  EXPECT_NO_THROW(
      session.gfx().transition_image(session.image, ImageLayout::transfer_dst));
  session.gfx().synchronize();
}

TEST(VulkanSharedImageLifetime, PendingCloseRetiresOnlyOneBorrowedReader) {
  check_pending_close(false);
}

TEST(VulkanSharedImageLifetime, PendingInstanceCloseCollectsItsQueueOperands) {
  check_pending_close(true);
}

TEST(VulkanSharedImageLifetime, PendingWriterCloseRetiresOnlyItsImage) {
  check_pending_close(true, true);
}

void check_reset_after_view_deletion(bool exclusive) {
  if (!vulkan::is_vulkan_api_available() || !texture_fixture(exclusive))
    GTEST_SKIP()
        << "Requires Vulkan and the existing texture_aot_test_ fixture";
  BorrowedImageSession session;
  if (!session.device().vk_caps().timeline_semaphore ||
      session.device().created_compute_queue_count() < 3)
    GTEST_SKIP();
  session.initialize(texture_fixture(exclusive), exclusive);
  session.submit_pending();
  EXPECT_EQ(ti_query_completion_ext(session.completions[0]), TI_FALSE);
  EXPECT_EQ(ti_query_completion_ext(session.completions[1]), TI_FALSE);
  session.program.delete_texture(session.texture);
  session.texture = nullptr;
  if (session.peer_texture) {
    session.program.delete_texture(session.peer_texture);
    session.peer_texture = nullptr;
  }
  EXPECT_FALSE(session.observed_image.expired());
  auto owner = session.program.runtime_fault_domain();
  PendingAction reset(session.gates, [&] { session.program.finalize(); });
  EXPECT_EQ(reset.status(100ms), std::future_status::timeout);
  EXPECT_FALSE(session.observed_image.expired());
  session.gates->release(0);
  // Whole-Program retirement drains every domain prefix before invoking any
  // resource callbacks. One released reader must therefore not close either
  // domain while its peer is still pending. Query host state only: external
  // completion backend queries are not allowed during finalization.
  TiVulkanExecutionDomainInfo first{};
  ti_get_vulkan_execution_domain_info_ext(session.domains[0], &first);
  require_success();
  EXPECT_FALSE(first.closed);
  TiVulkanExecutionDomainInfo second{};
  ti_get_vulkan_execution_domain_info_ext(session.domains[1], &second);
  require_success();
  EXPECT_FALSE(second.closed);
  EXPECT_EQ(reset.status(100ms), std::future_status::timeout);
  EXPECT_FALSE(session.observed_image.expired());
  session.gates->release(1);
  EXPECT_EQ(reset.status(2s), std::future_status::ready);
  EXPECT_NO_THROW(reset.finish());
  EXPECT_TRUE(session.borrower->closed());
  EXPECT_TRUE(session.observed_image.expired());
  EXPECT_EQ(owner->state(), RuntimeLifecycleState::kFinalized);
  for (auto completion : session.completions) {
    EXPECT_EQ(ti_query_completion_ext(completion), TI_TRUE);
    ti_wait_completion_ext(completion);
    require_success();
  }
}

TEST(VulkanSharedImageLifetime,
     ProgramResetDrainsBothReadersAfterSourceViewDeletion) {
  check_reset_after_view_deletion(false);
}

TEST(VulkanSharedImageLifetime,
     ProgramResetDrainsBothWritersAfterSourceViewDeletion) {
  check_reset_after_view_deletion(true);
}
}  // namespace taichi::lang
#endif
