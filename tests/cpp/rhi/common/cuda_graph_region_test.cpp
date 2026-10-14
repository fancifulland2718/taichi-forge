#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/graph_region_builder.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"

namespace taichi::lang {
namespace {
class CudaGraphRegion : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CUDADriver::get_instance_without_context()
             .nvidia_extensions_available())
      GTEST_SKIP();
    CUDAContext::get_instance().make_current();
    if (!cuda::GraphRegionBuilder::available())
      GTEST_SKIP();
    auto &driver = CUDADriver::get_instance();
    driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    driver.graph_create(&root, 0);
    driver.malloc(&data, 4 * sizeof(std::uint32_t));
    driver.malloc(reinterpret_cast<void **>(&control), sizeof(*control));
    const char *ptx = R"ptx(
.version 6.0
.target sm_50
.address_size 64
.visible .entry advance(.param .u64 address, .param .u32 increment) {
  .reg .b64 %p;
  .reg .b32 %r<5>;
  .reg .pred %test;
  ld.param.u64 %p, [address];
  ld.param.u32 %r0, [increment];
  ld.global.u32 %r1, [%p];
  add.u32 %r1, %r1, %r0;
  st.global.u32 [%p], %r1;
  ld.global.u32 %r2, [%p+12];
  setp.lt.u32 %test, %r1, %r2;
  selp.u32 %r3, 1, 0, %test;
  st.global.u32 [%p+4], %r3;
  ret;
}
)ptx";
    driver.module_load_data_ex(&module, ptx, 0, nullptr, nullptr);
    driver.module_get_function(&function, module, "advance");
    cuda::driver_graph_prepare_conditional_setter();
  }

  void TearDown() override {
    if (!stream)
      return;
    auto &driver = CUDADriver::get_instance();
    driver.stream_synchronize.call(stream);
    if (executable)
      driver.graph_exec_destroy.call(executable);
    if (root)
      driver.graph_destroy.call(root);
    if (module)
      driver.module_unload.call(module);
    if (control)
      driver.mem_free.call(control);
    if (data)
      driver.mem_free.call(data);
    driver.stream_destroy.call(stream);
  }
  void advance(void *stream, std::uint32_t increment) {
    void *args[] = {&data, &increment};
    CUDADriver::get_instance().launch_kernel(function, 1, 1, 1, 1, 1, 1, 0,
                                             stream, args, nullptr);
  }
  void *at(std::size_t index) {
    return static_cast<std::uint32_t *>(data) + index;
  }
  void write(std::uint32_t target, std::uint32_t budget) {
    auto &driver = CUDADriver::get_instance();
    std::array<std::uint32_t, 4> initial{99, 99, 99, target};
    driver.memcpy_host_to_device(data, initial.data(), sizeof(initial));
    cuda::CudaGraphConditionalControl description;
    description.predicate = reinterpret_cast<std::uintptr_t>(at(1));
    description.max_iterations = budget;
    driver.memcpy_host_to_device(control, &description, sizeof(description));
  }
  std::array<std::uint32_t, 4> read() {
    std::array<std::uint32_t, 4> result;
    CUDADriver::get_instance().memcpy_device_to_host(result.data(), data,
                                                     sizeof(result));
    return result;
  }
  void *stream{nullptr}, *module{nullptr}, *function{nullptr}, *data{nullptr};
  cuda::CudaGraphConditionalControl *control{nullptr};
  CUgraph root{nullptr};
  CUgraphExec executable{nullptr};
};

TEST_F(CudaGraphRegion, JoinsPrefixesAndReplaysDeviceWhileWithOneBody) {
  auto &cuda_context = CUDAContext::get_instance();
  auto guard = cuda_context.get_guard();
  auto lock = cuda_context.get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  write(7, 4);
  cuda::GraphRegionBuilder builder(root, stream);
  auto first = builder.capture(
      root, {}, [&](void *s) { driver.memsetd32_async(data, 0, 1, s); });
  auto second = builder.capture(
      root, {}, [&](void *s) { driver.memsetd32_async(at(2), 42, 1, s); });
  first.insert(first.end(), second.begin(), second.end());
  ASSERT_EQ(first.size(), 2);
  const auto handle = builder.conditional_handle();
  const auto entry = builder.capture(root, first, [&](void *s) {
    advance(s, 0);
    cuda::driver_graph_set_conditional(control, handle, s, true);
  });
  const auto loop = builder.add_while(root, entry, handle);
  builder.capture(loop.body, {}, [&](void *s) {
    advance(s, 1);
    cuda::driver_graph_set_conditional(control, handle, s);
  });
  builder.capture(root, {loop.node}, [&](void *s) {
    driver.memcpy_device_to_device_async(at(2), data, sizeof(std::uint32_t), s);
  });
  std::size_t body_nodes = 0;
  driver.graph_get_nodes(loop.body, nullptr, &body_nodes);
  ASSERT_EQ(body_nodes, 2);
  driver.graph_instantiate_with_flags(&executable, root, 0);
  EXPECT_EQ(read(), (std::array<std::uint32_t, 4>{99, 99, 99, 7}));
  // Compilation/assembly did no mathematical work. Destroy the source graph;
  // the executable retains its conditional structure until explicit retirement.
  driver.graph_destroy(root);
  root = nullptr;
  for (const auto [target, budget] :
       {std::pair{7u, 4u}, {2u, 4u}, {0u, 4u}, {5u, 0u}, {3u, 100000u}}) {
    write(target, budget);
    driver.graph_launch(executable, stream);
    driver.stream_synchronize(stream);
    const auto result = read();
    EXPECT_EQ(result[0], std::min(target, budget));
    EXPECT_EQ(result[2], result[0]);
  }
}

TEST_F(CudaGraphRegion, CallbackFailureEndsCaptureWithoutDestroyingOwner) {
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  cuda::GraphRegionBuilder builder(root, stream);
  EXPECT_THROW(builder.capture(root, {},
                               [&](void *s) {
                                 driver.memsetd32_async(data, 0, 1, s);
                                 throw std::runtime_error("recorder failure");
                               }),
               std::runtime_error);
  std::uint32_t status = 99;
  driver.stream_get_capture_info_v2(stream, &status, nullptr, nullptr, nullptr,
                                    nullptr);
  EXPECT_EQ(status, 0);
  std::size_t nodes = 0;
  EXPECT_EQ(driver.graph_get_nodes.call(root, nullptr, &nodes), CUDA_SUCCESS);
  EXPECT_EQ(nodes, 1);
  // Failed roots are caller-owned and discarded transactionally. Reuse the
  // stream for another capture to detect a leaked capture state.
  driver.graph_destroy(root);
  root = nullptr;
  driver.graph_create(&root, 0);
  cuda::GraphRegionBuilder fresh(root, stream);
  fresh.capture(root, {},
                [&](void *s) { driver.memsetd32_async(data, 8, 1, s); });
  driver.graph_instantiate_with_flags(&executable, root, 0);
  driver.graph_launch(executable, stream);
  driver.stream_synchronize(stream);
  EXPECT_EQ(read()[0], 8);
}
}  // namespace
}  // namespace taichi::lang
#endif
