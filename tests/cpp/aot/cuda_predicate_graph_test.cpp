#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/predicate_graph.h"

namespace taichi::lang {
namespace {

TEST(CudaPredicateGraphContract, RejectsInvalidAndUnrepresentableLayouts) {
  cuda::NestedPredicateGraphControls controls;
  cuda::NestedPredicateGraphLayout layout;
  EXPECT_EQ(controls.initialize(layout, false), CUDA_ERROR_NOT_SUPPORTED);
  layout = {4, 8, 1, 1, 4, {{12, 16, 3, 1, 2, 3}}};
  auto invalid = layout;
  invalid.inners[0].end = 5;
  EXPECT_EQ(controls.initialize(invalid, false), CUDA_ERROR_NOT_SUPPORTED);
  invalid = layout;
  invalid.inners[0].iterations = std::uint64_t{1} << 32;
  EXPECT_EQ(controls.initialize(invalid, false), CUDA_ERROR_NOT_SUPPORTED);
  invalid = layout;
  invalid.iterations = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(controls.initialize(invalid, false), CUDA_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(controls.capture(nullptr, {}), CUDA_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(controls.finalize_nodes(), CUDA_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(controls.upload(), CUDA_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(controls.device_bytes(), 0);
  controls.release(false);
  controls.release(false);
}

class CudaPredicateGraph : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &driver = CUDADriver::get_instance_without_context();
    if (!driver.nvidia_extensions_available() ||
        !driver.launch_kernel_ex.available() || !driver.graph_upload.available())
      GTEST_SKIP();
    CUDAContext::get_instance().make_current();
    std::uint32_t error = CUDA_SUCCESS;
    if (!cuda::driver_graph_prepare_bounded_update(&error))
      GTEST_SKIP() << "predicate updater unavailable: " << error;
    const auto qualification = cuda::driver_graph_bounded_probe(true);
    if (!qualification.passed)
      GTEST_SKIP() << qualification.reason;
    driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    const char *ptx = R"ptx(
.version 6.0
.target sm_50
.address_size 64
.visible .entry dispatch(.param .u64 data, .param .u32 operation) {
  .reg .b64 %p;
  .reg .b32 %op, %count, %limit, %pred, %total;
  .reg .pred %test;
  ld.param.u64 %p, [data];
  ld.param.u32 %op, [operation];
  setp.eq.u32 %test, %op, 0;
  @%test bra INITIAL;
  setp.eq.u32 %test, %op, 1;
  @%test bra INNER_INITIAL;
  setp.eq.u32 %test, %op, 2;
  @%test bra INNER_STEP;
  ld.global.u32 %count, [%p];
  add.u32 %count, %count, 1;
  st.global.u32 [%p], %count;
  bra OUTER_CONDITION;
INITIAL:
  mov.u32 %count, 0;
  st.global.u32 [%p], %count;
  st.global.u32 [%p+24], %count;
OUTER_CONDITION:
  ld.global.u32 %limit, [%p+16];
  setp.lt.u32 %test, %count, %limit;
  selp.u32 %pred, 1, 0, %test;
  st.global.u32 [%p+4], %pred;
  ret;
INNER_INITIAL:
  mov.u32 %count, 0;
  st.global.u32 [%p+8], %count;
  bra INNER_CONDITION;
INNER_STEP:
  ld.global.u32 %count, [%p+8];
  add.u32 %count, %count, 1;
  st.global.u32 [%p+8], %count;
  ld.global.u32 %total, [%p+24];
  add.u32 %total, %total, 1;
  st.global.u32 [%p+24], %total;
INNER_CONDITION:
  ld.global.u32 %limit, [%p+20];
  setp.lt.u32 %test, %count, %limit;
  selp.u32 %pred, 1, 0, %test;
  st.global.u32 [%p+12], %pred;
  ret;
}
)ptx";
    driver.module_load_data_ex(&module, ptx, 0, nullptr, nullptr);
    driver.module_get_function(&function, module, "dispatch");
  }
  void TearDown() override {
    if (!stream)
      return;
    auto &driver = CUDADriver::get_instance();
    driver.stream_synchronize(stream);
    if (module)
      driver.module_unload(module);
    driver.stream_destroy(stream);
  }

  struct Frame {
    CudaPredicateGraph &owner;
    cuda::NestedPredicateGraphControls controls;
    void *data{nullptr};
    CUgraph graph{nullptr};
    CUgraphExec executable{nullptr};
    std::uint64_t outer_budget, inner_budget;
    Frame(CudaPredicateGraph &owner, std::uint64_t outer, std::uint64_t inner)
        : owner(owner), outer_budget(outer), inner_budget(inner) {
      auto &driver = CUDADriver::get_instance();
      driver.malloc(&data, 10 * sizeof(std::uint32_t));
      std::array<std::uint32_t, 10> sentinel;
      sentinel.fill(99);
      driver.memcpy_host_to_device(data, sentinel.data(), sizeof(sentinel));
    }
    ~Frame() {
      reset_graph();
      CUDADriver::get_instance().mem_free(data);
    }
    void reset_graph() {
      auto &driver = CUDADriver::get_instance();
      driver.stream_synchronize(owner.stream);
      if (executable)
        driver.graph_exec_destroy(executable);
      if (graph)
        driver.graph_destroy(graph);
      executable = nullptr;
      graph = nullptr;
      controls.release(true);
    }
    std::uintptr_t at(std::size_t word) {
      return reinterpret_cast<std::uintptr_t>(data) + word * sizeof(std::uint32_t);
    }
    std::uint32_t prepare(bool fail_payload = false) {
      cuda::NestedPredicateGraphLayout layout{
          at(1), at(8), outer_budget, 1, 4,
          {{at(3), at(9), inner_budget, 1, 2, 3}}};
      auto error = controls.initialize(std::move(layout), true);
      if (error != CUDA_SUCCESS)
        return error;
      auto &driver = CUDADriver::get_instance();
      driver.stream_begin_capture(owner.stream, CU_STREAM_CAPTURE_MODE_RELAXED);
      error = controls.capture(
          owner.stream, [&](std::size_t index, std::vector<void *> *nodes) {
            if (fail_payload && index == 2)
              return static_cast<std::uint32_t>(CUDA_ERROR_NOT_SUPPORTED);
            std::uint32_t operation = static_cast<std::uint32_t>(index);
            void *arguments[] = {&data, &operation};
            TaichiCudaLaunchAttribute attribute{};
            attribute.id = TAICHI_CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
            attribute.value.device_updatable_kernel_node.device_updatable = 1;
            TaichiCudaLaunchConfig config{};
            config.grid_dim_x = config.grid_dim_y = config.grid_dim_z = 1;
            config.block_dim_x = config.block_dim_y = config.block_dim_z = 1;
            config.stream = owner.stream;
            config.attributes = nodes ? &attribute : nullptr;
            config.num_attributes = nodes ? 1 : 0;
            auto result = driver.launch_kernel_ex.call(
                &config, owner.function, arguments, nullptr);
            if (result == CUDA_SUCCESS && nodes)
              nodes->push_back(
                  attribute.value.device_updatable_kernel_node.device_node);
            return result;
          });
      const auto end_error = driver.stream_end_capture.call(owner.stream, &graph);
      if (error != CUDA_SUCCESS)
        return error;
      if (end_error != CUDA_SUCCESS)
        return end_error;
      error = controls.finalize_nodes();
      if (error != CUDA_SUCCESS)
        return error;
      driver.graph_instantiate_with_flags(&executable, graph, 0);
      driver.graph_upload(executable, owner.stream);
      driver.stream_synchronize(owner.stream);
      return controls.upload();
    }
    std::array<std::uint32_t, 10> read() {
      std::array<std::uint32_t, 10> result;
      CUDADriver::get_instance().memcpy_device_to_host(
          result.data(), data, sizeof(result));
      return result;
    }
    void run(std::uint32_t outer, std::uint32_t inner) {
      auto &driver = CUDADriver::get_instance();
      std::array<std::uint32_t, 2> limits{outer, inner};
      driver.memcpy_host_to_device(reinterpret_cast<void *>(at(4)), limits.data(),
                                   sizeof(limits));
      driver.graph_launch(executable, owner.stream);
      driver.stream_synchronize(owner.stream);
      const auto result = read();
      EXPECT_EQ(result[0], std::min<std::uint64_t>(outer, outer_budget));
      EXPECT_EQ(result[6], std::min<std::uint64_t>(outer, outer_budget) *
                               std::min<std::uint64_t>(inner, inner_budget));
    }
  };
  void *stream{nullptr}, *module{nullptr}, *function{nullptr};
};

TEST_F(CudaPredicateGraph, PreservesBudgetsBindingsAndPrepareOnly) {
  RecordProperty("host_owner_bytes",
                 static_cast<int>(sizeof(cuda::NestedPredicateGraphControls)));
  RecordProperty(
      "legacy_host_fields_bytes",
      static_cast<int>(2 * sizeof(void *) +
                       sizeof(std::vector<cuda::CudaGraphPredicateGroupControl>) +
                       sizeof(std::vector<std::uintptr_t>)));
  auto &context = CUDAContext::get_instance();
  auto guard = context.get_guard();
  auto lock = context.get_submission_lock_guard();
  Frame second(*this, 4, 3);
  ASSERT_EQ(second.prepare(), CUDA_SUCCESS);
  for (const auto shape : {std::array<std::uint64_t, 2>{4, 3}, {0, 3},
                           {4, 0}, {65, 1}}) {
    Frame first(*this, shape[0], shape[1]);
    ASSERT_EQ(first.prepare(), CUDA_SUCCESS);
    for (auto value : first.read())
      EXPECT_EQ(value, 99);
    for (const auto limits : {std::array<std::uint32_t, 2>{0, 3}, {2, 1},
                              {7, 7}, {1, 0}, {0, 2}}) {
      first.run(limits[0], limits[1]);
      second.run(limits[1], limits[0]);
    }
  }
  second.run(3, 2);  // Other owners have retired their controls and executables.
}

TEST_F(CudaPredicateGraph, FailedCaptureCanBeRetiredAndRebuilt) {
  auto &context = CUDAContext::get_instance();
  auto guard = context.get_guard();
  auto lock = context.get_submission_lock_guard();
  Frame frame(*this, 2, 3);
  EXPECT_EQ(frame.prepare(true), CUDA_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(frame.controls.finalize_nodes(), CUDA_ERROR_NOT_SUPPORTED);
  frame.reset_graph();
  EXPECT_EQ(frame.controls.device_bytes(), 0);
  ASSERT_EQ(frame.prepare(), CUDA_SUCCESS);
  frame.run(2, 2);
  frame.run(0, 0);
}

}  // namespace
}  // namespace taichi::lang
#endif
