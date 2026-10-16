#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/graph_region_builder.h"
#include "taichi/runtime/cuda/predicate_graph.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/rhi/cuda/cuda_device.h"

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

TEST(CudaPredicateGraphContract, MaskedRecordingRejectsBeforePayloadOrDriver) {
  std::size_t calls = 0;
  const auto payload = [&](std::size_t) { ++calls; };
  cuda::NestedPredicateGraphLayout layout{4, 8, 1, 1, 4,
                                         {{12, 16, 3, 1, 2, 3}}};
  auto invalid = layout;
  invalid.inners[0].gate = 0;
  EXPECT_THROW(cuda::capture_masked_while_graph(invalid, true, nullptr, payload),
               std::invalid_argument);
  invalid = layout;
  invalid.inners[0].body_begin = invalid.inners[0].condition_begin;
  EXPECT_THROW(cuda::capture_masked_while_graph(invalid, true, nullptr, payload),
               std::invalid_argument);
  invalid = layout;
  invalid.iterations = std::numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(cuda::capture_masked_while_graph(invalid, true, nullptr, payload),
               std::invalid_argument);
  EXPECT_THROW(cuda::capture_masked_while_graph(layout, true, nullptr, {}),
               std::invalid_argument);
  cuda::MaskedBranchGraphLayout branch{4, 8, 0, 3,
      std::numeric_limits<std::uint32_t>::max(), 3};
  EXPECT_THROW(cuda::capture_masked_branch_graph(branch, nullptr, payload),
               std::invalid_argument);
  branch.conditional_type = 2;
  branch.default_branch = 3;
  EXPECT_THROW(cuda::capture_masked_branch_graph(branch, nullptr, payload),
               std::invalid_argument);
  EXPECT_EQ(calls, 0);
}

TEST(CudaPredicateGraphContract, ZeroMaskedBudgetCapturesOnlyInitialCondition) {
  // A zero outer budget does not expand even a large inner budget or touch
  // device addresses. The initial condition still belongs to this region.
  cuda::NestedPredicateGraphLayout layout{
      4, 8, 0, 2, 6, {{12, 16, std::uint64_t{1} << 32, 2, 3, 5}}};
  std::vector<std::size_t> captured;
  cuda::capture_masked_while_graph(layout, true, nullptr,
      [&](std::size_t dispatch) { captured.push_back(dispatch); });
  EXPECT_EQ(captured, (std::vector<std::size_t>{0, 1}));
  layout.inners.clear();
  layout.initial_dispatches = 0;  // Flat while condition owned by caller.
  captured.clear();
  cuda::capture_masked_while_graph(layout, false, nullptr,
      [&](std::size_t dispatch) { captured.push_back(dispatch); });
  EXPECT_TRUE(captured.empty());
}

TEST(CudaPredicateGraphContract, FlatMaskedRecipeKeepsNestedValidation) {
  CudaControlRecipe<void *> recipe;
  EXPECT_THROW(recipe.validate(0, 1), std::invalid_argument);
  recipe.encoding = CudaControlEncoding::masked;
  for (const bool nonzero : {false, true}) {
    EXPECT_NO_THROW(recipe.validate(0, 1, nonzero));
    EXPECT_NO_THROW(recipe.validate(1, 2, nonzero));
    EXPECT_THROW(recipe.validate(1, 1, nonzero), std::invalid_argument);
  }
  recipe.inners.push_back({nullptr, 1, 2, 3, 2});
  EXPECT_NO_THROW(recipe.validate(1, 4));
  EXPECT_THROW(recipe.validate(0, 4), std::invalid_argument);
  EXPECT_THROW(recipe.validate(1, 4, false), std::invalid_argument);
}

TEST(CudaMaskedBranch, SignedSelectionDoesNotDependOnBranchCount) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto storage = device.allocate_memory_unique({8});
  ASSERT_EQ(storage.second, RhiResult::success);
  auto *selector = device.get_memory_addr(storage.first->get_ptr());
  auto *gate = static_cast<char *>(selector) + 4;
  cuda::driver_graph_prepare_mask_latch();
  // Only two device scalars: wide metadata never allocates a huge branch graph.
  for (const auto count : {4u, 0x80000001u, 0x80000002u,
                           std::numeric_limits<std::uint32_t>::max()}) {
    for (const bool fallback : {false, true}) {
      const BranchSelection selection{BranchSelection::Kind::indexed, fallback};
      for (const auto value : {std::numeric_limits<std::int32_t>::min(), -1, 0, 2,
                               std::numeric_limits<std::int32_t>::max()}) {
        SCOPED_TRACE(::testing::Message() << "count=" << count
                     << ", fallback=" << fallback << ", selector=" << value);
        auto input = value;
        driver.memcpy_host_to_device(selector, &input, 4);
        driver.memsetd32(gate, 99, 1);
        cuda::driver_graph_latch_branch(
            selector, gate, 2, count,
            fallback ? count - 1 : std::numeric_limits<std::uint32_t>::max());
        driver.stream_synchronize(nullptr);
        std::uint32_t actual = 99;
        driver.memcpy_device_to_host(&actual, gate, 4);
        const auto selected = selection.select(value, count);
        const auto expected = selected == count ? 0u :
            static_cast<std::uint32_t>(selected + 1);
        EXPECT_EQ(actual, expected);
      }
    }
  }
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
    fixed_ptx = ptx;
    const auto replace = [&](const std::string &from, const std::string &to) {
      const auto at = fixed_ptx.find(from);
      ASSERT_NE(at, std::string::npos);
      fixed_ptx.replace(at, from.size(), to);
    };
    replace(".param .u64 data, .param .u32 operation",
            ".param .align 8 .b8 context[" +
                std::to_string(sizeof(RuntimeContext)) + "]");
    replace(".reg .b64 %p;", ".reg .b64 %p, %args;");
    replace("ld.param.u64 %p, [data];",
            "ld.param.u64 %args, [context];\nld.global.u64 %p, [%args];");
    replace("ld.param.u32 %op, [operation];",
            "ld.global.u32 %op, [%args+8];");
    replace("st.global.u32 [%p+24], %count;",
            "st.global.u32 [%p+24], %count;\n"
            "st.global.u32 [%p+28], %count;");
    // A second physical task tests that every task in a dispatch is gated.
    fixed_ptx +=
        ".visible .entry audit(.param .align 8 .b8 context[" +
        std::to_string(sizeof(RuntimeContext)) + R"ptx(]) {
  .reg .b64 %args, %p;
  .reg .b32 %count;
  ld.param.u64 %args, [context];
  ld.global.u64 %p, [%args];
  ld.global.u32 %count, [%p+28];
  add.u32 %count, %count, 1;
  st.global.u32 [%p+28], %count;
  ret;
}
)ptx";
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
  std::string fixed_ptx;
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
  const auto bytes = frame.controls.device_bytes();
  RecordProperty("discarded_host_table_payload_bytes", static_cast<int>(bytes));
  ASSERT_EQ(frame.controls.upload(false), CUDA_SUCCESS);
  EXPECT_EQ(frame.controls.device_bytes(), bytes);
  EXPECT_EQ(frame.controls.host_controls.capacity(), 0);
  EXPECT_EQ(frame.controls.host_nodes.capacity(), 0);
  EXPECT_EQ(frame.controls.upload(), CUDA_ERROR_NOT_SUPPORTED);
  frame.run(0, 0);
}

TEST_F(CudaPredicateGraph, FixedExecutionLoadsOrdinaryArtifactsAndOwnsEachRegion) {
  auto &context = CUDAContext::get_instance();
  auto guard = context.get_guard();
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  cuda::KernelLauncher launcher({&executor});
  CUDAKernelArtifact artifact;
  artifact.payload.assign(fixed_ptx.begin(), fixed_ptx.end());
  artifact.payload.push_back('\0');
  artifact.entry_names = {"dispatch", "audit"};
  artifact.target_identity = "sm_50|+ptx60";
  artifact.provider_identity = "fixed_predicate_test";
  LLVM::CompiledKernelData::InternalData metadata;
  metadata.args_size = 16;
  metadata.compiled_data.tasks = {OffloadedTask("dispatch", 1, 1),
                                  OffloadedTask("audit", 1, 1)};
  llvm_aot::KernelImpl kernel(
      &launcher, launcher.register_cuda_artifact_kernel(metadata, artifact),
      Arch::cuda, "dispatch", metadata);
  auto &device = *executor.get_compute_device();
  auto &driver = CUDADriver::get_instance();
  auto first = device.allocate_memory_unique({40});
  auto second = device.allocate_memory_unique({40});
  ASSERT_EQ(first.second, RhiResult::success);
  ASSERT_EQ(second.second, RhiResult::success);
  auto ptr = [](DeviceAllocationUnique &allocation, std::size_t word) {
    auto result = allocation->get_ptr();
    result.offset = word * 4;
    return result;
  };
  auto address = [&](DevicePtr pointer) {
    auto &cuda_device = static_cast<cuda::CudaDevice &>(device);
    return static_cast<char *>(cuda_device.get_alloc_info(pointer).ptr) +
           pointer.offset;
  };
  auto prepare = [&](DeviceAllocationUnique &allocation, std::uint64_t outer,
                     std::uint64_t inner, bool conditional = false) {
    driver.memsetd32(address(ptr(allocation, 0)), 99, 10);
    auto loop = std::make_shared<FixedExecution::While>();
    loop->predicate = ptr(allocation, 1);
    loop->max_iterations = outer;
    loop->cuda_control.emplace();
    loop->cuda_control->inners.push_back({ptr(allocation, 3), 1, 2, 3, inner});
    std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
    for (std::uint32_t i = 0; i < 4; ++i) {
      auto launch = std::make_unique<LaunchContextBuilder>(&kernel);
      struct Arguments {
        void *data;
        std::uint32_t operation;
        std::uint32_t padding;
      } arguments{address(ptr(allocation, 0)), i, 0};
      std::memcpy(launch->get_context().arg_buffer, &arguments, sizeof(arguments));
      auto &region = i ? loop->body : loop->condition;
      region.emplace_back(FixedExecution::Dispatch{&kernel, launch.get()});
      contexts.push_back(std::move(launch));
    }
    std::vector<FixedExecution::Operation> operations{
        FixedExecution::BufferFill{ptr(allocation, 8), 4, 77}, loop, loop,
        FixedExecution::BufferCopy{ptr(allocation, 8), ptr(allocation, 6), 4}};
    if (conditional) {
      auto branch = std::make_shared<FixedExecution::Branch>();
      branch->selector = ptr(allocation, 4);
      branch->selection = {BranchSelection::Kind::nonzero, true};
      branch->bodies = {
          {FixedExecution::BufferFill{ptr(allocation, 9), 4, 11}},
          {FixedExecution::BufferFill{ptr(allocation, 9), 4, 22}}};
      operations.emplace_back(std::move(branch));
    }
    // Unsupported operation kinds must fail before preparing any replay.
    auto saved = loop->body.front();
    loop->body.front() = FixedExecution::BufferFill{ptr(allocation, 2), 4, 0};
    EXPECT_THROW(FixedExecution(Arch::cuda, device, nullptr, operations),
                 std::invalid_argument);
    loop->body.front() = saved;
    auto result = std::make_unique<FixedExecution>(Arch::cuda, device, nullptr,
                                                  operations);
    EXPECT_EQ(result->execution_kind(), 1);
    EXPECT_GT(result->workspace_bytes(), 0);
    std::array<std::uint32_t, 10> before;
    driver.memcpy_device_to_host(before.data(), address(ptr(allocation, 0)), 40);
    for (auto value : before)
      EXPECT_EQ(value, 99);
    return result;  // Cold descriptors/contexts are gone before execution.
  };
  auto run = [&](FixedExecution &call, DeviceAllocationUnique &allocation,
                 std::uint64_t outer_budget, std::uint64_t inner_budget,
                 std::uint32_t outer, std::uint32_t inner) {
    std::array<std::uint32_t, 2> limits{outer, inner};
    driver.memcpy_host_to_device(address(ptr(allocation, 4)), limits.data(), 8);
    call.run();
    driver.stream_synchronize(nullptr);
    std::array<std::uint32_t, 10> actual;
    driver.memcpy_device_to_host(actual.data(), address(ptr(allocation, 0)), 40);
    const auto count = std::min<std::uint64_t>(outer_budget, outer);
    const auto steps = std::min<std::uint64_t>(inner_budget, inner);
    EXPECT_EQ(actual[0], count);
    EXPECT_EQ(actual[6], count * steps);
    EXPECT_EQ(actual[7], 1 + count * (2 + steps));
    EXPECT_EQ(actual[8], count * steps);
  };
  auto sibling = prepare(second.first, 3, 4);
  for (const auto budgets : {std::array<std::uint64_t, 2>{4, 3}, {0, 3},
                             {4, 0}, {65, 1}}) {
    auto call = prepare(first.first, budgets[0], budgets[1]);
    for (const auto limits : {std::array<std::uint32_t, 2>{0, 0}, {2, 1},
                              {7, 7}, {0, 2}}) {
      run(*call, first.first, budgets[0], budgets[1], limits[0], limits[1]);
      run(*sibling, second.first, 3, 4, limits[1], limits[0]);
    }
    std::array<std::uint32_t, 2> limits{2, 2};
    driver.memcpy_host_to_device(address(ptr(first.first, 4)), limits.data(), 8);
    call->run();
    call.reset();  // Retirement waits for submitted work before freeing gates.
    std::uint32_t result = 99;
    driver.memcpy_device_to_host(&result, address(ptr(first.first, 8)), 4);
    EXPECT_EQ(result, std::min<std::uint64_t>(2, budgets[0]) *
                          std::min<std::uint64_t>(2, budgets[1]));
  }
  run(*sibling, second.first, 3, 4, 3, 2);
  if (cuda::GraphRegionBuilder::available()) {
    auto mixed = prepare(first.first, 2, 3, true);
    for (const auto outer : {2u, 0u}) {
      run(*mixed, first.first, 2, 3, outer, 1);
      std::uint32_t selected = 0;
      driver.memcpy_device_to_host(&selected, address(ptr(first.first, 9)), 4);
      EXPECT_EQ(selected, outer ? 11 : 22);
    }
  }
  sibling.reset();
  EXPECT_EQ(launcher.debug_registered_kernel_count(), 1);
  launcher.retire_all_kernels();
}

}  // namespace
}  // namespace taichi::lang
#endif
