#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include "taichi/runtime/branch_selection.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/gfx/runtime.h"
#endif

namespace taichi::lang {
namespace {
// Compare the production executor with the independently composed contract.
using Step = std::function<void()>;
using Region = std::vector<Step>;
using Frame = std::array<std::int32_t, 4>;
using Reader = std::function<std::int32_t()>;
using PrepareReader = std::function<Reader(DevicePtr)>;
using ReadFrame = std::function<Frame(DevicePtr)>;
using WriteFrame = std::function<void(DevicePtr, const Frame &)>;

class FactOperation final : public NativeOperation {
 public:
  FactOperation(Arch arch,
                Device &device,
                DevicePtr output,
                const NativeResource *marker = nullptr,
                bool opaque = false)
      : arch_(arch),
        device_(device),
        output_(output),
        marker_(marker),
        opaque_(opaque) {
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 16;
  }
  void validate_recording(NativeRecordingState &facts) const override {
    if (opaque_)
      facts.provider_state<int>(this) = 1;
    if (!marker_)
      facts.initialized.insert(this);
    else if (!facts.initialized.count(marker_))
      throw std::invalid_argument("read before initialization");
  }
  void run_cpu() override {
#ifdef TI_WITH_LLVM
    void *mapped = nullptr;
    ASSERT_EQ(static_cast<cpu::CpuDevice &>(device_).map_range_for_cpu_native(
                  output_, 4, &mapped),
              RhiResult::success);
    *static_cast<std::uint32_t *>(mapped) = marker_ ? 888 : 777;
#endif
  }
  void record_cuda(void *stream) override {
#ifdef TI_WITH_CUDA
    auto *address = static_cast<char *>(
        static_cast<cuda::CudaDevice &>(device_).get_memory_addr(output_));
    CUDADriver::get_instance().memsetd32_async(address + output_.offset,
                                               marker_ ? 888 : 777, 1, stream);
#endif
  }
  void record_vulkan(Device *, CommandList *commands) override {
    commands->buffer_fill(output_, 4, marker_ ? 888 : 777);
  }

 private:
  Arch arch_;
  Device &device_;
  DevicePtr output_;
  const NativeResource *marker_;
  bool opaque_;
};

void check_nested_and_facts(Arch arch,
                            Device &device,
                            gfx::GfxRuntime *runtime,
                            const WriteFrame &write,
                            const ReadFrame &read) {
  auto allocation = device.allocate_memory_unique({sizeof(Frame)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto ptr = [&](std::size_t i) { return allocation.first->get_ptr(i * 4); };
  auto fill = [&](std::size_t i, std::uint32_t value) {
    return FixedExecution::BufferFill{ptr(i), 4, value};
  };
  auto inner = std::make_shared<FixedExecution::Branch>();
  inner->host_selection = true;
  inner->selector = ptr(1);
  inner->selection.has_default = true;
  inner->condition = {fill(1, 0)};
  inner->bodies = {{fill(2, 10)}, {fill(2, 20)}};
  auto outer = std::make_shared<FixedExecution::Branch>();
  outer->host_selection = true;
  outer->selector = ptr(1);
  outer->selection.has_default = true;
  outer->condition = {FixedExecution::BufferCopy{ptr(1), ptr(0), 4}};
  outer->bodies = {{inner}, {fill(2, 30)}};
  {
    FixedExecution nested(
        arch, device, runtime,
        {outer, FixedExecution::BufferCopy{ptr(3), ptr(2), 4}});
    EXPECT_EQ(nested.workspace_bytes(), arch == Arch::vulkan ? 4 : 0);
    for (int input : {0, 1}) {
      write(ptr(0), Frame{input, 71, 72, 73});
      nested.run();
      EXPECT_EQ(read(ptr(0))[3], input ? 20 : 30);
    }
  }
  if (arch == Arch::cuda) {
    outer->host_selection = false;
    EXPECT_THROW(FixedExecution(arch, device, runtime, {outer}),
                 std::invalid_argument);
    outer->host_selection = true;
    // A native child under host selection remains a native CUDA graph.
    inner->host_selection = false;
    FixedExecution mixed(arch, device, runtime, {outer});
    write(ptr(0), Frame{1, 71, 72, 73});
    mixed.run();
    EXPECT_EQ(read(ptr(0))[2], 20);
    inner->host_selection = true;
  }
  auto writer = std::make_shared<FactOperation>(arch, device, ptr(2));
  auto reader =
      std::make_shared<FactOperation>(arch, device, ptr(2), writer.get());
  outer->bodies = {{writer}, {}};
  NativeRecordingState facts;
  EXPECT_THROW(FixedExecution(arch, device, runtime, {outer, reader}, &facts),
               std::invalid_argument);
  EXPECT_TRUE(facts.initialized.empty());
  outer->bodies = {{writer}, {writer}};
  {
    FixedExecution joined(arch, device, runtime, {outer, reader}, &facts);
    // Both branch captures retain one writer; its workspace is counted once.
    EXPECT_EQ(joined.workspace_bytes(), arch == Arch::vulkan ? 36 : 32);
    EXPECT_TRUE(facts.initialized.count(writer.get()));
    for (int input : {0, 1}) {
      write(ptr(0), Frame{input, 71, 72, 73});
      joined.run();
      EXPECT_EQ(read(ptr(0))[2], 888);
    }
  }
  outer->selection.has_default = false;
  outer->bodies.resize(1);
  EXPECT_THROW(FixedExecution(arch, device, runtime, {outer, reader}),
               std::invalid_argument);
  outer->condition.push_back(writer);
  outer->bodies = {{reader}};
  FixedExecution condition(arch, device, runtime, {outer, reader});
  write(ptr(0), Frame{0, 71, 72, 73});
  condition.run();
  EXPECT_EQ(read(ptr(0))[2], 888);
  auto opaque =
      std::make_shared<FactOperation>(arch, device, ptr(2), nullptr, true);
  outer->bodies = {{opaque}};
  EXPECT_THROW(FixedExecution(arch, device, runtime, {outer}),
               std::invalid_argument);
}

void check_production(Arch arch,
                      Device &device,
                      gfx::GfxRuntime *runtime,
                      const WriteFrame &write,
                      const ReadFrame &read) {
  using Kind = BranchSelection::Kind;
  for (const auto kind : {Kind::nonzero, Kind::indexed}) {
    for (bool fallback : {false, true}) {
      auto allocation = device.allocate_memory_unique({sizeof(Frame) * 2});
      ASSERT_EQ(allocation.second, RhiResult::success);
      auto ptr = [&](std::size_t index) {
        auto p = allocation.first->get_ptr();
        p.offset += index * sizeof(std::int32_t);
        return p;
      };
      auto fill = [&](std::size_t index, std::uint32_t value) {
        return FixedExecution::BufferFill{ptr(index), 4, value};
      };
      const std::size_t cases = kind == Kind::nonzero ? 1 : 8;
      std::vector<std::unique_ptr<FixedExecution>> calls;
      for (unsigned frame = 0; frame < 2; ++frame) {
        const auto begin = frame * 4;
        write(ptr(begin), Frame{0, 71, 72, 73});
        auto branch = std::make_shared<FixedExecution::Branch>();
        branch->host_selection = true;
        branch->selector = ptr(begin + 1);
        branch->selection = {kind, fallback};
        branch->condition = {
            FixedExecution::BufferCopy{ptr(begin + 1), ptr(begin), 4}};
        for (std::size_t i = 0; i < cases + unsigned(fallback); ++i)
          branch->bodies.push_back(
              {fill(begin + 2, 100 + i), fill(begin + 1, std::uint32_t(-77))});
        calls.push_back(std::make_unique<FixedExecution>(
            arch, device, runtime,
            std::vector<FixedExecution::Operation>{
                fill(begin + 2, 99), branch,
                FixedExecution::BufferCopy{ptr(begin + 3), ptr(begin + 1),
                                           4}}));
        // Branch descriptors and their vectors are not retained by replay.
        branch->bodies.clear();
        EXPECT_EQ(read(ptr(begin)), (Frame{0, 71, 72, 73}));
        EXPECT_EQ(calls.back()->execution_kind(), arch_is_cpu(arch) ? 0 : 5);
        EXPECT_EQ(calls.back()->workspace_bytes(),
                  arch == Arch::vulkan ? 4 : 0);
      }
      for (std::int32_t selected :
           {-1, 0, 1, 2, 7, 8, 9, std::numeric_limits<std::int32_t>::min(),
            std::numeric_limits<std::int32_t>::max()}) {
        for (unsigned frame = 0; frame < 2; ++frame) {
          const auto value = frame ? ~selected : selected;
          write(ptr(frame * 4), Frame{value, 71, 72, 73});
          calls[frame]->run();
          bool matched = kind == Kind::nonzero
                             ? value != 0
                             : value >= 0 && std::size_t(value) < cases;
          std::int32_t expected =
              matched    ? 100 + (kind == Kind::nonzero ? 0 : value)
              : fallback ? 100 + cases
                         : 99;
          const auto echo = matched || fallback ? -77 : value;
          EXPECT_EQ(read(ptr(frame * 4)), (Frame{value, echo, expected, echo}));
        }
      }
      calls.front()->run();
      // Retire pending work without invalidating the sibling binding.
      calls.front().reset();
      write(ptr(4), Frame{0, 71, 72, 73});
      calls.back()->run();
      EXPECT_EQ(read(ptr(4))[2],
                kind == Kind::nonzero ? (fallback ? 101 : 99) : 100);
    }
  }
}

void run(const Region &steps) {
  for (const auto &step : steps)
    step();
}

Step branch(BranchSelection selection,
            Reader reader,
            Region condition,
            std::vector<Region> bodies) {
  selection.validate(bodies.size());
  return [selection, reader = std::move(reader),
          condition = std::move(condition), bodies = std::move(bodies)] {
    run(condition);
    const auto index = selection.select(reader(), bodies.size());
    if (index < bodies.size())
      run(bodies[index]);
  };
}

void check(Arch arch,
           Device &device,
           gfx::GfxRuntime *runtime,
           const PrepareReader &reader,
           const WriteFrame &write,
           const ReadFrame &read) {
  check_production(arch, device, runtime, write, read);
  check_nested_and_facts(arch, device, runtime, write, read);
  auto memory = device.allocate_memory_unique({sizeof(Frame) * 2});
  ASSERT_EQ(memory.second, RhiResult::success);
  auto ptr = [&](std::size_t index) {
    return memory.first->get_ptr(index * sizeof(std::int32_t));
  };
  auto fill = [&](std::size_t index, std::int32_t value) {
    return FixedExecution::BufferFill{ptr(index), 4,
                                      static_cast<std::uint32_t>(value)};
  };
  auto copy = [&](std::size_t to, std::size_t from) {
    return FixedExecution::BufferCopy{ptr(to), ptr(from), 4};
  };
  auto native = [&](std::vector<FixedExecution::Operation> operations) -> Step {
    auto call =
        std::make_shared<FixedExecution>(arch, device, runtime, operations);
    return [call = std::move(call)] { call->run(); };
  };
  for (const auto kind :
       {BranchSelection::Kind::nonzero, BranchSelection::Kind::indexed}) {
    for (const bool fallback : {false, true}) {
      std::array<Region, 2> plans;
      const std::size_t cases = kind == BranchSelection::Kind::nonzero ? 1 : 8;
      for (std::size_t frame = 0; frame < plans.size(); ++frame) {
        const auto base = frame * 4;
        write(ptr(base), Frame{0, 71, 72, 73});
        std::vector<Region> bodies;
        for (std::size_t i = 0; i < cases + std::size_t(fallback); ++i)
          bodies.push_back({native(
              {fill(base + 2, 100 + i + frame * 1000), fill(base + 1, -77)})});
        auto selected = branch({kind, fallback}, reader(ptr(base + 1)),
                               {native({copy(base + 1, base)})}, bodies);
        bodies.clear();  // Prepared structure is independent of source vectors.
        plans[frame] = {native({fill(base + 2, 99)}), std::move(selected),
                        native({copy(base + 3, base + 1)})};
        EXPECT_EQ(read(ptr(base)), (Frame{0, 71, 72, 73}));
      }
      for (std::int32_t input :
           {0, 1, 7, 8, -1, std::numeric_limits<std::int32_t>::min(),
            std::numeric_limits<std::int32_t>::max()}) {
        for (std::size_t frame = 0; frame < plans.size(); ++frame) {
          const auto value = frame ? ~input : input;
          write(ptr(frame * 4), Frame{value, 71, 72, 73});
          run(plans[frame]);
          const bool matched = kind == BranchSelection::Kind::nonzero
                                   ? value != 0
                                   : value >= 0 && std::size_t(value) < cases;
          const auto index = matched ? (kind == BranchSelection::Kind::nonzero
                                            ? 0
                                            : std::size_t(value))
                                     : cases;
          const bool entered = matched || fallback;
          const auto output = entered ? 100 + index + frame * 1000 : 99;
          EXPECT_EQ(read(ptr(frame * 4)),
                    (Frame{value, entered ? -77 : value,
                           static_cast<std::int32_t>(output),
                           entered ? -77 : value}));
        }
      }
      run(plans.front());
      plans.front().clear();  // Native blocks retire pending suffix work.
      write(ptr(4), Frame{1, 71, 72, 73});
      run(plans.back());
      EXPECT_EQ(read(ptr(4))[2],
                kind == BranchSelection::Kind::nonzero ? 1100 : 1101);
    }
  }
  // Nested host selection reuses the same staging serially. Inner condition
  // mutates the outer selector; it cannot cause another outer body to run.
  auto inner = branch({BranchSelection::Kind::nonzero, true}, reader(ptr(1)),
                      {native({fill(1, 0)})},
                      {{native({fill(2, 10)})}, {native({fill(2, 20)})}});
  auto outer =
      branch({BranchSelection::Kind::nonzero, true}, reader(ptr(1)),
             {native({copy(1, 0)})}, {{inner}, {native({fill(2, 30)})}});
  for (const auto input : {0, 1}) {
    write(ptr(0), Frame{input, 71, 72, 73});
    outer();
    EXPECT_EQ(read(ptr(0))[2], input ? 20 : 30);
  }
}

#ifdef TI_WITH_LLVM
TEST(HostBranchContract, CpuPreparedBlocks) {
  cpu::CpuDevice device;
  auto address = [&](DevicePtr ptr) {
    void *value = nullptr;
    EXPECT_EQ(device.map_range_for_cpu_native(ptr, 4, &value),
              RhiResult::success);
    return value;
  };
  check(
      host_arch(), device, nullptr,
      [&](DevicePtr ptr) -> Reader {
        return [&, ptr] { return *static_cast<std::int32_t *>(address(ptr)); };
      },
      [&](DevicePtr ptr, const Frame &data) {
        std::memcpy(address(ptr), data.data(), sizeof(data));
      },
      [&](DevicePtr ptr) {
        Frame data;
        std::memcpy(data.data(), address(ptr), sizeof(data));
        return data;
      });
}
#endif

#ifdef TI_WITH_CUDA
TEST(HostBranchContract, FailedHostProviderCannotReplay) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto allocation = device.allocate_memory_unique({4});
  ASSERT_EQ(allocation.second, RhiResult::success);
  class Failure final : public NativeOperation {
   public:
    explicit Failure(Device &device) : device_(device) {}
    Arch arch() const override { return Arch::cuda; }
    Device &device() const override { return device_; }
    std::size_t workspace_bytes() const override { return 0; }
    NativeOperationMode execution_mode() const override {
      return NativeOperationMode::cuda_synchronous;
    }
    void run_cuda_synchronous() override {
      ++calls;
      throw std::runtime_error("injected provider failure");
    }
    int calls{0};
   private:
    Device &device_;
  };
  auto failure = std::make_shared<Failure>(device);
  auto branch = std::make_shared<FixedExecution::Branch>();
  branch->host_selection = true;
  branch->selector = allocation.first->get_ptr();
  branch->condition = {FixedExecution::BufferFill{branch->selector, 4, 1}};
  branch->bodies = {{failure}};
  FixedExecution execution(Arch::cuda, device, nullptr, {branch});
  EXPECT_EQ(failure->calls, 0);
  EXPECT_THROW(execution.run(), std::runtime_error);
  EXPECT_EQ(failure->calls, 1);
  EXPECT_THROW(execution.run(), std::runtime_error);
  EXPECT_EQ(failure->calls, 1);
}

TEST(HostBranchContract, CudaPreparedBlocks) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto address = [&](DevicePtr ptr) {
    return static_cast<char *>(device.get_memory_addr(ptr)) + ptr.offset;
  };
  check(
      Arch::cuda, device, nullptr,
      [&](DevicePtr ptr) -> Reader {
        return [&, source = address(ptr)] {
          std::int32_t value;
          driver.memcpy_device_to_host(&value, source, sizeof(value));
          return value;
        };
      },
      [&](DevicePtr ptr, const Frame &data) {
        driver.memcpy_host_to_device(address(ptr),
                                     const_cast<std::int32_t *>(data.data()),
                                     sizeof(data));
      },
      [&](DevicePtr ptr) {
        Frame data;
        driver.memcpy_device_to_host(data.data(), address(ptr), sizeof(data));
        return data;
      });
}
#endif

#ifdef TI_WITH_VULKAN
TEST(HostBranchContract, VulkanPersistentReadbackCommands) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator::Params parameters;
  vulkan::VulkanDeviceCreator creator(parameters);
  auto &device = *creator.device();
  gfx::GfxRuntime::Params runtime_parameters;
  runtime_parameters.device = &device;
  gfx::GfxRuntime runtime(std::move(runtime_parameters));
  auto staging =
      device.allocate_memory_unique({4, false, true, false, AllocUsage::None});
  ASSERT_EQ(staging.second, RhiResult::success);
  check(
      Arch::vulkan, device, &runtime,
      [&](DevicePtr ptr) -> Reader {
        auto command = device.get_compute_stream()->new_command_list_unique();
        EXPECT_EQ(command.second, RhiResult::success);
        std::shared_ptr<CommandList> commands(std::move(command.first));
        commands->memory_barrier();
        commands->buffer_copy(staging.first->get_ptr(), ptr, 4);
        commands->memory_barrier();
        return [&, commands] {
          (void)runtime.flush_if_pending();
          device.get_compute_stream()->submit_synced(commands.get());
          void *mapped = nullptr;
          EXPECT_EQ(device.map(*staging.first, &mapped), RhiResult::success);
          std::int32_t value;
          std::memcpy(&value, mapped, sizeof(value));
          device.unmap(*staging.first);
          return value;
        };
      },
      [&](DevicePtr ptr, const Frame &data) {
        (void)runtime.flush_if_pending();
        const void *source = data.data();
        std::size_t bytes = sizeof(data);
        EXPECT_EQ(device.upload_data(&ptr, &source, &bytes),
                  RhiResult::success);
      },
      [&](DevicePtr ptr) {
        (void)runtime.flush_if_pending();
        Frame data;
        void *target = data.data();
        std::size_t bytes = sizeof(data);
        EXPECT_EQ(device.readback_data(&ptr, &target, &bytes),
                  RhiResult::success);
        return data;
      });
}
#endif
}  // namespace
}  // namespace taichi::lang
