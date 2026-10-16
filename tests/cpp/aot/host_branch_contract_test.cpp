#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include "taichi/runtime/branch_selection.h"
#include "taichi/runtime/fixed_execution.h"
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
// Isolated contract proof only. No FixedExecution routing or ABI change.
using Step = std::function<void()>;
using Region = std::vector<Step>;
using Frame = std::array<std::int32_t, 4>;
using Reader = std::function<std::int32_t()>;
using PrepareReader = std::function<Reader(DevicePtr)>;
using ReadFrame = std::function<Frame(DevicePtr)>;
using WriteFrame = std::function<void(DevicePtr, const Frame &)>;

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
