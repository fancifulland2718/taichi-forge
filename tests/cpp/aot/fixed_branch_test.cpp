#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/graph_region_builder.h"
#endif

namespace taichi::lang {
namespace {
using Frame = std::array<std::int32_t, 4>;
using Write = std::function<void(DevicePtr, const Frame &)>;
using Read = std::function<Frame(DevicePtr)>;

void check_branches(Arch arch,
                    Device &device,
                    const Write &write,
                    const Read &read) {
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
        branch->selector = ptr(begin + 1);
        branch->selection = {kind, fallback};
        branch->condition = {
            FixedExecution::BufferCopy{ptr(begin + 1), ptr(begin), 4}};
        for (std::size_t i = 0; i < cases + unsigned(fallback); ++i)
          branch->bodies.push_back(
              {fill(begin + 2, 100 + i), fill(begin + 1, std::uint32_t(-77))});
        calls.push_back(std::make_unique<FixedExecution>(
            arch, device, nullptr,
            std::vector<FixedExecution::Operation>{
                fill(begin + 2, 99), branch,
                FixedExecution::BufferCopy{ptr(begin + 3), ptr(begin + 1),
                                           4}}));
        // Branch descriptors and their vectors are not retained by replay.
        branch->bodies.clear();
        EXPECT_EQ(read(ptr(begin)), (Frame{0, 71, 72, 73}));
        EXPECT_EQ(calls.back()->execution_kind(), arch == Arch::cuda ? 1 : 0);
        EXPECT_EQ(calls.back()->workspace_bytes(), arch == Arch::cuda ? 24 : 0);
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

void check_nested_branches(Arch arch,
                           Device &device,
                           const Write &write,
                           const Read &read) {
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
  auto inner = std::make_shared<FixedExecution::Branch>();
  inner->selector = ptr(5);
  inner->selection.has_default = true;
  inner->condition = {FixedExecution::BufferCopy{ptr(5), ptr(0), 4}};
  inner->bodies = {{fill(2, 11), fill(0, 0)}, {fill(2, 22), fill(0, 1)}};
  auto loop = std::make_shared<FixedExecution::While>();
  loop->predicate = ptr(4);
  loop->condition = {fill(4, 1)};
  loop->body = {inner};
  auto outer = std::make_shared<FixedExecution::Branch>();
  outer->selector = ptr(1);
  outer->selection.has_default = true;
  outer->condition = {FixedExecution::BufferCopy{ptr(1), ptr(0), 4}};
  outer->bodies = {{loop}, {fill(2, 777)}};
  for (const auto budget : {0u, 1u, 2u}) {
    loop->max_iterations = budget;
    FixedExecution execution(
        arch, device, nullptr,
        {fill(2, 99), outer, FixedExecution::BufferCopy{ptr(3), ptr(0), 4}});
    EXPECT_EQ(execution.workspace_bytes(), arch == Arch::cuda ? 80 : 0);
    for (std::int32_t input : {1, 0}) {
      write(ptr(0), Frame{input, 71, 72, 73});
      execution.run();
      const auto result = read(ptr(0));
      EXPECT_EQ(result[2], input ? (budget == 0   ? 99
                                    : budget == 1 ? 11
                                                  : 22)
                                 : 777);
      EXPECT_EQ(result[3], input && budget == 1 ? 0 : input);
    }
  }
}

#ifdef TI_WITH_LLVM
TEST(FixedBranch, CpuSelectsExactlyOneBodyAndRetainsIndependentBindings) {
  cpu::CpuDevice device;
  auto address = [&](DevicePtr ptr) {
    void *data = nullptr;
    EXPECT_EQ(device.map_range_for_cpu_native(ptr, sizeof(Frame), &data),
              RhiResult::success);
    return data;
  };
  const Write write = [&](DevicePtr ptr, const Frame &data) {
    std::memcpy(address(ptr), data.data(), sizeof(data));
  };
  const Read read = [&](DevicePtr ptr) {
    Frame data;
    std::memcpy(data.data(), address(ptr), sizeof(data));
    return data;
  };
  check_branches(host_arch(), device, write, read);
  check_nested_branches(host_arch(), device, write, read);
}
#endif

#ifdef TI_WITH_CUDA
TEST(FixedBranch, CudaSelectsOnDeviceWithOneRootAndIndependentBindings) {
  if (!CUDADriver::get_instance_without_context().detected() ||
      !cuda::GraphRegionBuilder::available())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto address = [&](DevicePtr ptr) {
    return static_cast<char *>(device.get_memory_addr(ptr)) + ptr.offset;
  };
  const Write write = [&](DevicePtr ptr, const Frame &data) {
    driver.memcpy_host_to_device(address(ptr),
                                const_cast<std::int32_t *>(data.data()),
                                sizeof(data));
  };
  const Read read = [&](DevicePtr ptr) {
    Frame data;
    driver.memcpy_device_to_host(data.data(), address(ptr), sizeof(data));
    return data;
  };
  check_branches(Arch::cuda, device, write, read);
  check_nested_branches(Arch::cuda, device, write, read);
}
#endif
}  // namespace
}  // namespace taichi::lang
