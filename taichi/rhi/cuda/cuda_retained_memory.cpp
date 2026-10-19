#include "taichi/rhi/cuda/cuda_retained_memory.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang::cuda {

struct RetainedMemoryPool::State {
  struct Block {
    void *pointer{nullptr};
    void *event{nullptr};
    std::size_t bytes{0};
    bool live{true};
  };
  std::mutex mutex;
  std::unordered_map<void *, std::unique_ptr<Block>> blocks;
  std::map<std::size_t, std::vector<Block *>> retired;
  std::uint64_t reserved{0}, reserved_high{0}, used{0}, used_high{0};
  std::uint64_t allocations{0}, frees{0}, reuses{0}, retirements{0};
  bool faulted{false};

  bool ready(Block &block) {
    auto &driver = CUDADriver::get_instance();
    const auto result = driver.event_query.call(block.event);
    if (result == CUDA_ERROR_NOT_READY)
      return false;
    if (result != CUDA_SUCCESS) {
      faulted = true;
      // Route the error through the normal structured fault reporter.
      driver.event_query.check_result(result);
    }
    return true;
  }
};

RetainedMemoryPool::RetainedMemoryPool() : state_(std::make_unique<State>()) {
}

RetainedMemoryPool::~RetainedMemoryPool() {
  auto &state = *state_;
  if (state.faulted)
    return;  // Keep the failed context's storage quarantined until process
             // exit.
  try {
    auto context = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    for (auto &[pointer, block] : state.blocks) {
      // A live allocation must retain its pool owner. Do not release a broken
      // owner's storage optimistically if that invariant is ever violated.
      if (block->live)
        continue;
      driver.event_synchronize(block->event);
      driver.mem_free(pointer);
      driver.event_destroy(block->event);
    }
  } catch (...) {
    // Destructor/fault cleanup must not throw or destroy unproven storage.
  }
}

RetainedMemoryPool &RetainedMemoryPool::process_pool() {
  // Same process lifetime as CUDAContext; fixed bounded workloads reach a
  // high-water mark. Explicit trim is the only synchronous reclamation path.
  static auto *pool = new RetainedMemoryPool;
  return *pool;
}

void *RetainedMemoryPool::allocate(std::size_t bytes) {
  if (bytes == 0)
    return nullptr;
  CUDAContext::check_runtime_health();
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  TI_ERROR_IF(state.faulted, "CiG retained memory pool has faulted");
  // Search only retired size classes, never all live allocations. Do not
  // consume an arbitrarily large block for a small transient parameter buffer.
  for (auto bin = state.retired.lower_bound(bytes);
       bin != state.retired.end() && bin->first - bytes <= bytes; ++bin) {
    auto &blocks = bin->second;
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      auto *block = blocks[index];
      if (!state.ready(*block))
        continue;
      block->live = true;
      blocks[index] = blocks.back();
      blocks.pop_back();
      // Retain empty bins and their vector capacity across reuse cycles.
      state.used += block->bytes;
      state.used_high = std::max(state.used_high, state.used);
      ++state.reuses;
      return block->pointer;
    }
  }
  auto block = std::make_unique<State::Block>();
  block->bytes = bytes;
  driver.event_create(&block->event, CU_EVENT_DISABLE_TIMING);
  try {
    driver.malloc(&block->pointer, bytes);
    auto *pointer = block->pointer;
    auto inserted = state.blocks.try_emplace(pointer);
    TI_ASSERT(inserted.second);
    inserted.first->second = std::move(block);
    state.reserved += bytes;
    state.reserved_high = std::max(state.reserved_high, state.reserved);
    state.used += bytes;
    state.used_high = std::max(state.used_high, state.used);
    ++state.allocations;
    return pointer;
  } catch (...) {
    if (block) {
      if (block->pointer)
        driver.mem_free.call(block->pointer);
      driver.event_destroy.call(block->event);
    }
    throw;
  }
}

void RetainedMemoryPool::release(void *pointer, void *stream) {
  if (!pointer)
    return;
  auto context = CUDAContext::get_instance().get_guard();
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  auto found = state.blocks.find(pointer);
  TI_ERROR_IF(found == state.blocks.end() || !found->second->live,
              "Invalid CiG retained allocation retirement");
  TI_ERROR_IF(state.faulted, "CiG retained memory pool has faulted");
  auto &block = *found->second;
  // Reserve host metadata before submitting the event, so an allocation
  // failure cannot make storage reusable without a completion dependency.
  auto &bin = state.retired[block.bytes];
  bin.reserve(bin.size() + 1);
  try {
    CUDADriver::get_instance().event_record(block.event, stream);
  } catch (...) {
    state.faulted = true;
    throw;
  }
  block.live = false;
  state.used -= block.bytes;
  ++state.retirements;
  bin.push_back(&block);
}

void RetainedMemoryPool::trim() {
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  TI_ERROR_IF(state.faulted, "CiG retained memory pool has faulted");
  for (auto bin = state.retired.begin(); bin != state.retired.end();) {
    auto &blocks = bin->second;
    for (std::size_t index = 0; index < blocks.size();) {
      auto *block = blocks[index];
      if (!state.ready(*block)) {
        ++index;
        continue;
      }
      try {
        driver.mem_free(block->pointer);
        driver.event_destroy(block->event);
      } catch (...) {
        state.faulted = true;
        throw;
      }
      state.reserved -= block->bytes;
      ++state.frees;
      state.blocks.erase(block->pointer);
      blocks[index] = blocks.back();
      blocks.pop_back();
    }
    if (blocks.empty())
      bin = state.retired.erase(bin);
    else
      ++bin;
  }
}

std::unordered_map<std::string, std::uint64_t> RetainedMemoryPool::snapshot() {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  return {{"native_async_pool", 0},
          {"retained_device_pool", 1},
          {"reserved_current_bytes", state.reserved},
          {"reserved_high_bytes", state.reserved_high},
          {"used_current_bytes", state.used},
          {"used_high_bytes", state.used_high},
          {"native_allocation_calls", state.allocations},
          {"native_free_calls", state.frees},
          {"reuse_calls", state.reuses},
          {"retirement_calls", state.retirements},
          {"faulted", state.faulted}};
}

}  // namespace taichi::lang::cuda
