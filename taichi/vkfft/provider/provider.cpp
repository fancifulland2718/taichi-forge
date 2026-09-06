#include "taichi/vkfft/forge_vkfft_provider.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

#include <glslang/build_info.h>

namespace {
// VkFFT allocates LUT and temporary storage itself. Account those actual
// requests at plan creation/destruction, rather than polling device memory
// during replay or claiming the input buffer is the whole plan footprint.
struct Allocations {
  std::unordered_map<VkDeviceMemory, VkDeviceSize> sizes;
  uint64_t live{0};
  uint64_t peak{0};
};
thread_local Allocations *active_allocations = nullptr;

struct AllocationScope {
  Allocations *previous;
  explicit AllocationScope(Allocations &allocations)
      : previous(active_allocations) {
    active_allocations = &allocations;
  }
  ~AllocationScope() {
    active_allocations = previous;
  }
};

VkResult allocate_memory(VkDevice device,
                         const VkMemoryAllocateInfo *info,
                         const VkAllocationCallbacks *callbacks,
                         VkDeviceMemory *memory) {
  auto result = vkAllocateMemory(device, info, callbacks, memory);
  if (result == VK_SUCCESS) {
    try {
      active_allocations->sizes.emplace(*memory, info->allocationSize);
    } catch (const std::bad_alloc &) {
      vkFreeMemory(device, *memory, callbacks);
      *memory = VK_NULL_HANDLE;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    active_allocations->live += info->allocationSize;
    active_allocations->peak =
        std::max(active_allocations->peak, active_allocations->live);
  }
  return result;
}

void free_memory(VkDevice device,
                 VkDeviceMemory memory,
                 const VkAllocationCallbacks *callbacks) {
  auto entry = active_allocations->sizes.find(memory);
  if (entry != active_allocations->sizes.end()) {
    active_allocations->live -= entry->second;
    active_allocations->sizes.erase(entry);
  }
  vkFreeMemory(device, memory, callbacks);
}
}  // namespace

// Local to this translation unit, covering upstream allocation calls only.
#define vkAllocateMemory allocate_memory
#define vkFreeMemory free_memory
#define VKFFT_BACKEND 0
#include <vkFFT.h>
#undef vkAllocateMemory
#undef vkFreeMemory

namespace {
thread_local char error_message[256]{};
std::mutex compiler_mutex;

int fail(const char *operation, int code) {
  std::snprintf(error_message, sizeof(error_message), "%s (%d)", operation,
                code);
  return code == 0 ? -1 : code;
}

struct Plan {
  TiForgeVkfftConfig config{};
  VkCommandPool pool{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  VkCommandBuffer executable{VK_NULL_HANDLE};
  VkFFTApplication application{};
  Allocations allocations;
  bool initialized{false};

  ~Plan() {
    AllocationScope scope(allocations);
    if (initialized) {
      deleteVkFFT(&application);
    }
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(config.device, fence, nullptr);
    }
    if (pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(config.device, pool, nullptr);
    }
  }
};

bool supported_size(uint64_t size) {
  if (size == 0) {
    return false;
  }
  for (uint64_t radix : {2, 3, 5, 7, 11, 13}) {
    while (size % radix == 0) {
      size /= radix;
    }
  }
  return size == 1;
}

int create_plan(const TiForgeVkfftConfig *config, TiForgeVkfftPlan *out) {
  if (!out) {
    return fail("missing plan output", -1);
  }
  *out = nullptr;
  if (!config || config->struct_size != sizeof(*config) || config->rank < 1 ||
      config->rank > 3 || config->batches == 0 || !config->physical_device ||
      !config->device || !config->queue || !config->buffer ||
      config->reserved || (config->direction != -1 && config->direction != 1) ||
      config->normalize_inverse > 1) {
    return fail("invalid compact complex-f32 plan configuration", -1);
  }
  uint64_t bytes = 2 * sizeof(float);
  for (uint32_t axis = 0; axis <= config->rank; ++axis) {
    const auto extent =
        axis == config->rank ? config->batches : config->dimensions[axis];
    if (axis != config->rank && !supported_size(extent)) {
      return fail("unsupported dimension: prime factors must be <= 13", -1);
    }
    if (extent > std::numeric_limits<uint64_t>::max() / bytes) {
      return fail("FFT storage size overflow", -1);
    }
    bytes *= extent;
  }
  if (bytes != config->buffer_bytes) {
    return fail("FFT buffer size must equal compact batched storage", -1);
  }
  try {
    auto plan = std::make_unique<Plan>();
    plan->config = *config;
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = config->queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    auto result =
        vkCreateCommandPool(config->device, &pool_info, nullptr, &plan->pool);
    if (result != VK_SUCCESS) {
      return fail("vkCreateCommandPool", result);
    }
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    result = vkCreateFence(config->device, &fence_info, nullptr, &plan->fence);
    if (result != VK_SUCCESS) {
      return fail("vkCreateFence", result);
    }
    VkFFTConfiguration parameters{};
    parameters.FFTdim = config->rank;
    for (uint32_t axis = 0; axis < config->rank; ++axis) {
      parameters.size[axis] = config->dimensions[config->rank - axis - 1];
    }
    parameters.numberBatches = config->batches;
    parameters.physicalDevice = &plan->config.physical_device;
    parameters.device = &plan->config.device;
    parameters.queue = &plan->config.queue;
    parameters.commandPool = &plan->pool;
    parameters.fence = &plan->fence;
    parameters.buffer = &plan->config.buffer;
    parameters.bufferSize = &plan->config.buffer_bytes;
    parameters.normalize = config->normalize_inverse;
    parameters.makeForwardPlanOnly = config->direction == -1;
    parameters.makeInversePlanOnly = config->direction == 1;
    // No shared global compiler lifetime races between cold plan builds.
    std::lock_guard<std::mutex> compiler_lock(compiler_mutex);
    AllocationScope scope(plan->allocations);
    const auto fft_result = initializeVkFFT(&plan->application, parameters);
    if (fft_result != VKFFT_SUCCESS) {
      // initializeVkFFT owns cleanup of partially initialized applications.
      return fail("initializeVkFFT", fft_result);
    }
    plan->initialized = true;
    VkCommandBufferAllocateInfo allocation{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = plan->pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocation.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(config->device, &allocation,
                                      &plan->executable);
    if (result != VK_SUCCESS) {
      return fail("vkAllocateCommandBuffers", result);
    }
    VkCommandBufferInheritanceInfo inheritance{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin.pInheritanceInfo = &inheritance;
    result = vkBeginCommandBuffer(plan->executable, &begin);
    if (result != VK_SUCCESS) {
      return fail("vkBeginCommandBuffer", result);
    }
    VkFFTLaunchParams launch{};
    launch.commandBuffer = &plan->executable;
    const auto record_result =
        VkFFTAppend(&plan->application, config->direction, &launch);
    if (record_result != VKFFT_SUCCESS) {
      return fail("VkFFTAppend during plan recording", record_result);
    }
    result = vkEndCommandBuffer(plan->executable);
    if (result != VK_SUCCESS) {
      return fail("vkEndCommandBuffer", result);
    }
    *out = plan.release();
    error_message[0] = '\0';
    return 0;
  } catch (const std::bad_alloc &) {
    return fail("host allocation failed during plan creation", -1);
  } catch (...) {
    return fail("unexpected failure during plan creation", -1);
  }
}

int append_plan(TiForgeVkfftPlan handle, VkCommandBuffer command) {
  auto *plan = static_cast<Plan *>(handle);
  // The complete vendor dispatch sequence was recorded once at plan creation.
  // Reuse it inside the caller's ordered primary command list, without a
  // vendor host dispatch loop, descriptor rebinding, JIT or extra submission.
  vkCmdExecuteCommands(command, 1, &plan->executable);
  return 0;
}

void plan_memory(TiForgeVkfftPlan handle, TiForgeVkfftMemory *out) {
  const auto *plan = static_cast<Plan *>(handle);
  const auto &config = plan->application.configuration;
  *out = {plan->allocations.live, plan->allocations.peak,
          static_cast<uint64_t>(plan->allocations.sizes.size()),
          config.allocateTempBuffer ? config.tempBufferSize[0] : 0};
}

void destroy_plan(TiForgeVkfftPlan handle) {
  delete static_cast<Plan *>(handle);
}

const char *last_error() {
  return error_message;
}
}  // namespace

int taichi_forge_vkfft_provider_query(uint32_t abi,
                                      size_t size,
                                      TiForgeVkfftApi *api) {
  if (abi != TI_FORGE_VKFFT_ABI_VERSION || size != sizeof(*api) || !api) {
    return fail("VkFFT adapter ABI mismatch", -1);
  }
  *api = {sizeof(*api),
          TI_FORGE_VKFFT_ABI_VERSION,
          static_cast<uint32_t>(VkFFTGetVersion()),
          GLSLANG_VERSION_MAJOR,
          GLSLANG_VERSION_MINOR,
          GLSLANG_VERSION_PATCH,
          create_plan,
          append_plan,
          plan_memory,
          destroy_plan,
          last_error};
  return 0;
}
