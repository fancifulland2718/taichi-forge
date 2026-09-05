#include <cub/cub.cuh>
#include <cub/version.cuh>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>

#include "segmented_scan.cuh"

#if defined(_WIN32)
#define TI_FORGE_CUB_EXPORT __declspec(dllexport)
#else
#define TI_FORGE_CUB_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr std::uint32_t kProviderAbiVersion = 1;
constexpr std::uint32_t kSuccess = 0;
constexpr std::uint32_t kInvalidArgument = 1;
constexpr std::uint32_t kCudaFailure = 2;
constexpr std::uint32_t kInternalFailure = 3;

constexpr std::uint32_t kRadixSortPairsU32 = 1;
constexpr std::uint32_t kRadixSortPairsU64 = 2;
constexpr std::uint32_t kExclusiveScanU32 = 3;
constexpr std::uint32_t kSelectFlaggedU32 = 4;
constexpr std::uint32_t kSegmentedInclusiveScanU32 = 5;
constexpr std::uint32_t kSegmentedExclusiveScanU32 = 6;

constexpr std::uint64_t kFeatureRadixSortPairsU32 = 1ull << 0;
constexpr std::uint64_t kFeatureRadixSortPairsU64 = 1ull << 1;
constexpr std::uint64_t kFeatureExclusiveScanU32 = 1ull << 2;
constexpr std::uint64_t kFeatureSelectFlaggedU32 = 1ull << 3;
constexpr std::uint64_t kFeatureSegmentedInclusiveScanU32 = 1ull << 4;
constexpr std::uint64_t kFeatureSegmentedExclusiveScanU32 = 1ull << 5;
constexpr std::uint64_t kFeatures =
    kFeatureRadixSortPairsU32 | kFeatureRadixSortPairsU64 |
    kFeatureExclusiveScanU32 | kFeatureSelectFlaggedU32 |
    kFeatureSegmentedInclusiveScanU32 | kFeatureSegmentedExclusiveScanU32;

thread_local std::string last_error;

struct ProviderInfo {
  std::uint32_t struct_size;
  std::uint32_t provider_abi_version;
  std::uint32_t cuda_runtime_version;
  std::uint32_t cub_version;
  std::uint64_t features;
};

struct Invocation {
  std::uint32_t struct_size;
  std::uint32_t operation;
  std::uint32_t flags;
  std::uint32_t reserved;
  std::uint64_t num_items;
  const void *input0;
  const void *input1;
  void *output0;
  void *output1;
  void *workspace;
  std::size_t workspace_bytes;
  void *stream;
};

std::uint32_t fail(std::uint32_t status, const std::string &message) {
  last_error = message;
  return status;
}

std::uint32_t check_invocation(const Invocation *invocation) {
  if (!invocation || invocation->struct_size != sizeof(Invocation)) {
    return fail(kInvalidArgument, "invalid CUB invocation structure");
  }
  if (invocation->flags != 0 || invocation->reserved != 0) {
    return fail(kInvalidArgument, "unsupported CUB invocation flags");
  }
  if (invocation->num_items >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return fail(kInvalidArgument, "CUB item count exceeds INT_MAX");
  }
  switch (invocation->operation) {
    case kRadixSortPairsU32:
    case kRadixSortPairsU64:
    case kExclusiveScanU32:
    case kSelectFlaggedU32:
    case kSegmentedInclusiveScanU32:
    case kSegmentedExclusiveScanU32:
      return kSuccess;
    default:
      return fail(kInvalidArgument, "unsupported CUB operation");
  }
}

template <typename Key>
cudaError_t sort_pairs(const Invocation &invocation,
                       std::size_t &workspace_bytes) {
  return cub::DeviceRadixSort::SortPairs(
      invocation.workspace, workspace_bytes,
      static_cast<const Key *>(invocation.input0),
      static_cast<Key *>(invocation.output0),
      static_cast<const std::uint32_t *>(invocation.input1),
      static_cast<std::uint32_t *>(invocation.output1),
      static_cast<int>(invocation.num_items), 0, sizeof(Key) * 8,
      static_cast<cudaStream_t>(invocation.stream));
}

cudaError_t invoke(const Invocation &invocation, std::size_t &workspace_bytes) {
  switch (invocation.operation) {
    case kSegmentedInclusiveScanU32:
      return forge_cub::segmented_scan<false>(
          invocation.workspace, workspace_bytes,
          static_cast<const std::uint32_t *>(invocation.input0),
          static_cast<const std::uint32_t *>(invocation.input1),
          static_cast<std::uint32_t *>(invocation.output0),
          static_cast<int>(invocation.num_items),
          static_cast<cudaStream_t>(invocation.stream));
    case kSegmentedExclusiveScanU32:
      return forge_cub::segmented_scan<true>(
          invocation.workspace, workspace_bytes,
          static_cast<const std::uint32_t *>(invocation.input0),
          static_cast<const std::uint32_t *>(invocation.input1),
          static_cast<std::uint32_t *>(invocation.output0),
          static_cast<int>(invocation.num_items),
          static_cast<cudaStream_t>(invocation.stream));
    case kRadixSortPairsU32:
      return sort_pairs<std::uint32_t>(invocation, workspace_bytes);
    case kRadixSortPairsU64:
      return sort_pairs<std::uint64_t>(invocation, workspace_bytes);
    case kExclusiveScanU32:
      return cub::DeviceScan::ExclusiveSum(
          invocation.workspace, workspace_bytes,
          static_cast<const std::uint32_t *>(invocation.input0),
          static_cast<std::uint32_t *>(invocation.output0),
          static_cast<int>(invocation.num_items),
          static_cast<cudaStream_t>(invocation.stream));
    case kSelectFlaggedU32:
      return cub::DeviceSelect::Flagged(
          invocation.workspace, workspace_bytes,
          static_cast<const std::uint32_t *>(invocation.input0),
          static_cast<const std::uint32_t *>(invocation.input1),
          static_cast<std::uint32_t *>(invocation.output0),
          static_cast<std::uint32_t *>(invocation.output1),
          static_cast<int>(invocation.num_items),
          static_cast<cudaStream_t>(invocation.stream));
    default:
      return cudaErrorInvalidValue;
  }
}

bool required_pointers_present(const Invocation &invocation) {
  if (invocation.num_items == 0) {
    return invocation.operation != kSelectFlaggedU32 || invocation.output1;
  }
  if (!invocation.input0 || !invocation.output0) {
    return false;
  }
  if ((invocation.operation == kRadixSortPairsU32 ||
       invocation.operation == kRadixSortPairsU64 ||
       invocation.operation == kSegmentedInclusiveScanU32 ||
       invocation.operation == kSegmentedExclusiveScanU32 ||
       invocation.operation == kSelectFlaggedU32) &&
      !invocation.input1) {
    return false;
  }
  if ((invocation.operation == kRadixSortPairsU32 ||
       invocation.operation == kRadixSortPairsU64 ||
       invocation.operation == kSelectFlaggedU32) &&
      !invocation.output1) {
    return false;
  }
  return true;
}

}  // namespace

extern "C" {

TI_FORGE_CUB_EXPORT std::uint32_t ti_forge_cub_source_provider_query(
    std::uint32_t requested_abi,
    std::size_t info_size,
    ProviderInfo *info) {
  last_error.clear();
  if (requested_abi != kProviderAbiVersion || !info ||
      info_size != sizeof(ProviderInfo)) {
    return fail(kInvalidArgument, "unsupported Forge CUB source-provider ABI");
  }
  ProviderInfo result{};
  result.struct_size = sizeof(ProviderInfo);
  result.provider_abi_version = kProviderAbiVersion;
  result.cuda_runtime_version = CUDART_VERSION;
  result.cub_version = CUB_VERSION;
  result.features = kFeatures;
  *info = result;
  return kSuccess;
}

TI_FORGE_CUB_EXPORT std::uint32_t ti_forge_cub_source_provider_workspace_bytes(
    const Invocation *invocation,
    std::size_t *workspace_bytes) {
  last_error.clear();
  const auto validation = check_invocation(invocation);
  if (validation != kSuccess || !workspace_bytes) {
    return validation != kSuccess
               ? validation
               : fail(kInvalidArgument, "workspace output is null");
  }
  try {
    Invocation query = *invocation;
    query.workspace = nullptr;
    query.workspace_bytes = 0;
    std::size_t required = 0;
    const auto result = invoke(query, required);
    if (result != cudaSuccess) {
      return fail(kCudaFailure, std::string("CUB workspace query failed: ") +
                                    cudaGetErrorString(result));
    }
    *workspace_bytes = required;
    return kSuccess;
  } catch (const std::exception &error) {
    return fail(kInternalFailure, error.what());
  } catch (...) {
    return fail(kInternalFailure, "unknown CUB workspace-query failure");
  }
}

TI_FORGE_CUB_EXPORT std::uint32_t ti_forge_cub_source_provider_execute(
    const Invocation *invocation) {
  last_error.clear();
  const auto validation = check_invocation(invocation);
  if (validation != kSuccess) {
    return validation;
  }
  if (!required_pointers_present(*invocation)) {
    return fail(kInvalidArgument, "CUB invocation has a null device pointer");
  }
  if (invocation->num_items == 0) {
    if (invocation->operation == kSelectFlaggedU32) {
      const auto result =
          cudaMemsetAsync(invocation->output1, 0, sizeof(std::uint32_t),
                          static_cast<cudaStream_t>(invocation->stream));
      if (result != cudaSuccess) {
        return fail(kCudaFailure, std::string("CUB empty select failed: ") +
                                      cudaGetErrorString(result));
      }
    }
    return kSuccess;
  }
  if (!invocation->workspace || invocation->workspace_bytes == 0) {
    return fail(kInvalidArgument,
                "CUB invocation requires caller-owned workspace");
  }
  try {
    std::size_t available = invocation->workspace_bytes;
    const auto result = invoke(*invocation, available);
    if (result != cudaSuccess) {
      return fail(kCudaFailure, std::string("CUB execution failed: ") +
                                    cudaGetErrorString(result));
    }
    return kSuccess;
  } catch (const std::exception &error) {
    return fail(kInternalFailure, error.what());
  } catch (...) {
    return fail(kInternalFailure, "unknown CUB execution failure");
  }
}

TI_FORGE_CUB_EXPORT std::size_t ti_forge_cub_source_provider_get_last_error(
    char *buffer,
    std::size_t capacity) {
  const auto required = last_error.size() + 1;
  if (buffer && capacity) {
    const auto count = capacity < required ? capacity - 1 : last_error.size();
    if (count) {
      std::memcpy(buffer, last_error.data(), count);
    }
    buffer[count] = '\0';
  }
  return required;
}

}  // extern "C"
