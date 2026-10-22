#include <gtest/gtest.h>

#ifdef TI_TEST_RUNTIME_C_API
#include <cstddef>
#include "c_api/src/completion.h"
#include "taichi/program/runtime_fault.h"

namespace taichi::lang {

static_assert(sizeof(TiCompletionInfo) == 24);
static_assert(offsetof(TiCompletionInfo, owner) == 0);
static_assert(offsetof(TiCompletionInfo, sequence) == 8);
static_assert(offsetof(TiCompletionInfo, has_backend_work) == 16);
static_assert(sizeof(TiCompletionGpuTiming) == 32);
static_assert(offsetof(TiCompletionGpuTiming, duration_ns) == 24);
static_assert(TI_COMPLETION_SOURCE_DOMAIN == TI_COMPLETION_SOURCE_CUDA_DOMAIN);

TEST(CompletionPublication, CoreContractAndWaitSetAcceptEveryBackendIdentity) {
  // Completed prefixes require no backend initialization. The common C API
  // must not depend on CUDA availability, nor reinterpret another handle type.
  for (const auto arch : {host_arch(), Arch::cuda, Arch::vulkan}) {
    const std::uint64_t owner_id = (std::uint64_t{1} << 51) + 17;
    const std::uint64_t sequence = (std::uint64_t{1} << 49) + 23;
    auto owner = std::make_shared<RuntimeFaultDomain>(arch, owner_id);
    const auto handle = capi::CompletionPublication{}.commit(
        RuntimeCompletion::completed(arch, owner_id, sequence, owner));
    ti_set_last_error(TI_ERROR_SUCCESS, "");
    EXPECT_TRUE(ti_query_completion_ext(handle));
    TiCompletionInfo info{};
    ti_get_completion_info_ext(handle, &info);
    EXPECT_EQ(info.owner, owner_id);
    EXPECT_EQ(info.sequence, sequence);
    EXPECT_FALSE(info.has_backend_work);
    TiCompletionGpuTiming timing{};
    ti_get_completion_gpu_timing_ext(handle, &timing);
    EXPECT_EQ(timing.status, TI_COMPLETION_GPU_TIMING_NOT_SAMPLED);
    EXPECT_EQ(timing.owner, owner_id);
    EXPECT_EQ(timing.sequence, sequence);
    const TiCompletionSource source{TI_COMPLETION_SOURCE_DOMAIN, handle};
    const auto set = ti_create_completion_set_ext(1, &source, 1000);
    ASSERT_NE(set, nullptr);
    ti_release_completion_ext(handle);
    TiCompletionWaitResult result{};
    ti_wait_completion_set_ext(set, 0, &result);
    EXPECT_EQ(result.status, TI_COMPLETION_WAIT_READY);
    EXPECT_EQ(result.index, 0u);
    ti_release_completion_set_ext(set);
    EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
    EXPECT_THROW(capi::get_completion(handle), std::exception);
  }
}

TEST(CompletionPublication, FailedReservationDoesNotPublishATicket) {
  {
    capi::CompletionPublication reservation;
    EXPECT_THROW(reservation.commit({}), std::logic_error);
    // Destruction rolls back the unpublished handle without a device wait.
  }
  const auto handle = capi::CompletionPublication{}.commit(
      RuntimeCompletion::completed(host_arch(), 7, 11));
  EXPECT_EQ(capi::get_completion(handle).sequence(), 11u);
  ti_release_completion_ext(handle);
  EXPECT_THROW(capi::get_completion(handle), std::exception);
}

}  // namespace taichi::lang
#endif
