#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"

#include <array>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#if defined(TI_WITH_VULKAN)

namespace taichi::lang {
namespace {

constexpr uint32_t kRadixBits = 4;
constexpr uint32_t kRadixBins = 1u << kRadixBits;
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kSingleChunkPrefixMaxBlocks = 32;
constexpr uint32_t kInlineChunkPrefixMaxChunks = 4;
constexpr uint32_t kRadix8Bins = 256;
constexpr uint32_t kRadix8PartitionSize = 2048;
constexpr uint32_t kHistogramPrivateChunkSize = 2048;
constexpr uint32_t kReducePrivateChunkSize = 2048;

struct VulkanSortCpuProfileSample {
  uint64_t sort_calls{0};
  uint64_t lambda_calls{0};
  uint64_t workspace_reallocs{0};
  uint64_t realloc_sync_calls{0};
  uint64_t internal_sync_calls{0};
  uint64_t dispatch_calls{0};
  uint64_t bind_pipeline_calls{0};
  uint64_t bind_shader_resources_calls{0};
  uint64_t resource_set_calls{0};
  uint64_t resource_set_create_calls{0};
  uint64_t rw_buffer_calls{0};
  uint64_t buffer_fill_calls{0};
  uint64_t buffer_barrier_calls{0};
  uint64_t buffer_copy_calls{0};
  double total_call_us{0.0};
  double get_cache_us{0.0};
  double ensure_workspace_us{0.0};
  double enqueue_us{0.0};
  double realloc_sync_us{0.0};
  double internal_sync_us{0.0};
  double lambda_total_us{0.0};
  double radix8_body_us{0.0};
  double bind_pipeline_us{0.0};
  double bind_shader_resources_us{0.0};
  double profiler_scope_us{0.0};
  double dispatch_us{0.0};
  double resource_set_us{0.0};
  double rw_buffer_us{0.0};
  double buffer_fill_us{0.0};
  double buffer_barrier_us{0.0};
  double buffer_copy_us{0.0};

  void add(const VulkanSortCpuProfileSample &other) {
#define TI_VULKAN_SORT_PROFILE_ADD_FIELD(name) name += other.name
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(sort_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(lambda_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(workspace_reallocs);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(realloc_sync_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(internal_sync_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(dispatch_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_pipeline_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_shader_resources_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(resource_set_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(resource_set_create_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(rw_buffer_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_fill_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_barrier_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_copy_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(total_call_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(get_cache_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(ensure_workspace_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(enqueue_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(realloc_sync_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(internal_sync_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(lambda_total_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(radix8_body_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_pipeline_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_shader_resources_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(profiler_scope_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(dispatch_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(resource_set_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(rw_buffer_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_fill_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_barrier_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_copy_us);
#undef TI_VULKAN_SORT_PROFILE_ADD_FIELD
  }
};

struct VulkanSortCpuProfileTotals {
  std::mutex mutex;
  VulkanSortCpuProfileSample totals;

  void clear() {
    std::lock_guard<std::mutex> guard(mutex);
    totals = {};
  }

  void merge(const VulkanSortCpuProfileSample &sample) {
    std::lock_guard<std::mutex> guard(mutex);
    totals.add(sample);
  }

  std::string report_json() {
    std::lock_guard<std::mutex> guard(mutex);
    const auto &t = totals;
    const double denom = t.sort_calls ? static_cast<double>(t.sort_calls) : 1.0;
    auto per_sort = [&](double value) { return value / denom; };
    auto count_per_sort = [&](uint64_t value) {
      return static_cast<double>(value) / denom;
    };
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{";
    os << "\"sort_calls\":" << t.sort_calls << ",";
    os << "\"lambda_calls\":" << t.lambda_calls << ",";
    os << "\"counts_per_sort\":{";
    os << "\"workspace_reallocs\":" << count_per_sort(t.workspace_reallocs)
       << ",";
    os << "\"realloc_sync_calls\":" << count_per_sort(t.realloc_sync_calls)
       << ",";
    os << "\"internal_sync_calls\":" << count_per_sort(t.internal_sync_calls)
       << ",";
    os << "\"dispatch_calls\":" << count_per_sort(t.dispatch_calls) << ",";
    os << "\"bind_pipeline_calls\":" << count_per_sort(t.bind_pipeline_calls)
       << ",";
    os << "\"bind_shader_resources_calls\":"
       << count_per_sort(t.bind_shader_resources_calls) << ",";
    os << "\"resource_set_calls\":" << count_per_sort(t.resource_set_calls)
       << ",";
    os << "\"resource_set_create_calls\":"
       << count_per_sort(t.resource_set_create_calls) << ",";
    os << "\"rw_buffer_calls\":" << count_per_sort(t.rw_buffer_calls) << ",";
    os << "\"buffer_fill_calls\":" << count_per_sort(t.buffer_fill_calls)
       << ",";
    os << "\"buffer_barrier_calls\":"
       << count_per_sort(t.buffer_barrier_calls) << ",";
    os << "\"buffer_copy_calls\":" << count_per_sort(t.buffer_copy_calls);
    os << "},";
    os << "\"per_sort_us\":{";
    os << "\"total_call_us\":" << per_sort(t.total_call_us) << ",";
    os << "\"get_cache_us\":" << per_sort(t.get_cache_us) << ",";
    os << "\"ensure_workspace_us\":" << per_sort(t.ensure_workspace_us) << ",";
    os << "\"enqueue_us\":" << per_sort(t.enqueue_us) << ",";
    os << "\"realloc_sync_us\":" << per_sort(t.realloc_sync_us) << ",";
    os << "\"internal_sync_us\":" << per_sort(t.internal_sync_us) << ",";
    os << "\"lambda_total_us\":" << per_sort(t.lambda_total_us) << ",";
    os << "\"radix8_body_us\":" << per_sort(t.radix8_body_us) << ",";
    os << "\"bind_pipeline_us\":" << per_sort(t.bind_pipeline_us) << ",";
    os << "\"bind_shader_resources_us\":"
       << per_sort(t.bind_shader_resources_us) << ",";
    os << "\"profiler_scope_us\":" << per_sort(t.profiler_scope_us) << ",";
    os << "\"dispatch_us\":" << per_sort(t.dispatch_us) << ",";
    os << "\"resource_set_us\":" << per_sort(t.resource_set_us) << ",";
    os << "\"rw_buffer_us\":" << per_sort(t.rw_buffer_us) << ",";
    os << "\"buffer_fill_us\":" << per_sort(t.buffer_fill_us) << ",";
    os << "\"buffer_barrier_us\":" << per_sort(t.buffer_barrier_us) << ",";
    os << "\"buffer_copy_us\":" << per_sort(t.buffer_copy_us);
    os << "}}";
    return os.str();
  }
};

VulkanSortCpuProfileTotals g_vulkan_sort_cpu_profile;

bool vulkan_sort_cpu_profile_enabled() {
  return get_environ_config("TI_VULKAN_SORT_CPU_PROFILE", 0) != 0;
}

double profile_time_us() {
  return Time::get_time() * 1000000.0;
}

static const uint32_t kInitI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/init_i32.comp.spv.h"
    ;
static const uint32_t kCopyI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/copy_i32.comp.spv.h"
    ;
static const uint32_t kPrefixBlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/prefix_block.comp.spv.h"
    ;
static const uint32_t kPrefixChunksSpv[] =
#include "taichi/program/vulkan_sort_shaders/prefix_chunks.comp.spv.h"
    ;
static const uint32_t kPrefixSingleChunkSpv[] =
#include "taichi/program/vulkan_sort_shaders/prefix_single_chunk.comp.spv.h"
    ;
static const uint32_t kScanI32BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block.comp.spv.h"
    ;
static const uint32_t kScanI32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanI32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kCompactI32FlagsSpv[] =
#include "taichi/program/vulkan_sort_shaders/compact_i32_flags.comp.spv.h"
    ;
static const uint32_t kCompactI32ScatterSpv[] =
#include "taichi/program/vulkan_sort_shaders/compact_i32_scatter.comp.spv.h"
    ;
static const uint32_t kHistogramI32ClearSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_clear.comp.spv.h"
    ;
static const uint32_t kHistogramI32CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramI32CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramI32CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_count_private_shared.comp.spv.h"
    ;
static const uint32_t kHistogramI32ReducePrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_reduce_private.comp.spv.h"
    ;
static const uint32_t kHistogramI32SingleSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_single_shared.comp.spv.h"
    ;
static const uint32_t kReduceI32SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceI32MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_private.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_private.comp.spv.h"
    ;
static const uint32_t kReduceI32SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceI32MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_final.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_final.comp.spv.h"
    ;
static const uint32_t kReduceI32SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceI32MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_single.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_single.comp.spv.h"
    ;
static const uint32_t kTransformI32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_i32_affine.comp.spv.h"
    ;
static const uint32_t kTransformF32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_f32_affine.comp.spv.h"
    ;
static const uint32_t kGatherU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_u32_by_i32.comp.spv.h"
    ;

static const uint32_t kRankHistShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift0.comp.spv.h"
    ;
static const uint32_t kRankHistShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift4.comp.spv.h"
    ;
static const uint32_t kRankHistShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift8.comp.spv.h"
    ;
static const uint32_t kRankHistShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift12.comp.spv.h"
    ;
static const uint32_t kRankHistShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift16.comp.spv.h"
    ;
static const uint32_t kRankHistShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift20.comp.spv.h"
    ;
static const uint32_t kRankHistShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift24.comp.spv.h"
    ;
static const uint32_t kRankHistShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift28.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift0.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift4.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift8.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift12.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift16.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift20.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift24.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift28.comp.spv.h"
    ;

static const uint32_t kScatterKeysShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift0.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift4.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift8.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift12.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift16.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift20.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift24.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift28.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift0.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift4.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift8.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift12.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift16.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift20.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift24.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift28.comp.spv.h"
    ;

static const uint32_t kScatterPairsShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift28.comp.spv.h"
    ;

static const uint32_t kRadix8UpsweepShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_upsweep_shift0.comp.spv.h"
    ;
static const uint32_t kRadix8UpsweepShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_upsweep_shift8.comp.spv.h"
    ;
static const uint32_t kRadix8UpsweepShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_upsweep_shift16.comp.spv.h"
    ;
static const uint32_t kRadix8UpsweepShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_upsweep_shift24.comp.spv.h"
    ;
static const uint32_t kRadix8SpineSpv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_spine.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepKeysShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_keys_shift0.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepKeysShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_keys_shift8.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepKeysShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_keys_shift16.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepKeysShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_keys_shift24.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift0.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift8.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift16.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift24.comp.spv.h"
    ;

template <size_t N>
std::unique_ptr<Pipeline> create_pipeline(Device *device,
                                          const uint32_t (&spv)[N],
                                          const std::string &name) {
  PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                          spv,
                          sizeof(spv),
                          PipelineStageType::compute};
  auto [pipeline, res] = device->create_pipeline_unique(desc, name);
  TI_ERROR_IF(res != RhiResult::success,
              "Failed to create Vulkan sort pipeline '{}': RhiResult({})",
              name, res);
  return std::move(pipeline);
}

struct VulkanRadixSortCache {
  Device *device{nullptr};
  size_t capacity{0};
  size_t num_blocks{0};
  size_t cached_bytes{0};
  DeviceAllocation key_in{kDeviceNullAllocation};
  DeviceAllocation key_out{kDeviceNullAllocation};
  DeviceAllocation rank{kDeviceNullAllocation};
  DeviceAllocation hist{kDeviceNullAllocation};
  DeviceAllocation offsets{kDeviceNullAllocation};
  DeviceAllocation chunk_sums{kDeviceNullAllocation};
  DeviceAllocation chunk_offsets{kDeviceNullAllocation};
  DeviceAllocation radix8_global_hist{kDeviceNullAllocation};
  DeviceAllocation radix8_partition_hist{kDeviceNullAllocation};
  DeviceAllocation value_in{kDeviceNullAllocation};
  DeviceAllocation value_out{kDeviceNullAllocation};
  bool has_value_buffers{false};
  bool workspace_uses_radix8{false};
  bool subgroup_rank_enabled{false};
  bool radix8_enabled{false};
  bool inline_chunk_prefix_allowed{false};

  std::unique_ptr<Pipeline> init_i32;
  std::unique_ptr<Pipeline> copy_i32;
  std::unique_ptr<Pipeline> prefix_block;
  std::unique_ptr<Pipeline> prefix_chunks;
  std::unique_ptr<Pipeline> prefix_single_chunk;
  std::array<std::unique_ptr<Pipeline>, 8> rank_hist;
  std::array<std::unique_ptr<Pipeline>, 8> rank_hist_subgroup;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_keys;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_keys_inline_chunks;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_inline_chunks;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_upsweep;
  std::unique_ptr<Pipeline> radix8_spine;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_downsweep_keys;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_downsweep_pairs;
  std::unique_ptr<ShaderResourceSet> radix8_spine_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 4> radix8_upsweep_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 4>
      radix8_downsweep_keys_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 4>
      radix8_downsweep_pairs_bindings;

  void clear_allocs() {
    if (!device) {
      return;
    }
    for (DeviceAllocation *alloc :
         {&key_in, &key_out, &rank, &hist, &offsets, &chunk_sums,
          &chunk_offsets, &radix8_global_hist, &radix8_partition_hist,
          &value_in, &value_out}) {
      if (*alloc != kDeviceNullAllocation) {
        device->dealloc_memory(*alloc);
        *alloc = kDeviceNullAllocation;
      }
    }
    capacity = 0;
    num_blocks = 0;
    cached_bytes = 0;
    has_value_buffers = false;
    workspace_uses_radix8 = false;
  }

  void clear_resource_sets() {
    radix8_spine_bindings.reset();
    for (auto &bindings : radix8_upsweep_bindings) {
      bindings.reset();
    }
    for (auto &bindings : radix8_downsweep_keys_bindings) {
      bindings.reset();
    }
    for (auto &bindings : radix8_downsweep_pairs_bindings) {
      bindings.reset();
    }
  }

  ~VulkanRadixSortCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && init_i32) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      init_i32.reset();
      copy_i32.reset();
      prefix_block.reset();
      prefix_chunks.reset();
      prefix_single_chunk.reset();
      for (auto &pipeline : rank_hist) {
        pipeline.reset();
      }
      for (auto &pipeline : rank_hist_subgroup) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_keys) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_keys_inline_chunks) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs_inline_chunks) {
        pipeline.reset();
      }
      for (auto &pipeline : radix8_upsweep) {
        pipeline.reset();
      }
      radix8_spine.reset();
      for (auto &pipeline : radix8_downsweep_keys) {
        pipeline.reset();
      }
      for (auto &pipeline : radix8_downsweep_pairs) {
        pipeline.reset();
      }
      clear_resource_sets();
    }
    device = dev;
    const bool subgroup_rank_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_ballot) != 0;
    const bool subgroup_arithmetic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_arithmetic) !=
        0;
    const bool subgroup_rank_allowed =
        get_environ_config("TI_VULKAN_SORT_ENABLE_SUBGROUP_RANK", 1) != 0;
    const bool subgroup_rank_requested =
        subgroup_rank_supported && subgroup_rank_allowed;
    const bool radix8_requested =
        get_environ_config("TI_VULKAN_SORT_ENABLE_RADIX8", 1) != 0;
    const bool radix8_supported =
        subgroup_rank_supported && subgroup_arithmetic_supported;
    inline_chunk_prefix_allowed =
        get_environ_config("TI_VULKAN_SORT_ENABLE_INLINE_CHUNK_PREFIX", 1) != 0;
    if (init_i32) {
      subgroup_rank_enabled =
          subgroup_rank_requested && rank_hist_subgroup[0] != nullptr;
      radix8_enabled = radix8_requested && radix8_spine != nullptr;
      return;
    }
    subgroup_rank_enabled = subgroup_rank_requested;
    radix8_enabled = radix8_requested && radix8_supported;
    init_i32 = create_pipeline(dev, kInitI32Spv, "vulkan_sort_init_i32");
    copy_i32 = create_pipeline(dev, kCopyI32Spv, "vulkan_sort_copy_i32");
    prefix_block =
        create_pipeline(dev, kPrefixBlockSpv, "vulkan_sort_prefix_block");
    prefix_chunks =
        create_pipeline(dev, kPrefixChunksSpv, "vulkan_sort_prefix_chunks");
    prefix_single_chunk = create_pipeline(dev, kPrefixSingleChunkSpv,
                                          "vulkan_sort_prefix_single_chunk");

    const std::array<const uint32_t *, 8> rank_data = {
        kRankHistShift0Spv,  kRankHistShift4Spv,  kRankHistShift8Spv,
        kRankHistShift12Spv, kRankHistShift16Spv, kRankHistShift20Spv,
        kRankHistShift24Spv, kRankHistShift28Spv};
    const std::array<size_t, 8> rank_sizes = {
        sizeof(kRankHistShift0Spv),  sizeof(kRankHistShift4Spv),
        sizeof(kRankHistShift8Spv),  sizeof(kRankHistShift12Spv),
        sizeof(kRankHistShift16Spv), sizeof(kRankHistShift20Spv),
        sizeof(kRankHistShift24Spv), sizeof(kRankHistShift28Spv)};
    const std::array<const uint32_t *, 8> rank_subgroup_data = {
        kRankHistSubgroupShift0Spv,  kRankHistSubgroupShift4Spv,
        kRankHistSubgroupShift8Spv,  kRankHistSubgroupShift12Spv,
        kRankHistSubgroupShift16Spv, kRankHistSubgroupShift20Spv,
        kRankHistSubgroupShift24Spv, kRankHistSubgroupShift28Spv};
    const std::array<size_t, 8> rank_subgroup_sizes = {
        sizeof(kRankHistSubgroupShift0Spv),
        sizeof(kRankHistSubgroupShift4Spv),
        sizeof(kRankHistSubgroupShift8Spv),
        sizeof(kRankHistSubgroupShift12Spv),
        sizeof(kRankHistSubgroupShift16Spv),
        sizeof(kRankHistSubgroupShift20Spv),
        sizeof(kRankHistSubgroupShift24Spv),
        sizeof(kRankHistSubgroupShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_key_data = {
        kScatterKeysShift0Spv,  kScatterKeysShift4Spv,
        kScatterKeysShift8Spv,  kScatterKeysShift12Spv,
        kScatterKeysShift16Spv, kScatterKeysShift20Spv,
        kScatterKeysShift24Spv, kScatterKeysShift28Spv};
    const std::array<size_t, 8> scatter_key_sizes = {
        sizeof(kScatterKeysShift0Spv),  sizeof(kScatterKeysShift4Spv),
        sizeof(kScatterKeysShift8Spv),  sizeof(kScatterKeysShift12Spv),
        sizeof(kScatterKeysShift16Spv), sizeof(kScatterKeysShift20Spv),
        sizeof(kScatterKeysShift24Spv), sizeof(kScatterKeysShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_key_inline_data = {
        kScatterKeysInlineChunksShift0Spv,
        kScatterKeysInlineChunksShift4Spv,
        kScatterKeysInlineChunksShift8Spv,
        kScatterKeysInlineChunksShift12Spv,
        kScatterKeysInlineChunksShift16Spv,
        kScatterKeysInlineChunksShift20Spv,
        kScatterKeysInlineChunksShift24Spv,
        kScatterKeysInlineChunksShift28Spv};
    const std::array<size_t, 8> scatter_key_inline_sizes = {
        sizeof(kScatterKeysInlineChunksShift0Spv),
        sizeof(kScatterKeysInlineChunksShift4Spv),
        sizeof(kScatterKeysInlineChunksShift8Spv),
        sizeof(kScatterKeysInlineChunksShift12Spv),
        sizeof(kScatterKeysInlineChunksShift16Spv),
        sizeof(kScatterKeysInlineChunksShift20Spv),
        sizeof(kScatterKeysInlineChunksShift24Spv),
        sizeof(kScatterKeysInlineChunksShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_data = {
        kScatterPairsShift0Spv,  kScatterPairsShift4Spv,
        kScatterPairsShift8Spv,  kScatterPairsShift12Spv,
        kScatterPairsShift16Spv, kScatterPairsShift20Spv,
        kScatterPairsShift24Spv, kScatterPairsShift28Spv};
    const std::array<size_t, 8> scatter_pair_sizes = {
        sizeof(kScatterPairsShift0Spv),  sizeof(kScatterPairsShift4Spv),
        sizeof(kScatterPairsShift8Spv),  sizeof(kScatterPairsShift12Spv),
        sizeof(kScatterPairsShift16Spv), sizeof(kScatterPairsShift20Spv),
        sizeof(kScatterPairsShift24Spv), sizeof(kScatterPairsShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_inline_data = {
        kScatterPairsInlineChunksShift0Spv,
        kScatterPairsInlineChunksShift4Spv,
        kScatterPairsInlineChunksShift8Spv,
        kScatterPairsInlineChunksShift12Spv,
        kScatterPairsInlineChunksShift16Spv,
        kScatterPairsInlineChunksShift20Spv,
        kScatterPairsInlineChunksShift24Spv,
        kScatterPairsInlineChunksShift28Spv};
    const std::array<size_t, 8> scatter_pair_inline_sizes = {
        sizeof(kScatterPairsInlineChunksShift0Spv),
        sizeof(kScatterPairsInlineChunksShift4Spv),
        sizeof(kScatterPairsInlineChunksShift8Spv),
        sizeof(kScatterPairsInlineChunksShift12Spv),
        sizeof(kScatterPairsInlineChunksShift16Spv),
        sizeof(kScatterPairsInlineChunksShift20Spv),
        sizeof(kScatterPairsInlineChunksShift24Spv),
        sizeof(kScatterPairsInlineChunksShift28Spv)};
    for (int pass = 0; pass < 8; ++pass) {
      PipelineSourceDesc rank_desc{PipelineSourceType::spirv_binary,
                                   rank_data[pass],
                                   rank_sizes[pass],
                                   PipelineStageType::compute};
      auto [rank_pipeline, rank_res] = dev->create_pipeline_unique(
          rank_desc, fmt::format("vulkan_sort_rank_hist_{}", pass));
      TI_ERROR_IF(rank_res != RhiResult::success,
                  "Failed to create Vulkan sort rank pipeline {}: "
                  "RhiResult({})",
                  pass, rank_res);
      rank_hist[pass] = std::move(rank_pipeline);

      if (subgroup_rank_enabled) {
        PipelineSourceDesc rank_subgroup_desc{
            PipelineSourceType::spirv_binary,
            rank_subgroup_data[pass],
            rank_subgroup_sizes[pass],
            PipelineStageType::compute};
        auto [rank_subgroup_pipeline, rank_subgroup_res] =
            dev->create_pipeline_unique(
                rank_subgroup_desc,
                fmt::format("vulkan_sort_rank_hist_subgroup_{}", pass));
        TI_ERROR_IF(rank_subgroup_res != RhiResult::success,
                    "Failed to create Vulkan sort subgroup rank pipeline {}: "
                    "RhiResult({})",
                    pass, rank_subgroup_res);
        rank_hist_subgroup[pass] = std::move(rank_subgroup_pipeline);
      }

      PipelineSourceDesc scatter_key_desc{PipelineSourceType::spirv_binary,
                                          scatter_key_data[pass],
                                          scatter_key_sizes[pass],
                                          PipelineStageType::compute};
      auto [scatter_key_pipeline, scatter_key_res] =
          dev->create_pipeline_unique(
              scatter_key_desc,
              fmt::format("vulkan_sort_scatter_keys_{}", pass));
      TI_ERROR_IF(scatter_key_res != RhiResult::success,
                  "Failed to create Vulkan sort key scatter pipeline {}: "
                  "RhiResult({})",
                  pass, scatter_key_res);
      scatter_keys[pass] = std::move(scatter_key_pipeline);

      PipelineSourceDesc scatter_key_inline_desc{
          PipelineSourceType::spirv_binary,
          scatter_key_inline_data[pass],
          scatter_key_inline_sizes[pass],
          PipelineStageType::compute};
      auto [scatter_key_inline_pipeline, scatter_key_inline_res] =
          dev->create_pipeline_unique(
              scatter_key_inline_desc,
              fmt::format("vulkan_sort_scatter_keys_inline_chunks_{}", pass));
      TI_ERROR_IF(scatter_key_inline_res != RhiResult::success,
                  "Failed to create Vulkan sort inline key scatter pipeline "
                  "{}: RhiResult({})",
                  pass, scatter_key_inline_res);
      scatter_keys_inline_chunks[pass] = std::move(scatter_key_inline_pipeline);

      PipelineSourceDesc scatter_pair_desc{PipelineSourceType::spirv_binary,
                                           scatter_pair_data[pass],
                                           scatter_pair_sizes[pass],
                                           PipelineStageType::compute};
      auto [scatter_pair_pipeline, scatter_pair_res] =
          dev->create_pipeline_unique(
              scatter_pair_desc,
              fmt::format("vulkan_sort_scatter_pairs_{}", pass));
      TI_ERROR_IF(scatter_pair_res != RhiResult::success,
                  "Failed to create Vulkan sort pair scatter pipeline {}: "
                  "RhiResult({})",
                  pass, scatter_pair_res);
      scatter_pairs[pass] = std::move(scatter_pair_pipeline);

      PipelineSourceDesc scatter_pair_inline_desc{
          PipelineSourceType::spirv_binary,
          scatter_pair_inline_data[pass],
          scatter_pair_inline_sizes[pass],
          PipelineStageType::compute};
      auto [scatter_pair_inline_pipeline, scatter_pair_inline_res] =
          dev->create_pipeline_unique(
              scatter_pair_inline_desc,
              fmt::format("vulkan_sort_scatter_pairs_inline_chunks_{}", pass));
      TI_ERROR_IF(scatter_pair_inline_res != RhiResult::success,
                  "Failed to create Vulkan sort inline pair scatter pipeline "
                  "{}: RhiResult({})",
                  pass, scatter_pair_inline_res);
      scatter_pairs_inline_chunks[pass] =
          std::move(scatter_pair_inline_pipeline);
    }

    if (radix8_supported) {
      radix8_spine =
          create_pipeline(dev, kRadix8SpineSpv, "vulkan_sort_radix8_spine");

      const std::array<const uint32_t *, 4> upsweep_data = {
          kRadix8UpsweepShift0Spv, kRadix8UpsweepShift8Spv,
          kRadix8UpsweepShift16Spv, kRadix8UpsweepShift24Spv};
      const std::array<size_t, 4> upsweep_sizes = {
          sizeof(kRadix8UpsweepShift0Spv), sizeof(kRadix8UpsweepShift8Spv),
          sizeof(kRadix8UpsweepShift16Spv),
          sizeof(kRadix8UpsweepShift24Spv)};
      const std::array<const uint32_t *, 4> downsweep_key_data = {
          kRadix8DownsweepKeysShift0Spv, kRadix8DownsweepKeysShift8Spv,
          kRadix8DownsweepKeysShift16Spv, kRadix8DownsweepKeysShift24Spv};
      const std::array<size_t, 4> downsweep_key_sizes = {
          sizeof(kRadix8DownsweepKeysShift0Spv),
          sizeof(kRadix8DownsweepKeysShift8Spv),
          sizeof(kRadix8DownsweepKeysShift16Spv),
          sizeof(kRadix8DownsweepKeysShift24Spv)};
      const std::array<const uint32_t *, 4> downsweep_pair_data = {
          kRadix8DownsweepPairsShift0Spv, kRadix8DownsweepPairsShift8Spv,
          kRadix8DownsweepPairsShift16Spv,
          kRadix8DownsweepPairsShift24Spv};
      const std::array<size_t, 4> downsweep_pair_sizes = {
          sizeof(kRadix8DownsweepPairsShift0Spv),
          sizeof(kRadix8DownsweepPairsShift8Spv),
          sizeof(kRadix8DownsweepPairsShift16Spv),
          sizeof(kRadix8DownsweepPairsShift24Spv)};

      for (int pass = 0; pass < 4; ++pass) {
        PipelineSourceDesc upsweep_desc{PipelineSourceType::spirv_binary,
                                        upsweep_data[pass],
                                        upsweep_sizes[pass],
                                        PipelineStageType::compute};
        auto [upsweep_pipeline, upsweep_res] = dev->create_pipeline_unique(
            upsweep_desc, fmt::format("vulkan_sort_radix8_upsweep_{}", pass));
        TI_ERROR_IF(upsweep_res != RhiResult::success,
                    "Failed to create Vulkan radix8 upsweep pipeline {}: "
                    "RhiResult({})",
                    pass, upsweep_res);
        radix8_upsweep[pass] = std::move(upsweep_pipeline);

        PipelineSourceDesc downsweep_key_desc{
            PipelineSourceType::spirv_binary,
            downsweep_key_data[pass],
            downsweep_key_sizes[pass],
            PipelineStageType::compute};
        auto [downsweep_key_pipeline, downsweep_key_res] =
            dev->create_pipeline_unique(
                downsweep_key_desc,
                fmt::format("vulkan_sort_radix8_downsweep_keys_{}", pass));
        TI_ERROR_IF(downsweep_key_res != RhiResult::success,
                    "Failed to create Vulkan radix8 key downsweep pipeline "
                    "{}: RhiResult({})",
                    pass, downsweep_key_res);
        radix8_downsweep_keys[pass] = std::move(downsweep_key_pipeline);

        PipelineSourceDesc downsweep_pair_desc{
            PipelineSourceType::spirv_binary,
            downsweep_pair_data[pass],
            downsweep_pair_sizes[pass],
            PipelineStageType::compute};
        auto [downsweep_pair_pipeline, downsweep_pair_res] =
            dev->create_pipeline_unique(
                downsweep_pair_desc,
                fmt::format("vulkan_sort_radix8_downsweep_pairs_{}", pass));
        TI_ERROR_IF(downsweep_pair_res != RhiResult::success,
                    "Failed to create Vulkan radix8 pair downsweep pipeline "
                    "{}: RhiResult({})",
                    pass, downsweep_pair_res);
        radix8_downsweep_pairs[pass] = std::move(downsweep_pair_pipeline);
      }
    }
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc;
    RhiResult res = device->allocate_memory(
        {bytes, /*host_write=*/false, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage},
        &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan sort workspace: RhiResult({})",
                res);
    return alloc;
  }

  ShaderResourceSet *cached_resource_set(
      std::unique_ptr<ShaderResourceSet> &bindings,
      VulkanSortCpuProfileSample *profile = nullptr) {
    double start = profile ? profile_time_us() : 0.0;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
      if (profile) {
        profile->resource_set_create_calls++;
      }
    }
    if (profile) {
      profile->resource_set_calls++;
      profile->resource_set_us += profile_time_us() - start;
    }
    return bindings.get();
  }

  bool has_workspace_allocs() const {
    return key_in != kDeviceNullAllocation;
  }

  bool needs_workspace_realloc(size_t n,
                               bool use_values,
                               bool use_radix8) const {
    const size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const size_t requested_partitions =
        (n + kRadix8PartitionSize - 1) / kRadix8PartitionSize;
    const size_t requested_units =
        use_radix8 ? requested_partitions : requested_blocks;
    return !(capacity >= n && num_blocks >= requested_units &&
             has_value_buffers >= use_values &&
             workspace_uses_radix8 == use_radix8);
  }

  void ensure_workspace(size_t n, bool use_values, bool use_radix8) {
    size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    size_t requested_partitions =
        (n + kRadix8PartitionSize - 1) / kRadix8PartitionSize;
    const size_t requested_units =
        use_radix8 ? requested_partitions : requested_blocks;
    if (!needs_workspace_realloc(n, use_values, use_radix8)) {
      return;
    }
    clear_resource_sets();
    clear_allocs();
    capacity = n;
    num_blocks = requested_units;
    has_value_buffers = use_values;
    workspace_uses_radix8 = use_radix8;
    const size_t key_bytes = n * sizeof(uint32_t);
    key_in = alloc_storage(key_bytes);
    key_out = alloc_storage(key_bytes);
    cached_bytes = key_bytes * 2;
    if (use_radix8) {
      const size_t global_hist_bytes = kRadix8Bins * sizeof(uint32_t);
      const size_t partition_hist_bytes =
          requested_partitions * kRadix8Bins * sizeof(uint32_t);
      radix8_global_hist = alloc_storage(global_hist_bytes);
      radix8_partition_hist = alloc_storage(partition_hist_bytes);
      cached_bytes += global_hist_bytes + partition_hist_bytes;
    } else {
      const size_t rank_bytes = n * sizeof(uint32_t);
      const size_t table_bytes =
          requested_blocks * kRadixBins * sizeof(uint32_t);
      const size_t requested_chunks =
          (requested_blocks + kBlockSize - 1) / kBlockSize;
      const size_t chunk_table_bytes =
          requested_chunks * kRadixBins * sizeof(uint32_t);
      rank = alloc_storage(rank_bytes);
      hist = alloc_storage(table_bytes);
      offsets = alloc_storage(table_bytes);
      chunk_sums = alloc_storage(chunk_table_bytes);
      chunk_offsets = alloc_storage(chunk_table_bytes);
      cached_bytes +=
          rank_bytes + table_bytes * 2 + chunk_table_bytes * 2;
    }
    if (use_values) {
      const size_t value_bytes = n * sizeof(int32_t);
      value_in = alloc_storage(value_bytes);
      value_out = alloc_storage(value_bytes);
      cached_bytes += value_bytes * 2;
    }
  }
};

struct VulkanScanCache {
  Device *device{nullptr};
  size_t capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation workspace{kDeviceNullAllocation};
  DeviceAllocation dummy_sums{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> scan_i32_block;
  std::unique_ptr<Pipeline> scan_i32_block_subgroup;
  std::unique_ptr<Pipeline> scan_i32_add;
  std::unique_ptr<Pipeline> scan_i32_small_subgroup;
  bool subgroup_scan_enabled{false};

  void clear_allocs() {
    if (device && workspace != kDeviceNullAllocation) {
      device->dealloc_memory(workspace);
    }
    if (device && dummy_sums != kDeviceNullAllocation) {
      device->dealloc_memory(dummy_sums);
    }
    workspace = kDeviceNullAllocation;
    dummy_sums = kDeviceNullAllocation;
    capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanScanCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && scan_i32_block) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      scan_i32_block.reset();
      scan_i32_block_subgroup.reset();
      scan_i32_add.reset();
      scan_i32_small_subgroup.reset();
    }
    device = dev;
    const bool subgroup_arithmetic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_arithmetic) !=
        0;
    const bool subgroup_scan_allowed =
        get_environ_config("TI_VULKAN_SCAN_ENABLE_SUBGROUP", 1) != 0;
    subgroup_scan_enabled =
        subgroup_arithmetic_supported && subgroup_scan_allowed;
    scan_i32_block =
        create_pipeline(dev, kScanI32BlockSpv, "vulkan_scan_i32_block");
    if (subgroup_scan_enabled) {
      scan_i32_block_subgroup = create_pipeline(
          dev, kScanI32BlockSubgroupSpv, "vulkan_scan_i32_block_subgroup");
      scan_i32_small_subgroup = create_pipeline(
          dev, kScanI32SmallSubgroupSpv, "vulkan_scan_i32_small_subgroup");
    }
    scan_i32_add = create_pipeline(dev, kScanI32AddSpv, "vulkan_scan_i32_add");
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan scan workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_workspace_realloc(size_t bytes) const {
    return capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return workspace != kDeviceNullAllocation ||
           dummy_sums != kDeviceNullAllocation;
  }

  void clear_workspace_alloc() {
    if (device && workspace != kDeviceNullAllocation) {
      device->dealloc_memory(workspace);
    }
    workspace = kDeviceNullAllocation;
    capacity = 0;
    cached_bytes = dummy_sums != kDeviceNullAllocation ? sizeof(int32_t) : 0;
  }

  void ensure_dummy_sums() {
    if (dummy_sums == kDeviceNullAllocation) {
      dummy_sums = alloc_storage(sizeof(int32_t));
    }
    cached_bytes = capacity +
                   (dummy_sums != kDeviceNullAllocation ? sizeof(int32_t) : 0);
  }

  void ensure_workspace(size_t bytes) {
    ensure_dummy_sums();
    if (bytes == 0 || !needs_workspace_realloc(bytes)) {
      return;
    }
    clear_workspace_alloc();
    workspace = alloc_storage(bytes);
    capacity = bytes;
    cached_bytes = bytes + sizeof(int32_t);
  }
};

struct VulkanCompactCache {
  Device *device{nullptr};
  size_t prefix_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation prefix{kDeviceNullAllocation};
  VulkanScanCache scan;
  std::unique_ptr<Pipeline> compact_i32_flags;
  std::unique_ptr<Pipeline> compact_i32_scatter;

  void clear_allocs() {
    if (device && prefix != kDeviceNullAllocation) {
      device->dealloc_memory(prefix);
    }
    prefix = kDeviceNullAllocation;
    prefix_capacity = 0;
    cached_bytes = scan.cached_bytes;
  }

  ~VulkanCompactCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && compact_i32_flags) {
      scan.ensure_pipelines(dev);
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      compact_i32_flags.reset();
      compact_i32_scatter.reset();
    }
    device = dev;
    scan.ensure_pipelines(dev);
    compact_i32_flags =
        create_pipeline(dev, kCompactI32FlagsSpv, "vulkan_compact_i32_flags");
    compact_i32_scatter = create_pipeline(
        dev, kCompactI32ScatterSpv, "vulkan_compact_i32_scatter");
    cached_bytes = prefix_capacity + scan.cached_bytes;
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan compact workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_prefix_realloc(size_t bytes) const {
    return prefix_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return prefix != kDeviceNullAllocation || scan.has_workspace_allocs();
  }

  void clear_prefix_alloc() {
    if (device && prefix != kDeviceNullAllocation) {
      device->dealloc_memory(prefix);
    }
    prefix = kDeviceNullAllocation;
    prefix_capacity = 0;
    cached_bytes = scan.cached_bytes;
  }

  void ensure_prefix(size_t bytes) {
    if (bytes == 0 || !needs_prefix_realloc(bytes)) {
      cached_bytes = prefix_capacity + scan.cached_bytes;
      return;
    }
    clear_prefix_alloc();
    prefix = alloc_storage(bytes);
    prefix_capacity = bytes;
    cached_bytes = prefix_capacity + scan.cached_bytes;
  }

  size_t allocated_bytes() const {
    return prefix_capacity + scan.cached_bytes;
  }
};

struct VulkanHistogramCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> histogram_i32_clear;
  std::unique_ptr<Pipeline> histogram_i32_count_direct;
  std::unique_ptr<Pipeline> histogram_i32_count_private;
  std::unique_ptr<Pipeline> histogram_i32_count_private_shared;
  std::unique_ptr<Pipeline> histogram_i32_reduce_private;
  std::unique_ptr<Pipeline> histogram_i32_single_shared;

  void clear_allocs() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanHistogramCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && histogram_i32_clear) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      histogram_i32_clear.reset();
      histogram_i32_count_direct.reset();
      histogram_i32_count_private.reset();
      histogram_i32_count_private_shared.reset();
      histogram_i32_reduce_private.reset();
      histogram_i32_single_shared.reset();
    }
    device = dev;
    histogram_i32_clear = create_pipeline(
        dev, kHistogramI32ClearSpv, "vulkan_histogram_i32_clear");
    histogram_i32_count_direct = create_pipeline(
        dev, kHistogramI32CountDirectSpv,
        "vulkan_histogram_i32_count_direct");
    histogram_i32_count_private = create_pipeline(
        dev, kHistogramI32CountPrivateSpv,
        "vulkan_histogram_i32_count_private");
    histogram_i32_count_private_shared = create_pipeline(
        dev, kHistogramI32CountPrivateSharedSpv,
        "vulkan_histogram_i32_count_private_shared");
    histogram_i32_reduce_private = create_pipeline(
        dev, kHistogramI32ReducePrivateSpv,
        "vulkan_histogram_i32_reduce_private");
    histogram_i32_single_shared = create_pipeline(
        dev, kHistogramI32SingleSharedSpv,
        "vulkan_histogram_i32_single_shared");
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan histogram workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_partial_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return partial != kDeviceNullAllocation;
  }

  void clear_partial_alloc() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  void ensure_partial(size_t bytes) {
    if (bytes == 0 || !needs_partial_realloc(bytes)) {
      cached_bytes = partial_capacity;
      return;
    }
    clear_partial_alloc();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = partial_capacity;
  }
};

struct VulkanReduceCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  std::array<std::unique_ptr<Pipeline>, 3> reduce_i32_private;
  std::array<std::unique_ptr<Pipeline>, 3> reduce_i32_final;
  std::array<std::unique_ptr<Pipeline>, 3> reduce_i32_single;

  void clear_allocs() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanReduceCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && reduce_i32_private[0]) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      for (int i = 0; i < 3; ++i) {
        reduce_i32_private[i].reset();
        reduce_i32_final[i].reset();
        reduce_i32_single[i].reset();
      }
    }
    device = dev;
    reduce_i32_private[0] = create_pipeline(
        dev, kReduceI32SumPrivateSpv, "vulkan_reduce_i32_sum_private");
    reduce_i32_private[1] = create_pipeline(
        dev, kReduceI32MinPrivateSpv, "vulkan_reduce_i32_min_private");
    reduce_i32_private[2] = create_pipeline(
        dev, kReduceI32MaxPrivateSpv, "vulkan_reduce_i32_max_private");
    reduce_i32_final[0] = create_pipeline(
        dev, kReduceI32SumFinalSpv, "vulkan_reduce_i32_sum_final");
    reduce_i32_final[1] = create_pipeline(
        dev, kReduceI32MinFinalSpv, "vulkan_reduce_i32_min_final");
    reduce_i32_final[2] = create_pipeline(
        dev, kReduceI32MaxFinalSpv, "vulkan_reduce_i32_max_final");
    reduce_i32_single[0] = create_pipeline(
        dev, kReduceI32SumSingleSpv, "vulkan_reduce_i32_sum_single");
    reduce_i32_single[1] = create_pipeline(
        dev, kReduceI32MinSingleSpv, "vulkan_reduce_i32_min_single");
    reduce_i32_single[2] = create_pipeline(
        dev, kReduceI32MaxSingleSpv, "vulkan_reduce_i32_max_single");
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan reduce workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_partial_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return partial != kDeviceNullAllocation;
  }

  void clear_partial_alloc() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  void ensure_partial(size_t bytes) {
    if (bytes == 0 || !needs_partial_realloc(bytes)) {
      cached_bytes = partial_capacity;
      return;
    }
    clear_partial_alloc();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = partial_capacity;
  }
};

struct VulkanTransformCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  DeviceAllocation params{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> transform_i32_affine;
  std::unique_ptr<Pipeline> transform_f32_affine;
  std::unique_ptr<ShaderResourceSet> affine_bindings;

  void clear_allocs() {
    if (device && params != kDeviceNullAllocation) {
      device->dealloc_memory(params);
    }
    params = kDeviceNullAllocation;
    affine_bindings.reset();
    cached_bytes = 0;
  }

  ~VulkanTransformCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && transform_i32_affine) {
      ensure_params();
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      transform_i32_affine.reset();
      transform_f32_affine.reset();
      affine_bindings.reset();
    }
    device = dev;
    transform_i32_affine = create_pipeline(
        dev, kTransformI32AffineSpv, "vulkan_transform_i32_affine");
    transform_f32_affine = create_pipeline(
        dev, kTransformF32AffineSpv, "vulkan_transform_f32_affine");
    ensure_params();
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams alloc_params;
    alloc_params.size = bytes;
    alloc_params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(alloc_params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan transform workspace: RhiResult({})",
                res);
    return alloc;
  }

  void ensure_params() {
    if (params == kDeviceNullAllocation) {
      params = alloc_storage(2 * sizeof(uint32_t));
    }
    cached_bytes = 2 * sizeof(uint32_t);
  }

  ShaderResourceSet *cached_affine_resource_set() {
    if (!affine_bindings) {
      affine_bindings.reset(device->create_resource_set());
    }
    return affine_bindings.get();
  }
};

struct VulkanIndexedCopyCache {
  Device *device{nullptr};
  std::unique_ptr<Pipeline> gather_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_u32_by_i32;
  std::unique_ptr<ShaderResourceSet> gather_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_bindings;

  void ensure_pipelines(Device *dev) {
    if (device == dev && gather_u32_by_i32) {
      return;
    }
    if (device && device != dev) {
      gather_u32_by_i32.reset();
      scatter_u32_by_i32.reset();
      gather_bindings.reset();
      scatter_bindings.reset();
    }
    device = dev;
    gather_u32_by_i32 =
        create_pipeline(dev, kGatherU32ByI32Spv, "vulkan_gather_u32_by_i32");
    scatter_u32_by_i32 = create_pipeline(dev, kScatterU32ByI32Spv,
                                         "vulkan_scatter_u32_by_i32");
  }

  ShaderResourceSet *cached_resource_set(bool scatter) {
    auto &bindings = scatter ? scatter_bindings : gather_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }
};

std::mutex g_vulkan_sort_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanRadixSortCache>>
    g_vulkan_sort_caches;
std::mutex g_vulkan_scan_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanScanCache>>
    g_vulkan_scan_caches;
std::mutex g_vulkan_compact_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanCompactCache>>
    g_vulkan_compact_caches;
std::mutex g_vulkan_histogram_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanHistogramCache>>
    g_vulkan_histogram_caches;
std::mutex g_vulkan_reduce_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanReduceCache>>
    g_vulkan_reduce_caches;
std::mutex g_vulkan_transform_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanTransformCache>>
    g_vulkan_transform_caches;
std::mutex g_vulkan_indexed_copy_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanIndexedCopyCache>>
    g_vulkan_indexed_copy_caches;

VulkanRadixSortCache &get_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_sort_mutex);
  auto &cache = g_vulkan_sort_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanRadixSortCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

VulkanScanCache &get_scan_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_scan_mutex);
  auto &cache = g_vulkan_scan_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanScanCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

VulkanCompactCache &get_compact_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_compact_mutex);
  auto &cache = g_vulkan_compact_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanCompactCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

VulkanHistogramCache &get_histogram_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_histogram_mutex);
  auto &cache = g_vulkan_histogram_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanHistogramCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

VulkanReduceCache &get_reduce_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_reduce_mutex);
  auto &cache = g_vulkan_reduce_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanReduceCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

VulkanTransformCache &get_transform_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_transform_mutex);
  auto &cache = g_vulkan_transform_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanTransformCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

VulkanIndexedCopyCache &get_indexed_copy_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_indexed_copy_mutex);
  auto &cache = g_vulkan_indexed_copy_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanIndexedCopyCache>();
  }
  cache->ensure_pipelines(device);
  return *cache;
}

void dispatch_pipeline(CommandList *cmdlist,
                       Pipeline *pipeline,
                       ShaderResourceSet *bindings,
                       uint32_t groups,
                       uint32_t groups_y,
                       uint32_t groups_z,
                       const char *scope_name,
                       VulkanSortCpuProfileSample *profile = nullptr) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->bind_pipeline(pipeline);
    profile->bind_pipeline_calls++;
    profile->bind_pipeline_us += profile_time_us() - start;
  } else {
    cmdlist->bind_pipeline(pipeline);
  }
  RhiResult bind_status;
  if (profile) {
    double start = profile_time_us();
    bind_status = cmdlist->bind_shader_resources(bindings);
    profile->bind_shader_resources_calls++;
    profile->bind_shader_resources_us += profile_time_us() - start;
  } else {
    bind_status = cmdlist->bind_shader_resources(bindings);
  }
  TI_ERROR_IF(bind_status != RhiResult::success,
              "Vulkan sort resource binding failed: RhiResult({})",
              bind_status);
  if (scope_name && profile) {
    double start = profile_time_us();
    cmdlist->begin_profiler_scope(scope_name);
    profile->profiler_scope_us += profile_time_us() - start;
  } else if (scope_name) {
    cmdlist->begin_profiler_scope(scope_name);
  }
  RhiResult dispatch_status;
  if (profile) {
    double start = profile_time_us();
    dispatch_status = cmdlist->dispatch(groups, groups_y, groups_z);
    profile->dispatch_calls++;
    profile->dispatch_us += profile_time_us() - start;
  } else {
    dispatch_status = cmdlist->dispatch(groups, groups_y, groups_z);
  }
  if (scope_name && profile) {
    double start = profile_time_us();
    cmdlist->end_profiler_scope();
    profile->profiler_scope_us += profile_time_us() - start;
  } else if (scope_name) {
    cmdlist->end_profiler_scope();
  }
  TI_ERROR_IF(dispatch_status != RhiResult::success,
              "Vulkan sort dispatch failed: RhiResult({})", dispatch_status);
}

template <typename Ptr>
void profiled_rw_buffer(ShaderResourceSet *bindings,
                        uint32_t binding,
                        Ptr ptr,
                        size_t bytes,
                        VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    bindings->rw_buffer(binding, ptr, bytes);
    profile->rw_buffer_calls++;
    profile->rw_buffer_us += profile_time_us() - start;
  } else {
    bindings->rw_buffer(binding, ptr, bytes);
  }
}

void profiled_buffer_barrier(CommandList *cmdlist,
                             DeviceAllocation alloc,
                             VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->buffer_barrier(alloc);
    profile->buffer_barrier_calls++;
    profile->buffer_barrier_us += profile_time_us() - start;
  } else {
    cmdlist->buffer_barrier(alloc);
  }
}

template <typename Ptr>
void profiled_buffer_fill(CommandList *cmdlist,
                          Ptr ptr,
                          size_t bytes,
                          uint32_t data,
                          VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->buffer_fill(ptr, bytes, data);
    profile->buffer_fill_calls++;
    profile->buffer_fill_us += profile_time_us() - start;
  } else {
    cmdlist->buffer_fill(ptr, bytes, data);
  }
}

template <typename DstPtr, typename SrcPtr>
void profiled_buffer_copy(CommandList *cmdlist,
                          DstPtr dst,
                          SrcPtr src,
                          size_t bytes,
                          VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->buffer_copy(dst, src, bytes);
    profile->buffer_copy_calls++;
    profile->buffer_copy_us += profile_time_us() - start;
  } else {
    cmdlist->buffer_copy(dst, src, bytes);
  }
}

void dispatch_unary(CommandList *cmdlist,
                    Device *device,
                    Pipeline *pipeline,
                    DeviceAllocation in,
                    DeviceAllocation out,
                    size_t bytes,
                    uint32_t groups,
                    const char *scope_name,
                    VulkanSortCpuProfileSample *profile = nullptr) {
  std::unique_ptr<ShaderResourceSet> bindings;
  if (profile) {
    double start = profile_time_us();
    bindings = device->create_resource_set_unique();
    profile->resource_set_calls++;
    profile->resource_set_create_calls++;
    profile->resource_set_us += profile_time_us() - start;
  } else {
    bindings = device->create_resource_set_unique();
  }
  profiled_rw_buffer(bindings.get(), 0, in.get_ptr(0), bytes, profile);
  profiled_rw_buffer(bindings.get(), 1, out.get_ptr(0), bytes, profile);
  dispatch_pipeline(cmdlist, pipeline, bindings.get(), groups, 1, 1,
                    scope_name, profile);
}

std::vector<size_t> scan_level_lengths(size_t n) {
  std::vector<size_t> levels;
  while (n > 0) {
    levels.push_back(n);
    if (n <= kBlockSize) {
      break;
    }
    n = (n + kBlockSize - 1) / kBlockSize;
  }
  return levels;
}

size_t scan_workspace_bytes(const std::vector<size_t> &levels) {
  size_t items = 0;
  for (size_t i = 1; i < levels.size(); ++i) {
    items += levels[i];
  }
  return items * sizeof(int32_t);
}

DevicePtr scan_level_ptr(DeviceAllocation data_alloc,
                         DeviceAllocation workspace,
                         const std::vector<size_t> &workspace_offsets,
                         size_t level) {
  if (level == 0) {
    return data_alloc.get_ptr(0);
  }
  return workspace.get_ptr(workspace_offsets[level - 1] * sizeof(int32_t));
}

struct VulkanScanDispatchPlan {
  DeviceAllocation data_alloc{kDeviceNullAllocation};
  size_t n{0};
  size_t workspace_bytes{0};
  bool use_small_subgroup{false};
  size_t data_bytes{0};
  DeviceAllocation workspace_alloc{kDeviceNullAllocation};
  DeviceAllocation dummy_sums_alloc{kDeviceNullAllocation};
  std::vector<size_t> levels;
  std::vector<size_t> workspace_offsets;
  Pipeline *scan_small{nullptr};
  Pipeline *scan_block{nullptr};
  Pipeline *scan_add{nullptr};
  const char *scan_block_scope{nullptr};
};

VulkanScanDispatchPlan prepare_vulkan_i32_scan(Program *program,
                                               VulkanScanCache &cache,
                                               DeviceAllocation data_alloc,
                                               size_t n) {
  VulkanScanDispatchPlan plan;
  plan.data_alloc = data_alloc;
  plan.n = n;
  if (n <= 1) {
    return plan;
  }

  const int small_subgroup_threshold =
      get_environ_config("TI_VULKAN_SCAN_SMALL_SUBGROUP_MAX_N", 4096);
  plan.use_small_subgroup =
      cache.subgroup_scan_enabled && cache.scan_i32_small_subgroup &&
      small_subgroup_threshold > 0 &&
      n <= static_cast<size_t>(small_subgroup_threshold);
  if (plan.use_small_subgroup) {
    plan.scan_small = cache.scan_i32_small_subgroup.get();
    plan.data_bytes = n * sizeof(int32_t);
    return plan;
  }

  plan.levels = scan_level_lengths(n);
  plan.workspace_bytes = scan_workspace_bytes(plan.levels);
  if (cache.has_workspace_allocs() &&
      cache.needs_workspace_realloc(plan.workspace_bytes)) {
    program->synchronize();
  }
  cache.ensure_workspace(plan.workspace_bytes);

  plan.workspace_offsets.reserve(plan.levels.size() > 0 ? plan.levels.size() - 1
                                                        : 0);
  size_t offset = 0;
  for (size_t i = 1; i < plan.levels.size(); ++i) {
    plan.workspace_offsets.push_back(offset);
    offset += plan.levels[i];
  }

  plan.workspace_alloc = cache.workspace;
  plan.dummy_sums_alloc = cache.dummy_sums;
  const int subgroup_block_min_n_config =
      get_environ_config("TI_VULKAN_SCAN_SUBGROUP_BLOCK_MIN_N", 1048576);
  const size_t subgroup_block_min_n =
      subgroup_block_min_n_config <= 0
          ? 0
          : static_cast<size_t>(subgroup_block_min_n_config);
  const bool use_subgroup_block = cache.subgroup_scan_enabled &&
                                  cache.scan_i32_block_subgroup &&
                                  n >= subgroup_block_min_n;
  plan.scan_block = use_subgroup_block ? cache.scan_i32_block_subgroup.get()
                                       : cache.scan_i32_block.get();
  plan.scan_block_scope = use_subgroup_block ? "vulkan_scan_i32_block_subgroup"
                                             : "vulkan_scan_i32_block";
  plan.scan_add = cache.scan_i32_add.get();
  return plan;
}

void record_vulkan_i32_scan(Device *op_device,
                            CommandList *cmdlist,
                            const VulkanScanDispatchPlan &plan,
                            bool profiler_scopes) {
  if (plan.n <= 1) {
    return;
  }
  if (plan.use_small_subgroup) {
    auto bindings = op_device->create_resource_set_unique();
    bindings->rw_buffer(0, plan.data_alloc.get_ptr(0), plan.data_bytes);
    dispatch_pipeline(cmdlist, plan.scan_small, bindings.get(), 1, 1, 1,
                      profiler_scopes ? "vulkan_scan_i32_small_subgroup"
                                      : nullptr);
    cmdlist->buffer_barrier(plan.data_alloc);
    return;
  }

  auto scope_name = [profiler_scopes](const char *name) {
    return profiler_scopes ? name : nullptr;
  };
  auto barrier_level = [&plan](CommandList *cmdlist, size_t level) {
    if (level == 0) {
      cmdlist->buffer_barrier(plan.data_alloc);
    } else {
      cmdlist->buffer_barrier(plan.workspace_alloc);
    }
  };
  for (size_t level = 0; level < plan.levels.size(); ++level) {
    DevicePtr level_ptr =
        scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                       plan.workspace_offsets, level);
    const size_t level_bytes = plan.levels[level] * sizeof(int32_t);
    DevicePtr sums_ptr = plan.dummy_sums_alloc.get_ptr(0);
    size_t sums_bytes = sizeof(int32_t);
    if (level + 1 < plan.levels.size()) {
      sums_ptr = scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                                plan.workspace_offsets, level + 1);
      sums_bytes = plan.levels[level + 1] * sizeof(int32_t);
    }
    auto bindings = op_device->create_resource_set_unique();
    bindings->rw_buffer(0, level_ptr, level_bytes);
    bindings->rw_buffer(1, sums_ptr, sums_bytes);
    const uint32_t groups = static_cast<uint32_t>(
        (plan.levels[level] + kBlockSize - 1) / kBlockSize);
    dispatch_pipeline(cmdlist, plan.scan_block, bindings.get(), groups, 1, 1,
                      scope_name(plan.scan_block_scope));
    barrier_level(cmdlist, level);
    if (level + 1 < plan.levels.size()) {
      cmdlist->buffer_barrier(plan.workspace_alloc);
    }
  }
  if (plan.levels.size() > 1) {
    for (size_t level = plan.levels.size() - 1; level-- > 0;) {
      DevicePtr level_ptr =
          scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                         plan.workspace_offsets, level);
      DevicePtr offsets_ptr =
          scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                         plan.workspace_offsets, level + 1);
      const size_t level_bytes = plan.levels[level] * sizeof(int32_t);
      const size_t offsets_bytes = plan.levels[level + 1] * sizeof(int32_t);
      auto bindings = op_device->create_resource_set_unique();
      bindings->rw_buffer(0, level_ptr, level_bytes);
      bindings->rw_buffer(1, offsets_ptr, offsets_bytes);
      const uint32_t groups = static_cast<uint32_t>(
          (plan.levels[level] + kBlockSize - 1) / kBlockSize);
      dispatch_pipeline(cmdlist, plan.scan_add, bindings.get(), groups, 1, 1,
                        scope_name("vulkan_scan_i32_add"));
      barrier_level(cmdlist, level);
    }
  }
}

size_t enqueue_vulkan_i32_scan(Program *program,
                               VulkanScanCache &cache,
                               DeviceAllocation data_alloc,
                               size_t n,
                               bool profiler_scopes) {
  auto plan = prepare_vulkan_i32_scan(program, cache, data_alloc, n);
  if (plan.n <= 1) {
    return 0;
  }
  program->enqueue_compute_op_lambda(
      [plan, profiler_scopes](Device *op_device, CommandList *cmdlist) {
        record_vulkan_i32_scan(op_device, cmdlist, plan, profiler_scopes);
      },
      {});
  return plan.workspace_bytes;
}

}  // namespace

bool Program::vulkan_radix_sort_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scan_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_compact_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_histogram_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_reduce_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_transform_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_indexed_copy_available() const {
  return compile_config().arch == Arch::vulkan;
}

std::size_t Program::vulkan_inclusive_scan_ndarray(Ndarray *data,
                                                   int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scan is only available on Vulkan.");
  TI_ERROR_IF(!data, "Vulkan native scan received null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "Vulkan native scan expects a 1D ndarray.");
  TI_ERROR_IF(value_type != 0,
              "Vulkan native scan currently supports only i32 values.");
  TI_ERROR_IF(data->get_element_size() != sizeof(int32_t),
              "Vulkan native scan currently expects i32 data.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scan requires a compute device.");
  auto &cache = get_scan_cache(this, device);
  return enqueue_vulkan_i32_scan(this, cache, data->ndarray_alloc_, n,
                                 profiler != nullptr);
}

std::size_t Program::vulkan_compact_i32_ndarray(Ndarray *values,
                                                Ndarray *flags,
                                                Ndarray *output,
                                                Ndarray *count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native compact is only available on Vulkan.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "Vulkan native compact received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "Vulkan native compact expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement(),
              "Vulkan native compact values and flags must have the same "
              "length.");
  TI_ERROR_IF(output->get_nelement() < values->get_nelement(),
              "Vulkan native compact output must have at least input length.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "Vulkan native compact count must contain at least one item.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "Vulkan native compact currently expects i32 values, flags, "
              "output, and count.");

  const size_t n = values->get_nelement();
  if (n == 0) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native compact requires a compute device.");
  auto &cache = get_compact_cache(this, device);
  const size_t prefix_bytes = n * sizeof(int32_t);
  if (cache.has_workspace_allocs() &&
      cache.needs_prefix_realloc(prefix_bytes)) {
    synchronize();
  }
  cache.ensure_prefix(prefix_bytes);

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation flags_alloc = flags->ndarray_alloc_;
  DeviceAllocation output_alloc = output->ndarray_alloc_;
  DeviceAllocation count_alloc = count->ndarray_alloc_;
  DeviceAllocation prefix_alloc = cache.prefix;
  Pipeline *flags_pipeline = cache.compact_i32_flags.get();
  Pipeline *scatter_pipeline = cache.compact_i32_scatter.get();
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const int compact_fuse_max_n_config =
      get_environ_config("TI_VULKAN_COMPACT_FUSE_MAX_N", 4096);
  const bool use_fused_recording =
      compact_fuse_max_n_config > 0 &&
      n <= static_cast<size_t>(compact_fuse_max_n_config);

  if (use_fused_recording) {
    auto scan_plan = prepare_vulkan_i32_scan(this, cache.scan, prefix_alloc, n);
    cache.cached_bytes = cache.allocated_bytes();
    enqueue_compute_op_lambda(
        [flags_alloc, prefix_alloc, prefix_bytes, flags_pipeline, groups,
         values_alloc, output_alloc, count_alloc, scatter_pipeline, scan_plan,
         profiler_scopes](Device *op_device, CommandList *cmdlist) {
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, flags_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(1, prefix_alloc.get_ptr(0), prefix_bytes);
            dispatch_pipeline(cmdlist, flags_pipeline, bindings.get(), groups,
                              1, 1,
                              profiler_scopes ? "vulkan_compact_i32_flags"
                                              : nullptr);
            cmdlist->buffer_barrier(prefix_alloc);
          }
          record_vulkan_i32_scan(op_device, cmdlist, scan_plan,
                                 profiler_scopes);
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, values_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(1, flags_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(2, prefix_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(3, output_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(4, count_alloc.get_ptr(0), sizeof(int32_t));
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                              groups, 1, 1,
                              profiler_scopes ? "vulkan_compact_i32_scatter"
                                              : nullptr);
          }
          cmdlist->buffer_barrier(output_alloc);
          cmdlist->buffer_barrier(count_alloc);
        },
        {});
    return cache.cached_bytes;
  }

  enqueue_compute_op_lambda(
      [flags_alloc, prefix_alloc, prefix_bytes, flags_pipeline, groups,
       profiler_scopes](Device *op_device, CommandList *cmdlist) {
        auto bindings = op_device->create_resource_set_unique();
        bindings->rw_buffer(0, flags_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(1, prefix_alloc.get_ptr(0), prefix_bytes);
        dispatch_pipeline(cmdlist, flags_pipeline, bindings.get(), groups, 1,
                          1,
                          profiler_scopes ? "vulkan_compact_i32_flags"
                                          : nullptr);
        cmdlist->buffer_barrier(prefix_alloc);
      },
      {});

  enqueue_vulkan_i32_scan(this, cache.scan, prefix_alloc, n, profiler_scopes);
  cache.cached_bytes = cache.allocated_bytes();

  enqueue_compute_op_lambda(
      [values_alloc, flags_alloc, prefix_alloc, output_alloc, count_alloc,
       prefix_bytes, scatter_pipeline, groups,
       profiler_scopes](Device *op_device, CommandList *cmdlist) {
        auto bindings = op_device->create_resource_set_unique();
        bindings->rw_buffer(0, values_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(1, flags_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(2, prefix_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(3, output_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(4, count_alloc.get_ptr(0), sizeof(int32_t));
        dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(), groups, 1,
                          1,
                          profiler_scopes ? "vulkan_compact_i32_scatter"
                                          : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        cmdlist->buffer_barrier(count_alloc);
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_histogram_i32_ndarray(Ndarray *values,
                                                  Ndarray *bins) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native histogram is only available on Vulkan.");
  TI_ERROR_IF(!values || !bins,
              "Vulkan native histogram received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "Vulkan native histogram expects 1D ndarrays.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  bins->get_element_size() != sizeof(int32_t),
              "Vulkan native histogram currently expects i32 values and bins.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "Vulkan native histogram expects at least one bin.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native histogram requires a compute device.");
  auto &cache = get_histogram_cache(this, device);

  const size_t n = values->get_nelement();
  const size_t num_bins = bins->get_nelement();
  const size_t value_bytes = n * sizeof(int32_t);
  const size_t bin_bytes = num_bins * sizeof(int32_t);
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N", 65536);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MAX_BINS", 512);
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N", 4096);
  const bool shared_bins_supported = num_bins <= 512;
  const bool use_single_shared =
      n > 0 && shared_bins_supported && single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);
  const bool use_private =
      n > 0 && !use_single_shared && shared_bins_supported &&
      (private_min_n_config <= 0 ||
       n >= static_cast<size_t>(private_min_n_config)) &&
      (private_max_bins_config <= 0 ||
       num_bins <= static_cast<size_t>(private_max_bins_config));

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (use_private) {
    num_chunks = (n + kHistogramPrivateChunkSize - 1) /
                 kHistogramPrivateChunkSize;
    partial_bytes = num_chunks * num_bins * sizeof(int32_t);
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation bins_alloc = bins->ndarray_alloc_;
  DeviceAllocation partial_alloc = cache.partial;
  Pipeline *clear_pipeline = cache.histogram_i32_clear.get();
  Pipeline *count_direct_pipeline = cache.histogram_i32_count_direct.get();
  Pipeline *count_private_pipeline = cache.histogram_i32_count_private.get();
  Pipeline *count_private_shared_pipeline =
      cache.histogram_i32_count_private_shared.get();
  Pipeline *reduce_private_pipeline = cache.histogram_i32_reduce_private.get();
  Pipeline *single_shared_pipeline = cache.histogram_i32_single_shared.get();
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t bin_groups = static_cast<uint32_t>(
      (num_bins + kBlockSize - 1) / kBlockSize);
  const uint32_t value_groups = static_cast<uint32_t>(
      (n + kBlockSize - 1) / kBlockSize);

  enqueue_compute_op_lambda(
      [values_alloc, bins_alloc, partial_alloc, value_bytes, bin_bytes,
       partial_bytes, clear_pipeline, count_direct_pipeline,
       count_private_pipeline, count_private_shared_pipeline,
       reduce_private_pipeline, single_shared_pipeline, value_groups,
       bin_groups, num_chunks, use_private, use_single_shared, profiler_scopes](
          Device *op_device, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        if (use_single_shared) {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
          dispatch_pipeline(cmdlist, single_shared_pipeline, bindings.get(), 1,
                            1, 1,
                            scope_name("vulkan_histogram_i32_single_shared"));
          cmdlist->buffer_barrier(bins_alloc);
          return;
        }
        if (use_private) {
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
            bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
            bindings->rw_buffer(2, partial_alloc.get_ptr(0), partial_bytes);
            Pipeline *count_pipeline = count_private_shared_pipeline
                                           ? count_private_shared_pipeline
                                           : count_private_pipeline;
            const char *count_scope =
                count_private_shared_pipeline
                    ? "vulkan_histogram_i32_count_private_shared"
                    : "vulkan_histogram_i32_count_private";
            uint32_t count_groups =
                count_private_shared_pipeline
                    ? static_cast<uint32_t>(num_chunks)
                    : value_groups;
            dispatch_pipeline(cmdlist, count_pipeline, bindings.get(),
                              count_groups, 1, 1, scope_name(count_scope));
            cmdlist->buffer_barrier(partial_alloc);
          }
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, partial_alloc.get_ptr(0), partial_bytes);
            bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
            dispatch_pipeline(cmdlist, reduce_private_pipeline, bindings.get(),
                              bin_groups, 1, 1,
                              scope_name(
                                  "vulkan_histogram_i32_reduce_private"));
            cmdlist->buffer_barrier(bins_alloc);
          }
          return;
        }
        {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, bins_alloc.get_ptr(0), bin_bytes);
          dispatch_pipeline(cmdlist, clear_pipeline, bindings.get(), bin_groups,
                            1, 1,
                            scope_name("vulkan_histogram_i32_clear_bins"));
          cmdlist->buffer_barrier(bins_alloc);
        }
        if (value_groups > 0) {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
          dispatch_pipeline(cmdlist, count_direct_pipeline, bindings.get(),
                            value_groups, 1, 1,
                            scope_name("vulkan_histogram_i32_count_direct"));
          cmdlist->buffer_barrier(bins_alloc);
        }
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_reduce_i32_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native reduce is only available on Vulkan.");
  TI_ERROR_IF(!values || !output,
              "Vulkan native reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native reduce output must contain at least one item.");
  TI_ERROR_IF(values->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != sizeof(int32_t),
              "Vulkan native reduce currently expects i32 values and output.");
  TI_ERROR_IF(op < 0 || op > 2,
              "Vulkan native reduce supports only sum/min/max operations.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native reduce requires a compute device.");
  auto &cache = get_reduce_cache(this, device);

  const size_t n = values->get_nelement();
  const size_t value_bytes = n * sizeof(int32_t);
  const size_t output_bytes = sizeof(int32_t);
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_REDUCE_SINGLE_SHARED_MAX_N", 4096);
  const bool use_single_shared =
      single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (!use_single_shared) {
    num_chunks = (n + kReducePrivateChunkSize - 1) / kReducePrivateChunkSize;
    partial_bytes = num_chunks * sizeof(int32_t);
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation output_alloc = output->ndarray_alloc_;
  DeviceAllocation partial_alloc = cache.partial;
  Pipeline *private_pipeline = cache.reduce_i32_private[op].get();
  Pipeline *final_pipeline = cache.reduce_i32_final[op].get();
  Pipeline *single_pipeline = cache.reduce_i32_single[op].get();
  const bool profiler_scopes = profiler != nullptr;

  enqueue_compute_op_lambda(
      [values_alloc, output_alloc, partial_alloc, value_bytes, output_bytes,
       partial_bytes, private_pipeline, final_pipeline, single_pipeline,
       num_chunks, use_single_shared,
       profiler_scopes](Device *op_device, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        if (use_single_shared) {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, output_alloc.get_ptr(0), output_bytes);
          dispatch_pipeline(cmdlist, single_pipeline, bindings.get(), 1, 1, 1,
                            scope_name("vulkan_reduce_i32_single"));
          cmdlist->buffer_barrier(output_alloc);
          return;
        }
        {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, partial_alloc.get_ptr(0), partial_bytes);
          dispatch_pipeline(cmdlist, private_pipeline, bindings.get(),
                            static_cast<uint32_t>(num_chunks), 1, 1,
                            scope_name("vulkan_reduce_i32_private"));
          cmdlist->buffer_barrier(partial_alloc);
        }
        {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, partial_alloc.get_ptr(0), partial_bytes);
          bindings->rw_buffer(1, output_alloc.get_ptr(0), output_bytes);
          dispatch_pipeline(cmdlist, final_pipeline, bindings.get(), 1, 1, 1,
                            scope_name("vulkan_reduce_i32_final"));
          cmdlist->buffer_barrier(output_alloc);
        }
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_transform_affine_ndarray(Ndarray *src,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     double scale,
                                                     double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native transform is only available on Vulkan.");
  TI_ERROR_IF(!src || !dst, "Vulkan native transform received null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1,
              "Vulkan native transform expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native transform source and destination sizes differ.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native transform source and destination dtypes differ.");
  TI_ERROR_IF(value_type < 0 || value_type > 1,
              "Vulkan native transform received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != sizeof(uint32_t),
              "Vulkan native transform currently expects 32-bit values.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native transform currently supports at most UINT32_MAX "
              "items.");

  const size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native transform requires a compute device.");
  auto &cache = get_transform_cache(this, device);

  uint32_t scale_bits = 0;
  uint32_t bias_bits = 0;
  if (value_type == 0) {
    scale_bits = static_cast<uint32_t>(static_cast<int32_t>(scale));
    bias_bits = static_cast<uint32_t>(static_cast<int32_t>(bias));
  } else {
    float scale_f32 = static_cast<float>(scale);
    float bias_f32 = static_cast<float>(bias);
    std::memcpy(&scale_bits, &scale_f32, sizeof(scale_bits));
    std::memcpy(&bias_bits, &bias_f32, sizeof(bias_bits));
  }

  DeviceAllocation src_alloc = src->ndarray_alloc_;
  DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  DeviceAllocation params_alloc = cache.params;
  const bool bind_static_params = !cache.affine_bindings;
  ShaderResourceSet *bindings = cache.cached_affine_resource_set();
  Pipeline *pipeline = value_type == 0 ? cache.transform_i32_affine.get()
                                       : cache.transform_f32_affine.get();
  const size_t bytes = n * sizeof(uint32_t);
  const size_t params_bytes = 2 * sizeof(uint32_t);
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;

  enqueue_compute_op_lambda(
      [src_alloc, dst_alloc, params_alloc, bindings, bind_static_params,
       pipeline, bytes, params_bytes, scale_bits, bias_bits, groups,
       profiler_scopes](Device *op_device, CommandList *cmdlist) {
        cmdlist->buffer_fill(params_alloc.get_ptr(0), sizeof(uint32_t),
                             scale_bits);
        cmdlist->buffer_fill(params_alloc.get_ptr(sizeof(uint32_t)),
                             sizeof(uint32_t), bias_bits);
        cmdlist->buffer_barrier(params_alloc);
        bindings->rw_buffer(0, src_alloc.get_ptr(0), bytes);
        bindings->rw_buffer(1, dst_alloc.get_ptr(0), bytes);
        if (bind_static_params) {
          bindings->rw_buffer(2, params_alloc.get_ptr(0), params_bytes);
        }
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_transform_affine"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_gather_ndarray(Ndarray *src,
                                           Ndarray *indices,
                                           Ndarray *dst) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native gather is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native gather received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native gather currently expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "Vulkan native gather expects indices and destination sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native gather source and destination dtypes differ.");
  TI_ERROR_IF(src->get_element_size() != sizeof(uint32_t) ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native gather currently expects 32-bit values and i32 "
              "indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native gather currently supports at most UINT32_MAX "
              "items.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native gather requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  ShaderResourceSet *bindings = cache.cached_resource_set(false);
  Pipeline *pipeline = cache.gather_u32_by_i32.get();
  const size_t value_bytes = n * sizeof(uint32_t);
  const size_t src_bytes = src->get_nelement() * sizeof(uint32_t);
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       value_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        bindings->rw_buffer(0, src_alloc.get_ptr(0), src_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), value_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), value_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_gather_u32_by_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return 0;
}

std::size_t Program::vulkan_scatter_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scatter is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native scatter received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native scatter currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "Vulkan native scatter expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native scatter source and destination dtypes differ.");
  TI_ERROR_IF(src->get_element_size() != sizeof(uint32_t) ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native scatter currently expects 32-bit values and i32 "
              "indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter currently supports at most UINT32_MAX "
              "items.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scatter requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  ShaderResourceSet *bindings = cache.cached_resource_set(true);
  Pipeline *pipeline = cache.scatter_u32_by_i32.get();
  const size_t value_bytes = n * sizeof(uint32_t);
  const size_t dst_bytes = dst->get_nelement() * sizeof(uint32_t);
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, value_bytes,
       dst_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        bindings->rw_buffer(0, src_alloc.get_ptr(0), value_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), value_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), dst_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_scatter_u32_by_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return 0;
}

std::size_t Program::vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   int key_type) {
  const bool cpu_profile_enabled = vulkan_sort_cpu_profile_enabled();
  VulkanSortCpuProfileSample front_profile;
  VulkanSortCpuProfileSample *front =
      cpu_profile_enabled ? &front_profile : nullptr;
  const double total_start = front ? profile_time_us() : 0.0;
  if (front) {
    front->sort_calls++;
  }
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native radix sort is only available on Vulkan.");
  TI_ERROR_IF(!keys, "Vulkan native radix sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "Vulkan native radix sort expects a 1D ndarray.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(uint32_t),
              "Vulkan native radix sort currently expects 32-bit keys.");
  TI_ERROR_IF(key_type < 0 || key_type > 1,
              "Vulkan native radix sort supports only u32/i32 keys.");
  const bool use_values = values != nullptr;
  if (use_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "Vulkan native radix sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "Vulkan native radix sort keys and values must have the same "
                "length.");
    TI_ERROR_IF(values->get_element_size() != sizeof(int32_t),
                "Vulkan native radix sort currently expects i32 payload "
                "values.");
  }

  const size_t n = keys->get_nelement();
  if (n <= 1) {
    if (front) {
      front->total_call_us += profile_time_us() - total_start;
      g_vulkan_sort_cpu_profile.merge(front_profile);
    }
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native radix sort requires a compute device.");
  double start = front ? profile_time_us() : 0.0;
  auto &cache = get_cache(this, device);
  if (front) {
    front->get_cache_us += profile_time_us() - start;
  }
  const bool use_radix8 = cache.radix8_enabled;
  const bool needs_realloc =
      cache.needs_workspace_realloc(n, use_values, use_radix8);
  if (front && needs_realloc) {
    front->workspace_reallocs++;
  }
  if (cache.has_workspace_allocs() && needs_realloc) {
    if (front) {
      start = profile_time_us();
    }
    synchronize();
    if (front) {
      front->realloc_sync_calls++;
      front->realloc_sync_us += profile_time_us() - start;
    }
  }
  if (front) {
    start = profile_time_us();
  }
  cache.ensure_workspace(n, use_values, use_radix8);
  if (front) {
    front->ensure_workspace_us += profile_time_us() - start;
  }

  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t radix8_partitions =
      static_cast<uint32_t>((n + kRadix8PartitionSize - 1) /
                            kRadix8PartitionSize);
  const bool signed_keys = key_type == 1;
  const bool profiler_scopes = profiler != nullptr;
  DeviceAllocation key_alloc = keys->ndarray_alloc_;
  DeviceAllocation value_alloc =
      use_values ? values->ndarray_alloc_ : kDeviceNullAllocation;
  const size_t key_bytes = n * sizeof(uint32_t);
  const size_t value_bytes = n * sizeof(int32_t);
  const size_t table_bytes =
      static_cast<size_t>(groups) * kRadixBins * sizeof(uint32_t);
  const uint32_t chunk_groups =
      (groups + kBlockSize - 1) / kBlockSize;
  const size_t chunk_table_bytes =
      static_cast<size_t>(chunk_groups) * kRadixBins * sizeof(uint32_t);
  const bool inline_chunk_offsets =
      groups > kSingleChunkPrefixMaxBlocks &&
      chunk_groups <= kInlineChunkPrefixMaxChunks &&
      cache.inline_chunk_prefix_allowed;
  const size_t radix8_global_hist_bytes = kRadix8Bins * sizeof(uint32_t);
  const size_t radix8_partition_hist_bytes =
      static_cast<size_t>(radix8_partitions) * kRadix8Bins * sizeof(uint32_t);

  std::shared_ptr<VulkanSortCpuProfileSample> lambda_profile;
  if (cpu_profile_enabled) {
    lambda_profile = std::make_shared<VulkanSortCpuProfileSample>();
  }
  if (front) {
    start = profile_time_us();
  }
  program_impl_->enqueue_compute_op_lambda(
      [&, groups, n, signed_keys, use_values, key_alloc, value_alloc, key_bytes,
       value_bytes, table_bytes, chunk_groups, chunk_table_bytes,
       inline_chunk_offsets, use_radix8, radix8_partitions,
       radix8_global_hist_bytes, radix8_partition_hist_bytes,
       lambda_profile, profiler_scopes](
          Device *op_device, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        VulkanSortCpuProfileSample *profile = lambda_profile.get();
        const double lambda_start = profile ? profile_time_us() : 0.0;
        if (profile) {
          profile->lambda_calls++;
        }
        if (use_radix8) {
          const double radix8_start = profile ? profile_time_us() : 0.0;
          DeviceAllocation key_read = signed_keys ? cache.key_in : key_alloc;
          DeviceAllocation key_write = cache.key_out;
          DeviceAllocation value_read =
              use_values ? value_alloc : cache.value_in;
          DeviceAllocation value_write = cache.value_out;
          if (signed_keys) {
            dispatch_unary(cmdlist, op_device, cache.init_i32.get(), key_alloc,
                           cache.key_in, key_bytes, groups,
                           scope_name("vulkan_sort_init_i32"), profile);
            profiled_buffer_barrier(cmdlist, cache.key_in, profile);
          }
          bool keys_written_to_user = false;
          bool values_written_to_user = false;
          for (int pass = 0; pass < 4; ++pass) {
            const bool last_pass = (pass == 3);
            const bool direct_key_output = last_pass && !signed_keys;
            const bool direct_value_output = last_pass && use_values;
            DeviceAllocation pass_key_write =
                direct_key_output ? key_alloc : key_write;
            DeviceAllocation pass_value_write =
                direct_value_output ? value_alloc : value_write;
            profiled_buffer_fill(cmdlist, cache.radix8_global_hist.get_ptr(0),
                                 radix8_global_hist_bytes, 0, profile);
            profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                    profile);
            {
              const bool init_static_bindings =
                  !cache.radix8_upsweep_bindings[pass];
              ShaderResourceSet *bindings =
                  cache.cached_resource_set(cache.radix8_upsweep_bindings[pass],
                                            profile);
              profiled_rw_buffer(bindings, 0, key_read.get_ptr(0), key_bytes,
                                 profile);
              if (init_static_bindings) {
                profiled_rw_buffer(bindings, 1,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, profile);
                profiled_rw_buffer(bindings, 2,
                                   cache.radix8_partition_hist.get_ptr(0),
                                   radix8_partition_hist_bytes, profile);
              }
              dispatch_pipeline(cmdlist, cache.radix8_upsweep[pass].get(),
                                bindings, radix8_partitions, 1, 1,
                                scope_name("vulkan_sort_radix8_upsweep"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                      profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                      profile);
            }
            {
              const bool init_static_bindings = !cache.radix8_spine_bindings;
              ShaderResourceSet *bindings =
                  cache.cached_resource_set(cache.radix8_spine_bindings,
                                            profile);
              if (init_static_bindings) {
                profiled_rw_buffer(bindings, 0,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, profile);
                profiled_rw_buffer(bindings, 1,
                                   cache.radix8_partition_hist.get_ptr(0),
                                   radix8_partition_hist_bytes, profile);
              }
              dispatch_pipeline(cmdlist, cache.radix8_spine.get(),
                                bindings, kRadix8Bins, 1, 1,
                                scope_name("vulkan_sort_radix8_spine"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                      profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                      profile);
            }
            if (use_values) {
              const bool init_static_bindings =
                  !cache.radix8_downsweep_pairs_bindings[pass];
              ShaderResourceSet *bindings = cache.cached_resource_set(
                  cache.radix8_downsweep_pairs_bindings[pass], profile);
              profiled_rw_buffer(bindings, 0, key_read.get_ptr(0), key_bytes,
                                 profile);
              profiled_rw_buffer(bindings, 1, pass_key_write.get_ptr(0),
                                 key_bytes, profile);
              if (init_static_bindings) {
                profiled_rw_buffer(bindings, 2,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, profile);
                profiled_rw_buffer(bindings, 3,
                                   cache.radix8_partition_hist.get_ptr(0),
                                   radix8_partition_hist_bytes, profile);
              }
              profiled_rw_buffer(bindings, 4, value_read.get_ptr(0),
                                 value_bytes, profile);
              profiled_rw_buffer(bindings, 5, pass_value_write.get_ptr(0),
                                 value_bytes, profile);
              dispatch_pipeline(cmdlist,
                                cache.radix8_downsweep_pairs[pass].get(),
                                bindings, radix8_partitions, 1, 1,
                                scope_name(
                                    "vulkan_sort_radix8_downsweep_pairs"),
                                profile);
              profiled_buffer_barrier(cmdlist, pass_key_write, profile);
              profiled_buffer_barrier(cmdlist, pass_value_write, profile);
              if (direct_value_output) {
                value_read = pass_value_write;
                values_written_to_user = true;
              } else {
                std::swap(value_read, value_write);
              }
            } else {
              const bool init_static_bindings =
                  !cache.radix8_downsweep_keys_bindings[pass];
              ShaderResourceSet *bindings = cache.cached_resource_set(
                  cache.radix8_downsweep_keys_bindings[pass], profile);
              profiled_rw_buffer(bindings, 0, key_read.get_ptr(0), key_bytes,
                                 profile);
              profiled_rw_buffer(bindings, 1, pass_key_write.get_ptr(0),
                                 key_bytes, profile);
              if (init_static_bindings) {
                profiled_rw_buffer(bindings, 2,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, profile);
                profiled_rw_buffer(bindings, 3,
                                   cache.radix8_partition_hist.get_ptr(0),
                                   radix8_partition_hist_bytes, profile);
              }
              dispatch_pipeline(cmdlist, cache.radix8_downsweep_keys[pass].get(),
                                bindings, radix8_partitions, 1, 1,
                                scope_name("vulkan_sort_radix8_downsweep_keys"),
                                profile);
              profiled_buffer_barrier(cmdlist, pass_key_write, profile);
            }
            if (direct_key_output) {
              key_read = pass_key_write;
              keys_written_to_user = true;
            } else {
              std::swap(key_read, key_write);
            }
          }

          if (signed_keys) {
            dispatch_unary(cmdlist, op_device, cache.copy_i32.get(), key_read,
                           key_alloc, key_bytes, groups,
                           scope_name("vulkan_sort_copy_i32"), profile);
            profiled_buffer_barrier(cmdlist, key_alloc, profile);
          } else if (!keys_written_to_user) {
            profiled_buffer_copy(cmdlist, key_alloc.get_ptr(0),
                                 key_read.get_ptr(0), key_bytes, profile);
            profiled_buffer_barrier(cmdlist, key_alloc, profile);
          }
          if (use_values && !values_written_to_user) {
            profiled_buffer_copy(cmdlist, value_alloc.get_ptr(0),
                                 value_read.get_ptr(0), value_bytes, profile);
            profiled_buffer_barrier(cmdlist, value_alloc, profile);
          }
          if (profile) {
            profile->radix8_body_us += profile_time_us() - radix8_start;
            profile->lambda_total_us += profile_time_us() - lambda_start;
            g_vulkan_sort_cpu_profile.merge(*profile);
          }
          return;
        }
        if (signed_keys) {
          dispatch_unary(cmdlist, op_device, cache.init_i32.get(), key_alloc,
                         cache.key_in, key_bytes, groups,
                         scope_name("vulkan_sort_init_i32"), profile);
          profiled_buffer_barrier(cmdlist, cache.key_in, profile);
        } else {
          profiled_buffer_copy(cmdlist, cache.key_in.get_ptr(0),
                               key_alloc.get_ptr(0), key_bytes, profile);
          profiled_buffer_barrier(cmdlist, cache.key_in, profile);
        }
        if (use_values) {
          profiled_buffer_copy(cmdlist, cache.value_in.get_ptr(0),
                               value_alloc.get_ptr(0), value_bytes, profile);
          profiled_buffer_barrier(cmdlist, cache.value_in, profile);
        }

        DeviceAllocation key_read = cache.key_in;
        DeviceAllocation key_write = cache.key_out;
        DeviceAllocation value_read = cache.value_in;
        DeviceAllocation value_write = cache.value_out;
        for (int pass = 0; pass < 8; ++pass) {
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
            bindings->rw_buffer(1, cache.rank.get_ptr(0), key_bytes);
            bindings->rw_buffer(2, cache.hist.get_ptr(0), table_bytes);
            Pipeline *rank_pipeline =
                cache.subgroup_rank_enabled
                    ? cache.rank_hist_subgroup[pass].get()
                    : cache.rank_hist[pass].get();
            const char *rank_scope =
                cache.subgroup_rank_enabled ? "vulkan_sort_rank_hist_subgroup"
                                            : "vulkan_sort_rank_hist";
            dispatch_pipeline(cmdlist, rank_pipeline, bindings.get(), groups,
                              1, 1, scope_name(rank_scope), profile);
            profiled_buffer_barrier(cmdlist, cache.rank, profile);
            profiled_buffer_barrier(cmdlist, cache.hist, profile);
          }
          if (groups <= kSingleChunkPrefixMaxBlocks) {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, cache.hist.get_ptr(0), table_bytes);
            bindings->rw_buffer(1, cache.offsets.get_ptr(0), table_bytes);
            bindings->rw_buffer(2, cache.chunk_offsets.get_ptr(0),
                                chunk_table_bytes);
            dispatch_pipeline(cmdlist, cache.prefix_single_chunk.get(),
                              bindings.get(), 1, kRadixBins, 1,
                              scope_name("vulkan_sort_prefix_single_chunk"),
                              profile);
            profiled_buffer_barrier(cmdlist, cache.offsets, profile);
            profiled_buffer_barrier(cmdlist, cache.chunk_offsets, profile);
          } else {
            {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, cache.hist.get_ptr(0), table_bytes);
              bindings->rw_buffer(1, cache.offsets.get_ptr(0), table_bytes);
              bindings->rw_buffer(2, cache.chunk_sums.get_ptr(0),
                                  chunk_table_bytes);
              dispatch_pipeline(cmdlist, cache.prefix_block.get(),
                                bindings.get(), chunk_groups, kRadixBins, 1,
                                scope_name("vulkan_sort_prefix_block"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.offsets, profile);
              profiled_buffer_barrier(cmdlist, cache.chunk_sums, profile);
            }
            if (!inline_chunk_offsets) {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, cache.chunk_sums.get_ptr(0),
                                  chunk_table_bytes);
              bindings->rw_buffer(1, cache.chunk_offsets.get_ptr(0),
                                  chunk_table_bytes);
              dispatch_pipeline(cmdlist, cache.prefix_chunks.get(),
                                bindings.get(), 1, 1, 1,
                                scope_name("vulkan_sort_prefix_chunks"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.chunk_offsets, profile);
            }
          }
          if (use_values) {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
            bindings->rw_buffer(1, key_write.get_ptr(0), key_bytes);
            bindings->rw_buffer(2, cache.rank.get_ptr(0), key_bytes);
            bindings->rw_buffer(3, cache.offsets.get_ptr(0), table_bytes);
            bindings->rw_buffer(4, value_read.get_ptr(0), value_bytes);
            bindings->rw_buffer(5, value_write.get_ptr(0), value_bytes);
            bindings->rw_buffer(6,
                                (inline_chunk_offsets ? cache.chunk_sums
                                                      : cache.chunk_offsets)
                                    .get_ptr(0),
                                chunk_table_bytes);
            Pipeline *scatter_pipeline =
                inline_chunk_offsets
                    ? cache.scatter_pairs_inline_chunks[pass].get()
                    : cache.scatter_pairs[pass].get();
            const char *scatter_scope =
                inline_chunk_offsets ? "vulkan_sort_scatter_pairs_inline_chunks"
                                     : "vulkan_sort_scatter_pairs";
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                              groups, 1, 1, scope_name(scatter_scope), profile);
            profiled_buffer_barrier(cmdlist, key_write, profile);
            profiled_buffer_barrier(cmdlist, value_write, profile);
            std::swap(value_read, value_write);
          } else {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
            bindings->rw_buffer(1, key_write.get_ptr(0), key_bytes);
            bindings->rw_buffer(2, cache.rank.get_ptr(0), key_bytes);
            bindings->rw_buffer(3, cache.offsets.get_ptr(0), table_bytes);
            bindings->rw_buffer(4,
                                (inline_chunk_offsets ? cache.chunk_sums
                                                      : cache.chunk_offsets)
                                    .get_ptr(0),
                                chunk_table_bytes);
            Pipeline *scatter_pipeline =
                inline_chunk_offsets
                    ? cache.scatter_keys_inline_chunks[pass].get()
                    : cache.scatter_keys[pass].get();
            const char *scatter_scope =
                inline_chunk_offsets ? "vulkan_sort_scatter_keys_inline_chunks"
                                     : "vulkan_sort_scatter_keys";
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                              groups, 1, 1, scope_name(scatter_scope), profile);
            profiled_buffer_barrier(cmdlist, key_write, profile);
          }
          std::swap(key_read, key_write);
        }

        if (signed_keys) {
          dispatch_unary(cmdlist, op_device, cache.copy_i32.get(), key_read,
                         key_alloc, key_bytes, groups,
                         scope_name("vulkan_sort_copy_i32"), profile);
          profiled_buffer_barrier(cmdlist, key_alloc, profile);
        } else {
          profiled_buffer_copy(cmdlist, key_alloc.get_ptr(0),
                               key_read.get_ptr(0), key_bytes, profile);
          profiled_buffer_barrier(cmdlist, key_alloc, profile);
        }
        if (use_values) {
          profiled_buffer_copy(cmdlist, value_alloc.get_ptr(0),
                               value_read.get_ptr(0), value_bytes, profile);
          profiled_buffer_barrier(cmdlist, value_alloc, profile);
        }
        if (profile) {
          profile->lambda_total_us += profile_time_us() - lambda_start;
          g_vulkan_sort_cpu_profile.merge(*profile);
        }
      },
      {});
  if (front) {
    front->enqueue_us += profile_time_us() - start;
  }
  if (get_environ_config("TI_VULKAN_SORT_INTERNAL_SYNC", 0) != 0) {
    if (front) {
      start = profile_time_us();
    }
    synchronize();
    if (front) {
      front->internal_sync_calls++;
      front->internal_sync_us += profile_time_us() - start;
    }
  }
  if (front) {
    front->total_call_us += profile_time_us() - total_start;
    g_vulkan_sort_cpu_profile.merge(front_profile);
  }
  return cache.cached_bytes;
}

void Program::vulkan_radix_sort_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_sort_mutex);
    auto it = g_vulkan_sort_caches.find(this);
    sync_before_clear =
        it != g_vulkan_sort_caches.end() && it->second->has_workspace_allocs();
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_sort_mutex);
  auto it = g_vulkan_sort_caches.find(this);
  if (it != g_vulkan_sort_caches.end()) {
    g_vulkan_sort_caches.erase(it);
  }
}

void Program::vulkan_scan_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_scan_mutex);
    auto it = g_vulkan_scan_caches.find(this);
    sync_before_clear =
        it != g_vulkan_scan_caches.end() && it->second->has_workspace_allocs();
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_scan_mutex);
  auto it = g_vulkan_scan_caches.find(this);
  if (it != g_vulkan_scan_caches.end()) {
    g_vulkan_scan_caches.erase(it);
  }
}

void Program::vulkan_compact_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_compact_mutex);
    auto it = g_vulkan_compact_caches.find(this);
    sync_before_clear = it != g_vulkan_compact_caches.end() &&
                        it->second->has_workspace_allocs();
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_compact_mutex);
  auto it = g_vulkan_compact_caches.find(this);
  if (it != g_vulkan_compact_caches.end()) {
    g_vulkan_compact_caches.erase(it);
  }
}

void Program::vulkan_histogram_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_histogram_mutex);
    auto it = g_vulkan_histogram_caches.find(this);
    sync_before_clear = it != g_vulkan_histogram_caches.end() &&
                        it->second->has_workspace_allocs();
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_histogram_mutex);
  auto it = g_vulkan_histogram_caches.find(this);
  if (it != g_vulkan_histogram_caches.end()) {
    g_vulkan_histogram_caches.erase(it);
  }
}

void Program::vulkan_reduce_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_reduce_mutex);
    auto it = g_vulkan_reduce_caches.find(this);
    sync_before_clear = it != g_vulkan_reduce_caches.end() &&
                        it->second->has_workspace_allocs();
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_reduce_mutex);
  auto it = g_vulkan_reduce_caches.find(this);
  if (it != g_vulkan_reduce_caches.end()) {
    g_vulkan_reduce_caches.erase(it);
  }
}

void Program::vulkan_transform_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_transform_mutex);
    auto it = g_vulkan_transform_caches.find(this);
    sync_before_clear =
        it != g_vulkan_transform_caches.end() &&
        it->second->params != kDeviceNullAllocation;
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_transform_mutex);
  auto it = g_vulkan_transform_caches.find(this);
  if (it != g_vulkan_transform_caches.end()) {
    g_vulkan_transform_caches.erase(it);
  }
}

void Program::vulkan_indexed_copy_clear_workspace() {
  std::lock_guard<std::mutex> guard(g_vulkan_indexed_copy_mutex);
  auto it = g_vulkan_indexed_copy_caches.find(this);
  if (it != g_vulkan_indexed_copy_caches.end()) {
    g_vulkan_indexed_copy_caches.erase(it);
  }
}

std::size_t Program::vulkan_radix_sort_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_sort_mutex);
  auto it = g_vulkan_sort_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_sort_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_scan_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_scan_mutex);
  auto it = g_vulkan_scan_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_scan_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_compact_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_compact_mutex);
  auto it = g_vulkan_compact_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_compact_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_histogram_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_histogram_mutex);
  auto it = g_vulkan_histogram_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_histogram_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_reduce_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_reduce_mutex);
  auto it = g_vulkan_reduce_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_reduce_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_transform_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_transform_mutex);
  auto it = g_vulkan_transform_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_transform_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_indexed_copy_workspace_bytes() const {
  return 0;
}

void Program::vulkan_radix_sort_cpu_profile_clear() {
  g_vulkan_sort_cpu_profile.clear();
}

std::string Program::vulkan_radix_sort_cpu_profile_report() const {
  return g_vulkan_sort_cpu_profile.report_json();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_radix_sort_available() const {
  return false;
}

bool Program::vulkan_scan_available() const {
  return false;
}

bool Program::vulkan_compact_available() const {
  return false;
}

bool Program::vulkan_histogram_available() const {
  return false;
}

bool Program::vulkan_reduce_available() const {
  return false;
}

bool Program::vulkan_transform_available() const {
  return false;
}

bool Program::vulkan_indexed_copy_available() const {
  return false;
}

std::size_t Program::vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   int key_type) {
  TI_ERROR("Vulkan native radix sort requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_ndarray(Ndarray *data,
                                                   int value_type) {
  TI_ERROR("Vulkan native scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_compact_i32_ndarray(Ndarray *values,
                                                Ndarray *flags,
                                                Ndarray *output,
                                                Ndarray *count) {
  TI_ERROR("Vulkan native compact requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_histogram_i32_ndarray(Ndarray *values,
                                                  Ndarray *bins) {
  TI_ERROR("Vulkan native histogram requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_i32_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int op) {
  TI_ERROR("Vulkan native reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_ndarray(Ndarray *src,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     double scale,
                                                     double bias) {
  TI_ERROR("Vulkan native transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_ndarray(Ndarray *src,
                                           Ndarray *indices,
                                           Ndarray *dst) {
  TI_ERROR("Vulkan native gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst) {
  TI_ERROR("Vulkan native scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

void Program::vulkan_radix_sort_clear_workspace() {
}

void Program::vulkan_scan_clear_workspace() {
}

void Program::vulkan_compact_clear_workspace() {
}

void Program::vulkan_histogram_clear_workspace() {
}

void Program::vulkan_reduce_clear_workspace() {
}

void Program::vulkan_transform_clear_workspace() {
}

void Program::vulkan_indexed_copy_clear_workspace() {
}

std::size_t Program::vulkan_radix_sort_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_scan_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_compact_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_histogram_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_reduce_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_transform_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_indexed_copy_workspace_bytes() const {
  return 0;
}

void Program::vulkan_radix_sort_cpu_profile_clear() {
}

std::string Program::vulkan_radix_sort_cpu_profile_report() const {
  return "{}";
}

}  // namespace taichi::lang

#endif
