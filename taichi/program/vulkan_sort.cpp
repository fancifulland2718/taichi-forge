#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"

#include <algorithm>
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
static const uint32_t kSortInitU32IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_u32_index.comp.spv.h"
    ;
static const uint32_t kSortInitI32IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_i32_index.comp.spv.h"
    ;
static const uint32_t kSortInitF32IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_f32_index.comp.spv.h"
    ;
static const uint32_t kSortInitU64IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_u64_index.comp.spv.h"
    ;
static const uint32_t kSortInitI64IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_i64_index.comp.spv.h"
    ;
static const uint32_t kSortInitF64IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_f64_index.comp.spv.h"
    ;
static const uint32_t kGatherU32ByU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_u32_by_u32.comp.spv.h"
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
static const uint32_t kScanI32BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_strided.comp.spv.h"
    ;
static const uint32_t kScanI32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanI32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add.comp.spv.h"
    ;
static const uint32_t kScanI32AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add_strided.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup_strided.comp.spv.h"
    ;
static const uint32_t kScanF32BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block.comp.spv.h"
    ;
static const uint32_t kScanF32BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block_strided.comp.spv.h"
    ;
static const uint32_t kScanF32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanF32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_add.comp.spv.h"
    ;
static const uint32_t kScanF32AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_add_strided.comp.spv.h"
    ;
static const uint32_t kScanF32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kScanF32SmallSubgroupStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_small_subgroup_strided.comp.spv.h"
    ;
static const uint32_t kScanU32BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block.comp.spv.h"
    ;
static const uint32_t kScanU32BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block_strided.comp.spv.h"
    ;
static const uint32_t kScanU32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanU32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_add.comp.spv.h"
    ;
static const uint32_t kScanU32AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_add_strided.comp.spv.h"
    ;
static const uint32_t kScanU32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kScanU32SmallSubgroupStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_small_subgroup_strided.comp.spv.h"
    ;
static const uint32_t kScanU64BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_block.comp.spv.h"
    ;
static const uint32_t kScanU64BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_block_strided.comp.spv.h"
    ;
static const uint32_t kScanU64AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_add.comp.spv.h"
    ;
static const uint32_t kScanU64AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_add_strided.comp.spv.h"
    ;
static const uint32_t kScanI64BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_block.comp.spv.h"
    ;
static const uint32_t kScanI64BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_block_strided.comp.spv.h"
    ;
static const uint32_t kScanI64AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_add.comp.spv.h"
    ;
static const uint32_t kScanI64AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_add_strided.comp.spv.h"
    ;
static const uint32_t kScanF64BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_block.comp.spv.h"
    ;
static const uint32_t kScanF64BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_block_strided.comp.spv.h"
    ;
static const uint32_t kScanF64AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_add.comp.spv.h"
    ;
static const uint32_t kScanF64AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_add_strided.comp.spv.h"
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
static const uint32_t kHistogramU32CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramU32CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramU32CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_count_private_shared.comp.spv.h"
    ;
static const uint32_t kHistogramU32SingleSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_single_shared.comp.spv.h"
    ;
static const uint32_t kHistogramI64ClearSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i64_clear.comp.spv.h"
    ;
static const uint32_t kHistogramI32I64CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_i64_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramI32I64CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_i64_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramI32I64CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_i64_count_private_shared.comp.spv.h"
    ;
static const uint32_t kHistogramI64ReducePrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i64_reduce_private.comp.spv.h"
    ;
static const uint32_t kHistogramU32I64CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_i64_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramU32I64CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_i64_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramU32I64CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_i64_count_private_shared.comp.spv.h"
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
static const uint32_t kReduceF32SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceF32MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_private.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_private.comp.spv.h"
    ;
static const uint32_t kReduceF32SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceF32MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_final.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_final.comp.spv.h"
    ;
static const uint32_t kReduceF32SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceF32MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_single.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_single.comp.spv.h"
    ;
static const uint32_t kReduceU32SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceU32MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_private.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_private.comp.spv.h"
    ;
static const uint32_t kReduceU32SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceU32MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_final.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_final.comp.spv.h"
    ;
static const uint32_t kReduceU32SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceU32MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_single.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_single.comp.spv.h"
    ;
static const uint32_t kReduceU64SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceU64MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_private.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_private.comp.spv.h"
    ;
static const uint32_t kReduceU64SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceU64MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_final.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_final.comp.spv.h"
    ;
static const uint32_t kReduceU64SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceU64MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_single.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_single.comp.spv.h"
    ;
static const uint32_t kReduceI64SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceI64MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_private.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_private.comp.spv.h"
    ;
static const uint32_t kReduceI64SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceI64MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_final.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_final.comp.spv.h"
    ;
static const uint32_t kReduceI64SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceI64MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_single.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_single.comp.spv.h"
    ;
static const uint32_t kReduceF64SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceF64MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_private.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_private.comp.spv.h"
    ;
static const uint32_t kReduceF64SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceF64MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_final.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_final.comp.spv.h"
    ;
static const uint32_t kReduceF64SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceF64MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_single.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_single.comp.spv.h"
    ;
static const uint32_t kReduceI32SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_single_strided.comp.spv.h"
    ;
static const uint32_t kTransformI32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_i32_affine.comp.spv.h"
    ;
static const uint32_t kTransformF32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_f32_affine.comp.spv.h"
    ;
static const uint32_t kTransformU64AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_u64_affine.comp.spv.h"
    ;
static const uint32_t kTransformF64AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_f64_affine.comp.spv.h"
    ;
static const uint32_t kGatherU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kGatherStridedU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_strided_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterStridedU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_strided_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddI32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddF32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddU64ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u64_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddI64ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i64_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddF64ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f64_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddI32ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i32_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddF32ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f32_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddU32ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u32_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddU64ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u64_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddI64ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i64_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddF64ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f64_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kBucketClearI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_clear_i32.comp.spv.h"
    ;
static const uint32_t kBucketCountI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_count_i32.comp.spv.h"
    ;
static const uint32_t kBucketCountPrivateSharedI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_count_private_shared_i32.comp.spv.h"
    ;
static const uint32_t kBucketPrefixI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_prefix_i32.comp.spv.h"
    ;
static const uint32_t kBucketPrefixChunksI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_prefix_chunks_i32.comp.spv.h"
    ;
static const uint32_t kBucketScatterI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_i32.comp.spv.h"
    ;
static const uint32_t kBucketScatterF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_f32.comp.spv.h"
    ;
static const uint32_t kBucketScatterU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_u32.comp.spv.h"
    ;
static const uint32_t kBucketScatterRaw64Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_raw64.comp.spv.h"
    ;
static const uint32_t kBucketScatterRawWordsSpv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_raw_words.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_i32.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_f32.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_u32.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedRaw64Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_raw64.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedRawWordsSpv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_raw_words.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_i32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_f32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_u32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_u64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_i64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_f64.comp.spv.h"
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
static const uint32_t kScatterPairsRaw64Shift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift28.comp.spv.h"
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
static const uint32_t kRadix8DownsweepPairsRaw64Shift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_raw64_shift0.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift8.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsRaw64Shift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_raw64_shift8.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift16.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsRaw64Shift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_raw64_shift16.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_shift24.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsRaw64Shift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_raw64_shift24.comp.spv.h"
    ;

std::unique_ptr<Pipeline> create_pipeline_from_spv(Device *device,
                                                   const uint32_t *spv,
                                                   size_t spv_bytes,
                                                   const std::string &name) {
  PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                          spv,
                          spv_bytes,
                          PipelineStageType::compute};
  auto [pipeline, res] = device->create_pipeline_unique(desc, name);
  TI_ERROR_IF(res != RhiResult::success,
              "Failed to create Vulkan sort pipeline '{}': RhiResult({})",
              name, res);
  return std::move(pipeline);
}

template <size_t N>
std::unique_ptr<Pipeline> create_pipeline(Device *device,
                                          const uint32_t (&spv)[N],
                                          const std::string &name) {
  return create_pipeline_from_spv(device, spv, sizeof(spv), name);
}

struct VulkanScanStridedSpvSet {
  const uint32_t *block_spv;
  size_t block_bytes;
  const uint32_t *add_spv;
  size_t add_bytes;
  const uint32_t *small_spv;
  size_t small_bytes;
  const char *dtype_name;
};

const VulkanScanStridedSpvSet &vulkan_scan_strided_spv_set(int value_type) {
  static const VulkanScanStridedSpvSet sets[] = {
      {kScanI32BlockStridedSpv, sizeof(kScanI32BlockStridedSpv),
       kScanI32AddStridedSpv, sizeof(kScanI32AddStridedSpv),
       kScanI32SmallSubgroupStridedSpv,
       sizeof(kScanI32SmallSubgroupStridedSpv), "i32"},
      {kScanF32BlockStridedSpv, sizeof(kScanF32BlockStridedSpv),
       kScanF32AddStridedSpv, sizeof(kScanF32AddStridedSpv),
       kScanF32SmallSubgroupStridedSpv,
       sizeof(kScanF32SmallSubgroupStridedSpv), "f32"},
      {kScanU32BlockStridedSpv, sizeof(kScanU32BlockStridedSpv),
       kScanU32AddStridedSpv, sizeof(kScanU32AddStridedSpv),
       kScanU32SmallSubgroupStridedSpv,
       sizeof(kScanU32SmallSubgroupStridedSpv), "u32"},
      {kScanU64BlockStridedSpv, sizeof(kScanU64BlockStridedSpv),
       kScanU64AddStridedSpv, sizeof(kScanU64AddStridedSpv), nullptr, 0,
       "u64"},
      {kScanI64BlockStridedSpv, sizeof(kScanI64BlockStridedSpv),
       kScanI64AddStridedSpv, sizeof(kScanI64AddStridedSpv), nullptr, 0,
       "i64"},
      {kScanF64BlockStridedSpv, sizeof(kScanF64BlockStridedSpv),
       kScanF64AddStridedSpv, sizeof(kScanF64AddStridedSpv), nullptr, 0,
       "f64"},
  };
  TI_ERROR_IF(value_type < 0 || value_type >= 6,
              "Unsupported Vulkan strided scan value type.");
  return sets[value_type];
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
  DeviceAllocation key_high{kDeviceNullAllocation};
  bool has_value_buffers{false};
  bool has_high_key_buffer{false};
  size_t key_bytes_per_item{sizeof(uint32_t)};
  size_t value_bytes_per_item{0};
  bool workspace_uses_radix8{false};
  bool subgroup_rank_enabled{false};
  bool radix8_enabled{false};
  bool inline_chunk_prefix_allowed{false};

  std::unique_ptr<Pipeline> init_i32;
  std::unique_ptr<Pipeline> copy_i32;
  std::unique_ptr<Pipeline> sort_init_u32_index;
  std::unique_ptr<Pipeline> sort_init_i32_index;
  std::unique_ptr<Pipeline> sort_init_f32_index;
  std::unique_ptr<Pipeline> sort_init_u64_index;
  std::unique_ptr<Pipeline> sort_init_i64_index;
  std::unique_ptr<Pipeline> sort_init_f64_index;
  std::unique_ptr<Pipeline> gather_u32_by_u32;
  std::unique_ptr<Pipeline> prefix_block;
  std::unique_ptr<Pipeline> prefix_chunks;
  std::unique_ptr<Pipeline> prefix_single_chunk;
  std::array<std::unique_ptr<Pipeline>, 8> rank_hist;
  std::array<std::unique_ptr<Pipeline>, 8> rank_hist_subgroup;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_keys;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_keys_inline_chunks;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_raw64;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_inline_chunks;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_inline_chunks_raw64;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_upsweep;
  std::unique_ptr<Pipeline> radix8_spine;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_downsweep_keys;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_downsweep_pairs;
  std::array<std::unique_ptr<Pipeline>, 4> radix8_downsweep_pairs_raw64;
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
          &value_in, &value_out, &key_high}) {
      if (*alloc != kDeviceNullAllocation) {
        device->dealloc_memory(*alloc);
        *alloc = kDeviceNullAllocation;
      }
    }
    capacity = 0;
    num_blocks = 0;
    cached_bytes = 0;
    has_value_buffers = false;
    has_high_key_buffer = false;
    key_bytes_per_item = sizeof(uint32_t);
    value_bytes_per_item = 0;
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
      sort_init_u32_index.reset();
      sort_init_i32_index.reset();
      sort_init_f32_index.reset();
      sort_init_u64_index.reset();
      sort_init_i64_index.reset();
      sort_init_f64_index.reset();
      gather_u32_by_u32.reset();
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
      for (auto &pipeline : scatter_pairs_raw64) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs_inline_chunks) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs_inline_chunks_raw64) {
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
      for (auto &pipeline : radix8_downsweep_pairs_raw64) {
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
    sort_init_u32_index = create_pipeline(
        dev, kSortInitU32IndexSpv, "vulkan_sort_init_u32_index");
    sort_init_i32_index = create_pipeline(
        dev, kSortInitI32IndexSpv, "vulkan_sort_init_i32_index");
    sort_init_f32_index = create_pipeline(
        dev, kSortInitF32IndexSpv, "vulkan_sort_init_f32_index");
    sort_init_u64_index = create_pipeline(
        dev, kSortInitU64IndexSpv, "vulkan_sort_init_u64_index");
    sort_init_i64_index = create_pipeline(
        dev, kSortInitI64IndexSpv, "vulkan_sort_init_i64_index");
    sort_init_f64_index = create_pipeline(
        dev, kSortInitF64IndexSpv, "vulkan_sort_init_f64_index");
    gather_u32_by_u32 =
        create_pipeline(dev, kGatherU32ByU32Spv, "vulkan_sort_gather_u32_by_u32");
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
    const std::array<const uint32_t *, 8> scatter_pair_raw64_data = {
        kScatterPairsRaw64Shift0Spv,  kScatterPairsRaw64Shift4Spv,
        kScatterPairsRaw64Shift8Spv,  kScatterPairsRaw64Shift12Spv,
        kScatterPairsRaw64Shift16Spv, kScatterPairsRaw64Shift20Spv,
        kScatterPairsRaw64Shift24Spv, kScatterPairsRaw64Shift28Spv};
    const std::array<size_t, 8> scatter_pair_raw64_sizes = {
        sizeof(kScatterPairsRaw64Shift0Spv),
        sizeof(kScatterPairsRaw64Shift4Spv),
        sizeof(kScatterPairsRaw64Shift8Spv),
        sizeof(kScatterPairsRaw64Shift12Spv),
        sizeof(kScatterPairsRaw64Shift16Spv),
        sizeof(kScatterPairsRaw64Shift20Spv),
        sizeof(kScatterPairsRaw64Shift24Spv),
        sizeof(kScatterPairsRaw64Shift28Spv)};
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
    const std::array<const uint32_t *, 8> scatter_pair_inline_raw64_data = {
        kScatterPairsInlineChunksRaw64Shift0Spv,
        kScatterPairsInlineChunksRaw64Shift4Spv,
        kScatterPairsInlineChunksRaw64Shift8Spv,
        kScatterPairsInlineChunksRaw64Shift12Spv,
        kScatterPairsInlineChunksRaw64Shift16Spv,
        kScatterPairsInlineChunksRaw64Shift20Spv,
        kScatterPairsInlineChunksRaw64Shift24Spv,
        kScatterPairsInlineChunksRaw64Shift28Spv};
    const std::array<size_t, 8> scatter_pair_inline_raw64_sizes = {
        sizeof(kScatterPairsInlineChunksRaw64Shift0Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift4Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift8Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift12Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift16Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift20Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift24Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift28Spv)};
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

      PipelineSourceDesc scatter_pair_raw64_desc{
          PipelineSourceType::spirv_binary,
          scatter_pair_raw64_data[pass],
          scatter_pair_raw64_sizes[pass],
          PipelineStageType::compute};
      auto [scatter_pair_raw64_pipeline, scatter_pair_raw64_res] =
          dev->create_pipeline_unique(
              scatter_pair_raw64_desc,
              fmt::format("vulkan_sort_scatter_pairs_raw64_{}", pass));
      TI_ERROR_IF(scatter_pair_raw64_res != RhiResult::success,
                  "Failed to create Vulkan sort raw64 pair scatter pipeline "
                  "{}: RhiResult({})",
                  pass, scatter_pair_raw64_res);
      scatter_pairs_raw64[pass] = std::move(scatter_pair_raw64_pipeline);

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

      PipelineSourceDesc scatter_pair_inline_raw64_desc{
          PipelineSourceType::spirv_binary,
          scatter_pair_inline_raw64_data[pass],
          scatter_pair_inline_raw64_sizes[pass],
          PipelineStageType::compute};
      auto [scatter_pair_inline_raw64_pipeline,
            scatter_pair_inline_raw64_res] =
          dev->create_pipeline_unique(
              scatter_pair_inline_raw64_desc,
              fmt::format("vulkan_sort_scatter_pairs_inline_chunks_raw64_{}",
                          pass));
      TI_ERROR_IF(
          scatter_pair_inline_raw64_res != RhiResult::success,
          "Failed to create Vulkan sort inline raw64 pair scatter pipeline "
          "{}: RhiResult({})",
          pass, scatter_pair_inline_raw64_res);
      scatter_pairs_inline_chunks_raw64[pass] =
          std::move(scatter_pair_inline_raw64_pipeline);
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
      const std::array<const uint32_t *, 4> downsweep_pair_raw64_data = {
          kRadix8DownsweepPairsRaw64Shift0Spv,
          kRadix8DownsweepPairsRaw64Shift8Spv,
          kRadix8DownsweepPairsRaw64Shift16Spv,
          kRadix8DownsweepPairsRaw64Shift24Spv};
      const std::array<size_t, 4> downsweep_pair_raw64_sizes = {
          sizeof(kRadix8DownsweepPairsRaw64Shift0Spv),
          sizeof(kRadix8DownsweepPairsRaw64Shift8Spv),
          sizeof(kRadix8DownsweepPairsRaw64Shift16Spv),
          sizeof(kRadix8DownsweepPairsRaw64Shift24Spv)};

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

        PipelineSourceDesc downsweep_pair_raw64_desc{
            PipelineSourceType::spirv_binary,
            downsweep_pair_raw64_data[pass],
            downsweep_pair_raw64_sizes[pass],
            PipelineStageType::compute};
        auto [downsweep_pair_raw64_pipeline, downsweep_pair_raw64_res] =
            dev->create_pipeline_unique(
                downsweep_pair_raw64_desc,
                fmt::format("vulkan_sort_radix8_downsweep_pairs_raw64_{}",
                            pass));
        TI_ERROR_IF(downsweep_pair_raw64_res != RhiResult::success,
                    "Failed to create Vulkan radix8 raw64 pair downsweep "
                    "pipeline {}: RhiResult({})",
                    pass, downsweep_pair_raw64_res);
        radix8_downsweep_pairs_raw64[pass] =
            std::move(downsweep_pair_raw64_pipeline);
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
                               bool use_radix8,
                               size_t value_size,
                               size_t key_item_size = sizeof(uint32_t),
                               bool use_high_keys = false) const {
    const size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const size_t requested_partitions =
        (n + kRadix8PartitionSize - 1) / kRadix8PartitionSize;
    const size_t requested_units =
        use_radix8 ? requested_partitions : requested_blocks;
    return !(capacity >= n && num_blocks >= requested_units &&
             has_value_buffers >= use_values &&
             has_high_key_buffer >= use_high_keys &&
             workspace_uses_radix8 == use_radix8 &&
             key_bytes_per_item >= key_item_size &&
             (!use_values || value_bytes_per_item >= value_size));
  }

  void ensure_workspace(size_t n,
                        bool use_values,
                        bool use_radix8,
                        size_t value_size,
                        size_t key_item_size = sizeof(uint32_t),
                        bool use_high_keys = false) {
    size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    size_t requested_partitions =
        (n + kRadix8PartitionSize - 1) / kRadix8PartitionSize;
    const size_t requested_units =
        use_radix8 ? requested_partitions : requested_blocks;
    if (!needs_workspace_realloc(n, use_values, use_radix8, value_size,
                                 key_item_size, use_high_keys)) {
      return;
    }
    clear_resource_sets();
    clear_allocs();
    capacity = n;
    num_blocks = requested_units;
    has_value_buffers = use_values;
    has_high_key_buffer = use_high_keys;
    key_bytes_per_item = key_item_size;
    value_bytes_per_item = use_values ? value_size : 0;
    workspace_uses_radix8 = use_radix8;
    const size_t sort_key_bytes = n * sizeof(uint32_t);
    const size_t key_storage_bytes = n * key_item_size;
    key_in = alloc_storage(key_storage_bytes);
    key_out = alloc_storage(key_storage_bytes);
    cached_bytes = key_storage_bytes * 2;
    if (use_high_keys) {
      key_high = alloc_storage(sort_key_bytes);
      cached_bytes += sort_key_bytes;
    }
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
      const size_t value_bytes = n * value_size;
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
  DeviceAllocation params{kDeviceNullAllocation};
  size_t dummy_sums_capacity{0};
  size_t params_capacity{0};
  std::unique_ptr<Pipeline> scan_i32_block;
  std::unique_ptr<Pipeline> scan_i32_block_subgroup;
  std::unique_ptr<Pipeline> scan_i32_add;
  std::unique_ptr<Pipeline> scan_i32_small_subgroup;
  std::unique_ptr<Pipeline> scan_f32_block;
  std::unique_ptr<Pipeline> scan_f32_block_subgroup;
  std::unique_ptr<Pipeline> scan_f32_add;
  std::unique_ptr<Pipeline> scan_f32_small_subgroup;
  std::unique_ptr<Pipeline> scan_u32_block;
  std::unique_ptr<Pipeline> scan_u32_block_subgroup;
  std::unique_ptr<Pipeline> scan_u32_add;
  std::unique_ptr<Pipeline> scan_u32_small_subgroup;
  std::unique_ptr<Pipeline> scan_u64_block;
  std::unique_ptr<Pipeline> scan_u64_add;
  std::unique_ptr<Pipeline> scan_i64_block;
  std::unique_ptr<Pipeline> scan_i64_add;
  std::unique_ptr<Pipeline> scan_f64_block;
  std::unique_ptr<Pipeline> scan_f64_add;
  std::array<std::unique_ptr<Pipeline>, 6> scan_block_strided;
  std::array<std::unique_ptr<Pipeline>, 6> scan_add_strided;
  std::array<std::unique_ptr<Pipeline>, 3> scan_small_strided;
  bool subgroup_scan_enabled{false};

  void clear_allocs() {
    if (device && workspace != kDeviceNullAllocation) {
      device->dealloc_memory(workspace);
    }
    if (device && dummy_sums != kDeviceNullAllocation) {
      device->dealloc_memory(dummy_sums);
    }
    if (device && params != kDeviceNullAllocation) {
      device->dealloc_memory(params);
    }
    workspace = kDeviceNullAllocation;
    dummy_sums = kDeviceNullAllocation;
    params = kDeviceNullAllocation;
    dummy_sums_capacity = 0;
    params_capacity = 0;
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
      scan_f32_block.reset();
      scan_f32_block_subgroup.reset();
      scan_f32_add.reset();
      scan_f32_small_subgroup.reset();
      scan_u32_block.reset();
      scan_u32_block_subgroup.reset();
      scan_u32_add.reset();
      scan_u32_small_subgroup.reset();
      scan_u64_block.reset();
      scan_u64_add.reset();
      scan_i64_block.reset();
      scan_i64_add.reset();
      scan_f64_block.reset();
      scan_f64_add.reset();
      for (auto &pipeline : scan_block_strided) {
        pipeline.reset();
      }
      for (auto &pipeline : scan_add_strided) {
        pipeline.reset();
      }
      for (auto &pipeline : scan_small_strided) {
        pipeline.reset();
      }
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
    scan_f32_block =
        create_pipeline(dev, kScanF32BlockSpv, "vulkan_scan_f32_block");
    scan_u32_block =
        create_pipeline(dev, kScanU32BlockSpv, "vulkan_scan_u32_block");
    const bool int64_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_int64) != 0;
    const bool float64_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_float64) != 0;
    if (int64_supported) {
      scan_u64_block =
          create_pipeline(dev, kScanU64BlockSpv, "vulkan_scan_u64_block");
      scan_i64_block =
          create_pipeline(dev, kScanI64BlockSpv, "vulkan_scan_i64_block");
    }
    if (float64_supported) {
      scan_f64_block =
          create_pipeline(dev, kScanF64BlockSpv, "vulkan_scan_f64_block");
    }
    if (subgroup_scan_enabled) {
      scan_i32_block_subgroup = create_pipeline(
          dev, kScanI32BlockSubgroupSpv, "vulkan_scan_i32_block_subgroup");
      scan_i32_small_subgroup = create_pipeline(
          dev, kScanI32SmallSubgroupSpv, "vulkan_scan_i32_small_subgroup");
      scan_f32_block_subgroup = create_pipeline(
          dev, kScanF32BlockSubgroupSpv, "vulkan_scan_f32_block_subgroup");
      scan_f32_small_subgroup = create_pipeline(
          dev, kScanF32SmallSubgroupSpv, "vulkan_scan_f32_small_subgroup");
      scan_u32_block_subgroup = create_pipeline(
          dev, kScanU32BlockSubgroupSpv, "vulkan_scan_u32_block_subgroup");
      scan_u32_small_subgroup = create_pipeline(
          dev, kScanU32SmallSubgroupSpv, "vulkan_scan_u32_small_subgroup");
    }
    scan_i32_add = create_pipeline(dev, kScanI32AddSpv, "vulkan_scan_i32_add");
    scan_f32_add = create_pipeline(dev, kScanF32AddSpv, "vulkan_scan_f32_add");
    scan_u32_add = create_pipeline(dev, kScanU32AddSpv, "vulkan_scan_u32_add");
    if (int64_supported) {
      scan_u64_add =
          create_pipeline(dev, kScanU64AddSpv, "vulkan_scan_u64_add");
      scan_i64_add =
          create_pipeline(dev, kScanI64AddSpv, "vulkan_scan_i64_add");
    }
    if (float64_supported) {
      scan_f64_add =
          create_pipeline(dev, kScanF64AddSpv, "vulkan_scan_f64_add");
    }
  }

  void ensure_strided_pipelines(Device *dev, int value_type) {
    ensure_pipelines(dev);
    TI_ERROR_IF(value_type < 0 || value_type >= 6,
                "Unsupported Vulkan strided scan value type.");
    if (scan_block_strided[value_type]) {
      return;
    }
    const auto &spv = vulkan_scan_strided_spv_set(value_type);
    scan_block_strided[value_type] = create_pipeline_from_spv(
        dev, spv.block_spv, spv.block_bytes,
        fmt::format("vulkan_scan_{}_block_strided", spv.dtype_name));
    scan_add_strided[value_type] = create_pipeline_from_spv(
        dev, spv.add_spv, spv.add_bytes,
        fmt::format("vulkan_scan_{}_add_strided", spv.dtype_name));
    if (value_type < 3 && spv.small_spv && subgroup_scan_enabled) {
      scan_small_strided[value_type] = create_pipeline_from_spv(
          dev, spv.small_spv, spv.small_bytes,
          fmt::format("vulkan_scan_{}_small_subgroup_strided",
                      spv.dtype_name));
    }
    ensure_params();
  }

  Pipeline *scan_small_pipeline(int value_type) const {
    switch (value_type) {
      case 0:
        return scan_i32_small_subgroup.get();
      case 1:
        return scan_f32_small_subgroup.get();
      case 2:
        return scan_u32_small_subgroup.get();
      default:
        return nullptr;
    }
  }

  Pipeline *scan_block_pipeline(int value_type, bool subgroup) const {
    switch (value_type) {
      case 0:
        return subgroup ? scan_i32_block_subgroup.get() : scan_i32_block.get();
      case 1:
        return subgroup ? scan_f32_block_subgroup.get() : scan_f32_block.get();
      case 2:
        return subgroup ? scan_u32_block_subgroup.get() : scan_u32_block.get();
      case 3:
        return subgroup ? nullptr : scan_u64_block.get();
      case 4:
        return subgroup ? nullptr : scan_i64_block.get();
      case 5:
        return subgroup ? nullptr : scan_f64_block.get();
      default:
        return nullptr;
    }
  }

  Pipeline *scan_add_pipeline(int value_type) const {
    switch (value_type) {
      case 0:
        return scan_i32_add.get();
      case 1:
        return scan_f32_add.get();
      case 2:
        return scan_u32_add.get();
      case 3:
        return scan_u64_add.get();
      case 4:
        return scan_i64_add.get();
      case 5:
        return scan_f64_add.get();
      default:
        return nullptr;
    }
  }

  Pipeline *scan_block_strided_pipeline(int value_type) const {
    if (value_type < 0 || value_type >= 6) {
      return nullptr;
    }
    return scan_block_strided[value_type].get();
  }

  Pipeline *scan_add_strided_pipeline(int value_type) const {
    if (value_type < 0 || value_type >= 6) {
      return nullptr;
    }
    return scan_add_strided[value_type].get();
  }

  Pipeline *scan_small_strided_pipeline(int value_type) const {
    if (value_type < 0 || value_type >= 3) {
      return nullptr;
    }
    return scan_small_strided[value_type].get();
  }

  const char *scan_small_scope(int value_type) const {
    switch (value_type) {
      case 0:
        return "vulkan_scan_i32_small_subgroup";
      case 1:
        return "vulkan_scan_f32_small_subgroup";
      case 2:
        return "vulkan_scan_u32_small_subgroup";
      default:
        return "vulkan_scan_unknown_small_subgroup";
    }
  }

  const char *scan_block_scope(int value_type, bool subgroup) const {
    switch (value_type) {
      case 0:
        return subgroup ? "vulkan_scan_i32_block_subgroup"
                        : "vulkan_scan_i32_block";
      case 1:
        return subgroup ? "vulkan_scan_f32_block_subgroup"
                        : "vulkan_scan_f32_block";
      case 2:
        return subgroup ? "vulkan_scan_u32_block_subgroup"
                        : "vulkan_scan_u32_block";
      case 3:
        return subgroup ? "vulkan_scan_unknown_block"
                        : "vulkan_scan_u64_block";
      case 4:
        return subgroup ? "vulkan_scan_unknown_block"
                        : "vulkan_scan_i64_block";
      case 5:
        return subgroup ? "vulkan_scan_unknown_block"
                        : "vulkan_scan_f64_block";
      default:
        return "vulkan_scan_unknown_block";
    }
  }

  const char *scan_add_scope(int value_type) const {
    switch (value_type) {
      case 0:
        return "vulkan_scan_i32_add";
      case 1:
        return "vulkan_scan_f32_add";
      case 2:
        return "vulkan_scan_u32_add";
      case 3:
        return "vulkan_scan_u64_add";
      case 4:
        return "vulkan_scan_i64_add";
      case 5:
        return "vulkan_scan_f64_add";
      default:
        return "vulkan_scan_unknown_add";
    }
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

  size_t allocated_bytes() const {
    return capacity + dummy_sums_capacity + params_capacity;
  }

  void clear_workspace_alloc() {
    if (device && workspace != kDeviceNullAllocation) {
      device->dealloc_memory(workspace);
    }
    workspace = kDeviceNullAllocation;
    capacity = 0;
    cached_bytes = allocated_bytes();
  }

  void ensure_dummy_sums() {
    if (dummy_sums == kDeviceNullAllocation) {
      dummy_sums_capacity = sizeof(uint64_t);
      dummy_sums = alloc_storage(dummy_sums_capacity);
    }
    cached_bytes = allocated_bytes();
  }

  void ensure_params() {
    constexpr size_t kParamsBytes = 3 * sizeof(uint32_t);
    if (params == kDeviceNullAllocation) {
      params = alloc_storage(kParamsBytes);
      params_capacity = kParamsBytes;
    }
    cached_bytes = allocated_bytes();
  }

  void ensure_workspace(size_t bytes) {
    ensure_dummy_sums();
    if (bytes == 0 || !needs_workspace_realloc(bytes)) {
      return;
    }
    clear_workspace_alloc();
    workspace = alloc_storage(bytes);
    capacity = bytes;
    cached_bytes = allocated_bytes();
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
  std::unique_ptr<Pipeline> histogram_u32_count_direct;
  std::unique_ptr<Pipeline> histogram_u32_count_private;
  std::unique_ptr<Pipeline> histogram_u32_count_private_shared;
  std::unique_ptr<Pipeline> histogram_u32_single_shared;
  std::unique_ptr<Pipeline> histogram_i64_clear;
  std::unique_ptr<Pipeline> histogram_i32_i64_count_direct;
  std::unique_ptr<Pipeline> histogram_i32_i64_count_private;
  std::unique_ptr<Pipeline> histogram_i32_i64_count_private_shared;
  std::unique_ptr<Pipeline> histogram_i64_reduce_private;
  std::unique_ptr<Pipeline> histogram_u32_i64_count_direct;
  std::unique_ptr<Pipeline> histogram_u32_i64_count_private;
  std::unique_ptr<Pipeline> histogram_u32_i64_count_private_shared;

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
      histogram_u32_count_direct.reset();
      histogram_u32_count_private.reset();
      histogram_u32_count_private_shared.reset();
      histogram_u32_single_shared.reset();
      histogram_i64_clear.reset();
      histogram_i32_i64_count_direct.reset();
      histogram_i32_i64_count_private.reset();
      histogram_i32_i64_count_private_shared.reset();
      histogram_i64_reduce_private.reset();
      histogram_u32_i64_count_direct.reset();
      histogram_u32_i64_count_private.reset();
      histogram_u32_i64_count_private_shared.reset();
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
    histogram_u32_count_direct = create_pipeline(
        dev, kHistogramU32CountDirectSpv,
        "vulkan_histogram_u32_count_direct");
    histogram_u32_count_private = create_pipeline(
        dev, kHistogramU32CountPrivateSpv,
        "vulkan_histogram_u32_count_private");
    histogram_u32_count_private_shared = create_pipeline(
        dev, kHistogramU32CountPrivateSharedSpv,
        "vulkan_histogram_u32_count_private_shared");
    histogram_u32_single_shared = create_pipeline(
        dev, kHistogramU32SingleSharedSpv,
        "vulkan_histogram_u32_single_shared");
    const bool supports_i64_bins =
        dev->get_caps().get(DeviceCapability::spirv_has_int64) != 0 &&
        dev->get_caps().get(DeviceCapability::spirv_has_atomic_int64) != 0;
    if (supports_i64_bins) {
      histogram_i64_clear = create_pipeline(
          dev, kHistogramI64ClearSpv, "vulkan_histogram_i64_clear");
      histogram_i32_i64_count_direct = create_pipeline(
          dev, kHistogramI32I64CountDirectSpv,
          "vulkan_histogram_i32_i64_count_direct");
      histogram_i32_i64_count_private = create_pipeline(
          dev, kHistogramI32I64CountPrivateSpv,
          "vulkan_histogram_i32_i64_count_private");
      histogram_i32_i64_count_private_shared = create_pipeline(
          dev, kHistogramI32I64CountPrivateSharedSpv,
          "vulkan_histogram_i32_i64_count_private_shared");
      histogram_i64_reduce_private = create_pipeline(
          dev, kHistogramI64ReducePrivateSpv,
          "vulkan_histogram_i64_reduce_private");
      histogram_u32_i64_count_direct = create_pipeline(
          dev, kHistogramU32I64CountDirectSpv,
          "vulkan_histogram_u32_i64_count_direct");
      histogram_u32_i64_count_private = create_pipeline(
          dev, kHistogramU32I64CountPrivateSpv,
          "vulkan_histogram_u32_i64_count_private");
      histogram_u32_i64_count_private_shared = create_pipeline(
          dev, kHistogramU32I64CountPrivateSharedSpv,
          "vulkan_histogram_u32_i64_count_private_shared");
    }
  }

  Pipeline *clear_pipeline(int bin_type) const {
    return bin_type == 4 ? histogram_i64_clear.get() : histogram_i32_clear.get();
  }

  Pipeline *reduce_private_pipeline(int bin_type) const {
    return bin_type == 4 ? histogram_i64_reduce_private.get()
                         : histogram_i32_reduce_private.get();
  }

  Pipeline *count_direct_pipeline(int value_type, int bin_type) const {
    if (bin_type == 4) {
      return value_type == 2 ? histogram_u32_i64_count_direct.get()
                             : histogram_i32_i64_count_direct.get();
    }
    return value_type == 2 ? histogram_u32_count_direct.get()
                           : histogram_i32_count_direct.get();
  }

  Pipeline *count_private_pipeline(int value_type, int bin_type) const {
    if (bin_type == 4) {
      return value_type == 2 ? histogram_u32_i64_count_private.get()
                             : histogram_i32_i64_count_private.get();
    }
    return value_type == 2 ? histogram_u32_count_private.get()
                           : histogram_i32_count_private.get();
  }

  Pipeline *count_private_shared_pipeline(int value_type) const {
    return value_type == 2 ? histogram_u32_count_private_shared.get()
                           : histogram_i32_count_private_shared.get();
  }

  Pipeline *count_private_shared_pipeline(int value_type, int bin_type) const {
    if (bin_type == 4) {
      return value_type == 2 ? histogram_u32_i64_count_private_shared.get()
                             : histogram_i32_i64_count_private_shared.get();
    }
    return count_private_shared_pipeline(value_type);
  }

  Pipeline *single_shared_pipeline(int value_type) const {
    return value_type == 2 ? histogram_u32_single_shared.get()
                           : histogram_i32_single_shared.get();
  }

  const char *clear_scope(int bin_type) const {
    return bin_type == 4 ? "vulkan_histogram_i64_clear_bins"
                         : "vulkan_histogram_i32_clear_bins";
  }

  const char *reduce_private_scope(int bin_type) const {
    return bin_type == 4 ? "vulkan_histogram_i64_reduce_private"
                         : "vulkan_histogram_i32_reduce_private";
  }

  const char *count_direct_scope(int value_type, int bin_type) const {
    if (bin_type == 4) {
      return value_type == 2 ? "vulkan_histogram_u32_i64_count_direct"
                             : "vulkan_histogram_i32_i64_count_direct";
    }
    return value_type == 2 ? "vulkan_histogram_u32_count_direct"
                           : "vulkan_histogram_i32_count_direct";
  }

  const char *count_private_scope(int value_type,
                                  int bin_type,
                                  bool shared) const {
    if (bin_type == 4) {
      if (value_type == 2) {
        return shared ? "vulkan_histogram_u32_i64_count_private_shared"
                      : "vulkan_histogram_u32_i64_count_private";
      }
      return shared ? "vulkan_histogram_i32_i64_count_private_shared"
                    : "vulkan_histogram_i32_i64_count_private";
    }
    if (value_type == 2) {
      return shared ? "vulkan_histogram_u32_count_private_shared"
                    : "vulkan_histogram_u32_count_private";
    }
    return shared ? "vulkan_histogram_i32_count_private_shared"
                  : "vulkan_histogram_i32_count_private";
  }

  const char *single_shared_scope(int value_type) const {
    return value_type == 2 ? "vulkan_histogram_u32_single_shared"
                           : "vulkan_histogram_i32_single_shared";
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

struct VulkanReduceSpvSet {
  const uint32_t *private_spv{nullptr};
  size_t private_bytes{0};
  const uint32_t *private_strided_spv{nullptr};
  size_t private_strided_bytes{0};
  const uint32_t *final_spv{nullptr};
  size_t final_bytes{0};
  const uint32_t *single_spv{nullptr};
  size_t single_bytes{0};
  const uint32_t *single_strided_spv{nullptr};
  size_t single_strided_bytes{0};
  const char *dtype_name{nullptr};
  const char *op_name{nullptr};
};

#define TI_REDUCE_SPV_SET(TYPE_CAPS, TYPE_NAME, OP_CAPS, OP_NAME)       \
  VulkanReduceSpvSet {                                                  \
    kReduce##TYPE_CAPS##OP_CAPS##PrivateSpv,                            \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##PrivateSpv),                \
        kReduce##TYPE_CAPS##OP_CAPS##PrivateStridedSpv,                 \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##PrivateStridedSpv),         \
        kReduce##TYPE_CAPS##OP_CAPS##FinalSpv,                          \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##FinalSpv),                  \
        kReduce##TYPE_CAPS##OP_CAPS##SingleSpv,                         \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##SingleSpv),                 \
        kReduce##TYPE_CAPS##OP_CAPS##SingleStridedSpv,                  \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##SingleStridedSpv),          \
        TYPE_NAME, OP_NAME                                               \
  }

const VulkanReduceSpvSet &vulkan_reduce_spv_set(int value_type, int op) {
  static const std::array<std::array<VulkanReduceSpvSet, 3>, 6> table = {{
      {TI_REDUCE_SPV_SET(I32, "i32", Sum, "sum"),
       TI_REDUCE_SPV_SET(I32, "i32", Min, "min"),
       TI_REDUCE_SPV_SET(I32, "i32", Max, "max")},
      {TI_REDUCE_SPV_SET(F32, "f32", Sum, "sum"),
       TI_REDUCE_SPV_SET(F32, "f32", Min, "min"),
       TI_REDUCE_SPV_SET(F32, "f32", Max, "max")},
      {TI_REDUCE_SPV_SET(U32, "u32", Sum, "sum"),
       TI_REDUCE_SPV_SET(U32, "u32", Min, "min"),
       TI_REDUCE_SPV_SET(U32, "u32", Max, "max")},
      {TI_REDUCE_SPV_SET(U64, "u64", Sum, "sum"),
       TI_REDUCE_SPV_SET(U64, "u64", Min, "min"),
       TI_REDUCE_SPV_SET(U64, "u64", Max, "max")},
      {TI_REDUCE_SPV_SET(I64, "i64", Sum, "sum"),
       TI_REDUCE_SPV_SET(I64, "i64", Min, "min"),
       TI_REDUCE_SPV_SET(I64, "i64", Max, "max")},
      {TI_REDUCE_SPV_SET(F64, "f64", Sum, "sum"),
       TI_REDUCE_SPV_SET(F64, "f64", Min, "min"),
       TI_REDUCE_SPV_SET(F64, "f64", Max, "max")},
  }};
  return table[value_type][op];
}

#undef TI_REDUCE_SPV_SET

struct VulkanReducePipelineSet {
  std::array<std::unique_ptr<Pipeline>, 3> private_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> private_strided_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> final_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> single_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> single_strided_pipelines;

  void reset() {
    for (int i = 0; i < 3; ++i) {
      private_pipelines[i].reset();
      private_strided_pipelines[i].reset();
      final_pipelines[i].reset();
      single_pipelines[i].reset();
      single_strided_pipelines[i].reset();
    }
  }
};

struct VulkanReduceCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t params_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  DeviceAllocation params{kDeviceNullAllocation};
  std::array<VulkanReducePipelineSet, 6> reduce_pipelines;

  void clear_allocs() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    if (device && params != kDeviceNullAllocation) {
      device->dealloc_memory(params);
    }
    partial = kDeviceNullAllocation;
    params = kDeviceNullAllocation;
    partial_capacity = 0;
    params_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanReduceCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      for (auto &pipelines : reduce_pipelines) {
        pipelines.reset();
      }
    }
    device = dev;
  }

  VulkanReducePipelineSet &pipeline_set(int value_type) {
    return reduce_pipelines[value_type];
  }

  void ensure_pipeline_set(Device *dev, int value_type) {
    ensure_pipelines(dev);
    auto &pipelines = pipeline_set(value_type);
    if (pipelines.private_pipelines[0]) {
      return;
    }
    for (int op = 0; op < 3; ++op) {
      const auto &spv = vulkan_reduce_spv_set(value_type, op);
      pipelines.private_pipelines[op] = create_pipeline_from_spv(
          dev, spv.private_spv, spv.private_bytes,
          fmt::format("vulkan_reduce_{}_{}_private", spv.dtype_name,
                      spv.op_name));
      pipelines.final_pipelines[op] = create_pipeline_from_spv(
          dev, spv.final_spv, spv.final_bytes,
          fmt::format("vulkan_reduce_{}_{}_final", spv.dtype_name,
                      spv.op_name));
      pipelines.single_pipelines[op] = create_pipeline_from_spv(
          dev, spv.single_spv, spv.single_bytes,
          fmt::format("vulkan_reduce_{}_{}_single", spv.dtype_name,
                      spv.op_name));
    }
  }

  void ensure_strided_pipeline_set(Device *dev, int value_type) {
    ensure_pipeline_set(dev, value_type);
    auto &pipelines = pipeline_set(value_type);
    if (pipelines.private_strided_pipelines[0]) {
      return;
    }
    for (int op = 0; op < 3; ++op) {
      const auto &spv = vulkan_reduce_spv_set(value_type, op);
      pipelines.private_strided_pipelines[op] = create_pipeline_from_spv(
          dev, spv.private_strided_spv, spv.private_strided_bytes,
          fmt::format("vulkan_reduce_{}_{}_private_strided", spv.dtype_name,
                      spv.op_name));
      pipelines.single_strided_pipelines[op] = create_pipeline_from_spv(
          dev, spv.single_strided_spv, spv.single_strided_bytes,
          fmt::format("vulkan_reduce_{}_{}_single_strided", spv.dtype_name,
                      spv.op_name));
    }
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
    return partial != kDeviceNullAllocation || params != kDeviceNullAllocation;
  }

  size_t allocated_bytes() const {
    return partial_capacity + params_capacity;
  }

  void clear_partial_alloc() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = allocated_bytes();
  }

  void ensure_partial(size_t bytes) {
    if (bytes == 0 || !needs_partial_realloc(bytes)) {
      cached_bytes = allocated_bytes();
      return;
    }
    clear_partial_alloc();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = allocated_bytes();
  }

  void ensure_params() {
    constexpr size_t kParamsBytes = 3 * sizeof(uint32_t);
    if (params == kDeviceNullAllocation) {
      params = alloc_storage(kParamsBytes);
      params_capacity = kParamsBytes;
    }
    cached_bytes = allocated_bytes();
  }
};

struct VulkanTransformCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  DeviceAllocation params{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> transform_i32_affine;
  std::unique_ptr<Pipeline> transform_f32_affine;
  std::unique_ptr<Pipeline> transform_u64_affine;
  std::unique_ptr<Pipeline> transform_f64_affine;
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
      transform_u64_affine.reset();
      transform_f64_affine.reset();
      affine_bindings.reset();
    }
    device = dev;
    transform_i32_affine = create_pipeline(
        dev, kTransformI32AffineSpv, "vulkan_transform_i32_affine");
    transform_f32_affine = create_pipeline(
        dev, kTransformF32AffineSpv, "vulkan_transform_f32_affine");
    ensure_params();
  }

  Pipeline *pipeline_for(Device *dev, int value_type, bool has_float64) {
    ensure_pipelines(dev);
    if (value_type == 0 || value_type == 2) {
      return transform_i32_affine.get();
    }
    if (value_type == 1) {
      return transform_f32_affine.get();
    }
    if (value_type == 3 || value_type == 4) {
      if (!transform_u64_affine) {
        transform_u64_affine = create_pipeline(
            dev, kTransformU64AffineSpv, "vulkan_transform_u64_affine");
      }
      return transform_u64_affine.get();
    }
    if (value_type == 5) {
      TI_ERROR_IF(!has_float64,
                  "Vulkan native f64 transform requires shader Float64 "
                  "device capability.");
      if (!transform_f64_affine) {
        transform_f64_affine = create_pipeline(
            dev, kTransformF64AffineSpv, "vulkan_transform_f64_affine");
      }
      return transform_f64_affine.get();
    }
    TI_ERROR("Unsupported Vulkan transform value type.");
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
      params = alloc_storage(10 * sizeof(uint32_t));
    }
    cached_bytes = 10 * sizeof(uint32_t);
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
  size_t cached_bytes{0};
  DeviceAllocation indexed_copy_params{kDeviceNullAllocation};
  DeviceAllocation scatter_add_params{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> gather_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_u32_by_i32;
  std::unique_ptr<Pipeline> gather_strided_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_strided_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_i32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_f32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_u64_by_i32;
  std::unique_ptr<Pipeline> scatter_add_i64_by_i32;
  std::unique_ptr<Pipeline> scatter_add_f64_by_i32;
  std::array<std::unique_ptr<Pipeline>, 6> scatter_add_strided;
  std::unique_ptr<ShaderResourceSet> gather_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_bindings;
  std::unique_ptr<ShaderResourceSet> gather_strided_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_strided_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_i32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_f32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_u32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_u64_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_i64_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_f64_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 6> scatter_add_strided_bindings;

  void clear_allocs() {
    if (device && indexed_copy_params != kDeviceNullAllocation) {
      device->dealloc_memory(indexed_copy_params);
    }
    if (device && scatter_add_params != kDeviceNullAllocation) {
      device->dealloc_memory(scatter_add_params);
    }
    indexed_copy_params = kDeviceNullAllocation;
    scatter_add_params = kDeviceNullAllocation;
    cached_bytes = 0;
  }

  ~VulkanIndexedCopyCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && gather_u32_by_i32) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      gather_u32_by_i32.reset();
      scatter_u32_by_i32.reset();
      gather_strided_u32_by_i32.reset();
      scatter_strided_u32_by_i32.reset();
      scatter_add_i32_by_i32.reset();
      scatter_add_f32_by_i32.reset();
      scatter_add_u32_by_i32.reset();
      scatter_add_u64_by_i32.reset();
      scatter_add_i64_by_i32.reset();
      scatter_add_f64_by_i32.reset();
      for (auto &pipeline : scatter_add_strided) {
        pipeline.reset();
      }
      gather_bindings.reset();
      scatter_bindings.reset();
      gather_strided_bindings.reset();
      scatter_strided_bindings.reset();
      scatter_add_i32_bindings.reset();
      scatter_add_f32_bindings.reset();
      scatter_add_u32_bindings.reset();
      scatter_add_u64_bindings.reset();
      scatter_add_i64_bindings.reset();
      scatter_add_f64_bindings.reset();
      for (auto &bindings : scatter_add_strided_bindings) {
        bindings.reset();
      }
    }
    device = dev;
    gather_u32_by_i32 =
        create_pipeline(dev, kGatherU32ByI32Spv, "vulkan_gather_u32_by_i32");
    scatter_u32_by_i32 = create_pipeline(dev, kScatterU32ByI32Spv,
                                         "vulkan_scatter_u32_by_i32");
    gather_strided_u32_by_i32 =
        create_pipeline(dev, kGatherStridedU32ByI32Spv,
                        "vulkan_gather_strided_u32_by_i32");
    scatter_strided_u32_by_i32 =
        create_pipeline(dev, kScatterStridedU32ByI32Spv,
                        "vulkan_scatter_strided_u32_by_i32");
    scatter_add_i32_by_i32 = create_pipeline(
        dev, kScatterAddI32ByI32Spv, "vulkan_scatter_add_i32_by_i32");
    if (dev->get_caps().get(DeviceCapability::spirv_has_atomic_float_add) !=
        0) {
      scatter_add_f32_by_i32 = create_pipeline(
          dev, kScatterAddF32ByI32Spv, "vulkan_scatter_add_f32_by_i32");
    }
    scatter_add_u32_by_i32 = create_pipeline(
        dev, kScatterAddU32ByI32Spv, "vulkan_scatter_add_u32_by_i32");
    const bool int64_atomic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_int64) != 0 &&
        dev->get_caps().get(DeviceCapability::spirv_has_atomic_int64) != 0;
    if (int64_atomic_supported) {
      scatter_add_u64_by_i32 = create_pipeline(
          dev, kScatterAddU64ByI32Spv, "vulkan_scatter_add_u64_by_i32");
      scatter_add_i64_by_i32 = create_pipeline(
          dev, kScatterAddI64ByI32Spv, "vulkan_scatter_add_i64_by_i32");
    }
    const bool float64_atomic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_float64) != 0 &&
        dev->get_caps().get(DeviceCapability::spirv_has_atomic_float64_add) !=
            0;
    if (float64_atomic_supported) {
      scatter_add_f64_by_i32 = create_pipeline(
          dev, kScatterAddF64ByI32Spv, "vulkan_scatter_add_f64_by_i32");
    }
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams alloc_params;
    alloc_params.size = bytes;
    alloc_params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(alloc_params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan indexed-copy workspace: "
                "RhiResult({})",
                res);
    return alloc;
  }

  void ensure_scatter_add_params() {
    if (scatter_add_params == kDeviceNullAllocation) {
      scatter_add_params = alloc_storage(6 * sizeof(uint32_t));
    }
    cached_bytes = std::max(cached_bytes, 6 * sizeof(uint32_t));
  }

  void ensure_indexed_copy_params() {
    constexpr size_t params_bytes = 7 * sizeof(uint32_t);
    if (indexed_copy_params == kDeviceNullAllocation) {
      indexed_copy_params = alloc_storage(params_bytes);
    }
    cached_bytes = std::max(cached_bytes, params_bytes);
  }

  void ensure_scatter_add_strided_pipeline(int value_type) {
    if (scatter_add_strided[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddF32ByI32StridedSpv,
            "vulkan_scatter_add_f32_by_i32_strided");
        return;
      case 2:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddU32ByI32StridedSpv,
            "vulkan_scatter_add_u32_by_i32_strided");
        return;
      case 3:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddU64ByI32StridedSpv,
            "vulkan_scatter_add_u64_by_i32_strided");
        return;
      case 4:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddI64ByI32StridedSpv,
            "vulkan_scatter_add_i64_by_i32_strided");
        return;
      case 5:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddF64ByI32StridedSpv,
            "vulkan_scatter_add_f64_by_i32_strided");
        return;
      default:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddI32ByI32StridedSpv,
            "vulkan_scatter_add_i32_by_i32_strided");
        return;
    }
  }

  ShaderResourceSet *cached_resource_set(bool scatter) {
    auto &bindings = scatter ? scatter_bindings : gather_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_strided_resource_set(bool scatter) {
    auto &bindings = scatter ? scatter_strided_bindings : gather_strided_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_scatter_add_resource_set(int value_type) {
    auto &bindings = value_type == 1   ? scatter_add_f32_bindings
                     : value_type == 2 ? scatter_add_u32_bindings
                     : value_type == 3 ? scatter_add_u64_bindings
                     : value_type == 4 ? scatter_add_i64_bindings
                     : value_type == 5 ? scatter_add_f64_bindings
                                       : scatter_add_i32_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_scatter_add_strided_resource_set(int value_type) {
    auto &bindings = scatter_add_strided_bindings[value_type];
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  Pipeline *scatter_add_pipeline(int value_type) const {
    if (value_type == 1) {
      return scatter_add_f32_by_i32.get();
    }
    if (value_type == 3) {
      return scatter_add_u64_by_i32.get();
    }
    if (value_type == 4) {
      return scatter_add_i64_by_i32.get();
    }
    if (value_type == 5) {
      return scatter_add_f64_by_i32.get();
    }
    return value_type == 2 ? scatter_add_u32_by_i32.get()
                           : scatter_add_i32_by_i32.get();
  }

  Pipeline *scatter_add_strided_pipeline(int value_type) const {
    return scatter_add_strided[value_type].get();
  }
};

struct VulkanBucketBuilderCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  DeviceAllocation grouped_reduce_params{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> clear_i32;
  std::unique_ptr<Pipeline> count_i32;
  std::unique_ptr<Pipeline> count_private_shared_i32;
  std::unique_ptr<Pipeline> prefix_i32;
  std::unique_ptr<Pipeline> prefix_chunks_i32;
  std::unique_ptr<Pipeline> scatter_i32;
  std::unique_ptr<Pipeline> scatter_f32;
  std::unique_ptr<Pipeline> scatter_u32;
  std::unique_ptr<Pipeline> scatter_raw64;
  std::unique_ptr<Pipeline> scatter_raw_words;
  std::unique_ptr<Pipeline> scatter_private_shared_i32;
  std::unique_ptr<Pipeline> scatter_private_shared_f32;
  std::unique_ptr<Pipeline> scatter_private_shared_u32;
  std::unique_ptr<Pipeline> scatter_private_shared_raw64;
  std::unique_ptr<Pipeline> scatter_private_shared_raw_words;
  std::unique_ptr<Pipeline> grouped_reduce_zero_i32;
  std::unique_ptr<Pipeline> grouped_reduce_zero_f32;
  std::unique_ptr<Pipeline> grouped_reduce_zero_u32;
  std::unique_ptr<Pipeline> grouped_reduce_zero_u64;
  std::unique_ptr<Pipeline> grouped_reduce_zero_i64;
  std::unique_ptr<Pipeline> grouped_reduce_zero_f64;
  std::array<std::unique_ptr<Pipeline>, 6> grouped_reduce_zero_strided;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_i32;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_f32;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_u32;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_u64;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_i64;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_f64;
  std::array<std::unique_ptr<Pipeline>, 6> grouped_reduce_atomic_sum_strided;
  std::unique_ptr<Pipeline> grouped_reduce_sum_i32;
  std::unique_ptr<Pipeline> grouped_reduce_sum_f32;
  std::unique_ptr<Pipeline> grouped_reduce_sum_u32;
  std::unique_ptr<Pipeline> grouped_reduce_sum_u64;
  std::unique_ptr<Pipeline> grouped_reduce_sum_i64;
  std::unique_ptr<Pipeline> grouped_reduce_sum_f64;
  std::unique_ptr<ShaderResourceSet> clear_bindings;
  std::unique_ptr<ShaderResourceSet> count_bindings;
  std::unique_ptr<ShaderResourceSet> count_private_bindings;
  std::unique_ptr<ShaderResourceSet> prefix_bindings;
  std::unique_ptr<ShaderResourceSet> prefix_chunks_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_f32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_u32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_raw64_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_raw_words_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_private_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_private_f32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_private_u32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_private_raw64_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_private_raw_words_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_zero_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_zero_f32_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_zero_u32_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_zero_u64_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_zero_i64_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_zero_f64_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 6>
      grouped_reduce_zero_strided_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_atomic_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_atomic_f32_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_atomic_u32_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_atomic_u64_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_atomic_i64_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_atomic_f64_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 6>
      grouped_reduce_atomic_strided_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_f32_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_u32_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_u64_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_i64_bindings;
  std::unique_ptr<ShaderResourceSet> grouped_reduce_f64_bindings;

  void clear_allocs() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    if (device && grouped_reduce_params != kDeviceNullAllocation) {
      device->dealloc_memory(grouped_reduce_params);
    }
    partial = kDeviceNullAllocation;
    grouped_reduce_params = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanBucketBuilderCache() {
    clear_allocs();
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && clear_i32) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      clear_i32.reset();
      count_i32.reset();
      count_private_shared_i32.reset();
      prefix_i32.reset();
      prefix_chunks_i32.reset();
      scatter_i32.reset();
      scatter_f32.reset();
      scatter_u32.reset();
      scatter_raw64.reset();
      scatter_raw_words.reset();
      scatter_private_shared_i32.reset();
      scatter_private_shared_f32.reset();
      scatter_private_shared_u32.reset();
      scatter_private_shared_raw64.reset();
      scatter_private_shared_raw_words.reset();
      grouped_reduce_zero_i32.reset();
      grouped_reduce_zero_f32.reset();
      grouped_reduce_zero_u32.reset();
      grouped_reduce_zero_u64.reset();
      grouped_reduce_zero_i64.reset();
      grouped_reduce_zero_f64.reset();
      for (auto &pipeline : grouped_reduce_zero_strided) {
        pipeline.reset();
      }
      grouped_reduce_atomic_sum_i32.reset();
      grouped_reduce_atomic_sum_f32.reset();
      grouped_reduce_atomic_sum_u32.reset();
      grouped_reduce_atomic_sum_u64.reset();
      grouped_reduce_atomic_sum_i64.reset();
      grouped_reduce_atomic_sum_f64.reset();
      for (auto &pipeline : grouped_reduce_atomic_sum_strided) {
        pipeline.reset();
      }
      grouped_reduce_sum_i32.reset();
      grouped_reduce_sum_f32.reset();
      grouped_reduce_sum_u32.reset();
      grouped_reduce_sum_u64.reset();
      grouped_reduce_sum_i64.reset();
      grouped_reduce_sum_f64.reset();
      clear_bindings.reset();
      count_bindings.reset();
      count_private_bindings.reset();
      prefix_bindings.reset();
      prefix_chunks_bindings.reset();
      scatter_bindings.reset();
      scatter_f32_bindings.reset();
      scatter_u32_bindings.reset();
      scatter_raw64_bindings.reset();
      scatter_raw_words_bindings.reset();
      scatter_private_bindings.reset();
      scatter_private_f32_bindings.reset();
      scatter_private_u32_bindings.reset();
      scatter_private_raw64_bindings.reset();
      scatter_private_raw_words_bindings.reset();
      grouped_reduce_zero_bindings.reset();
      grouped_reduce_zero_f32_bindings.reset();
      grouped_reduce_zero_u32_bindings.reset();
      grouped_reduce_zero_u64_bindings.reset();
      grouped_reduce_zero_i64_bindings.reset();
      grouped_reduce_zero_f64_bindings.reset();
      for (auto &bindings : grouped_reduce_zero_strided_bindings) {
        bindings.reset();
      }
      grouped_reduce_atomic_bindings.reset();
      grouped_reduce_atomic_f32_bindings.reset();
      grouped_reduce_atomic_u32_bindings.reset();
      grouped_reduce_atomic_u64_bindings.reset();
      grouped_reduce_atomic_i64_bindings.reset();
      grouped_reduce_atomic_f64_bindings.reset();
      for (auto &bindings : grouped_reduce_atomic_strided_bindings) {
        bindings.reset();
      }
      grouped_reduce_bindings.reset();
      grouped_reduce_f32_bindings.reset();
      grouped_reduce_u32_bindings.reset();
      grouped_reduce_u64_bindings.reset();
      grouped_reduce_i64_bindings.reset();
      grouped_reduce_f64_bindings.reset();
    }
    device = dev;
    clear_i32 =
        create_pipeline(dev, kBucketClearI32Spv, "vulkan_bucket_clear_i32");
    count_i32 =
        create_pipeline(dev, kBucketCountI32Spv, "vulkan_bucket_count_i32");
    count_private_shared_i32 = create_pipeline(
        dev, kBucketCountPrivateSharedI32Spv,
        "vulkan_bucket_count_private_shared_i32");
    prefix_i32 =
        create_pipeline(dev, kBucketPrefixI32Spv, "vulkan_bucket_prefix_i32");
    prefix_chunks_i32 = create_pipeline(
        dev, kBucketPrefixChunksI32Spv, "vulkan_bucket_prefix_chunks_i32");
    scatter_i32 =
        create_pipeline(dev, kBucketScatterI32Spv, "vulkan_bucket_scatter_i32");
    scatter_f32 =
        create_pipeline(dev, kBucketScatterF32Spv, "vulkan_bucket_scatter_f32");
    scatter_u32 =
        create_pipeline(dev, kBucketScatterU32Spv, "vulkan_bucket_scatter_u32");
    scatter_raw64 = create_pipeline(dev, kBucketScatterRaw64Spv,
                                    "vulkan_bucket_scatter_raw64");
    scatter_raw_words = create_pipeline(dev, kBucketScatterRawWordsSpv,
                                        "vulkan_bucket_scatter_raw_words");
    scatter_private_shared_i32 = create_pipeline(
        dev, kBucketScatterPrivateSharedI32Spv,
        "vulkan_bucket_scatter_private_shared_i32");
    scatter_private_shared_f32 = create_pipeline(
        dev, kBucketScatterPrivateSharedF32Spv,
        "vulkan_bucket_scatter_private_shared_f32");
    scatter_private_shared_u32 = create_pipeline(
        dev, kBucketScatterPrivateSharedU32Spv,
        "vulkan_bucket_scatter_private_shared_u32");
    scatter_private_shared_raw64 = create_pipeline(
        dev, kBucketScatterPrivateSharedRaw64Spv,
        "vulkan_bucket_scatter_private_shared_raw64");
    scatter_private_shared_raw_words = create_pipeline(
        dev, kBucketScatterPrivateSharedRawWordsSpv,
        "vulkan_bucket_scatter_private_shared_raw_words");
    grouped_reduce_zero_i32 = create_pipeline(
        dev, kGroupedReduceZeroI32Spv, "vulkan_grouped_reduce_zero_i32");
    if (dev->get_caps().get(DeviceCapability::spirv_has_atomic_float_add) !=
        0) {
      grouped_reduce_zero_f32 = create_pipeline(
          dev, kGroupedReduceZeroF32Spv, "vulkan_grouped_reduce_zero_f32");
    }
    grouped_reduce_zero_u32 = create_pipeline(
        dev, kGroupedReduceZeroU32Spv, "vulkan_grouped_reduce_zero_u32");
    grouped_reduce_atomic_sum_i32 =
        create_pipeline(dev, kGroupedReduceAtomicSumI32Spv,
                        "vulkan_grouped_reduce_atomic_sum_i32");
    if (dev->get_caps().get(DeviceCapability::spirv_has_atomic_float_add) !=
        0) {
      grouped_reduce_atomic_sum_f32 =
          create_pipeline(dev, kGroupedReduceAtomicSumF32Spv,
                          "vulkan_grouped_reduce_atomic_sum_f32");
    }
    grouped_reduce_atomic_sum_u32 =
        create_pipeline(dev, kGroupedReduceAtomicSumU32Spv,
                        "vulkan_grouped_reduce_atomic_sum_u32");
    const bool int64_atomic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_int64) != 0 &&
        dev->get_caps().get(DeviceCapability::spirv_has_atomic_int64) != 0;
    if (int64_atomic_supported) {
      grouped_reduce_zero_u64 = create_pipeline(
          dev, kGroupedReduceZeroU64Spv, "vulkan_grouped_reduce_zero_u64");
      grouped_reduce_zero_i64 = create_pipeline(
          dev, kGroupedReduceZeroI64Spv, "vulkan_grouped_reduce_zero_i64");
      grouped_reduce_atomic_sum_u64 =
          create_pipeline(dev, kGroupedReduceAtomicSumU64Spv,
                          "vulkan_grouped_reduce_atomic_sum_u64");
      grouped_reduce_atomic_sum_i64 =
          create_pipeline(dev, kGroupedReduceAtomicSumI64Spv,
                          "vulkan_grouped_reduce_atomic_sum_i64");
    }
    const bool float64_atomic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_float64) != 0 &&
        dev->get_caps().get(DeviceCapability::spirv_has_atomic_float64_add) !=
            0;
    if (float64_atomic_supported) {
      grouped_reduce_zero_f64 = create_pipeline(
          dev, kGroupedReduceZeroF64Spv, "vulkan_grouped_reduce_zero_f64");
      grouped_reduce_atomic_sum_f64 =
          create_pipeline(dev, kGroupedReduceAtomicSumF64Spv,
                          "vulkan_grouped_reduce_atomic_sum_f64");
    }
    grouped_reduce_sum_i32 = create_pipeline(
        dev, kGroupedReduceSumI32Spv, "vulkan_grouped_reduce_sum_i32");
    grouped_reduce_sum_f32 = create_pipeline(
        dev, kGroupedReduceSumF32Spv, "vulkan_grouped_reduce_sum_f32");
    grouped_reduce_sum_u32 = create_pipeline(
        dev, kGroupedReduceSumU32Spv, "vulkan_grouped_reduce_sum_u32");
    const bool int64_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_int64) != 0;
    const bool float64_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_float64) != 0;
    if (int64_supported) {
      grouped_reduce_sum_u64 = create_pipeline(
          dev, kGroupedReduceSumU64Spv, "vulkan_grouped_reduce_sum_u64");
      grouped_reduce_sum_i64 = create_pipeline(
          dev, kGroupedReduceSumI64Spv, "vulkan_grouped_reduce_sum_i64");
    }
    if (float64_supported) {
      grouped_reduce_sum_f64 = create_pipeline(
          dev, kGroupedReduceSumF64Spv, "vulkan_grouped_reduce_sum_f64");
    }
  }

  Pipeline *grouped_reduce_zero_pipeline(int value_type) const {
    if (value_type == 1) {
      return grouped_reduce_zero_f32.get();
    }
    if (value_type == 3) {
      return grouped_reduce_zero_u64.get();
    }
    if (value_type == 4) {
      return grouped_reduce_zero_i64.get();
    }
    if (value_type == 5) {
      return grouped_reduce_zero_f64.get();
    }
    return value_type == 2 ? grouped_reduce_zero_u32.get()
                           : grouped_reduce_zero_i32.get();
  }

  void ensure_grouped_reduce_zero_strided_pipeline(int value_type) {
    if (grouped_reduce_zero_strided[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroF32StridedSpv,
            "vulkan_grouped_reduce_zero_f32_strided");
        return;
      case 2:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroU32StridedSpv,
            "vulkan_grouped_reduce_zero_u32_strided");
        return;
      case 3:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroU64StridedSpv,
            "vulkan_grouped_reduce_zero_u64_strided");
        return;
      case 4:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroI64StridedSpv,
            "vulkan_grouped_reduce_zero_i64_strided");
        return;
      case 5:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroF64StridedSpv,
            "vulkan_grouped_reduce_zero_f64_strided");
        return;
      default:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroI32StridedSpv,
            "vulkan_grouped_reduce_zero_i32_strided");
        return;
    }
  }

  Pipeline *grouped_reduce_zero_strided_pipeline(int value_type) const {
    return grouped_reduce_zero_strided[value_type].get();
  }

  Pipeline *grouped_reduce_atomic_pipeline(int value_type) const {
    if (value_type == 1) {
      return grouped_reduce_atomic_sum_f32.get();
    }
    if (value_type == 3) {
      return grouped_reduce_atomic_sum_u64.get();
    }
    if (value_type == 4) {
      return grouped_reduce_atomic_sum_i64.get();
    }
    if (value_type == 5) {
      return grouped_reduce_atomic_sum_f64.get();
    }
    return value_type == 2 ? grouped_reduce_atomic_sum_u32.get()
                           : grouped_reduce_atomic_sum_i32.get();
  }

  void ensure_grouped_reduce_atomic_strided_pipeline(int value_type) {
    if (grouped_reduce_atomic_sum_strided[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumF32StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_f32_strided");
        return;
      case 2:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumU32StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_u32_strided");
        return;
      case 3:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumU64StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_u64_strided");
        return;
      case 4:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumI64StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_i64_strided");
        return;
      case 5:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumF64StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_f64_strided");
        return;
      default:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumI32StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_i32_strided");
        return;
    }
  }

  Pipeline *grouped_reduce_atomic_strided_pipeline(int value_type) const {
    return grouped_reduce_atomic_sum_strided[value_type].get();
  }

  Pipeline *bucket_scatter_pipeline(int value_type) const {
    if (value_type == 7) {
      return scatter_raw_words.get();
    }
    if (value_type == 1) {
      return scatter_f32.get();
    }
    if (value_type == 2) {
      return scatter_u32.get();
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return scatter_raw64.get();
    }
    return scatter_i32.get();
  }

  Pipeline *bucket_scatter_private_pipeline(int value_type) const {
    if (value_type == 7) {
      return scatter_private_shared_raw_words.get();
    }
    if (value_type == 1) {
      return scatter_private_shared_f32.get();
    }
    if (value_type == 2) {
      return scatter_private_shared_u32.get();
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return scatter_private_shared_raw64.get();
    }
    return scatter_private_shared_i32.get();
  }

  ShaderResourceSet *bucket_scatter_resource_set(int value_type) {
    if (value_type == 7) {
      return resource_set(scatter_raw_words_bindings);
    }
    if (value_type == 1) {
      return resource_set(scatter_f32_bindings);
    }
    if (value_type == 2) {
      return resource_set(scatter_u32_bindings);
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return resource_set(scatter_raw64_bindings);
    }
    return resource_set(scatter_bindings);
  }

  ShaderResourceSet *bucket_scatter_private_resource_set(int value_type) {
    if (value_type == 7) {
      return resource_set(scatter_private_raw_words_bindings);
    }
    if (value_type == 1) {
      return resource_set(scatter_private_f32_bindings);
    }
    if (value_type == 2) {
      return resource_set(scatter_private_u32_bindings);
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return resource_set(scatter_private_raw64_bindings);
    }
    return resource_set(scatter_private_bindings);
  }

  const char *bucket_scatter_scope(int value_type) const {
    if (value_type == 7) {
      return "vulkan_bucket_scatter_raw_words";
    }
    if (value_type == 1) {
      return "vulkan_bucket_scatter_f32";
    }
    if (value_type == 2) {
      return "vulkan_bucket_scatter_u32";
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return "vulkan_bucket_scatter_raw64";
    }
    return "vulkan_bucket_scatter_i32";
  }

  const char *bucket_scatter_private_scope(int value_type) const {
    if (value_type == 7) {
      return "vulkan_bucket_scatter_private_shared_raw_words";
    }
    if (value_type == 1) {
      return "vulkan_bucket_scatter_private_shared_f32";
    }
    if (value_type == 2) {
      return "vulkan_bucket_scatter_private_shared_u32";
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return "vulkan_bucket_scatter_private_shared_raw64";
    }
    return "vulkan_bucket_scatter_private_shared_i32";
  }

  Pipeline *grouped_reduce_sum_pipeline(int value_type) const {
    switch (value_type) {
      case 1:
        return grouped_reduce_sum_f32.get();
      case 2:
        return grouped_reduce_sum_u32.get();
      case 3:
        return grouped_reduce_sum_u64.get();
      case 4:
        return grouped_reduce_sum_i64.get();
      case 5:
        return grouped_reduce_sum_f64.get();
      default:
        return grouped_reduce_sum_i32.get();
    }
  }

  ShaderResourceSet *grouped_reduce_zero_resource_set(int value_type) {
    if (value_type == 1) {
      return resource_set(grouped_reduce_zero_f32_bindings);
    }
    if (value_type == 3) {
      return resource_set(grouped_reduce_zero_u64_bindings);
    }
    if (value_type == 4) {
      return resource_set(grouped_reduce_zero_i64_bindings);
    }
    if (value_type == 5) {
      return resource_set(grouped_reduce_zero_f64_bindings);
    }
    return resource_set(value_type == 2 ? grouped_reduce_zero_u32_bindings
                                        : grouped_reduce_zero_bindings);
  }

  ShaderResourceSet *grouped_reduce_zero_strided_resource_set(int value_type) {
    auto &bindings = grouped_reduce_zero_strided_bindings[value_type];
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *grouped_reduce_atomic_resource_set(int value_type) {
    if (value_type == 1) {
      return resource_set(grouped_reduce_atomic_f32_bindings);
    }
    if (value_type == 3) {
      return resource_set(grouped_reduce_atomic_u64_bindings);
    }
    if (value_type == 4) {
      return resource_set(grouped_reduce_atomic_i64_bindings);
    }
    if (value_type == 5) {
      return resource_set(grouped_reduce_atomic_f64_bindings);
    }
    return resource_set(value_type == 2 ? grouped_reduce_atomic_u32_bindings
                                        : grouped_reduce_atomic_bindings);
  }

  ShaderResourceSet *grouped_reduce_atomic_strided_resource_set(
      int value_type) {
    auto &bindings = grouped_reduce_atomic_strided_bindings[value_type];
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *grouped_reduce_sum_resource_set(int value_type) {
    switch (value_type) {
      case 1:
        return resource_set(grouped_reduce_f32_bindings);
      case 2:
        return resource_set(grouped_reduce_u32_bindings);
      case 3:
        return resource_set(grouped_reduce_u64_bindings);
      case 4:
        return resource_set(grouped_reduce_i64_bindings);
      case 5:
        return resource_set(grouped_reduce_f64_bindings);
      default:
        return resource_set(grouped_reduce_bindings);
    }
  }

  const char *grouped_reduce_sum_scope(int value_type) const {
    switch (value_type) {
      case 1:
        return "vulkan_grouped_reduce_sum_f32";
      case 2:
        return "vulkan_grouped_reduce_sum_u32";
      case 3:
        return "vulkan_grouped_reduce_sum_u64";
      case 4:
        return "vulkan_grouped_reduce_sum_i64";
      case 5:
        return "vulkan_grouped_reduce_sum_f64";
      default:
        return "vulkan_grouped_reduce_sum_i32";
    }
  }

  ShaderResourceSet *resource_set(std::unique_ptr<ShaderResourceSet> &bindings) {
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan bucket builder workspace: "
                "RhiResult({})",
                res);
    return alloc;
  }

  void ensure_grouped_reduce_params() {
    if (grouped_reduce_params == kDeviceNullAllocation) {
      grouped_reduce_params = alloc_storage(8 * sizeof(uint32_t));
    }
    cached_bytes = partial_capacity + 8 * sizeof(uint32_t);
  }

  bool needs_workspace_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  void ensure_workspace(size_t bytes) {
    if (bytes == 0 || !needs_workspace_realloc(bytes)) {
      return;
    }
    clear_allocs();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes =
        bytes + (grouped_reduce_params != kDeviceNullAllocation
                     ? 8 * sizeof(uint32_t)
                     : 0);
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
std::mutex g_vulkan_bucket_builder_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanBucketBuilderCache>>
    g_vulkan_bucket_builder_caches;

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

VulkanReduceCache &get_reduce_cache(void *owner,
                                    Device *device,
                                    int value_type) {
  std::lock_guard<std::mutex> guard(g_vulkan_reduce_mutex);
  auto &cache = g_vulkan_reduce_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanReduceCache>();
  }
  cache->ensure_pipeline_set(device, value_type);
  return *cache;
}

VulkanReduceCache &get_reduce_cache(void *owner,
                                    Device *device,
                                    int value_type,
                                    bool strided_source) {
  std::lock_guard<std::mutex> guard(g_vulkan_reduce_mutex);
  auto &cache = g_vulkan_reduce_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanReduceCache>();
  }
  if (strided_source) {
    cache->ensure_strided_pipeline_set(device, value_type);
  } else {
    cache->ensure_pipeline_set(device, value_type);
  }
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

VulkanBucketBuilderCache &get_bucket_builder_cache(void *owner,
                                                   Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_bucket_builder_mutex);
  auto &cache = g_vulkan_bucket_builder_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanBucketBuilderCache>();
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

size_t scan_workspace_bytes(const std::vector<size_t> &levels,
                            size_t item_size) {
  size_t items = 0;
  for (size_t i = 1; i < levels.size(); ++i) {
    items += levels[i];
  }
  return items * item_size;
}

size_t vulkan_scan_value_type_size(int value_type) {
  switch (value_type) {
    case 0:
    case 1:
    case 2:
      return sizeof(uint32_t);
    case 3:
    case 4:
    case 5:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}

size_t vulkan_sort_key_type_size(int key_type) {
  switch (key_type) {
    case 0:
    case 1:
    case 2:
      return sizeof(uint32_t);
    case 3:
    case 4:
    case 5:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}

DevicePtr scan_level_ptr(DeviceAllocation data_alloc,
                         DeviceAllocation workspace,
                         const std::vector<size_t> &workspace_offsets,
                         size_t level,
                         size_t item_size) {
  if (level == 0) {
    return data_alloc.get_ptr(0);
  }
  return workspace.get_ptr(workspace_offsets[level - 1] * item_size);
}

struct VulkanScanDispatchPlan {
  DeviceAllocation data_alloc{kDeviceNullAllocation};
  size_t n{0};
  int value_type{0};
  size_t item_size{sizeof(int32_t)};
  size_t workspace_bytes{0};
  bool use_small_subgroup{false};
  size_t data_bytes{0};
  DeviceAllocation workspace_alloc{kDeviceNullAllocation};
  DeviceAllocation dummy_sums_alloc{kDeviceNullAllocation};
  DeviceAllocation params_alloc{kDeviceNullAllocation};
  std::vector<size_t> levels;
  std::vector<size_t> workspace_offsets;
  Pipeline *scan_small{nullptr};
  Pipeline *scan_block{nullptr};
  Pipeline *scan_add{nullptr};
  Pipeline *scan_block_strided{nullptr};
  Pipeline *scan_add_strided{nullptr};
  const char *scan_small_scope{nullptr};
  const char *scan_block_scope{nullptr};
  const char *scan_add_scope{nullptr};
  const char *scan_block_strided_scope{nullptr};
  const char *scan_add_strided_scope{nullptr};
  bool member_source{false};
  size_t offset{0};
  size_t stride{0};
  size_t params_bytes{0};
};

VulkanScanDispatchPlan prepare_vulkan_scan(Program *program,
                                           VulkanScanCache &cache,
                                           DeviceAllocation data_alloc,
                                           size_t n,
                                           int value_type,
                                           bool member_source = false,
                                           size_t offset = 0,
                                           size_t stride = 0) {
  VulkanScanDispatchPlan plan;
  plan.data_alloc = data_alloc;
  plan.n = n;
  plan.value_type = value_type;
  plan.item_size = vulkan_scan_value_type_size(value_type);
  plan.member_source = member_source;
  plan.offset = offset;
  plan.stride = stride;
  TI_ERROR_IF(plan.item_size == 0,
              "Vulkan native scan received an unsupported value type.");
  if (n <= 1) {
    return plan;
  }
  if (member_source) {
    cache.ensure_strided_pipelines(cache.device, value_type);
    cache.ensure_params();
    plan.params_alloc = cache.params;
    plan.params_bytes = 3 * sizeof(uint32_t);
  }

  const int small_subgroup_threshold =
      get_environ_config("TI_VULKAN_SCAN_SMALL_SUBGROUP_MAX_N", 4096);
  const bool use_32bit_value = plan.item_size == sizeof(uint32_t);
  plan.use_small_subgroup = use_32bit_value && cache.subgroup_scan_enabled &&
                            (member_source
                                 ? cache.scan_small_strided_pipeline(value_type)
                                 : cache.scan_small_pipeline(value_type)) &&
                            small_subgroup_threshold > 0 &&
                            n <= static_cast<size_t>(small_subgroup_threshold);
  if (plan.use_small_subgroup) {
    plan.scan_small =
        member_source ? cache.scan_small_strided_pipeline(value_type)
                      : cache.scan_small_pipeline(value_type);
    plan.scan_small_scope =
        member_source ? "vulkan_scan_small_subgroup_strided"
                      : cache.scan_small_scope(value_type);
    plan.data_bytes = member_source ? n * stride : n * plan.item_size;
    TI_ERROR_IF(!plan.scan_small,
                "Vulkan native scan could not find a small-scan pipeline.");
    return plan;
  }

  plan.levels = scan_level_lengths(n);
  plan.workspace_bytes = scan_workspace_bytes(plan.levels, plan.item_size);
  if (cache.has_workspace_allocs() &&
      cache.needs_workspace_realloc(plan.workspace_bytes)) {
    program->synchronize();
  }
  cache.ensure_workspace(plan.workspace_bytes);

  plan.workspace_offsets.reserve(plan.levels.size() > 0 ? plan.levels.size() - 1
                                                        : 0);
  size_t workspace_offset = 0;
  for (size_t i = 1; i < plan.levels.size(); ++i) {
    plan.workspace_offsets.push_back(workspace_offset);
    workspace_offset += plan.levels[i];
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
                                  !member_source &&
                                  cache.scan_block_pipeline(value_type, true) &&
                                  n >= subgroup_block_min_n;
  plan.scan_block = cache.scan_block_pipeline(value_type, use_subgroup_block);
  plan.scan_block_scope =
      cache.scan_block_scope(value_type, use_subgroup_block);
  plan.scan_add = cache.scan_add_pipeline(value_type);
  plan.scan_add_scope = cache.scan_add_scope(value_type);
  if (member_source) {
    plan.scan_block_strided = cache.scan_block_strided_pipeline(value_type);
    plan.scan_add_strided = cache.scan_add_strided_pipeline(value_type);
    plan.scan_block_strided_scope = "vulkan_scan_block_strided";
    plan.scan_add_strided_scope = "vulkan_scan_add_strided";
  }
  TI_ERROR_IF(!plan.scan_block || !plan.scan_add ||
                  (member_source &&
                   (!plan.scan_block_strided || !plan.scan_add_strided)),
              "Vulkan native scan could not find a scan pipeline.");
  return plan;
}

VulkanScanDispatchPlan prepare_vulkan_i32_scan(Program *program,
                                               VulkanScanCache &cache,
                                               DeviceAllocation data_alloc,
                                               size_t n) {
  return prepare_vulkan_scan(program, cache, data_alloc, n, 0);
}

void record_vulkan_scan(Device *op_device,
                        CommandList *cmdlist,
                        const VulkanScanDispatchPlan &plan,
                        bool profiler_scopes) {
  if (plan.n <= 1) {
    return;
  }
  auto bind_params = [&plan, cmdlist](ShaderResourceSet *bindings) {
    if (!plan.member_source) {
      return;
    }
    const std::array<uint32_t, 3> param_words{
        static_cast<uint32_t>(plan.n),
        static_cast<uint32_t>(plan.offset / sizeof(uint32_t)),
        static_cast<uint32_t>(plan.stride / sizeof(uint32_t)),
    };
    for (uint32_t i = 0; i < param_words.size(); ++i) {
      cmdlist->buffer_fill(plan.params_alloc.get_ptr(i * sizeof(uint32_t)),
                           sizeof(uint32_t), param_words[i]);
    }
    cmdlist->buffer_barrier(plan.params_alloc);
    bindings->rw_buffer(plan.use_small_subgroup ? 1 : 2,
                        plan.params_alloc.get_ptr(0), plan.params_bytes);
  };
  if (plan.use_small_subgroup) {
    auto bindings = op_device->create_resource_set_unique();
    bindings->rw_buffer(0, plan.data_alloc.get_ptr(0), plan.data_bytes);
    bind_params(bindings.get());
    dispatch_pipeline(cmdlist, plan.scan_small, bindings.get(), 1, 1, 1,
                      profiler_scopes ? plan.scan_small_scope : nullptr);
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
                       plan.workspace_offsets, level, plan.item_size);
    const bool strided_level = plan.member_source && level == 0;
    const size_t level_bytes =
        strided_level ? plan.n * plan.stride
                      : plan.levels[level] * plan.item_size;
    DevicePtr sums_ptr = plan.dummy_sums_alloc.get_ptr(0);
    size_t sums_bytes = plan.item_size;
    if (level + 1 < plan.levels.size()) {
      sums_ptr = scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                                plan.workspace_offsets, level + 1,
                                plan.item_size);
      sums_bytes = plan.levels[level + 1] * plan.item_size;
    }
    auto bindings = op_device->create_resource_set_unique();
    bindings->rw_buffer(0, level_ptr, level_bytes);
    bindings->rw_buffer(1, sums_ptr, sums_bytes);
    if (strided_level) {
      bind_params(bindings.get());
    }
    const uint32_t groups = static_cast<uint32_t>(
        (plan.levels[level] + kBlockSize - 1) / kBlockSize);
    Pipeline *pipeline =
        strided_level ? plan.scan_block_strided : plan.scan_block;
    const char *scope =
        strided_level ? plan.scan_block_strided_scope : plan.scan_block_scope;
    dispatch_pipeline(cmdlist, pipeline, bindings.get(), groups, 1, 1,
                      scope_name(scope));
    barrier_level(cmdlist, level);
    if (level + 1 < plan.levels.size()) {
      cmdlist->buffer_barrier(plan.workspace_alloc);
    }
  }
  if (plan.levels.size() > 1) {
    for (size_t level = plan.levels.size() - 1; level-- > 0;) {
      DevicePtr level_ptr =
          scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                         plan.workspace_offsets, level, plan.item_size);
      DevicePtr offsets_ptr =
          scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                         plan.workspace_offsets, level + 1, plan.item_size);
      const bool strided_level = plan.member_source && level == 0;
      const size_t level_bytes =
          strided_level ? plan.n * plan.stride
                        : plan.levels[level] * plan.item_size;
      const size_t offsets_bytes = plan.levels[level + 1] * plan.item_size;
      auto bindings = op_device->create_resource_set_unique();
      bindings->rw_buffer(0, level_ptr, level_bytes);
      bindings->rw_buffer(1, offsets_ptr, offsets_bytes);
      if (strided_level) {
        bind_params(bindings.get());
      }
      const uint32_t groups = static_cast<uint32_t>(
          (plan.levels[level] + kBlockSize - 1) / kBlockSize);
      Pipeline *pipeline =
          strided_level ? plan.scan_add_strided : plan.scan_add;
      const char *scope =
          strided_level ? plan.scan_add_strided_scope : plan.scan_add_scope;
      dispatch_pipeline(cmdlist, pipeline, bindings.get(), groups, 1, 1,
                        scope_name(scope));
      barrier_level(cmdlist, level);
    }
  }
}

void record_vulkan_i32_scan(Device *op_device,
                            CommandList *cmdlist,
                            const VulkanScanDispatchPlan &plan,
                            bool profiler_scopes) {
  record_vulkan_scan(op_device, cmdlist, plan, profiler_scopes);
}

size_t enqueue_vulkan_scan(Program *program,
                           VulkanScanCache &cache,
                           DeviceAllocation data_alloc,
                           size_t n,
                           int value_type,
                           bool profiler_scopes,
                           bool member_source = false,
                           size_t offset = 0,
                           size_t stride = 0) {
  auto plan = prepare_vulkan_scan(program, cache, data_alloc, n, value_type,
                                  member_source, offset, stride);
  if (plan.n <= 1) {
    return 0;
  }
  program->enqueue_compute_op_lambda(
      [plan, profiler_scopes](Device *op_device, CommandList *cmdlist) {
        record_vulkan_scan(op_device, cmdlist, plan, profiler_scopes);
      },
      {});
  return plan.workspace_bytes;
}

size_t enqueue_vulkan_i32_scan(Program *program,
                               VulkanScanCache &cache,
                               DeviceAllocation data_alloc,
                               size_t n,
                               bool profiler_scopes) {
  return enqueue_vulkan_scan(program, cache, data_alloc, n, 0,
                             profiler_scopes);
}

}  // namespace

bool Program::vulkan_radix_sort_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scan_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scan_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

bool Program::vulkan_compact_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_histogram_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_histogram_value_type_available(int value_type,
                                                    int bin_type) const {
  if (compile_config().arch != Arch::vulkan ||
      !(value_type == 0 || value_type == 2)) {
    return false;
  }
  if (bin_type == 0) {
    return true;
  }
  if (bin_type == 4) {
    auto &caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_int64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_int64) != 0;
  }
  return false;
}

bool Program::vulkan_reduce_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_reduce_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

bool Program::vulkan_transform_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_transform_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 4) {
    return true;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

namespace {

std::size_t vulkan_transform_value_size(int value_type) {
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "Vulkan native transform received an unsupported value type.");
  return (value_type == 3 || value_type == 4 || value_type == 5)
             ? sizeof(uint64_t)
             : sizeof(uint32_t);
}

void check_vulkan_transform_member_request(Ndarray *src,
                                           Ndarray *dst,
                                           int value_type,
                                           std::size_t offset,
                                           std::size_t stride) {
  TI_ERROR_IF(!src || !dst,
              "Vulkan native strided transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native strided transform source and destination sizes "
              "differ.");
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(dst->get_element_size() != value_size,
              "Vulkan native strided transform destination dtype does not "
              "match value type.");
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided transform source stride is smaller than "
              "value size.");
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided transform source offset/stride must "
              "align to value size.");
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * src->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "Vulkan native strided transform source buffer is smaller than "
              "value size.");
  TI_ERROR_IF(offset > src_bytes - value_size,
              "Vulkan native strided transform source offset is out of "
              "bounds.");
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "Vulkan native strided transform source range is out of bounds.");
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 || stride % sizeof(uint32_t) != 0,
              "Vulkan native strided transform source offset/stride must be "
              "uint32-word aligned.");
}

void check_vulkan_transform_strided_range(const char *role,
                                          Ndarray *arr,
                                          size_t logical_items,
                                          size_t value_size,
                                          size_t offset,
                                          size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided transform {} stride is smaller than "
              "value size.",
              role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided transform {} offset/stride must align "
              "to value size.",
              role);
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "Vulkan native strided transform {} offset/stride must be "
              "uint32-word aligned.",
              role);
  if (logical_items == 0) {
    return;
  }
  const size_t buffer_bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(buffer_bytes < value_size,
              "Vulkan native strided transform {} buffer is smaller than "
              "value size.",
              role);
  TI_ERROR_IF(offset > buffer_bytes - value_size,
              "Vulkan native strided transform {} offset is out of bounds.",
              role);
  const size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > buffer_bytes,
              "Vulkan native strided transform {} range is out of bounds.",
              role);
}

void check_vulkan_transform_strided_request(Ndarray *src,
                                            Ndarray *dst,
                                            int value_type,
                                            size_t src_offset,
                                            size_t src_stride,
                                            size_t dst_offset,
                                            size_t dst_stride) {
  TI_ERROR_IF(!src || !dst,
              "Vulkan native strided transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native strided transform source and destination sizes "
              "differ.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  check_vulkan_transform_strided_range("source", src, src->get_nelement(),
                                       value_size, src_offset, src_stride);
  check_vulkan_transform_strided_range("destination", dst, dst->get_nelement(),
                                       value_size, dst_offset, dst_stride);
}

void check_vulkan_transform_packed_strided_range(const char *role,
                                                 Ndarray *arr,
                                                 size_t logical_items,
                                                 size_t value_size,
                                                 int lane_count,
                                                 size_t offset,
                                                 size_t stride) {
  TI_ERROR_IF(lane_count <= 0,
              "Vulkan native packed strided transform lane count must be "
              "positive.");
  const size_t payload_bytes = static_cast<size_t>(lane_count) * value_size;
  TI_ERROR_IF(stride < payload_bytes,
              "Vulkan native packed strided transform {} stride is smaller "
              "than payload.",
              role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native packed strided transform {} offset/stride must "
              "align to value size.",
              role);
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "Vulkan native packed strided transform {} offset/stride must "
              "be uint32-word aligned.",
              role);
  if (logical_items == 0) {
    return;
  }
  const size_t buffer_bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(buffer_bytes < payload_bytes,
              "Vulkan native packed strided transform {} buffer is smaller "
              "than payload.",
              role);
  TI_ERROR_IF(offset > buffer_bytes - payload_bytes,
              "Vulkan native packed strided transform {} offset is out of "
              "bounds.",
              role);
  const size_t last =
      offset + (logical_items - 1) * stride + payload_bytes;
  TI_ERROR_IF(last > buffer_bytes,
              "Vulkan native packed strided transform {} range is out of "
              "bounds.",
              role);
}

void check_vulkan_transform_packed_strided_request(Ndarray *src,
                                                   Ndarray *dst,
                                                   int value_type,
                                                   int lane_count,
                                                   size_t src_offset,
                                                   size_t src_stride,
                                                   size_t dst_offset,
                                                   size_t dst_stride) {
  TI_ERROR_IF(!src || !dst,
              "Vulkan native packed strided transform received a null "
              "ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native packed strided transform source and destination "
              "sizes differ.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  check_vulkan_transform_packed_strided_range(
      "source", src, src->get_nelement(), value_size, lane_count, src_offset,
      src_stride);
  check_vulkan_transform_packed_strided_range(
      "destination", dst, dst->get_nelement(), value_size, lane_count,
      dst_offset, dst_stride);
}

void check_vulkan_indexed_copy_strided_request(Ndarray *src,
                                               Ndarray *indices,
                                               Ndarray *dst,
                                               std::size_t item_bytes,
                                               std::size_t src_offset,
                                               std::size_t src_stride,
                                               std::size_t dst_offset,
                                               std::size_t dst_stride,
                                               bool scatter) {
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native strided indexed-copy received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native strided indexed-copy expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "Vulkan native strided indexed-copy expects i32 indices.");
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native strided indexed-copy item size must be a "
              "positive uint32-word multiple.");
  if (scatter) {
    TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
                "Vulkan native strided scatter expects source and indices "
                "sizes to match.");
  } else {
    TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
                "Vulkan native strided gather expects indices and destination "
                "sizes to match.");
  }
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < item_bytes,
                "Vulkan native strided indexed-copy {} stride is smaller "
                "than item size.",
                role);
    TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                    stride % sizeof(uint32_t) != 0,
                "Vulkan native strided indexed-copy {} offset/stride must "
                "be uint32-word aligned.",
                role);
    if (logical_items == 0) {
      return;
    }
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < item_bytes,
                "Vulkan native strided indexed-copy {} buffer is smaller "
                "than item size.",
                role);
    TI_ERROR_IF(offset > bytes - item_bytes,
                "Vulkan native strided indexed-copy {} offset is out of "
                "bounds.",
                role);
    const std::size_t last = offset + (logical_items - 1) * stride + item_bytes;
    TI_ERROR_IF(last > bytes,
                "Vulkan native strided indexed-copy {} range is out of "
                "bounds.",
                role);
  };
  check_range("source", src, src->get_nelement(), src_offset, src_stride);
  check_range("destination", dst, dst->get_nelement(), dst_offset,
              dst_stride);
}

void check_vulkan_reduce_member_request(Ndarray *values,
                                        Ndarray *output,
                                        int value_type,
                                        std::size_t offset,
                                        std::size_t stride,
                                        int op) {
  TI_ERROR_IF(!values || !output,
              "Vulkan native strided reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native strided reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native strided reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native strided reduce output must contain at least one "
              "item.");
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(output->get_element_size() != value_size,
              "Vulkan native strided reduce output dtype does not match value "
              "type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "Vulkan native strided reduce supports only sum/min/max "
              "operations.");
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided reduce source stride is smaller than "
              "value size.");
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided reduce source offset/stride must align "
              "to value size.");
  const std::size_t n = values->get_nelement();
  const std::size_t src_bytes = n * values->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "Vulkan native strided reduce source buffer is smaller than "
              "value size.");
  TI_ERROR_IF(offset > src_bytes - value_size,
              "Vulkan native strided reduce source offset is out of bounds.");
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "Vulkan native strided reduce source range is out of bounds.");
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 || stride % sizeof(uint32_t) != 0,
              "Vulkan native strided reduce source offset/stride must be "
              "uint32-word aligned.");
}

void check_vulkan_reduce_strided_request(Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         std::size_t values_offset,
                                         std::size_t values_stride,
                                         std::size_t output_offset,
                                         std::size_t output_stride,
                                         int op) {
  TI_ERROR_IF(!values || !output,
              "Vulkan native strided reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native strided reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native strided reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native strided reduce output must contain at least one "
              "item.");
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(op < 0 || op > 2,
              "Vulkan native strided reduce supports only sum/min/max "
              "operations.");
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < value_size,
                "Vulkan native strided reduce {} stride is smaller than "
                "value size.",
                role);
    TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
                "Vulkan native strided reduce {} offset/stride must align "
                "to value size.",
                role);
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < value_size,
                "Vulkan native strided reduce {} buffer is smaller than "
                "value size.",
                role);
    TI_ERROR_IF(offset > bytes - value_size,
                "Vulkan native strided reduce {} offset is out of bounds.",
                role);
    const std::size_t last =
        offset + (logical_items - 1) * stride + value_size;
    TI_ERROR_IF(last > bytes,
                "Vulkan native strided reduce {} range is out of bounds.",
                role);
    TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                    stride % sizeof(uint32_t) != 0,
                "Vulkan native strided reduce {} offset/stride must be "
                "uint32-word aligned.",
                role);
  };
  check_range("source", values, values->get_nelement(), values_offset,
              values_stride);
  check_range("destination", output, 1, output_offset, output_stride);
}

void check_vulkan_scan_member_request(Ndarray *data,
                                      int value_type,
                                      std::size_t offset,
                                      std::size_t stride) {
  TI_ERROR_IF(!data, "Vulkan native strided scan received null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "Vulkan native strided scan expects a 1D ndarray.");
  const std::size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native strided scan received an unsupported value type.");
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided scan source stride is smaller than value "
              "size.");
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided scan source offset/stride must align to "
              "value size.");
  const std::size_t n = data->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * data->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "Vulkan native strided scan source buffer is smaller than "
              "value size.");
  TI_ERROR_IF(offset > src_bytes - value_size,
              "Vulkan native strided scan source offset is out of bounds.");
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "Vulkan native strided scan source range is out of bounds.");
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 || stride % sizeof(uint32_t) != 0,
              "Vulkan native strided scan source offset/stride must be "
              "uint32-word aligned.");
}

void check_vulkan_scatter_add_member_request(Ndarray *src,
                                             Ndarray *indices,
                                             Ndarray *dst,
                                             int value_type,
                                             std::size_t offset,
                                             std::size_t stride) {
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native strided scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native strided scatter-add expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "Vulkan native strided scatter-add source and indices sizes "
              "differ.");
  const std::size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native strided scatter-add received an unsupported "
              "value type.");
  TI_ERROR_IF(dst->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native strided scatter-add destination dtype or i32 "
              "index size mismatch.");
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided scatter-add source stride is smaller "
              "than value size.");
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided scatter-add source offset/stride must "
              "align to value size.");
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * src->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "Vulkan native strided scatter-add source buffer is smaller "
              "than value size.");
  TI_ERROR_IF(offset > src_bytes - value_size,
              "Vulkan native strided scatter-add source offset is out of "
              "bounds.");
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "Vulkan native strided scatter-add source range is out of "
              "bounds.");
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 || stride % sizeof(uint32_t) != 0,
              "Vulkan native strided scatter-add source offset/stride must be "
              "uint32-word aligned.");
}

void check_vulkan_grouped_reduce_member_request(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                std::size_t offset,
                                                std::size_t stride,
                                                int op) {
  TI_ERROR_IF(!keys || !values || !output,
              "Vulkan native strided grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "Vulkan native strided grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native strided grouped reduce keys and values sizes "
              "differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "Vulkan native strided grouped reduce output must contain at "
              "least one group.");
  const std::size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native strided grouped reduce received an unsupported "
              "value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != value_size,
              "Vulkan native strided grouped reduce output dtype or i32 key "
              "size mismatch.");
  TI_ERROR_IF(op != 0,
              "Vulkan native strided grouped reduce currently supports only "
              "sum.");
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided grouped reduce source stride is smaller "
              "than value size.");
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided grouped reduce source offset/stride must "
              "align to value size.");
  const std::size_t n = values->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t values_bytes = n * values->get_element_size();
  TI_ERROR_IF(values_bytes < value_size,
              "Vulkan native strided grouped reduce source buffer is smaller "
              "than value size.");
  TI_ERROR_IF(offset > values_bytes - value_size,
              "Vulkan native strided grouped reduce source offset is out of "
              "bounds.");
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > values_bytes,
              "Vulkan native strided grouped reduce source range is out of "
              "bounds.");
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 || stride % sizeof(uint32_t) != 0,
              "Vulkan native strided grouped reduce source offset/stride must "
              "be uint32-word aligned.");
}

void check_vulkan_strided_range(const char *op_name,
                                const char *arg_name,
                                Ndarray *arr,
                                size_t logical_items,
                                size_t value_size,
                                size_t offset,
                                size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "{} {} stride is smaller than value size.", op_name, arg_name);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} {} offset/stride must align to value size.", op_name,
              arg_name);
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "{} {} offset/stride must be uint32-word aligned.", op_name,
              arg_name);
  if (logical_items == 0) {
    return;
  }
  const size_t buffer_bytes = logical_items * arr->get_element_size();
  TI_ERROR_IF(buffer_bytes < value_size,
              "{} {} buffer is smaller than value size.", op_name, arg_name);
  TI_ERROR_IF(offset > buffer_bytes - value_size,
              "{} {} offset is out of bounds.", op_name, arg_name);
  const size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > buffer_bytes, "{} {} range is out of bounds.", op_name,
              arg_name);
}

void check_vulkan_scatter_add_strided_request(Ndarray *src,
                                              Ndarray *indices,
                                              Ndarray *dst,
                                              int value_type,
                                              size_t src_offset,
                                              size_t src_stride,
                                              size_t dst_offset,
                                              size_t dst_stride) {
  const char *op_name = "Vulkan native strided scatter-add";
  TI_ERROR_IF(!src || !indices || !dst, "{} received a null ndarray.",
              op_name);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} expects 1D ndarrays.", op_name);
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "{} source and indices sizes differ.", op_name);
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "{} received an unsupported value type.",
              op_name);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} expects i32 indices.", op_name);
  check_vulkan_strided_range(op_name, "source", src, src->get_nelement(),
                             value_size, src_offset, src_stride);
  check_vulkan_strided_range(op_name, "destination", dst, dst->get_nelement(),
                             value_size, dst_offset, dst_stride);
}

void check_vulkan_grouped_reduce_strided_keys_request(Ndarray *keys,
                                                      Ndarray *values,
                                                      Ndarray *output,
                                                      int value_type,
                                                      size_t keys_offset,
                                                      size_t keys_stride,
                                                      size_t values_offset,
                                                      size_t values_stride,
                                                      size_t output_offset,
                                                      size_t output_stride,
                                                      int op) {
  const char *op_name = "Vulkan native strided grouped reduce";
  TI_ERROR_IF(!keys || !values || !output, "{} received a null ndarray.",
              op_name);
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "{} expects 1D ndarrays.", op_name);
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "{} keys and values sizes differ.", op_name);
  TI_ERROR_IF(output->get_nelement() == 0,
              "{} output must contain at least one group.", op_name);
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "{} received an unsupported value type.",
              op_name);
  TI_ERROR_IF(op != 0, "{} currently supports only sum.", op_name);
  check_vulkan_strided_range(op_name, "keys", keys, keys->get_nelement(),
                             sizeof(int32_t), keys_offset, keys_stride);
  check_vulkan_strided_range(op_name, "values", values, values->get_nelement(),
                             value_size, values_offset, values_stride);
  check_vulkan_strided_range(op_name, "output", output, output->get_nelement(),
                             value_size, output_offset, output_stride);
}

void check_vulkan_grouped_reduce_strided_request(Ndarray *keys,
                                                 Ndarray *values,
                                                 Ndarray *output,
                                                 int value_type,
                                                 size_t values_offset,
                                                 size_t values_stride,
                                                 size_t output_offset,
                                                 size_t output_stride,
                                                 int op) {
  TI_ERROR_IF(keys && keys->get_element_size() != sizeof(int32_t),
              "Vulkan native strided grouped reduce expects i32 keys.");
  check_vulkan_grouped_reduce_strided_keys_request(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t vulkan_reduce_ndarray_impl(Program *program,
                                       Ndarray *values,
                                       Ndarray *output,
                                       int value_type,
                                       int op,
                                       std::size_t offset,
                                       std::size_t stride,
                                       std::size_t output_offset,
                                       std::size_t output_stride,
                                       bool member_source,
                                       bool member_destination) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native reduce is only available on Vulkan.");
  TI_ERROR_IF(!values || !output,
              "Vulkan native reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native reduce output must contain at least one item.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "Vulkan native reduce received an unsupported value type.");
  TI_ERROR_IF(!program->vulkan_reduce_value_type_available(value_type),
              "Vulkan native reduce dtype is not supported by this device.");
  const size_t element_size = vulkan_transform_value_size(value_type);
  if (member_source || member_destination) {
    check_vulkan_reduce_strided_request(values, output, value_type, offset,
                                        stride, output_offset, output_stride,
                                        op);
  } else if (member_source) {
    check_vulkan_reduce_member_request(values, output, value_type, offset,
                                       stride, op);
  } else {
    TI_ERROR_IF(values->get_element_size() != element_size ||
                    output->get_element_size() != element_size,
                "Vulkan native reduce dtype does not match value type.");
    offset = 0;
    stride = element_size;
    output_offset = 0;
    output_stride = element_size;
  }
  TI_ERROR_IF(op < 0 || op > 2,
              "Vulkan native reduce supports only sum/min/max operations.");

  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native reduce requires a compute device.");
  auto &cache = get_reduce_cache(program, device, value_type, member_source);

  const size_t n = values->get_nelement();
  const size_t value_bytes = member_source ? n * values->get_element_size()
                                           : n * element_size;
  const size_t output_bytes = element_size;
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_REDUCE_SINGLE_SHARED_MAX_N", 4096);
  const bool use_single_shared =
      single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (!use_single_shared) {
    num_chunks = (n + kReducePrivateChunkSize - 1) / kReducePrivateChunkSize;
    partial_bytes = num_chunks * element_size;
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      program->synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }
  if (member_source) {
    cache.ensure_params();
  }

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation output_alloc = output->ndarray_alloc_;
  DeviceAllocation partial_alloc = cache.partial;
  DeviceAllocation params_alloc = cache.params;
  auto &pipeline_set = cache.pipeline_set(value_type);
  Pipeline *private_pipeline =
      member_source ? pipeline_set.private_strided_pipelines[op].get()
                    : pipeline_set.private_pipelines[op].get();
  Pipeline *final_pipeline = pipeline_set.final_pipelines[op].get();
  Pipeline *single_pipeline =
      member_source ? pipeline_set.single_strided_pipelines[op].get()
                    : pipeline_set.single_pipelines[op].get();
  const bool profiler_scopes = program->profiler != nullptr;
  std::array<uint32_t, 3> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(offset / sizeof(uint32_t)),
      static_cast<uint32_t>(stride / sizeof(uint32_t)),
  };
  constexpr size_t params_bytes = 3 * sizeof(uint32_t);

  program->enqueue_compute_op_lambda(
      [values_alloc, output_alloc, partial_alloc, params_alloc, value_bytes,
       output_bytes, partial_bytes, private_pipeline, final_pipeline,
       single_pipeline, num_chunks, use_single_shared, member_source,
       output_offset, param_words, profiler_scopes,
       params_bytes](Device *op_device, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        auto bind_params = [&](ShaderResourceSet *bindings) {
          if (!member_source) {
            return;
          }
          for (uint32_t i = 0; i < param_words.size(); ++i) {
            cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                                 sizeof(uint32_t), param_words[i]);
          }
          cmdlist->buffer_barrier(params_alloc);
          bindings->rw_buffer(2, params_alloc.get_ptr(0), params_bytes);
        };
        if (use_single_shared) {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, output_alloc.get_ptr(output_offset),
                              output_bytes);
          bind_params(bindings.get());
          dispatch_pipeline(cmdlist, single_pipeline, bindings.get(), 1, 1, 1,
                            scope_name(member_source
                                           ? "vulkan_reduce_single_strided"
                                           : "vulkan_reduce_single"));
          cmdlist->buffer_barrier(output_alloc);
          return;
        }
        {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, partial_alloc.get_ptr(0), partial_bytes);
          bind_params(bindings.get());
          dispatch_pipeline(cmdlist, private_pipeline, bindings.get(),
                            static_cast<uint32_t>(num_chunks), 1, 1,
                            scope_name(member_source
                                           ? "vulkan_reduce_private_strided"
                                           : "vulkan_reduce_private"));
          cmdlist->buffer_barrier(partial_alloc);
        }
        {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, partial_alloc.get_ptr(0), partial_bytes);
          bindings->rw_buffer(1, output_alloc.get_ptr(output_offset),
                              output_bytes);
          dispatch_pipeline(cmdlist, final_pipeline, bindings.get(), 1, 1, 1,
                            scope_name("vulkan_reduce_final"));
          cmdlist->buffer_barrier(output_alloc);
        }
      },
      {});
  return cache.cached_bytes;
}

std::size_t vulkan_transform_affine_ndarray_impl(Program *program,
                                                 Ndarray *src,
                                                 Ndarray *dst,
                                                 int value_type,
                                                 int lane_count,
                                                 std::size_t src_offset,
                                                 std::size_t src_stride,
                                                 std::size_t dst_offset,
                                                 std::size_t dst_stride,
                                                 double scale,
                                                 double bias,
                                                 bool member_source,
                                                 bool member_destination) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native transform is only available on Vulkan.");
  TI_ERROR_IF(!src || !dst, "Vulkan native transform received null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native transform source and destination sizes differ.");
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  if (!member_source) {
    TI_ERROR_IF(!member_destination &&
                    src->get_element_size() != dst->get_element_size(),
                "Vulkan native transform source and destination dtypes "
                "differ.");
    TI_ERROR_IF(src->get_element_size() != value_size,
                "Vulkan native transform dtype does not match value type.");
    src_offset = 0;
    src_stride = value_size;
  }
  if (!member_destination) {
    dst_offset = 0;
    dst_stride = value_size;
  }
  if (member_source || member_destination) {
    if (lane_count == 1) {
      check_vulkan_transform_strided_request(src, dst, value_type, src_offset,
                                             src_stride, dst_offset,
                                             dst_stride);
    } else {
      check_vulkan_transform_packed_strided_request(
          src, dst, value_type, lane_count, src_offset, src_stride, dst_offset,
          dst_stride);
    }
  }
  TI_ERROR_IF(!program->vulkan_transform_value_type_available(value_type),
              "Vulkan native transform value type is not supported by this "
              "device.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native transform currently supports at most UINT32_MAX "
              "items.");

  const size_t n = src->get_nelement();
  const size_t scalar_count = n * static_cast<size_t>(lane_count);
  TI_ERROR_IF(scalar_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native transform scalar item count exceeds UINT32_MAX.");
  if (n == 0) {
    return 0;
  }
  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native transform requires a compute device.");
  auto &cache = get_transform_cache(program, device);

  std::array<uint32_t, 10> param_words{0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  if (value_type == 0) {
    param_words[0] = static_cast<uint32_t>(static_cast<int32_t>(scale));
    param_words[1] = static_cast<uint32_t>(static_cast<int32_t>(bias));
  } else if (value_type == 2) {
    param_words[0] = static_cast<uint32_t>(scale);
    param_words[1] = static_cast<uint32_t>(bias);
  } else if (value_type == 1) {
    float scale_f32 = static_cast<float>(scale);
    float bias_f32 = static_cast<float>(bias);
    std::memcpy(&param_words[0], &scale_f32, sizeof(param_words[0]));
    std::memcpy(&param_words[1], &bias_f32, sizeof(param_words[1]));
  } else {
    uint64_t scale_u64 = 0;
    uint64_t bias_u64 = 0;
    if (value_type == 4) {
      scale_u64 = static_cast<uint64_t>(static_cast<int64_t>(scale));
      bias_u64 = static_cast<uint64_t>(static_cast<int64_t>(bias));
    } else if (value_type == 5) {
      std::memcpy(&scale_u64, &scale, sizeof(scale_u64));
      std::memcpy(&bias_u64, &bias, sizeof(bias_u64));
    } else {
      scale_u64 = static_cast<uint64_t>(scale);
      bias_u64 = static_cast<uint64_t>(bias);
    }
    param_words[0] = static_cast<uint32_t>(scale_u64);
    param_words[1] = static_cast<uint32_t>(scale_u64 >> 32);
    param_words[2] = static_cast<uint32_t>(bias_u64);
    param_words[3] = static_cast<uint32_t>(bias_u64 >> 32);
  }
  param_words[4] = static_cast<uint32_t>(n);
  param_words[5] = static_cast<uint32_t>(src_offset / sizeof(uint32_t));
  param_words[6] = static_cast<uint32_t>(src_stride / sizeof(uint32_t));
  param_words[7] = static_cast<uint32_t>(dst_offset / sizeof(uint32_t));
  param_words[8] = static_cast<uint32_t>(dst_stride / sizeof(uint32_t));
  param_words[9] = static_cast<uint32_t>(lane_count);

  DeviceAllocation src_alloc = src->ndarray_alloc_;
  DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  DeviceAllocation params_alloc = cache.params;
  const bool bind_static_params = !cache.affine_bindings;
  ShaderResourceSet *bindings = cache.cached_affine_resource_set();
  const bool has_float64 =
      program->get_device_caps().get(DeviceCapability::spirv_has_float64) != 0;
  Pipeline *pipeline = cache.pipeline_for(device, value_type, has_float64);
  const size_t src_bytes = member_source ? n * src->get_element_size()
                                         : n * value_size;
  const size_t dst_bytes = member_destination ? n * dst->get_element_size()
                                              : n * value_size;
  const size_t params_bytes = param_words.size() * sizeof(uint32_t);
  const uint32_t groups =
      static_cast<uint32_t>((scalar_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = program->profiler != nullptr;

  program->enqueue_compute_op_lambda(
      [src_alloc, dst_alloc, params_alloc, bindings, bind_static_params,
       pipeline, src_bytes, dst_bytes, params_bytes, param_words, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < param_words.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), param_words[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        bindings->rw_buffer(0, src_alloc.get_ptr(0), src_bytes);
        bindings->rw_buffer(1, dst_alloc.get_ptr(0), dst_bytes);
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

}  // namespace

bool Program::vulkan_indexed_copy_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scatter_add_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scatter_add_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type == 0 || value_type == 2) {
    return true;
  }
  if (value_type == 1) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_atomic_float_add) != 0;
  }
  if (value_type == 3 || value_type == 4) {
    auto &caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_int64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_int64) != 0;
  }
  if (value_type == 5) {
    auto &caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_float64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_float64_add) != 0;
  }
  return false;
}

bool Program::vulkan_bucket_builder_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_grouped_reduce_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_grouped_reduce_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

bool Program::vulkan_grouped_reduce_atomic_value_type_available(
    int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type == 0 || value_type == 2) {
    return true;
  }
  if (value_type == 1) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_atomic_float_add) != 0;
  }
  if (value_type == 3 || value_type == 4) {
    auto &caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_int64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_int64) != 0;
  }
  if (value_type == 5) {
    auto &caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_float64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_float64_add) != 0;
  }
  return false;
}

std::size_t Program::vulkan_grouped_reduce_i32_atomic_ndarray(Ndarray *keys,
                                                              Ndarray *values,
                                                              Ndarray *output,
                                                              int op) {
  return vulkan_grouped_reduce_atomic_ndarray(keys, values, output, 0, op);
}

std::size_t Program::vulkan_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                          Ndarray *values,
                                                          Ndarray *output,
                                                          int value_type,
                                                          int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native grouped reduce is only available on Vulkan.");
  TI_ERROR_IF(!keys || !values || !output,
              "Vulkan native grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "Vulkan native grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "Vulkan native grouped reduce output must contain at least one group.");
  TI_ERROR_IF(!vulkan_grouped_reduce_atomic_value_type_available(value_type),
              "Vulkan native grouped reduce does not support the requested "
              "value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "Unsupported Vulkan grouped reduce value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != value_size ||
                  output->get_element_size() != value_size,
              "Vulkan native grouped reduce dtype does not match the "
              "requested value type.");
  TI_ERROR_IF(op != 0, "Vulkan native grouped reduce currently supports only sum.");
  const size_t n = keys->get_nelement();
  const size_t num_groups = output->get_nelement();
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native grouped reduce input is too large for u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native grouped reduce requires a compute device.");
  auto &cache = get_bucket_builder_cache(this, device);
  const DeviceAllocation keys_alloc = keys->ndarray_alloc_;
  const DeviceAllocation values_alloc = values->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const size_t input_bytes = n * sizeof(int32_t);
  const size_t values_bytes = n * value_size;
  const size_t output_bytes = num_groups * value_size;
  Pipeline *zero_pipeline = cache.grouped_reduce_zero_pipeline(value_type);
  Pipeline *atomic_pipeline = cache.grouped_reduce_atomic_pipeline(value_type);
  TI_ERROR_IF(!zero_pipeline || !atomic_pipeline,
              "Vulkan native grouped reduce could not find a pipeline for the "
              "requested value type.");
  ShaderResourceSet *zero_bindings =
      cache.grouped_reduce_zero_resource_set(value_type);
  ShaderResourceSet *atomic_bindings =
      cache.grouped_reduce_atomic_resource_set(value_type);
  const uint32_t zero_groups =
      static_cast<uint32_t>((num_groups + kBlockSize - 1) / kBlockSize);
  const uint32_t reduce_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [keys_alloc, values_alloc, output_alloc, input_bytes, output_bytes,
       zero_pipeline, atomic_pipeline, zero_bindings, atomic_bindings,
       zero_groups, reduce_groups, value_type, values_bytes,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        zero_bindings->rw_buffer(0, output_alloc.get_ptr(0), output_bytes);
        dispatch_pipeline(cmdlist, zero_pipeline, zero_bindings, zero_groups, 1,
                          1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_zero_f32"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_zero_u32"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_zero_u64"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_zero_i64"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_zero_f64"
                                     : "vulkan_grouped_reduce_zero_i32")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        if (reduce_groups == 0) {
          return;
        }
        atomic_bindings->rw_buffer(0, keys_alloc.get_ptr(0), input_bytes);
        atomic_bindings->rw_buffer(1, values_alloc.get_ptr(0), values_bytes);
        atomic_bindings->rw_buffer(2, output_alloc.get_ptr(0), output_bytes);
        dispatch_pipeline(cmdlist, atomic_pipeline, atomic_bindings,
                          reduce_groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_atomic_sum_f32"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_atomic_sum_u32"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_atomic_sum_u64"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_atomic_sum_i64"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_atomic_sum_f64"
                                     : "vulkan_grouped_reduce_atomic_sum_i32")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      },
      {});
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_member_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  return vulkan_grouped_reduce_atomic_strided_ndarray(
      keys, values, output, value_type, offset, stride, 0,
      vulkan_scan_value_type_size(value_type), op);
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  return vulkan_grouped_reduce_atomic_strided_keys_ndarray(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided grouped reduce is only available on "
              "Vulkan.");
  check_vulkan_grouped_reduce_strided_keys_request(
      keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  TI_ERROR_IF(!vulkan_grouped_reduce_atomic_value_type_available(value_type),
              "Vulkan native strided grouped reduce does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Unsupported Vulkan strided grouped reduce value type.");
  const size_t n = keys->get_nelement();
  const size_t num_groups = output->get_nelement();
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided grouped reduce input is too large for "
              "u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(device == nullptr,
              "Vulkan native strided grouped reduce requires a compute "
              "device.");
  auto &cache = get_bucket_builder_cache(this, device);
  cache.ensure_grouped_reduce_params();
  cache.ensure_grouped_reduce_zero_strided_pipeline(value_type);
  cache.ensure_grouped_reduce_atomic_strided_pipeline(value_type);
  const DeviceAllocation keys_alloc = keys->ndarray_alloc_;
  const DeviceAllocation values_alloc = values->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const DeviceAllocation params_alloc = cache.grouped_reduce_params;
  const size_t keys_bytes = n * keys->get_element_size();
  const size_t values_bytes = n * values->get_element_size();
  const size_t output_bytes = num_groups * output->get_element_size();
  const std::array<uint32_t, 3> zero_param_words{
      static_cast<uint32_t>(num_groups),
      static_cast<uint32_t>(output_offset / value_size),
      static_cast<uint32_t>(output_stride / value_size),
  };
  const std::array<uint32_t, 8> reduce_param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(keys_offset / sizeof(int32_t)),
      static_cast<uint32_t>(keys_stride / sizeof(int32_t)),
      static_cast<uint32_t>(values_offset / value_size),
      static_cast<uint32_t>(values_stride / value_size),
      static_cast<uint32_t>(num_groups),
      static_cast<uint32_t>(output_offset / value_size),
      static_cast<uint32_t>(output_stride / value_size),
  };
  const size_t zero_params_bytes = zero_param_words.size() * sizeof(uint32_t);
  const size_t reduce_params_bytes =
      reduce_param_words.size() * sizeof(uint32_t);
  Pipeline *zero_pipeline = cache.grouped_reduce_zero_strided_pipeline(value_type);
  Pipeline *atomic_pipeline =
      cache.grouped_reduce_atomic_strided_pipeline(value_type);
  TI_ERROR_IF(!zero_pipeline || !atomic_pipeline,
              "Vulkan native strided grouped reduce could not find a pipeline "
              "for the requested value type.");
  ShaderResourceSet *zero_bindings =
      cache.grouped_reduce_zero_strided_resource_set(value_type);
  ShaderResourceSet *atomic_bindings =
      cache.grouped_reduce_atomic_strided_resource_set(value_type);
  const uint32_t zero_groups =
      static_cast<uint32_t>((num_groups + kBlockSize - 1) / kBlockSize);
  const uint32_t reduce_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [keys_alloc, values_alloc, output_alloc, params_alloc, keys_bytes,
       values_bytes, output_bytes, zero_params_bytes, reduce_params_bytes,
       zero_param_words, reduce_param_words, zero_pipeline, atomic_pipeline,
       zero_bindings, atomic_bindings, zero_groups,
       reduce_groups, value_type, profiler_scopes](Device * /*op_device*/,
                                                   CommandList *cmdlist) {
        for (uint32_t i = 0; i < zero_param_words.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), zero_param_words[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        zero_bindings->rw_buffer(0, output_alloc.get_ptr(0), output_bytes);
        zero_bindings->rw_buffer(1, params_alloc.get_ptr(0),
                                 zero_params_bytes);
        dispatch_pipeline(cmdlist, zero_pipeline, zero_bindings, zero_groups, 1,
                          1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_zero_f32_strided"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_zero_u32_strided"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_zero_u64_strided"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_zero_i64_strided"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_zero_f64_strided"
                                     : "vulkan_grouped_reduce_zero_i32_strided")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        if (reduce_groups == 0) {
          return;
        }
        for (uint32_t i = 0; i < reduce_param_words.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), reduce_param_words[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        atomic_bindings->rw_buffer(0, keys_alloc.get_ptr(0), keys_bytes);
        atomic_bindings->rw_buffer(1, values_alloc.get_ptr(0), values_bytes);
        atomic_bindings->rw_buffer(2, output_alloc.get_ptr(0), output_bytes);
        atomic_bindings->rw_buffer(3, params_alloc.get_ptr(0),
                                   reduce_params_bytes);
        dispatch_pipeline(
            cmdlist, atomic_pipeline, atomic_bindings, reduce_groups, 1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_grouped_reduce_atomic_sum_f32_strided"
                       : value_type == 2
                       ? "vulkan_grouped_reduce_atomic_sum_u32_strided"
                       : value_type == 3
                       ? "vulkan_grouped_reduce_atomic_sum_u64_strided"
                       : value_type == 4
                       ? "vulkan_grouped_reduce_atomic_sum_i64_strided"
                       : value_type == 5
                       ? "vulkan_grouped_reduce_atomic_sum_f64_strided"
                       : "vulkan_grouped_reduce_atomic_sum_i32_strided")
                : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_inclusive_scan_ndarray(Ndarray *data,
                                                   int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scan is only available on Vulkan.");
  TI_ERROR_IF(!data, "Vulkan native scan received null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "Vulkan native scan expects a 1D ndarray.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native scan received an unsupported value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(data->get_element_size() != value_size,
              "Vulkan native scan dtype does not match the requested value type.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scan requires a compute device.");
  auto &cache = get_scan_cache(this, device);
  return enqueue_vulkan_scan(this, cache, data->ndarray_alloc_, n, value_type,
                             profiler != nullptr);
}

std::size_t Program::vulkan_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided scan is only available on Vulkan.");
  check_vulkan_scan_member_request(data, value_type, offset, stride);
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native strided scan received an unsupported value type.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native strided scan requires a compute device.");
  auto &cache = get_scan_cache(this, device);
  return enqueue_vulkan_scan(this, cache, data->ndarray_alloc_, n, value_type,
                             profiler != nullptr, true, offset, stride);
}

std::size_t Program::vulkan_compact_ndarray(Ndarray *values,
                                            Ndarray *flags,
                                            Ndarray *output,
                                            Ndarray *count,
                                            int value_type) {
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
  const size_t expected_value_bytes =
      (value_type == 0 || value_type == 1 || value_type == 2)
          ? sizeof(uint32_t)
          : (value_type == 3 || value_type == 4 || value_type == 5)
                ? sizeof(uint64_t)
                : 0;
  TI_ERROR_IF(expected_value_bytes == 0,
              "Vulkan native compact received an unsupported value type.");
  const size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "Vulkan native compact received mismatched value/flag/count "
              "dtypes or a non-4-byte-aligned payload.");

  const size_t n = values->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native compact word count exceeds UINT32_MAX.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native compact requires a compute device.");
  auto &cache = get_compact_cache(this, device);
  const size_t prefix_bytes = n * sizeof(int32_t);
  if (cache.has_workspace_allocs() &&
      cache.needs_prefix_realloc(prefix_bytes)) {
    synchronize();
  }
  cache.ensure_prefix(prefix_bytes);
  const size_t value_total_bytes = n * item_bytes;

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation flags_alloc = flags->ndarray_alloc_;
  DeviceAllocation output_alloc = output->ndarray_alloc_;
  DeviceAllocation count_alloc = count->ndarray_alloc_;
  DeviceAllocation prefix_alloc = cache.prefix;
  Pipeline *flags_pipeline = cache.compact_i32_flags.get();
  Pipeline *scatter_pipeline = cache.compact_i32_scatter.get();
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t flag_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t word_groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const int compact_fuse_max_n_config =
      get_environ_config("TI_VULKAN_COMPACT_FUSE_MAX_N", 4096);
  const bool use_fused_recording =
      compact_fuse_max_n_config > 0 &&
      n <= static_cast<size_t>(compact_fuse_max_n_config);

  if (use_fused_recording) {
    auto scan_plan = prepare_vulkan_i32_scan(this, cache.scan, prefix_alloc, n);
    cache.cached_bytes = cache.allocated_bytes();
    enqueue_compute_op_lambda(
        [flags_alloc, prefix_alloc, prefix_bytes, flags_pipeline, flag_groups,
         values_alloc, output_alloc, count_alloc, scatter_pipeline, scan_plan,
         value_total_bytes, word_groups,
         profiler_scopes](Device *op_device, CommandList *cmdlist) {
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, flags_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(1, prefix_alloc.get_ptr(0), prefix_bytes);
            dispatch_pipeline(cmdlist, flags_pipeline, bindings.get(),
                              flag_groups, 1, 1,
                              profiler_scopes ? "vulkan_compact_i32_flags"
                                              : nullptr);
            cmdlist->buffer_barrier(prefix_alloc);
          }
          record_vulkan_i32_scan(op_device, cmdlist, scan_plan,
                                 profiler_scopes);
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, values_alloc.get_ptr(0), value_total_bytes);
            bindings->rw_buffer(1, flags_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(2, prefix_alloc.get_ptr(0), prefix_bytes);
            bindings->rw_buffer(3, output_alloc.get_ptr(0), value_total_bytes);
            bindings->rw_buffer(4, count_alloc.get_ptr(0), sizeof(int32_t));
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                              word_groups, 1, 1,
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
      [flags_alloc, prefix_alloc, prefix_bytes, flags_pipeline, flag_groups,
       profiler_scopes](Device *op_device, CommandList *cmdlist) {
        auto bindings = op_device->create_resource_set_unique();
        bindings->rw_buffer(0, flags_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(1, prefix_alloc.get_ptr(0), prefix_bytes);
        dispatch_pipeline(cmdlist, flags_pipeline, bindings.get(), flag_groups,
                          1, 1,
                          profiler_scopes ? "vulkan_compact_i32_flags"
                                          : nullptr);
        cmdlist->buffer_barrier(prefix_alloc);
      },
      {});

  enqueue_vulkan_i32_scan(this, cache.scan, prefix_alloc, n, profiler_scopes);
  cache.cached_bytes = cache.allocated_bytes();

  enqueue_compute_op_lambda(
      [values_alloc, flags_alloc, prefix_alloc, output_alloc, count_alloc,
       prefix_bytes, value_total_bytes, scatter_pipeline, word_groups,
       profiler_scopes](Device *op_device, CommandList *cmdlist) {
        auto bindings = op_device->create_resource_set_unique();
        bindings->rw_buffer(0, values_alloc.get_ptr(0), value_total_bytes);
        bindings->rw_buffer(1, flags_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(2, prefix_alloc.get_ptr(0), prefix_bytes);
        bindings->rw_buffer(3, output_alloc.get_ptr(0), value_total_bytes);
        bindings->rw_buffer(4, count_alloc.get_ptr(0), sizeof(int32_t));
        dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                          word_groups, 1, 1,
                          profiler_scopes ? "vulkan_compact_i32_scatter"
                                          : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        cmdlist->buffer_barrier(count_alloc);
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_compact_i32_ndarray(Ndarray *values,
                                                Ndarray *flags,
                                                Ndarray *output,
                                                Ndarray *count) {
  return vulkan_compact_ndarray(values, flags, output, count, 0);
}

std::size_t Program::vulkan_histogram_i32_ndarray(Ndarray *values,
                                                  Ndarray *bins) {
  return vulkan_histogram_ndarray(values, bins, 0, 0);
}

std::size_t Program::vulkan_histogram_ndarray(Ndarray *values,
                                              Ndarray *bins,
                                              int value_type,
                                              int bin_type) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native histogram is only available on Vulkan.");
  TI_ERROR_IF(!values || !bins,
              "Vulkan native histogram received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "Vulkan native histogram expects 1D ndarrays.");
  TI_ERROR_IF(!vulkan_histogram_value_type_available(value_type, bin_type),
              "Vulkan native histogram received an unsupported value/bin type.");
  const size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                            : sizeof(int32_t);
  const size_t bin_size = bin_type == 4 ? sizeof(int64_t) : sizeof(int32_t);
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "Vulkan native histogram received mismatched value/bin dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "Vulkan native histogram expects at least one bin.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native histogram requires a compute device.");
  auto &cache = get_histogram_cache(this, device);

  const size_t n = values->get_nelement();
  const size_t num_bins = bins->get_nelement();
  const size_t value_bytes = n * value_size;
  const size_t bin_bytes = num_bins * bin_size;
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N", 65536);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MAX_BINS", 512);
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N", 4096);
  const bool shared_bins_supported = bin_type == 0 && num_bins <= 512;
  const bool private_shared_supported = num_bins <= 512;
  const bool use_single_shared =
      n > 0 && shared_bins_supported && single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);
  const bool use_private =
      n > 0 && !use_single_shared &&
      (private_min_n_config <= 0 ||
       n >= static_cast<size_t>(private_min_n_config)) &&
      (private_max_bins_config <= 0 ||
       num_bins <= static_cast<size_t>(private_max_bins_config));

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (use_private) {
    num_chunks = (n + kHistogramPrivateChunkSize - 1) /
                 kHistogramPrivateChunkSize;
    partial_bytes = num_chunks * num_bins * bin_size;
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation bins_alloc = bins->ndarray_alloc_;
  DeviceAllocation partial_alloc = cache.partial;
  Pipeline *clear_pipeline = cache.clear_pipeline(bin_type);
  Pipeline *count_direct_pipeline =
      cache.count_direct_pipeline(value_type, bin_type);
  Pipeline *count_private_pipeline =
      cache.count_private_pipeline(value_type, bin_type);
  Pipeline *count_private_shared_pipeline =
      private_shared_supported
          ? cache.count_private_shared_pipeline(value_type, bin_type)
          : nullptr;
  Pipeline *reduce_private_pipeline = cache.reduce_private_pipeline(bin_type);
  Pipeline *single_shared_pipeline = cache.single_shared_pipeline(value_type);
  const char *single_shared_scope = cache.single_shared_scope(value_type);
  const char *clear_scope = cache.clear_scope(bin_type);
  const char *reduce_private_scope = cache.reduce_private_scope(bin_type);
  const char *count_direct_scope =
      cache.count_direct_scope(value_type, bin_type);
  const char *count_private_scope =
      cache.count_private_scope(value_type, bin_type, false);
  const char *count_private_shared_scope =
      cache.count_private_scope(value_type, bin_type, true);
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t bin_groups = static_cast<uint32_t>(
      (num_bins + kBlockSize - 1) / kBlockSize);
  const uint32_t partial_groups = static_cast<uint32_t>(
      ((use_private ? num_chunks * num_bins : 0) + kBlockSize - 1) /
      kBlockSize);
  const uint32_t value_groups = static_cast<uint32_t>(
      (n + kBlockSize - 1) / kBlockSize);

  enqueue_compute_op_lambda(
      [values_alloc, bins_alloc, partial_alloc, value_bytes, bin_bytes,
       partial_bytes, clear_pipeline, count_direct_pipeline,
       count_private_pipeline, count_private_shared_pipeline,
       reduce_private_pipeline, single_shared_pipeline, single_shared_scope,
       clear_scope, reduce_private_scope, count_direct_scope,
       count_private_scope, count_private_shared_scope, value_groups,
       bin_groups, partial_groups, num_chunks, use_private, use_single_shared,
       profiler_scopes](
          Device *op_device, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        if (use_single_shared) {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
          dispatch_pipeline(cmdlist, single_shared_pipeline, bindings.get(), 1,
                            1, 1, scope_name(single_shared_scope));
          cmdlist->buffer_barrier(bins_alloc);
          return;
        }
        if (use_private) {
          if (!count_private_shared_pipeline) {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, partial_alloc.get_ptr(0), partial_bytes);
            dispatch_pipeline(cmdlist, clear_pipeline, bindings.get(),
                              partial_groups, 1, 1, scope_name(clear_scope));
            cmdlist->buffer_barrier(partial_alloc);
          }
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
            bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
            bindings->rw_buffer(2, partial_alloc.get_ptr(0), partial_bytes);
            Pipeline *count_pipeline = count_private_shared_pipeline
                                           ? count_private_shared_pipeline
                                           : count_private_pipeline;
            const char *count_scope =
                count_private_shared_pipeline ? count_private_shared_scope
                                              : count_private_scope;
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
                              scope_name(reduce_private_scope));
            cmdlist->buffer_barrier(bins_alloc);
          }
          return;
        }
        {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, bins_alloc.get_ptr(0), bin_bytes);
          dispatch_pipeline(cmdlist, clear_pipeline, bindings.get(), bin_groups,
                            1, 1, scope_name(clear_scope));
          cmdlist->buffer_barrier(bins_alloc);
        }
        if (value_groups > 0) {
          auto bindings = op_device->create_resource_set_unique();
          bindings->rw_buffer(0, values_alloc.get_ptr(0), value_bytes);
          bindings->rw_buffer(1, bins_alloc.get_ptr(0), bin_bytes);
          dispatch_pipeline(cmdlist, count_direct_pipeline, bindings.get(),
                            value_groups, 1, 1,
                            scope_name(count_direct_scope));
          cmdlist->buffer_barrier(bins_alloc);
        }
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_reduce_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           int op) {
  return vulkan_reduce_ndarray_impl(this, values, output, value_type, op, 0,
                                    vulkan_transform_value_size(value_type), 0,
                                    vulkan_transform_value_size(value_type),
                                    false, false);
}

std::size_t Program::vulkan_reduce_member_ndarray(Ndarray *values,
                                                  Ndarray *output,
                                                  int value_type,
                                                  std::size_t offset,
                                                  std::size_t stride,
                                                  int op) {
  return vulkan_reduce_ndarray_impl(this, values, output, value_type, op,
                                    offset, stride, 0,
                                    vulkan_transform_value_size(value_type),
                                    true, false);
}

std::size_t Program::vulkan_reduce_i32_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int op) {
  return vulkan_reduce_ndarray(values, output, 0, op);
}

std::size_t Program::vulkan_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  return vulkan_reduce_ndarray_impl(this, values, output, value_type, op,
                                    values_offset, values_stride,
                                    output_offset, output_stride, true, true);
}

std::size_t Program::vulkan_transform_affine_ndarray(Ndarray *src,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     double scale,
                                                     double bias) {
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, 0,
      vulkan_transform_value_size(value_type), 0,
      vulkan_transform_value_size(value_type), scale, bias, false, false);
}

std::size_t Program::vulkan_transform_affine_member_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    double scale,
    double bias) {
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, offset, stride, 0,
      vulkan_transform_value_size(value_type), scale, bias, true, false);
}

std::size_t Program::vulkan_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, src_offset, src_stride, dst_offset,
      dst_stride, scale, bias, true, true);
}

std::size_t Program::vulkan_transform_affine_packed_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, lane_count, src_offset, src_stride,
      dst_offset, dst_stride, scale, bias, true, true);
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
  const size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native gather currently expects 4-byte aligned values "
              "and i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native gather currently supports at most UINT32_MAX "
              "items.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native gather word count exceeds UINT32_MAX.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native gather requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  ShaderResourceSet *bindings = cache.cached_resource_set(false);
  Pipeline *pipeline = cache.gather_u32_by_i32.get();
  const size_t value_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src->get_nelement() * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       indices_bytes, value_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        bindings->rw_buffer(0, src_alloc.get_ptr(0), src_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), indices_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), value_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_gather_u32_by_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return 0;
}

std::size_t Program::vulkan_gather_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided gather is only available on Vulkan.");
  check_vulkan_indexed_copy_strided_request(
      src, indices, dst, item_bytes, src_offset, src_stride, dst_offset,
      dst_stride, false);
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided gather word count exceeds UINT32_MAX.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided gather source size exceeds UINT32_MAX.");
  auto check_word_param = [](const char *name, size_t value) {
    TI_ERROR_IF(value / sizeof(uint32_t) >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native strided gather {} exceeds UINT32_MAX words.",
                name);
  };
  check_word_param("source offset", src_offset);
  check_word_param("source stride", src_stride);
  check_word_param("destination offset", dst_offset);
  check_word_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native strided gather requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  cache.ensure_indexed_copy_params();
  ShaderResourceSet *bindings = cache.cached_strided_resource_set(false);
  Pipeline *pipeline = cache.gather_strided_u32_by_i32.get();
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src->get_nelement() * src->get_element_size();
  const size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const DeviceAllocation params_alloc = cache.indexed_copy_params;
  const bool profiler_scopes = profiler != nullptr;
  std::array<uint32_t, 7> params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src->get_nelement()),
      static_cast<uint32_t>(item_words),
      static_cast<uint32_t>(src_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(src_stride / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_stride / sizeof(uint32_t)),
  };
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, params_alloc, pipeline, bindings,
       src_bytes, indices_bytes, dst_bytes, groups, params,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < params.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), params[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        bindings->rw_buffer(0, src_alloc.get_ptr(0), src_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), indices_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), dst_bytes);
        bindings->rw_buffer(3, params_alloc.get_ptr(0),
                            params.size() * sizeof(uint32_t));
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_gather_strided_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return cache.cached_bytes;
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
  const size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native scatter currently expects 4-byte aligned values "
              "and i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter currently supports at most UINT32_MAX "
              "items.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter word count exceeds UINT32_MAX.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scatter requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  ShaderResourceSet *bindings = cache.cached_resource_set(true);
  Pipeline *pipeline = cache.scatter_u32_by_i32.get();
  const size_t value_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, value_bytes,
       indices_bytes, dst_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        bindings->rw_buffer(0, src_alloc.get_ptr(0), value_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), indices_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), dst_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_scatter_u32_by_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return 0;
}

std::size_t Program::vulkan_scatter_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided scatter is only available on Vulkan.");
  check_vulkan_indexed_copy_strided_request(
      src, indices, dst, item_bytes, src_offset, src_stride, dst_offset,
      dst_stride, true);
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided scatter word count exceeds UINT32_MAX.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided scatter destination size exceeds "
              "UINT32_MAX.");
  auto check_word_param = [](const char *name, size_t value) {
    TI_ERROR_IF(value / sizeof(uint32_t) >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native strided scatter {} exceeds UINT32_MAX words.",
                name);
  };
  check_word_param("source offset", src_offset);
  check_word_param("source stride", src_stride);
  check_word_param("destination offset", dst_offset);
  check_word_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native strided scatter requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  cache.ensure_indexed_copy_params();
  ShaderResourceSet *bindings = cache.cached_strided_resource_set(true);
  Pipeline *pipeline = cache.scatter_strided_u32_by_i32.get();
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src->get_nelement() * src->get_element_size();
  const size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const DeviceAllocation params_alloc = cache.indexed_copy_params;
  const bool profiler_scopes = profiler != nullptr;
  std::array<uint32_t, 7> params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(dst->get_nelement()),
      static_cast<uint32_t>(item_words),
      static_cast<uint32_t>(src_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(src_stride / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_stride / sizeof(uint32_t)),
  };
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, params_alloc, pipeline, bindings,
       src_bytes, indices_bytes, dst_bytes, groups, params,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < params.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), params[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        bindings->rw_buffer(0, src_alloc.get_ptr(0), src_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), indices_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), dst_bytes);
        bindings->rw_buffer(3, params_alloc.get_ptr(0),
                            params.size() * sizeof(uint32_t));
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_scatter_strided_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_add_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scatter-add is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native scatter-add currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "Vulkan native scatter-add expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native scatter-add source and destination dtypes differ.");
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native scatter-add does not support the requested value "
              "type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "Unsupported Vulkan scatter-add value type.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native scatter-add dtype does not match the requested "
              "value type and i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter-add currently supports at most UINT32_MAX "
              "source items.");
  const size_t n = indices->get_nelement();
  if (n == 0 || dst->get_nelement() == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scatter-add requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native scatter-add could not find a pipeline for the "
              "requested value type.");
  const size_t value_bytes = n * value_size;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * value_size;
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, value_bytes,
       indices_bytes, dst_bytes, groups, value_type,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        bindings->rw_buffer(0, src_alloc.get_ptr(0), value_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), indices_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), dst_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_scatter_add_f32_by_i32"
                                     : value_type == 2
                                     ? "vulkan_scatter_add_u32_by_i32"
                                     : value_type == 3
                                     ? "vulkan_scatter_add_u64_by_i32"
                                     : value_type == 4
                                     ? "vulkan_scatter_add_i64_by_i32"
                                     : value_type == 5
                                     ? "vulkan_scatter_add_f64_by_i32"
                                     : "vulkan_scatter_add_i32_by_i32")
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return 0;
}

std::size_t Program::vulkan_scatter_add_member_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  return vulkan_scatter_add_strided_ndarray(
      src, indices, dst, value_type, offset, stride, 0,
      vulkan_scan_value_type_size(value_type));
}

std::size_t Program::vulkan_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided scatter-add is only available on Vulkan.");
  check_vulkan_scatter_add_strided_request(
      src, indices, dst, value_type, src_offset, src_stride, dst_offset,
      dst_stride);
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native strided scatter-add does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Unsupported Vulkan strided scatter-add value type.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided scatter-add currently supports at most "
              "UINT32_MAX source items.");
  const size_t n = indices->get_nelement();
  if (n == 0 || dst->get_nelement() == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native strided scatter-add requires a compute device.");
  auto &cache = get_indexed_copy_cache(this, device);
  cache.ensure_scatter_add_params();
  cache.ensure_scatter_add_strided_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_strided_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_strided_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native strided scatter-add could not find a pipeline "
              "for the requested value type.");
  const std::array<uint32_t, 6> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src_offset / value_size),
      static_cast<uint32_t>(src_stride / value_size),
      static_cast<uint32_t>(dst->get_nelement()),
      static_cast<uint32_t>(dst_offset / value_size),
      static_cast<uint32_t>(dst_stride / value_size),
  };
  const size_t src_bytes = src->get_nelement() * src->get_element_size();
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const size_t params_bytes = param_words.size() * sizeof(uint32_t);
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const DeviceAllocation params_alloc = cache.scatter_add_params;
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [src_alloc, indices_alloc, dst_alloc, params_alloc, pipeline, bindings,
       src_bytes, indices_bytes, dst_bytes, params_bytes, param_words, groups,
       value_type, profiler_scopes](Device * /*op_device*/,
                                    CommandList *cmdlist) {
        for (uint32_t i = 0; i < param_words.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), param_words[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        bindings->rw_buffer(0, src_alloc.get_ptr(0), src_bytes);
        bindings->rw_buffer(1, indices_alloc.get_ptr(0), indices_bytes);
        bindings->rw_buffer(2, dst_alloc.get_ptr(0), dst_bytes);
        bindings->rw_buffer(3, params_alloc.get_ptr(0), params_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_scatter_add_f32_by_i32_strided"
                                     : value_type == 2
                                     ? "vulkan_scatter_add_u32_by_i32_strided"
                                     : value_type == 3
                                     ? "vulkan_scatter_add_u64_by_i32_strided"
                                     : value_type == 4
                                     ? "vulkan_scatter_add_i64_by_i32_strided"
                                     : value_type == 5
                                     ? "vulkan_scatter_add_f64_by_i32_strided"
                                     : "vulkan_scatter_add_i32_by_i32_strided")
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc);
      },
      {});
  return cache.cached_bytes;
}

bool Program::vulkan_bucket_builder_value_type_available(int value_type) const {
  return compile_config().arch == Arch::vulkan && value_type >= 0 &&
         value_type <= 5;
}

std::size_t Program::vulkan_bucket_builder_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *offsets,
                                                       Ndarray *output,
                                                       Ndarray *cursor) {
  return vulkan_bucket_builder_ndarray(keys, values, offsets, output, cursor,
                                       0);
}

std::size_t Program::vulkan_bucket_builder_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *offsets,
                                                   Ndarray *output,
                                                   Ndarray *cursor,
                                                   int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native bucket builder is only available on Vulkan.");
  TI_ERROR_IF(!keys || !values || !offsets || !output || !cursor,
              "Vulkan native bucket builder received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  offsets->shape.size() != 1 || output->shape.size() != 1 ||
                  cursor->shape.size() != 1,
              "Vulkan native bucket builder expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native bucket builder keys and values sizes differ.");
  TI_ERROR_IF(offsets->get_nelement() < 2,
              "Vulkan native bucket builder offsets must contain num_bins + 1 items.");
  const size_t n = keys->get_nelement();
  const size_t num_bins = offsets->get_nelement() - 1;
  TI_ERROR_IF(cursor->get_nelement() < num_bins,
              "Vulkan native bucket builder cursor is smaller than num_bins.");
  TI_ERROR_IF(output->get_nelement() < n,
              "Vulkan native bucket builder output is smaller than input values.");
  TI_ERROR_IF(!vulkan_bucket_builder_value_type_available(value_type),
              "Vulkan native bucket builder received an unsupported value type.");
  const bool value_is_64bit =
      value_type == 3 || value_type == 4 || value_type == 5;
  const size_t expected_value_size =
      value_is_64bit ? sizeof(uint64_t) : sizeof(uint32_t);
  const size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  item_bytes == 0 ||
                  item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  cursor->get_element_size() != sizeof(int32_t),
              "Vulkan native bucket builder dtype does not match value type or "
              "keys/offsets/cursor are not i32, or payload is not 4-byte aligned.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_bins >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native bucket builder input is too large for u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native bucket builder requires a compute device.");
  auto &cache = get_bucket_builder_cache(this, device);
  const DeviceAllocation keys_alloc = keys->ndarray_alloc_;
  const DeviceAllocation values_alloc = values->ndarray_alloc_;
  const DeviceAllocation offsets_alloc = offsets->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const DeviceAllocation cursor_alloc = cursor->ndarray_alloc_;
  const size_t key_bytes = n * sizeof(int32_t);
  const size_t value_bytes = n * item_bytes;
  const size_t offset_bytes = (num_bins + 1) * sizeof(int32_t);
  const size_t cursor_bytes = num_bins * sizeof(int32_t);
  const uint32_t item_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t offset_groups = static_cast<uint32_t>(
      (num_bins + 1 + kBlockSize - 1) / kBlockSize);
  constexpr size_t kPrivateChunkSize = 2048;
  const size_t private_chunks =
      n == 0 ? 0 : (n + kPrivateChunkSize - 1) / kPrivateChunkSize;
  const size_t private_partial_bytes =
      private_chunks * num_bins * sizeof(int32_t);
  const int private_enabled =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE", 1);
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MIN_N", 65536);
  const size_t private_min_n =
      private_min_n_config <= 0 ? 0 : static_cast<size_t>(private_min_n_config);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BINS", 1024);
  const size_t private_max_bins =
      private_max_bins_config <= 0 ? 0
                                   : static_cast<size_t>(private_max_bins_config);
  const int private_max_bytes_config = get_environ_config(
      "TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BYTES", 4 * 1024 * 1024);
  const size_t private_max_bytes =
      private_max_bytes_config <= 0
          ? 0
          : static_cast<size_t>(private_max_bytes_config);
  const bool use_private =
      private_enabled != 0 && n >= private_min_n && n > 0 &&
      num_bins <= private_max_bins && num_bins <= 4096 &&
      private_partial_bytes > 0 && private_partial_bytes <= private_max_bytes;
  if (use_private && cache.needs_workspace_realloc(private_partial_bytes) &&
      cache.partial != kDeviceNullAllocation) {
    synchronize();
  }
  if (use_private) {
    cache.ensure_workspace(private_partial_bytes);
  }
  Pipeline *clear_pipeline = cache.clear_i32.get();
  Pipeline *count_pipeline = cache.count_i32.get();
  Pipeline *count_private_pipeline = cache.count_private_shared_i32.get();
  Pipeline *prefix_pipeline = cache.prefix_i32.get();
  Pipeline *prefix_chunks_pipeline = cache.prefix_chunks_i32.get();
  const int scatter_value_type =
      item_bytes == expected_value_size ? value_type : 7;
  Pipeline *scatter_pipeline = cache.bucket_scatter_pipeline(scatter_value_type);
  Pipeline *scatter_private_pipeline =
      cache.bucket_scatter_private_pipeline(scatter_value_type);
  ShaderResourceSet *clear_bindings = cache.resource_set(cache.clear_bindings);
  ShaderResourceSet *count_bindings = cache.resource_set(cache.count_bindings);
  ShaderResourceSet *count_private_bindings =
      cache.resource_set(cache.count_private_bindings);
  ShaderResourceSet *prefix_bindings = cache.resource_set(cache.prefix_bindings);
  ShaderResourceSet *prefix_chunks_bindings =
      cache.resource_set(cache.prefix_chunks_bindings);
  ShaderResourceSet *scatter_bindings =
      cache.bucket_scatter_resource_set(scatter_value_type);
  ShaderResourceSet *scatter_private_bindings =
      cache.bucket_scatter_private_resource_set(scatter_value_type);
  const char *scatter_scope = cache.bucket_scatter_scope(scatter_value_type);
  const char *scatter_private_scope =
      cache.bucket_scatter_private_scope(scatter_value_type);
  const DeviceAllocation partial_alloc = cache.partial;
  const uint32_t private_groups = static_cast<uint32_t>(private_chunks);
  const uint32_t prefix_chunk_groups =
      static_cast<uint32_t>((num_bins + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [keys_alloc, values_alloc, offsets_alloc, output_alloc, cursor_alloc,
       partial_alloc, key_bytes, value_bytes, offset_bytes, cursor_bytes,
       private_partial_bytes, item_groups, offset_groups, use_private,
       private_groups, prefix_chunk_groups, clear_pipeline, count_pipeline,
       count_private_pipeline, prefix_pipeline, prefix_chunks_pipeline,
       scatter_pipeline, scatter_private_pipeline, clear_bindings,
       count_bindings, count_private_bindings, prefix_bindings,
       prefix_chunks_bindings, scatter_bindings, scatter_private_bindings,
       scatter_scope, scatter_private_scope, profiler_scopes](
          Device * /*op_device*/, CommandList *cmdlist) {
        if (use_private) {
          count_private_bindings->rw_buffer(0, keys_alloc.get_ptr(0),
                                            key_bytes);
          count_private_bindings->rw_buffer(1, partial_alloc.get_ptr(0),
                                            private_partial_bytes);
          dispatch_pipeline(
              cmdlist, count_private_pipeline, count_private_bindings,
              private_groups, 1, 1,
              profiler_scopes ? "vulkan_bucket_count_private_shared_i32"
                              : nullptr);
          cmdlist->buffer_barrier(partial_alloc);

          prefix_chunks_bindings->rw_buffer(0, partial_alloc.get_ptr(0),
                                            private_partial_bytes);
          prefix_chunks_bindings->rw_buffer(1, offsets_alloc.get_ptr(0),
                                            offset_bytes);
          dispatch_pipeline(cmdlist, prefix_chunks_pipeline,
                            prefix_chunks_bindings, prefix_chunk_groups, 1, 1,
                            profiler_scopes
                                ? "vulkan_bucket_prefix_chunks_i32"
                                : nullptr);
          cmdlist->buffer_barrier(partial_alloc);
          cmdlist->buffer_barrier(offsets_alloc);

          prefix_bindings->rw_buffer(0, offsets_alloc.get_ptr(0), offset_bytes);
          prefix_bindings->rw_buffer(1, cursor_alloc.get_ptr(0), cursor_bytes);
          dispatch_pipeline(cmdlist, prefix_pipeline, prefix_bindings, 1, 1, 1,
                            profiler_scopes ? "vulkan_bucket_prefix_i32"
                                            : nullptr);
          cmdlist->buffer_barrier(offsets_alloc);
          cmdlist->buffer_barrier(cursor_alloc);

          scatter_private_bindings->rw_buffer(0, keys_alloc.get_ptr(0),
                                              key_bytes);
          scatter_private_bindings->rw_buffer(1, values_alloc.get_ptr(0),
                                              value_bytes);
          scatter_private_bindings->rw_buffer(2, partial_alloc.get_ptr(0),
                                              private_partial_bytes);
          scatter_private_bindings->rw_buffer(3, offsets_alloc.get_ptr(0),
                                              offset_bytes);
          scatter_private_bindings->rw_buffer(4, output_alloc.get_ptr(0),
                                              value_bytes);
          dispatch_pipeline(
              cmdlist, scatter_private_pipeline, scatter_private_bindings,
              private_groups, 1, 1,
              profiler_scopes ? scatter_private_scope : nullptr);
          cmdlist->buffer_barrier(output_alloc);
          return;
        }

        clear_bindings->rw_buffer(0, offsets_alloc.get_ptr(0), offset_bytes);
        clear_bindings->rw_buffer(1, cursor_alloc.get_ptr(0), cursor_bytes);
        dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings,
                          offset_groups, 1, 1,
                          profiler_scopes ? "vulkan_bucket_clear_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(offsets_alloc);
        cmdlist->buffer_barrier(cursor_alloc);

        if (item_groups > 0) {
          count_bindings->rw_buffer(0, keys_alloc.get_ptr(0), key_bytes);
          count_bindings->rw_buffer(1, offsets_alloc.get_ptr(0), offset_bytes);
          dispatch_pipeline(cmdlist, count_pipeline, count_bindings,
                            item_groups, 1, 1,
                            profiler_scopes ? "vulkan_bucket_count_i32"
                                            : nullptr);
          cmdlist->buffer_barrier(offsets_alloc);
        }

        prefix_bindings->rw_buffer(0, offsets_alloc.get_ptr(0), offset_bytes);
        prefix_bindings->rw_buffer(1, cursor_alloc.get_ptr(0), cursor_bytes);
        dispatch_pipeline(cmdlist, prefix_pipeline, prefix_bindings, 1, 1, 1,
                          profiler_scopes ? "vulkan_bucket_prefix_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(offsets_alloc);
        cmdlist->buffer_barrier(cursor_alloc);

        if (item_groups > 0) {
          scatter_bindings->rw_buffer(0, keys_alloc.get_ptr(0), key_bytes);
          scatter_bindings->rw_buffer(1, values_alloc.get_ptr(0), value_bytes);
          scatter_bindings->rw_buffer(2, cursor_alloc.get_ptr(0), cursor_bytes);
          scatter_bindings->rw_buffer(3, output_alloc.get_ptr(0), value_bytes);
          dispatch_pipeline(cmdlist, scatter_pipeline, scatter_bindings,
                            item_groups, 1, 1,
                            profiler_scopes ? scatter_scope : nullptr);
          cmdlist->buffer_barrier(output_alloc);
        }
      },
      {});
  return use_private ? cache.cached_bytes : 0;
}

std::size_t Program::vulkan_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       Ndarray *offsets,
                                                       Ndarray *scratch,
                                                       Ndarray *cursor,
                                                       int op) {
  return vulkan_grouped_reduce_ndarray(keys, values, output, offsets, scratch,
                                       cursor, 0, op);
}

std::size_t Program::vulkan_grouped_reduce_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *output,
                                                   Ndarray *offsets,
                                                   Ndarray *scratch,
                                                   Ndarray *cursor,
                                                   int value_type,
                                                   int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native grouped reduce is only available on Vulkan.");
  TI_ERROR_IF(!keys || !values || !output || !offsets || !scratch || !cursor,
              "Vulkan native grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1 || offsets->shape.size() != 1 ||
                  scratch->shape.size() != 1 || cursor->shape.size() != 1,
              "Vulkan native grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "Vulkan native grouped reduce output must contain at least one group.");
  const size_t n = keys->get_nelement();
  const size_t num_groups = output->get_nelement();
  TI_ERROR_IF(offsets->get_nelement() < num_groups + 1,
              "Vulkan native grouped reduce offsets must contain num_groups + 1 items.");
  TI_ERROR_IF(scratch->get_nelement() < n,
              "Vulkan native grouped reduce scratch is smaller than input values.");
  TI_ERROR_IF(cursor->get_nelement() < num_groups,
              "Vulkan native grouped reduce cursor is smaller than num_groups.");
  TI_ERROR_IF(!vulkan_grouped_reduce_value_type_available(value_type),
              "Vulkan native grouped reduce received an unsupported value type.");
  const bool value_is_64bit =
      value_type == 3 || value_type == 4 || value_type == 5;
  const size_t value_size =
      value_is_64bit ? sizeof(uint64_t) : sizeof(uint32_t);
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != value_size ||
                  output->get_element_size() != value_size ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  scratch->get_element_size() != value_size ||
                  cursor->get_element_size() != sizeof(int32_t),
              "Vulkan native grouped reduce value type or i32 metadata size mismatch.");
  TI_ERROR_IF(op != 0, "Vulkan native grouped reduce currently supports only sum.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native grouped reduce input is too large for u32 dispatch.");

  std::size_t bucket_workspace =
      vulkan_bucket_builder_ndarray(keys, values, offsets, scratch, cursor,
                                    value_type);
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native grouped reduce requires a compute device.");
  auto &cache = get_bucket_builder_cache(this, device);
  const DeviceAllocation offsets_alloc = offsets->ndarray_alloc_;
  const DeviceAllocation scratch_alloc = scratch->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const size_t offset_bytes = (num_groups + 1) * sizeof(int32_t);
  const size_t scratch_bytes = n * value_size;
  const size_t output_bytes = num_groups * value_size;
  Pipeline *pipeline = cache.grouped_reduce_sum_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache.grouped_reduce_sum_resource_set(value_type);
  const char *reduce_scope = cache.grouped_reduce_sum_scope(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native grouped reduce could not find a sum pipeline.");
  const uint32_t groups = static_cast<uint32_t>(num_groups);
  const bool profiler_scopes = profiler != nullptr;
  enqueue_compute_op_lambda(
      [offsets_alloc, scratch_alloc, output_alloc, offset_bytes, scratch_bytes,
       output_bytes, pipeline, bindings, groups, reduce_scope,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        bindings->rw_buffer(0, offsets_alloc.get_ptr(0), offset_bytes);
        bindings->rw_buffer(1, scratch_alloc.get_ptr(0), scratch_bytes);
        bindings->rw_buffer(2, output_alloc.get_ptr(0), output_bytes);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? reduce_scope : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      },
      {});
  return bucket_workspace;
}

std::size_t Program::vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   int key_type,
                                                   int value_type) {
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
  const size_t key_size = vulkan_sort_key_type_size(key_type);
  TI_ERROR_IF(key_size == 0,
              "Vulkan native radix sort received an unsupported key type.");
  TI_ERROR_IF(keys->get_element_size() != key_size,
              "Vulkan native radix sort key dtype does not match the "
              "requested key type.");
  const bool use_values = values != nullptr;
  const size_t expected_value_size =
      use_values ? vulkan_scan_value_type_size(value_type) : sizeof(uint32_t);
  size_t value_size = expected_value_size;
  if (use_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "Vulkan native radix sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "Vulkan native radix sort keys and values must have the same "
                "length.");
    TI_ERROR_IF(expected_value_size == 0,
                "Vulkan native radix sort received an unsupported value type.");
    value_size = values->get_element_size();
    TI_ERROR_IF(value_size == 0 || value_size % sizeof(uint32_t) != 0,
                "Vulkan native radix sort value payload must be 4-byte aligned.");
  }

  const size_t n = keys->get_nelement();
  if (n <= 1) {
    if (front) {
      front->total_call_us += profile_time_us() - total_start;
      g_vulkan_sort_cpu_profile.merge(front_profile);
    }
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native radix sort currently supports at most "
              "UINT32_MAX items.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native radix sort requires a compute device.");
  double start = front ? profile_time_us() : 0.0;
  auto &cache = get_cache(this, device);
  if (front) {
    front->get_cache_us += profile_time_us() - start;
  }
  const bool use_radix8 = cache.radix8_enabled;
  const bool raw_value_payload =
      use_values && value_size != expected_value_size;
  const bool use_index_sort = key_type >= 2 || raw_value_payload;
  const bool workspace_use_values = use_values || use_index_sort;
  const size_t workspace_value_size =
      use_index_sort ? std::max(sizeof(uint32_t), value_size) : value_size;
  const size_t workspace_key_size =
      use_index_sort ? std::max(sizeof(uint32_t), key_size) : sizeof(uint32_t);
  const bool needs_realloc =
      cache.needs_workspace_realloc(n, workspace_use_values, use_radix8,
                                    workspace_value_size, workspace_key_size,
                                    use_index_sort);
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
  cache.ensure_workspace(n, workspace_use_values, use_radix8,
                         workspace_value_size, workspace_key_size,
                         use_index_sort);
  if (front) {
    front->ensure_workspace_us += profile_time_us() - start;
  }

  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t radix8_partitions =
      static_cast<uint32_t>((n + kRadix8PartitionSize - 1) /
                            kRadix8PartitionSize);
  const bool signed_keys = key_type == 1;
  const bool float32_keys = key_type == 2;
  const bool wide_keys = key_type >= 3;
  const bool profiler_scopes = profiler != nullptr;
  DeviceAllocation key_alloc = keys->ndarray_alloc_;
  DeviceAllocation value_alloc =
      use_values ? values->ndarray_alloc_ : kDeviceNullAllocation;
  const size_t key_bytes = n * sizeof(uint32_t);
  const size_t user_key_bytes = n * key_size;
  const size_t index_bytes = n * sizeof(uint32_t);
  const size_t value_bytes = n * value_size;
  const bool raw64_values = use_values && value_size == sizeof(uint64_t);
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
      [&, groups, n, key_type, signed_keys, float32_keys, wide_keys,
       use_index_sort, use_values, key_alloc, value_alloc, key_bytes,
       user_key_bytes, index_bytes, value_bytes, table_bytes, chunk_groups,
       chunk_table_bytes, raw64_values, inline_chunk_offsets, use_radix8,
       radix8_partitions,
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
        auto gather_words_by_index = [&](DeviceAllocation src_alloc,
                                         size_t src_bytes,
                                         DeviceAllocation indices_alloc,
                                         DeviceAllocation dst_alloc,
                                         size_t dst_bytes,
                                         const char *scope) {
          auto bindings = op_device->create_resource_set_unique();
          profiled_rw_buffer(bindings.get(), 0, src_alloc.get_ptr(0),
                             src_bytes, profile);
          profiled_rw_buffer(bindings.get(), 1, indices_alloc.get_ptr(0),
                             index_bytes, profile);
          profiled_rw_buffer(bindings.get(), 2, dst_alloc.get_ptr(0),
                             dst_bytes, profile);
          const uint32_t word_groups = static_cast<uint32_t>(
              ((dst_bytes / sizeof(uint32_t)) + kBlockSize - 1) / kBlockSize);
          dispatch_pipeline(cmdlist, cache.gather_u32_by_u32.get(),
                            bindings.get(), word_groups, 1, 1,
                            scope_name(scope), profile);
          profiled_buffer_barrier(cmdlist, dst_alloc, profile);
        };

        auto record_radix32_index_sort = [&]() {
          if (use_radix8) {
            DeviceAllocation key_read = cache.key_in;
            DeviceAllocation key_write = cache.key_out;
            DeviceAllocation value_read = cache.value_in;
            DeviceAllocation value_write = cache.value_out;
            for (int pass = 0; pass < 4; ++pass) {
              profiled_buffer_fill(cmdlist,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, 0, profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                      profile);
              {
                const bool init_static_bindings =
                    !cache.radix8_upsweep_bindings[pass];
                ShaderResourceSet *bindings =
                    cache.cached_resource_set(
                        cache.radix8_upsweep_bindings[pass], profile);
                profiled_rw_buffer(bindings, 0, key_read.get_ptr(0),
                                   key_bytes, profile);
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
                const bool init_static_bindings =
                    !cache.radix8_spine_bindings;
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
                dispatch_pipeline(cmdlist, cache.radix8_spine.get(), bindings,
                                  kRadix8Bins, 1, 1,
                                  scope_name("vulkan_sort_radix8_spine"),
                                  profile);
                profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                        profile);
                profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                        profile);
              }
              {
                const bool init_static_bindings =
                    !cache.radix8_downsweep_pairs_bindings[pass];
                ShaderResourceSet *bindings = cache.cached_resource_set(
                    cache.radix8_downsweep_pairs_bindings[pass], profile);
                profiled_rw_buffer(bindings, 0, key_read.get_ptr(0),
                                   key_bytes, profile);
                profiled_rw_buffer(bindings, 1, key_write.get_ptr(0),
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
                                   index_bytes, profile);
                profiled_rw_buffer(bindings, 5, value_write.get_ptr(0),
                                   index_bytes, profile);
                dispatch_pipeline(cmdlist,
                                  cache.radix8_downsweep_pairs[pass].get(),
                                  bindings, radix8_partitions, 1, 1,
                                  scope_name(
                                      "vulkan_sort_radix8_downsweep_pairs"),
                                  profile);
                profiled_buffer_barrier(cmdlist, key_write, profile);
                profiled_buffer_barrier(cmdlist, value_write, profile);
              }
              std::swap(key_read, key_write);
              std::swap(value_read, value_write);
            }
            return;
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
            {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
              bindings->rw_buffer(1, key_write.get_ptr(0), key_bytes);
              bindings->rw_buffer(2, cache.rank.get_ptr(0), key_bytes);
              bindings->rw_buffer(3, cache.offsets.get_ptr(0), table_bytes);
              bindings->rw_buffer(4, value_read.get_ptr(0), index_bytes);
              bindings->rw_buffer(5, value_write.get_ptr(0), index_bytes);
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
                  inline_chunk_offsets
                      ? "vulkan_sort_scatter_pairs_inline_chunks"
                      : "vulkan_sort_scatter_pairs";
              dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                                groups, 1, 1, scope_name(scatter_scope),
                                profile);
              profiled_buffer_barrier(cmdlist, key_write, profile);
              profiled_buffer_barrier(cmdlist, value_write, profile);
            }
            std::swap(key_read, key_write);
            std::swap(value_read, value_write);
          }
        };

        if (use_index_sort) {
          const double radix_start = profile ? profile_time_us() : 0.0;
          Pipeline *init_pipeline =
              key_type == 0
                  ? cache.sort_init_u32_index.get()
                  : key_type == 1
                        ? cache.sort_init_i32_index.get()
                        : float32_keys
                              ? cache.sort_init_f32_index.get()
                              : (key_type == 3
                                     ? cache.sort_init_u64_index.get()
                                     : key_type == 4
                                           ? cache.sort_init_i64_index.get()
                                           : cache.sort_init_f64_index.get());
          {
            auto bindings = op_device->create_resource_set_unique();
            profiled_rw_buffer(bindings.get(), 0, key_alloc.get_ptr(0),
                               user_key_bytes, profile);
            profiled_rw_buffer(bindings.get(), 1, cache.key_in.get_ptr(0),
                               key_bytes, profile);
            profiled_rw_buffer(bindings.get(), 2, cache.key_high.get_ptr(0),
                               key_bytes, profile);
            profiled_rw_buffer(bindings.get(), 3, cache.value_in.get_ptr(0),
                               index_bytes, profile);
            dispatch_pipeline(cmdlist, init_pipeline, bindings.get(), groups,
                              1, 1,
                              scope_name("vulkan_sort_init_key_index"),
                              profile);
            profiled_buffer_barrier(cmdlist, cache.key_in, profile);
            profiled_buffer_barrier(cmdlist, cache.value_in, profile);
            if (wide_keys) {
              profiled_buffer_barrier(cmdlist, cache.key_high, profile);
            }
          }

          record_radix32_index_sort();

          if (wide_keys) {
            gather_words_by_index(cache.key_high, key_bytes, cache.value_in,
                                  cache.key_in, key_bytes,
                                  "vulkan_sort_gather_high32");
            record_radix32_index_sort();
          }

          gather_words_by_index(key_alloc, user_key_bytes, cache.value_in,
                                cache.key_out, user_key_bytes,
                                "vulkan_sort_gather_keys");
          profiled_buffer_copy(cmdlist, key_alloc.get_ptr(0),
                               cache.key_out.get_ptr(0), user_key_bytes,
                               profile);
          profiled_buffer_barrier(cmdlist, key_alloc, profile);
          if (use_values) {
            gather_words_by_index(value_alloc, value_bytes, cache.value_in,
                                  cache.value_out, value_bytes,
                                  "vulkan_sort_gather_values");
            profiled_buffer_copy(cmdlist, value_alloc.get_ptr(0),
                                 cache.value_out.get_ptr(0), value_bytes,
                                 profile);
            profiled_buffer_barrier(cmdlist, value_alloc, profile);
          }
          if (profile) {
            if (use_radix8) {
              profile->radix8_body_us += profile_time_us() - radix_start;
            }
            profile->lambda_total_us += profile_time_us() - lambda_start;
            g_vulkan_sort_cpu_profile.merge(*profile);
          }
          return;
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
                                (raw64_values
                                     ? cache.radix8_downsweep_pairs_raw64[pass]
                                           .get()
                                     : cache.radix8_downsweep_pairs[pass].get()),
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
                raw64_values
                    ? (inline_chunk_offsets
                           ? cache.scatter_pairs_inline_chunks_raw64[pass].get()
                           : cache.scatter_pairs_raw64[pass].get())
                    : (inline_chunk_offsets
                           ? cache.scatter_pairs_inline_chunks[pass].get()
                           : cache.scatter_pairs[pass].get());
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
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_indexed_copy_mutex);
    auto it = g_vulkan_indexed_copy_caches.find(this);
    sync_before_clear =
        it != g_vulkan_indexed_copy_caches.end() &&
        it->second->scatter_add_params != kDeviceNullAllocation;
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_indexed_copy_mutex);
  auto it = g_vulkan_indexed_copy_caches.find(this);
  if (it != g_vulkan_indexed_copy_caches.end()) {
    g_vulkan_indexed_copy_caches.erase(it);
  }
}

void Program::vulkan_scatter_add_clear_workspace() {
  vulkan_indexed_copy_clear_workspace();
}

void Program::vulkan_bucket_builder_clear_workspace() {
  bool sync_before_clear = false;
  {
    std::lock_guard<std::mutex> guard(g_vulkan_bucket_builder_mutex);
    auto it = g_vulkan_bucket_builder_caches.find(this);
    sync_before_clear =
        it != g_vulkan_bucket_builder_caches.end() &&
        (it->second->partial != kDeviceNullAllocation ||
         it->second->grouped_reduce_params != kDeviceNullAllocation);
  }
  if (sync_before_clear) {
    synchronize();
  }
  std::lock_guard<std::mutex> guard(g_vulkan_bucket_builder_mutex);
  auto it = g_vulkan_bucket_builder_caches.find(this);
  if (it != g_vulkan_bucket_builder_caches.end()) {
    g_vulkan_bucket_builder_caches.erase(it);
  }
}

void Program::vulkan_grouped_reduce_clear_workspace() {
  vulkan_bucket_builder_clear_workspace();
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
  std::lock_guard<std::mutex> guard(g_vulkan_indexed_copy_mutex);
  auto it = g_vulkan_indexed_copy_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_indexed_copy_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_scatter_add_workspace_bytes() const {
  return vulkan_indexed_copy_workspace_bytes();
}

std::size_t Program::vulkan_bucket_builder_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_bucket_builder_mutex);
  auto it = g_vulkan_bucket_builder_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_bucket_builder_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

std::size_t Program::vulkan_grouped_reduce_workspace_bytes() const {
  return vulkan_bucket_builder_workspace_bytes();
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

bool Program::vulkan_scan_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_compact_available() const {
  return false;
}

bool Program::vulkan_histogram_available() const {
  return false;
}

bool Program::vulkan_histogram_value_type_available(int value_type,
                                                    int bin_type) const {
  return false;
}

bool Program::vulkan_reduce_available() const {
  return false;
}

bool Program::vulkan_reduce_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_transform_available() const {
  return false;
}

bool Program::vulkan_transform_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_indexed_copy_available() const {
  return false;
}

bool Program::vulkan_scatter_add_available() const {
  return false;
}

bool Program::vulkan_scatter_add_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_bucket_builder_available() const {
  return false;
}

bool Program::vulkan_bucket_builder_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_grouped_reduce_available() const {
  return false;
}

bool Program::vulkan_grouped_reduce_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_grouped_reduce_atomic_value_type_available(
    int value_type) const {
  return false;
}

std::size_t Program::vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   int key_type,
                                                   int value_type) {
  TI_ERROR("Vulkan native radix sort requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_ndarray(Ndarray *data,
                                                   int value_type) {
  TI_ERROR("Vulkan native scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  TI_ERROR("Vulkan native strided scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_compact_ndarray(Ndarray *values,
                                            Ndarray *flags,
                                            Ndarray *output,
                                            Ndarray *count,
                                            int value_type) {
  TI_ERROR("Vulkan native compact requires TI_WITH_VULKAN=ON.");
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

std::size_t Program::vulkan_histogram_ndarray(Ndarray *values,
                                              Ndarray *bins,
                                              int value_type,
                                              int bin_type) {
  TI_ERROR("Vulkan native histogram requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           int op) {
  TI_ERROR("Vulkan native reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_member_ndarray(Ndarray *values,
                                                  Ndarray *output,
                                                  int value_type,
                                                  std::size_t offset,
                                                  std::size_t stride,
                                                  int op) {
  TI_ERROR("Vulkan native strided reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR("Vulkan native strided reduce requires TI_WITH_VULKAN=ON.");
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

std::size_t Program::vulkan_transform_affine_member_ndarray(Ndarray *src,
                                                            Ndarray *dst,
                                                            int value_type,
                                                            std::size_t offset,
                                                            std::size_t stride,
                                                            double scale,
                                                            double bias) {
  TI_ERROR("Vulkan native strided transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  TI_ERROR("Vulkan native strided transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_packed_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  TI_ERROR("Vulkan native packed strided transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_ndarray(Ndarray *src,
                                           Ndarray *indices,
                                           Ndarray *dst) {
  TI_ERROR("Vulkan native gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR("Vulkan native strided gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst) {
  TI_ERROR("Vulkan native scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR("Vulkan native strided scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                int value_type) {
  TI_ERROR("Vulkan native scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_member_ndarray(Ndarray *src,
                                                       Ndarray *indices,
                                                       Ndarray *dst,
                                                       int value_type,
                                                       std::size_t offset,
                                                       std::size_t stride) {
  TI_ERROR("Vulkan native strided scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR("Vulkan native strided scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_bucket_builder_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *offsets,
                                                       Ndarray *output,
                                                       Ndarray *cursor) {
  TI_ERROR("Vulkan native bucket builder requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_bucket_builder_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *offsets,
                                                   Ndarray *output,
                                                   Ndarray *cursor,
                                                   int value_type) {
  TI_ERROR("Vulkan native bucket builder requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_i32_atomic_ndarray(Ndarray *keys,
                                                              Ndarray *values,
                                                              Ndarray *output,
                                                              int op) {
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                          Ndarray *values,
                                                          Ndarray *output,
                                                          int value_type,
                                                          int op) {
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_member_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  TI_ERROR("Vulkan native strided grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR("Vulkan native strided grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR("Vulkan native strided grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       Ndarray *offsets,
                                                       Ndarray *scratch,
                                                       Ndarray *cursor,
                                                       int op) {
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *output,
                                                   Ndarray *offsets,
                                                   Ndarray *scratch,
                                                   Ndarray *cursor,
                                                   int value_type,
                                                   int op) {
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
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

void Program::vulkan_scatter_add_clear_workspace() {
}

void Program::vulkan_bucket_builder_clear_workspace() {
}

void Program::vulkan_grouped_reduce_clear_workspace() {
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

std::size_t Program::vulkan_scatter_add_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_bucket_builder_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_workspace_bytes() const {
  return 0;
}

void Program::vulkan_radix_sort_cpu_profile_clear() {
}

std::string Program::vulkan_radix_sort_cpu_profile_report() const {
  return "{}";
}

}  // namespace taichi::lang

#endif
