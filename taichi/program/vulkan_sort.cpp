#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"

#include <array>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace taichi::lang {
namespace {

constexpr uint32_t kRadixBits = 4;
constexpr uint32_t kRadixBins = 1u << kRadixBits;
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kSingleChunkPrefixMaxBlocks = 32;
constexpr uint32_t kInlineChunkPrefixMaxChunks = 4;
constexpr uint32_t kRadix8Bins = 256;
constexpr uint32_t kRadix8PartitionSize = 2048;

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

std::mutex g_vulkan_sort_mutex;
std::unordered_map<void *, std::unique_ptr<VulkanRadixSortCache>>
    g_vulkan_sort_caches;

VulkanRadixSortCache &get_cache(void *owner, Device *device) {
  std::lock_guard<std::mutex> guard(g_vulkan_sort_mutex);
  auto &cache = g_vulkan_sort_caches[owner];
  if (!cache) {
    cache = std::make_unique<VulkanRadixSortCache>();
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

}  // namespace

bool Program::vulkan_radix_sort_available() const {
  return compile_config().arch == Arch::vulkan;
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

std::size_t Program::vulkan_radix_sort_workspace_bytes() const {
  std::lock_guard<std::mutex> guard(g_vulkan_sort_mutex);
  auto it = g_vulkan_sort_caches.find(const_cast<Program *>(this));
  if (it == g_vulkan_sort_caches.end()) {
    return 0;
  }
  return it->second->cached_bytes;
}

void Program::vulkan_radix_sort_cpu_profile_clear() {
  g_vulkan_sort_cpu_profile.clear();
}

std::string Program::vulkan_radix_sort_cpu_profile_report() const {
  return g_vulkan_sort_cpu_profile.report_json();
}

}  // namespace taichi::lang
