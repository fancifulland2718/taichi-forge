#pragma once

#include <ctime>
#include <atomic>
#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>

#include "taichi/util/offline_cache.h"
#include "taichi/codegen/kernel_compiler.h"
#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

struct KernelExecutableLifecycleStatistics {
  bool enabled{false};
  std::uint64_t memory_cache_hits{0};
  std::uint64_t loaded_cache_hits{0};
  std::uint64_t disk_loads{0};
  std::uint64_t compiler_invocations{0};
  std::uint64_t templates_installed{0};
  std::uint64_t templates_retired{0};
  std::uint64_t resident_templates{0};
  std::uint64_t in_progress_compiles{0};
  std::uint64_t live_handles{0};
  std::uint64_t pinned_handles{0};
  std::uint64_t retired_handles{0};
  std::uint64_t retired_generation_bound_handles{0};
  std::uint64_t relocatable_templates{0};
  std::uint64_t relocatable_template_hits{0};
  std::uint64_t relocatable_bindings_created{0};
  std::uint64_t relocatable_template_reclaims{0};
  std::uint64_t handle_inline_bytes{0};
};

struct CacheData {
  enum CacheMode {
    MemCache,        // Cache the kernel in memory
    MemAndDiskCache  // Cache the kernel in memory and disk
  };
  using Version = std::uint16_t[3];

  struct KernelData {
    std::string kernel_key;
    std::size_t size{0};          // byte
    std::time_t created_at{0};    // sec
    std::time_t last_used_at{0};  // sec

    // Dump the kernel to disk if `cache_mode` == `MemAndDiskCache`
    CacheMode cache_mode{MemCache};

    std::shared_ptr<lang::CompiledKernelData> compiled_kernel_data;
    // JIT-only strong owner. Excluded from TI_IO_DEF so the on-disk metadata
    // schema remains a description of bytes, not an in-process lease graph.
    std::shared_ptr<lang::KernelExecutionHandle> execution_handle;

    TI_IO_DEF(kernel_key, size, created_at, last_used_at);
  };

  using KernelMetadata = KernelData;  // Required by CacheCleaner

  Version version{};
  std::size_t size{0};
  std::unordered_map<std::string, KernelData> kernels;

  // NOTE: The "version" must be the first field to be serialized
  TI_IO_DEF(version, size, kernels);
};

class KernelCompilationManager final {
 public:
  static constexpr char kMetadataFilenameFormat[] = "ticache_{}_s{}.tcb";
  static constexpr char kCacheFilenameFormat[] = "{}_{}.tic";
  static constexpr char kMetadataLockNameFormat[] = "ticache_{}_s{}.lock";

  using KernelCacheData = CacheData::KernelData;
  using CachingKernels = std::unordered_map<std::string, KernelCacheData>;

  struct Config {
    std::string offline_cache_path;
    std::unique_ptr<KernelCompiler> kernel_compiler;
  };

  explicit KernelCompilationManager(Config init_params);

  // Load from memory || Load from disk || (Compile && Cache in memory)
  const CompiledKernelData &load_or_compile(const CompileConfig &compile_config,
                                            const DeviceCapabilityConfig &caps,
                                            const Kernel &kernel_def);

  std::shared_ptr<KernelExecutionHandle> load_or_compile_execution_handle(
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &caps,
      const Kernel &kernel_def);

  // Return an already materialized in-memory kernel for a known specialization
  // key. This intentionally does not load from disk or compile; callers that
  // miss must fall back to load_or_compile().
  const CompiledKernelData *find_cached_kernel(const std::string &kernel_key,
                                               const Kernel &kernel_def,
                                               Arch arch,
                                               bool offline_cache);

  std::shared_ptr<KernelExecutionHandle> find_cached_execution_handle(
      const std::string &kernel_key,
      const Kernel &kernel_def,
      Arch arch,
      bool offline_cache);

  // Dump the cached data in memory to disk
  void dump();

  // Drop in-memory compiled kernels without touching disk metadata. Used when
  // offline_cache is disabled, while preserving finalize-time ownership order.
  void clear();

  // Retire in-memory compiled artifacts that contain a static binding to the
  // specified SNodeTree. Explicit tree destruction is a cold transaction, so
  // this waits for outstanding compilation before removing matching entries.
  void invalidate_snode_tree(
      int tree_id,
      const std::vector<SNodeTreeDependency> &active_dependencies);

  bool has_relocatable_template(const std::string &kernel_key) const;

  bool register_relocatable_template_candidate(
      const std::string &kernel_key);

  std::shared_ptr<KernelExecutionHandle>
  instantiate_relocatable_execution_handle(
      const std::string &kernel_key,
      const std::vector<SNodeTreeDependency> &current_dependencies);

  std::uint64_t reclaim_relocatable_templates(std::size_t maximum_resident);

  bool relocatable_reuse_enabled() const noexcept {
    return relocatable_reuse_enabled_;
  }

  void set_executable_lifecycle_telemetry_enabled(bool enabled) noexcept;

  KernelExecutableLifecycleStatistics executable_lifecycle_statistics(
      bool reset);

  // Run offline cache cleaning
  void clean_offline_cache(offline_cache::CleanCachePolicy policy,
                           int max_bytes,
                           double cleaning_factor,
                           Arch arch);

 private:
  static std::string cache_file_prefix(Arch arch);

  static std::string metadata_filename(Arch arch);

  static std::string metadata_lock_name(Arch arch);

  void ensure_metadata_loaded_locked(Arch arch);

  std::string make_filename(const std::string &kernel_key) const;

  std::unique_ptr<CompiledKernelData> compile_kernel(
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &caps,
      const Kernel &kernel_def) const;

  std::string make_kernel_key(const CompileConfig &compile_config,
                              const DeviceCapabilityConfig &caps,
                              const Kernel &kernel_def) const;

  const CompiledKernelData *try_load_cached_kernel_locked(
      const Kernel &kernel_def,
      const std::string &kernel_key,
      Arch arch,
      CacheData::CacheMode cache_mode);

  // Inserts a freshly compiled kernel into the in-memory cache and returns
  // a stable reference to it. Caller must hold `cache_mutex_` and must have
  // previously registered `kernel_key` in `in_progress_keys_`.
  const CompiledKernelData &install_compiled_kernel_locked(
      const std::string &kernel_key,
      CacheData::CacheMode cache_mode,
      std::unique_ptr<CompiledKernelData> compiled);

  std::unique_ptr<CompiledKernelData> load_ckd(const std::string &kernel_key,
                                               Arch arch);

  static CacheData::CacheMode get_cache_mode(
      const CompileConfig &compile_config,
      const Kernel &kernel_def);

  std::shared_ptr<KernelExecutionHandle> ensure_execution_handle_locked(
      KernelCacheData &kernel);

  Config config_;
  CachingKernels caching_kernels_;
  CacheData cached_data_;
  std::vector<KernelCacheData *> updated_data_;
  std::string metadata_filename_;
  std::string metadata_lock_name_;
  std::string cache_file_prefix_;
  bool metadata_loaded_{false};

  // P5.a — thread-safety for parallel kernel compilation.
  //
  // `cache_mutex_` protects every access to `caching_kernels_`,
  // `cached_data_.kernels`, `updated_data_`, and `in_progress_keys_`.
  // The mutex is intentionally dropped across the actual
  // `KernelCompiler::compile()` call (the heavy work) so multiple worker
  // threads can compile *different* kernels concurrently.
  //
  // `in_progress_keys_` prevents duplicate work: if two threads request the
  // same kernel_key, only one compiles and the other waits on `cache_cv_`.
  //
  // Reference-stability note: `load_or_compile` returns
  // `const CompiledKernelData&` whose target lives on the heap behind the
  // entry's shared execution handle. The heap address is stable across map
  // rehashes, and Graph leases can keep it alive after cache retirement.
  mutable std::mutex cache_mutex_;
  std::condition_variable cache_cv_;
  std::unordered_set<std::string> in_progress_keys_;
  std::atomic<std::uint64_t> next_execution_handle_identity_{1};
  std::vector<std::weak_ptr<KernelExecutionHandle>> execution_handles_;

  struct RelocatableExecutableTemplate {
    std::shared_ptr<CompiledKernelData> compiled;
    std::vector<SNodeTreeDependency> dependencies;
    std::uint64_t last_used{0};
  };
  std::unordered_map<std::string, RelocatableExecutableTemplate>
      relocatable_templates_;
  std::unordered_set<std::string> relocatable_candidate_keys_;
  std::uint64_t relocatable_template_clock_{0};
  // One process-local rollback decision, sampled when the Program backend is
  // created. It is consulted only on compile/destroy/rebind cold paths.
  bool relocatable_reuse_enabled_{true};

  struct ExecutableLifecycleTelemetry {
    std::atomic<bool> enabled{false};
    std::atomic<std::uint64_t> memory_cache_hits{0};
    std::atomic<std::uint64_t> loaded_cache_hits{0};
    std::atomic<std::uint64_t> disk_loads{0};
    std::atomic<std::uint64_t> compiler_invocations{0};
    std::atomic<std::uint64_t> templates_installed{0};
    std::atomic<std::uint64_t> templates_retired{0};
    std::atomic<std::uint64_t> relocatable_template_hits{0};
    std::atomic<std::uint64_t> relocatable_bindings_created{0};
    std::atomic<std::uint64_t> relocatable_template_reclaims{0};
  } executable_lifecycle_telemetry_;
};

}  // namespace taichi::lang
