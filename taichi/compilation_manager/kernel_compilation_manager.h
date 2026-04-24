#pragma once

#include <ctime>
#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>

#include "taichi/util/offline_cache.h"
#include "taichi/codegen/kernel_compiler.h"
#include "taichi/codegen/compiled_kernel_data.h"

namespace taichi::lang {

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

    std::unique_ptr<lang::CompiledKernelData> compiled_kernel_data;

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
  static constexpr char kMetadataFilename[] = "ticache.tcb";
  static constexpr char kCacheFilenameFormat[] = "{}.tic";
  static constexpr char kMetadataLockName[] = "ticache.lock";

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

  // Dump the cached data in memory to disk
  void dump();

  // Run offline cache cleaning
  void clean_offline_cache(offline_cache::CleanCachePolicy policy,
                           int max_bytes,
                           double cleaning_factor) const;

 private:
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

  Config config_;
  CachingKernels caching_kernels_;
  CacheData cached_data_;
  std::vector<KernelCacheData *> updated_data_;

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
  // `const CompiledKernelData&` whose target lives on the heap (owned by a
  // `unique_ptr` inside `KernelCacheData`). The heap address is stable across
  // map rehashes, so the returned reference remains valid even if other
  // threads insert into `caching_kernels_` afterwards.
  mutable std::mutex cache_mutex_;
  std::condition_variable cache_cv_;
  std::unordered_set<std::string> in_progress_keys_;
};

}  // namespace taichi::lang
