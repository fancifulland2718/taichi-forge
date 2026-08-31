#include "taichi/compilation_manager/kernel_compilation_manager.h"
#include "taichi/system/profiler.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/compilation_manager/inproc_disk_mirror.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

namespace {

std::string cache_prefix_from_metadata_filename(
    const std::string &metadata_filename) {
  constexpr std::string_view kMetadataExtension = ".tcb";
  if (metadata_filename.size() > kMetadataExtension.size() &&
      metadata_filename.compare(metadata_filename.size() -
                                    kMetadataExtension.size(),
                                kMetadataExtension.size(),
                                kMetadataExtension.data(),
                                kMetadataExtension.size()) == 0) {
    return metadata_filename.substr(0,
                                    metadata_filename.size() -
                                        kMetadataExtension.size());
  }
  return metadata_filename;
}

void record_if_enabled(const std::atomic<bool> &enabled,
                       std::atomic<std::uint64_t> &counter,
                       std::uint64_t value = 1) noexcept {
  if (enabled.load(std::memory_order_relaxed)) {
    counter.fetch_add(value, std::memory_order_relaxed);
  }
}

bool relocatable_reuse_enabled_from_environment() {
  const char *value = std::getenv("TI_ENABLE_SNODE_EXECUTABLE_REUSE");
  if (value == nullptr) {
    return true;
  }
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return normalized != "0" && normalized != "false" &&
         normalized != "off" && normalized != "no";
}

}  // namespace

namespace offline_cache {

template <>
struct CacheCleanerUtils<CacheData> {
  using MetadataType = CacheData;
  using KernelMetaData = typename MetadataType::KernelMetadata;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
    write_to_binary_file(
        data, taichi::join_path(config.path, config.metadata_filename));
    return true;
  }

  static bool save_debugging_metadata(const CacheCleanerConfig &config,
                                      const MetadataType &data) {
    return true;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const CacheCleanerConfig &config,
      const KernelMetaData &kernel_meta) {
    auto fn = fmt::format(KernelCompilationManager::kCacheFilenameFormat,
                          cache_prefix_from_metadata_filename(
                              config.metadata_filename),
                          kernel_meta.kernel_key);
    return {fn};
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
    // Do nothing
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    std::string ext = filename_extension(name);
    const auto prefix =
        cache_prefix_from_metadata_filename(config.metadata_filename) + "_";
    return ext == kTiCacheFilenameExt && name.rfind(prefix, 0) == 0;
  }
};

}  // namespace offline_cache

KernelCompilationManager::KernelCompilationManager(Config config)
    : config_(std::move(config)),
      relocatable_reuse_enabled_(
          relocatable_reuse_enabled_from_environment()) {
  TI_DEBUG("Create KernelCompilationManager with offline_cache_file_path = {}",
           config_.offline_cache_path);
}

std::string KernelCompilationManager::cache_file_prefix(Arch arch) {
  return fmt::format("ticache_{}_s{}", arch_name(arch),
                     kOfflineCacheSchemaVersion);
}

std::string KernelCompilationManager::metadata_filename(Arch arch) {
  return fmt::format(kMetadataFilenameFormat, arch_name(arch),
                     kOfflineCacheSchemaVersion);
}

std::string KernelCompilationManager::metadata_lock_name(Arch arch) {
  return fmt::format(kMetadataLockNameFormat, arch_name(arch),
                     kOfflineCacheSchemaVersion);
}

void KernelCompilationManager::ensure_metadata_loaded_locked(Arch arch) {
  const auto next_metadata_filename = metadata_filename(arch);
  if (metadata_loaded_) {
    TI_ASSERT_INFO(
        metadata_filename_ == next_metadata_filename,
        "KernelCompilationManager cannot switch offline-cache metadata shard "
        "from {} to {} within one Program",
        metadata_filename_, next_metadata_filename);
    return;
  }

  metadata_filename_ = next_metadata_filename;
  metadata_lock_name_ = metadata_lock_name(arch);
  cache_file_prefix_ = cache_file_prefix(arch);
  metadata_loaded_ = true;

  auto filepath = join_path(config_.offline_cache_path, metadata_filename_);
  auto lock_path = join_path(config_.offline_cache_path, metadata_lock_name_);
  if (path_exists(filepath)) {
    if (offline_cache::lock_metadata_file(lock_path)) {
      auto _ = offline_cache::make_metadata_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
    } else {
      TI_WARN(
          "Offline-cache metadata lock {} is busy; skipping metadata load "
          "from {}.",
          lock_path, config_.offline_cache_path);
    }
  }
}

const CompiledKernelData &KernelCompilationManager::load_or_compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto cache_mode = get_cache_mode(compile_config, kernel_def);
  const auto kernel_key = make_kernel_key(compile_config, caps, kernel_def);

  // P5.a — serialize all cache-map mutation with cache_mutex_. Heavy
  // compile work happens OUTSIDE the lock inside
  // compile_and_cache_kernel().
  std::unique_lock<std::mutex> lock(cache_mutex_);
  if (cache_mode == CacheData::MemAndDiskCache) {
    ensure_metadata_loaded_locked(compile_config.arch);
  }

  // Wait-loop: another worker may already be compiling this exact key. If
  // so, we block on cache_cv_ and re-probe the cache on wake-up.
  while (true) {
    if (const auto *cached = try_load_cached_kernel_locked(
            kernel_def, kernel_key, compile_config.arch, cache_mode)) {
      return *cached;
    }
    if (in_progress_keys_.count(kernel_key) == 0) {
      break;  // We are responsible for compiling this key.
    }
    cache_cv_.wait(lock);
  }

  // P-Compile-2-B4: snapshot whether the offline cache metadata advertises
  // this key, while we still hold the lock. We use this hint to attempt a
  // disk load (outside the lock) BEFORE falling back to a full compile —
  // multiple workers can therefore do their respective disk reads in
  // parallel rather than serializing through `cache_mutex_`.
  bool disk_metadata_hit = false;
  if (cache_mode == CacheData::MemAndDiskCache) {
    auto it = cached_data_.kernels.find(kernel_key);
    // peek above guarantees `compiled_kernel_data` is null here, so a
    // metadata entry implies a disk-only candidate.
    disk_metadata_hit = (it != cached_data_.kernels.end());
  }

  in_progress_keys_.insert(kernel_key);
  // Drop the lock across compile() — the compile step is pure work on
  // kernel_def / compile_config (both const here) plus per-thread LLVM
  // context, so it is safe to run concurrently with other workers.
  lock.unlock();

  std::unique_ptr<CompiledKernelData> compiled;
  std::string logical_kernel_key;
  bool from_disk = false;
  try {
    if (disk_metadata_hit) {
      compiled = load_ckd(kernel_key, compile_config.arch);
      if (compiled) {
        from_disk = true;
        record_if_enabled(executable_lifecycle_telemetry_.enabled,
                          executable_lifecycle_telemetry_.disk_loads);
        TI_DEBUG("Create kernel '{}' from disk cache (key='{}', unlocked)",
                 kernel_def.get_name(), kernel_key);
      }
    }
    if (!compiled) {
      record_if_enabled(executable_lifecycle_telemetry_.enabled,
                        executable_lifecycle_telemetry_.compiler_invocations);
      compiled = compile_kernel(compile_config, caps, kernel_def);
    }
    // Logical identity is needed only when this thread installs a cache miss.
    // Keep its serialization and SHA-256 work off the memory-cache hit path,
    // and outside cache_mutex_ so unrelated cache traffic remains concurrent.
    logical_kernel_key =
        make_kernel_semantic_key(compile_config, caps, kernel_def);
  } catch (...) {
    lock.lock();
    in_progress_keys_.erase(kernel_key);
    cache_cv_.notify_all();
    throw;
  }

  lock.lock();
  if (from_disk) {
    // Refresh the metadata entry's `last_used_at` so cache pruning sees a
    // recent access. The previous (locked) `try_load_cached_kernel_locked`
    // path used to do this; preserve that behavior under the unlocked
    // disk-load path.
    auto it = cached_data_.kernels.find(kernel_key);
    if (it != cached_data_.kernels.end()) {
      it->second.last_used_at = std::time(nullptr);
      updated_data_.push_back(&it->second);
    }
  }
  const auto &result = install_compiled_kernel_locked(
      kernel_key, logical_kernel_key, kernel_def.optimization_spec_identity(),
      cache_mode, std::move(compiled));
  in_progress_keys_.erase(kernel_key);
  cache_cv_.notify_all();
  return result;
}

std::shared_ptr<KernelExecutionHandle>
KernelCompilationManager::load_or_compile_execution_handle(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  const auto &compiled = load_or_compile(compile_config, caps, kernel_def);
  const auto &kernel_key = compiled.kernel_identity();
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (auto found = caching_kernels_.find(kernel_key);
      found != caching_kernels_.end()) {
    return ensure_execution_handle_locked(found->second);
  }
  if (auto found = cached_data_.kernels.find(kernel_key);
      found != cached_data_.kernels.end() &&
      found->second.compiled_kernel_data != nullptr) {
    return ensure_execution_handle_locked(found->second);
  }
  TI_ERROR("Compiled kernel {} lost its cache owner before handle creation",
           kernel_key);
  return nullptr;
}

const CompiledKernelData *KernelCompilationManager::find_cached_kernel(
    const std::string &kernel_key,
    const Kernel &kernel_def,
    Arch arch,
    bool offline_cache) {
  if (kernel_key.empty()) {
    return nullptr;
  }
  const auto cache_mode =
      offline_cache && kernel_def.ir_is_ast() ? CacheData::MemAndDiskCache
                                             : CacheData::MemCache;
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (cache_mode == CacheData::MemAndDiskCache) {
    ensure_metadata_loaded_locked(arch);
  }
  return try_load_cached_kernel_locked(kernel_def, kernel_key, arch, cache_mode);
}

std::shared_ptr<KernelExecutionHandle>
KernelCompilationManager::find_cached_execution_handle(
    const std::string &kernel_key,
    const Kernel &kernel_def,
    Arch arch,
    bool offline_cache) {
  if (kernel_key.empty()) {
    return nullptr;
  }
  const auto cache_mode =
      offline_cache && kernel_def.ir_is_ast() ? CacheData::MemAndDiskCache
                                             : CacheData::MemCache;
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (cache_mode == CacheData::MemAndDiskCache) {
    ensure_metadata_loaded_locked(arch);
  }
  if (auto found = caching_kernels_.find(kernel_key);
      found != caching_kernels_.end()) {
    record_if_enabled(executable_lifecycle_telemetry_.enabled,
                      executable_lifecycle_telemetry_.memory_cache_hits);
    return ensure_execution_handle_locked(found->second);
  }
  if (cache_mode == CacheData::MemAndDiskCache) {
    if (auto found = cached_data_.kernels.find(kernel_key);
        found != cached_data_.kernels.end() &&
        found->second.compiled_kernel_data != nullptr) {
      record_if_enabled(executable_lifecycle_telemetry_.enabled,
                        executable_lifecycle_telemetry_.loaded_cache_hits);
      return ensure_execution_handle_locked(found->second);
    }
  }
  return nullptr;
}

void KernelCompilationManager::dump() {
  // P5.a — take a consistent snapshot of the in-memory caches before
  // touching disk. `dump()` is typically called at Program shutdown from
  // the main thread, but lock defensively so it stays correct if a worker
  // is still finishing a compile during shutdown.
  std::lock_guard<std::mutex> guard(cache_mutex_);

  if (caching_kernels_.empty() && updated_data_.empty()) {
    return;
  }
  bool has_disk_cache_work = !updated_data_.empty();
  if (!has_disk_cache_work) {
    for (const auto &[_, kernel] : caching_kernels_) {
      if (kernel.cache_mode == CacheData::MemAndDiskCache) {
        has_disk_cache_work = true;
        break;
      }
    }
  }
  if (!has_disk_cache_work) {
    caching_kernels_.clear();
    return;
  }
  if (metadata_filename_.empty()) {
    caching_kernels_.clear();
    updated_data_.clear();
    return;
  }

  taichi::create_directories(config_.offline_cache_path);
  auto filepath = join_path(config_.offline_cache_path, metadata_filename_);
  auto lock_path = join_path(config_.offline_cache_path, metadata_lock_name_);

  if (!offline_cache::lock_metadata_file(lock_path)) {
    TI_WARN("Offline-cache metadata lock {} is busy; skipping metadata dump "
            "to {}.",
            lock_path, config_.offline_cache_path);
    caching_kernels_.clear();  // Ignore the caching kernels
    updated_data_.clear();
    return;
  }

  auto _ = offline_cache::make_metadata_unlocker(lock_path);
  CacheData data;
  data.version[0] = TI_VERSION_MAJOR;
  data.version[1] = TI_VERSION_MINOR;
  data.version[2] = TI_VERSION_PATCH;
  auto &kernels = data.kernels;
  // Load old cached data
  offline_cache::load_metadata_with_checking(data, filepath);
  // Update the cached data
  for (const auto *e : updated_data_) {
    auto iter = kernels.find(e->kernel_key);
    if (iter != kernels.end()) {
      iter->second.last_used_at = e->last_used_at;
    }
  }
  // Add new data
  for (auto &[kernel_key, kernel] : caching_kernels_) {
    if (kernel.cache_mode == CacheData::MemAndDiskCache) {
      auto [iter, ok] = kernels.insert({kernel_key, std::move(kernel)});
      TI_ASSERT(!ok || iter->second.size == 0);
    }
  }
  // Clear caching_kernels_
  caching_kernels_.clear();
  // Dump cached CompiledKernelData to disk
  for (auto &[_, k] : kernels) {
    if (k.compiled_kernel_data) {
      auto cache_filename = make_filename(k.kernel_key);
      // Serialize once into an in-memory buffer so we can both write the
      // file and populate the in-process mirror without re-serializing.
      std::ostringstream oss(std::ios::out | std::ios::binary);
      auto err = k.compiled_kernel_data->dump(oss);
      if (err == CompiledKernelData::Err::kNoError) {
        std::string bytes = oss.str();
        std::ofstream fs{cache_filename,
                         std::ios::out | std::ios::binary};
        TI_ASSERT(fs.is_open());
        fs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        TI_ASSERT(!!fs);
        k.size = bytes.size();
        data.size += k.size;
        // P1.b — seed the mirror so the *next* Program within the same
        // process skips disk entirely.
        InprocDiskMirror::put(k.kernel_key, std::move(bytes));
      } else {
        TI_DEBUG("Dump cached CompiledKernelData(kernel_key={}) failed: {}",
                 k.kernel_key, CompiledKernelData::get_err_msg(err));
      }
    }
  }
  // Dump offline cache metadata
  if (!kernels.empty()) {
    write_to_binary_file(data, filepath);
  }
  updated_data_.clear();
}

void KernelCompilationManager::clear() {
  std::lock_guard<std::mutex> guard(cache_mutex_);
  std::uint64_t retired = caching_kernels_.size();
  for (auto &[_, kernel] : caching_kernels_) {
    if (kernel.execution_handle != nullptr) {
      kernel.execution_handle->retire();
    }
  }
  for (const auto &[_, kernel] : cached_data_.kernels) {
    retired += kernel.compiled_kernel_data != nullptr ? 1 : 0;
  }
  for (auto &[_, kernel] : cached_data_.kernels) {
    if (kernel.execution_handle != nullptr) {
      kernel.execution_handle->retire();
    }
  }
  record_if_enabled(executable_lifecycle_telemetry_.enabled,
                    executable_lifecycle_telemetry_.templates_retired,
                    retired);
  caching_kernels_.clear();
  cached_data_.kernels.clear();
  cached_data_.size = 0;
  updated_data_.clear();
  in_progress_keys_.clear();
  relocatable_templates_.clear();
  relocatable_candidate_keys_.clear();
}

void KernelCompilationManager::invalidate_snode_tree(
    int tree_id,
    const std::vector<SNodeTreeDependency> &active_dependencies) {
  auto dependency_snapshot = [&](const KernelCacheData &kernel) {
    std::vector<SNodeTreeDependency> result;
    if (kernel.compiled_kernel_data == nullptr) {
      return result;
    }
    for (int dependency_id : kernel.compiled_kernel_data->snode_tree_ids()) {
      const auto found = std::find_if(
          active_dependencies.begin(), active_dependencies.end(),
          [dependency_id](const auto &dependency) {
            return dependency.tree_id == dependency_id;
          });
      if (found == active_dependencies.end()) {
        return std::vector<SNodeTreeDependency>{};
      }
      result.push_back(*found);
    }
    return result;
  };
  auto retain_relocatable_template = [&](const std::string &kernel_key,
                                         KernelCacheData &kernel) {
    if (!relocatable_reuse_enabled_ ||
        relocatable_candidate_keys_.find(kernel_key) ==
            relocatable_candidate_keys_.end() ||
        kernel.compiled_kernel_data == nullptr ||
        !kernel.compiled_kernel_data->snode_relocation_descriptor()
             .reuse_admitted) {
      return;
    }
    auto dependencies = dependency_snapshot(kernel);
    if (dependencies.size() !=
        kernel.compiled_kernel_data->snode_tree_ids().size()) {
      return;
    }
    auto &entry = relocatable_templates_[kernel_key];
    entry.compiled = kernel.compiled_kernel_data;
    entry.dependencies = std::move(dependencies);
    entry.last_used = ++relocatable_template_clock_;
  };
  auto depends_on_tree = [tree_id](const KernelCacheData &kernel) {
    if (!kernel.compiled_kernel_data) {
      return false;
    }
    const auto &tree_ids = kernel.compiled_kernel_data->snode_tree_ids();
    return std::find(tree_ids.begin(), tree_ids.end(), tree_id) !=
           tree_ids.end();
  };

  std::unique_lock<std::mutex> lock(cache_mutex_);
  cache_cv_.wait(lock, [this] { return in_progress_keys_.empty(); });

  for (auto iter = caching_kernels_.begin();
       iter != caching_kernels_.end();) {
    if (depends_on_tree(iter->second)) {
      retain_relocatable_template(iter->first, iter->second);
      if (iter->second.execution_handle != nullptr) {
        iter->second.execution_handle->retire();
      }
      record_if_enabled(executable_lifecycle_telemetry_.enabled,
                        executable_lifecycle_telemetry_.templates_retired);
      iter = caching_kernels_.erase(iter);
    } else {
      ++iter;
    }
  }

  for (auto iter = cached_data_.kernels.begin();
       iter != cached_data_.kernels.end();) {
    if (!depends_on_tree(iter->second)) {
      ++iter;
      continue;
    }
    KernelCacheData *entry = &iter->second;
    retain_relocatable_template(iter->first, *entry);
    if (entry->execution_handle != nullptr) {
      entry->execution_handle->retire();
    }
    record_if_enabled(executable_lifecycle_telemetry_.enabled,
                      executable_lifecycle_telemetry_.templates_retired);
    updated_data_.erase(
        std::remove(updated_data_.begin(), updated_data_.end(), entry),
        updated_data_.end());
    cached_data_.size =
        iter->second.size > cached_data_.size
            ? 0
            : cached_data_.size - iter->second.size;
    iter = cached_data_.kernels.erase(iter);
  }
}

bool KernelCompilationManager::register_relocatable_template_candidate(
    const std::string &kernel_key) {
  if (!relocatable_reuse_enabled_ || kernel_key.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  // Frontend materialization precedes backend compilation. Registration only
  // records that this exact key has a structurally complete direct-Field
  // consumer; archiving later still requires the compiler-emitted descriptor
  // to admit reuse.
  relocatable_candidate_keys_.insert(kernel_key);
  return true;
}

bool KernelCompilationManager::has_relocatable_template(
    const std::string &kernel_key) const {
  if (!relocatable_reuse_enabled_ || kernel_key.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return relocatable_templates_.find(kernel_key) !=
         relocatable_templates_.end();
}

std::shared_ptr<KernelExecutionHandle>
KernelCompilationManager::instantiate_relocatable_execution_handle(
    const std::string &kernel_key,
    const std::vector<SNodeTreeDependency> &current_dependencies) {
  if (!relocatable_reuse_enabled_) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto found = relocatable_templates_.find(kernel_key);
  if (found == relocatable_templates_.end()) {
    return nullptr;
  }
  auto &source = found->second;
  if (source.dependencies.size() != current_dependencies.size()) {
    return nullptr;
  }
  for (std::size_t i = 0; i < source.dependencies.size(); ++i) {
    if (source.dependencies[i].tree_id != current_dependencies[i].tree_id ||
        source.dependencies[i].layout_fingerprint !=
            current_dependencies[i].layout_fingerprint) {
      return nullptr;
    }
  }
  source.last_used = ++relocatable_template_clock_;
  auto [active, inserted] = caching_kernels_.try_emplace(kernel_key);
  auto &binding = active->second;
  if (inserted) {
    binding.kernel_key = kernel_key;
    binding.created_at = binding.last_used_at = std::time(nullptr);
    binding.cache_mode = CacheData::MemCache;
    binding.compiled_kernel_data = source.compiled;
  } else {
    TI_ASSERT(binding.compiled_kernel_data == source.compiled);
  }
  auto handle = ensure_execution_handle_locked(binding);
  if (inserted) {
    record_if_enabled(executable_lifecycle_telemetry_.enabled,
                      executable_lifecycle_telemetry_.relocatable_template_hits);
    record_if_enabled(
        executable_lifecycle_telemetry_.enabled,
        executable_lifecycle_telemetry_.relocatable_bindings_created);
  }
  return handle;
}

std::uint64_t KernelCompilationManager::reclaim_relocatable_templates(
    std::size_t maximum_resident) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  std::uint64_t reclaimed = 0;
  while (relocatable_templates_.size() > maximum_resident) {
    auto victim = relocatable_templates_.end();
    for (auto iter = relocatable_templates_.begin();
         iter != relocatable_templates_.end(); ++iter) {
      // An archive-only payload has exactly one strong owner. A Graph/AOT
      // lease or active generation binding makes the template pinned; never
      // erase the accounting owner while such a lease is live.
      if (iter->second.compiled.use_count() != 1) {
        continue;
      }
      if (victim == relocatable_templates_.end() ||
          iter->second.last_used < victim->second.last_used) {
        victim = iter;
      }
    }
    if (victim == relocatable_templates_.end()) {
      break;
    }
    relocatable_candidate_keys_.erase(victim->first);
    relocatable_templates_.erase(victim);
    ++reclaimed;
  }
  record_if_enabled(
      executable_lifecycle_telemetry_.enabled,
      executable_lifecycle_telemetry_.relocatable_template_reclaims,
      reclaimed);
  return reclaimed;
}

void KernelCompilationManager::set_executable_lifecycle_telemetry_enabled(
    bool enabled) noexcept {
  executable_lifecycle_telemetry_.enabled.store(enabled,
                                                std::memory_order_relaxed);
}

KernelExecutableLifecycleStatistics
KernelCompilationManager::executable_lifecycle_statistics(bool reset) {
  KernelExecutableLifecycleStatistics result;
  result.enabled = executable_lifecycle_telemetry_.enabled.load(
      std::memory_order_relaxed);
  const auto read = [reset](std::atomic<std::uint64_t> &counter) {
    return reset ? counter.exchange(0, std::memory_order_relaxed)
                 : counter.load(std::memory_order_relaxed);
  };
  result.memory_cache_hits =
      read(executable_lifecycle_telemetry_.memory_cache_hits);
  result.loaded_cache_hits =
      read(executable_lifecycle_telemetry_.loaded_cache_hits);
  result.disk_loads = read(executable_lifecycle_telemetry_.disk_loads);
  result.compiler_invocations =
      read(executable_lifecycle_telemetry_.compiler_invocations);
  result.templates_installed =
      read(executable_lifecycle_telemetry_.templates_installed);
  result.templates_retired =
      read(executable_lifecycle_telemetry_.templates_retired);
  result.relocatable_template_hits =
      read(executable_lifecycle_telemetry_.relocatable_template_hits);
  result.relocatable_bindings_created =
      read(executable_lifecycle_telemetry_.relocatable_bindings_created);
  result.relocatable_template_reclaims =
      read(executable_lifecycle_telemetry_.relocatable_template_reclaims);
  {
    std::lock_guard<std::mutex> guard(cache_mutex_);
    result.resident_templates = caching_kernels_.size();
    for (const auto &[_, kernel] : cached_data_.kernels) {
      result.resident_templates +=
          kernel.compiled_kernel_data != nullptr ? 1 : 0;
    }
    result.in_progress_compiles = in_progress_keys_.size();
    result.relocatable_templates = relocatable_templates_.size();
    for (auto iter = execution_handles_.begin();
         iter != execution_handles_.end();) {
      auto handle = iter->lock();
      if (handle == nullptr) {
        iter = execution_handles_.erase(iter);
        continue;
      }
      if (handle->active()) {
        ++result.live_handles;
      } else {
        ++result.retired_handles;
        if (!handle->compiled().snode_relocation_descriptor()
                 .reuse_admitted) {
          ++result.retired_generation_bound_handles;
        }
      }
      const auto owner_floor = handle->active() ? 2L : 1L;
      if (handle.use_count() > owner_floor) {
        // The snapshot is always one owner. Active handles normally have a
        // second cache owner; retired handles do not. Any owner above that
        // state-specific floor is a Graph/AOT/in-flight lease.
        ++result.pinned_handles;
      }
      ++iter;
    }
    result.handle_inline_bytes =
        (result.live_handles + result.retired_handles) *
        sizeof(KernelExecutionHandle);
  }
  return result;
}

void KernelCompilationManager::clean_offline_cache(
    offline_cache::CleanCachePolicy policy,
    int max_bytes,
    double cleaning_factor,
    Arch arch) {
  using CacheCleaner = offline_cache::CacheCleaner<CacheData>;
  offline_cache::CacheCleanerConfig config;
  config.path = config_.offline_cache_path;
  config.policy = policy;
  config.cleaning_factor = cleaning_factor;
  config.max_size = max_bytes;
  {
    std::lock_guard<std::mutex> guard(cache_mutex_);
    const auto clean_metadata_filename = metadata_filename(arch);
    TI_ASSERT_INFO(
        !metadata_loaded_ || metadata_filename_ == clean_metadata_filename,
        "KernelCompilationManager cannot clean offline-cache metadata shard "
        "{} while loaded shard is {}",
        clean_metadata_filename, metadata_filename_);
    config.metadata_filename = clean_metadata_filename;
    config.metadata_lock_name = metadata_lock_name(arch);
  }
  config.debugging_metadata_filename = "";
  CacheCleaner::run(config);
}

std::string KernelCompilationManager::make_filename(
    const std::string &kernel_key) const {
  TI_ASSERT(!cache_file_prefix_.empty());
  return join_path(config_.offline_cache_path,
                   fmt::format(kCacheFilenameFormat, cache_file_prefix_,
                               kernel_key));
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  TI_COMPILE_PROFILER("cpp.compile.kernel");
  auto &compiler = *config_.kernel_compiler;
  GraphKernelMetadata graph_metadata;
  auto ir = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.ir_pipeline");
    return compiler.compile(compile_config, kernel_def, &graph_metadata);
  }();
  auto ckd = [&]() {
    TI_COMPILE_PROFILER("cpp.compile.backend_codegen");
    return compiler.compile(compile_config, caps, kernel_def, *ir);
  }();
  ckd->set_graph_metadata(std::move(graph_metadata));
  {
    TI_COMPILE_PROFILER("cpp.compile.ckd_check");
    TI_ASSERT(ckd->check() == CompiledKernelData::Err::kNoError);
  }
  return ckd;
}

std::string KernelCompilationManager::make_kernel_key(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto kernel_key = kernel_def.get_cached_kernel_key();
  if (kernel_key.empty()) {
    if (!kernel_def.ir_is_ast()) {
      const auto cache_context_key = get_hashed_offline_cache_key_context(
          compile_config, caps, (Kernel *)&kernel_def);
      kernel_key = "N" + cache_context_key + "_" + kernel_def.get_name();
    } else {  // The kernel key is generated from AST
      kernel_key = get_hashed_offline_cache_key(compile_config, caps,
                                                (Kernel *)&kernel_def);
    }

    kernel_def.set_kernel_key_for_cache(kernel_key);
  }
  return kernel_key;
}

std::string KernelCompilationManager::make_kernel_semantic_key(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  if (!kernel_def.ir_is_ast()) {
    const auto context_key = get_hashed_offline_cache_semantic_key_context(
        compile_config, caps, (Kernel *)&kernel_def);
    return "N" + context_key + "_" + kernel_def.get_name();
  }
  return get_hashed_offline_cache_semantic_key(
      compile_config, caps, (Kernel *)&kernel_def);
}

const CompiledKernelData *KernelCompilationManager::try_load_cached_kernel_locked(
    const Kernel &kernel_def,
    const std::string &kernel_key,
    Arch arch,
    CacheData::CacheMode cache_mode) {
  // Precondition: cache_mutex_ held by caller.
  {  // Find in memory-cache (caching_kernels_)
    const auto &kernels = caching_kernels_;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      record_if_enabled(executable_lifecycle_telemetry_.enabled,
                        executable_lifecycle_telemetry_.memory_cache_hits);
      TI_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
               kernel_def.get_name(), kernel_key);
      return iter->second.compiled_kernel_data.get();
    }
  }
  // Find in disk-cache (cached_data_)
  if (cache_mode == CacheData::MemAndDiskCache) {
    auto &kernels = cached_data_.kernels;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      auto &k = iter->second;
      if (k.compiled_kernel_data) {
        record_if_enabled(executable_lifecycle_telemetry_.enabled,
                          executable_lifecycle_telemetry_.loaded_cache_hits);
        TI_DEBUG("Create kernel '{}' from cache (key='{}')",
                 kernel_def.get_name(), kernel_key);
        return k.compiled_kernel_data.get();
      }
      // P-Compile-2-B4: do NOT call `load_ckd` while holding
      // `cache_mutex_`. Disk I/O is moved out into `load_or_compile`'s
      // unlocked region; we just signal "not in mem yet" by returning
      // nullptr, and the caller will probe disk after dropping the lock.
    }
  }
  return nullptr;
}

std::shared_ptr<KernelExecutionHandle>
KernelCompilationManager::ensure_execution_handle_locked(
    KernelCacheData &kernel) {
  TI_ASSERT(kernel.compiled_kernel_data != nullptr);
  if (kernel.execution_handle == nullptr ||
      !kernel.execution_handle->active()) {
    const auto identity = next_execution_handle_identity_.fetch_add(
        1, std::memory_order_relaxed);
    TI_ERROR_IF(identity == 0 ||
                    identity == std::numeric_limits<std::uint64_t>::max(),
                "Kernel execution handle identity space exhausted; call "
                "ti.reset().");
    kernel.execution_handle = std::make_shared<KernelExecutionHandle>(
        identity, kernel.compiled_kernel_data);
    execution_handles_.push_back(kernel.execution_handle);
  }
  return kernel.execution_handle;
}

const CompiledKernelData &
KernelCompilationManager::install_compiled_kernel_locked(
    const std::string &kernel_key,
    const std::string &logical_kernel_key,
    const std::string &optimization_spec_identity,
    CacheData::CacheMode cache_mode,
    std::unique_ptr<CompiledKernelData> compiled) {
  // Precondition: cache_mutex_ held by caller; `kernel_key` is present in
  // `in_progress_keys_` and is NOT yet in caching_kernels_.
  TI_DEBUG_IF(cache_mode == CacheData::MemAndDiskCache,
              "Cache kernel (key='{}')", kernel_key);
  // Another thread may have raced us to the cache while we held
  // in_progress_keys_, but by contract only one thread owns any given
  // kernel_key at a time, so this must still be absent.
  TI_ASSERT(caching_kernels_.find(kernel_key) == caching_kernels_.end());
  compiled->set_kernel_identity(kernel_key);
  compiled->set_logical_kernel_identity(logical_kernel_key);
  compiled->set_optimization_spec_identity(optimization_spec_identity);
  KernelCacheData k;
  k.kernel_key = kernel_key;
  k.created_at = k.last_used_at = std::time(nullptr);
  k.compiled_kernel_data = std::move(compiled);
  k.size = 0;  // Populate `size` within the KernelCompilationManager::dump()
  k.cache_mode = cache_mode;
  auto &kernel_data = (caching_kernels_[kernel_key] = std::move(k));
  ensure_execution_handle_locked(kernel_data);
  record_if_enabled(executable_lifecycle_telemetry_.enabled,
                    executable_lifecycle_telemetry_.templates_installed);
  return *kernel_data.compiled_kernel_data;
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::load_ckd(
    const std::string &kernel_key,
    Arch arch) {
  // P1.b — try the in-process bytes mirror first. On a hit we skip the
  // file open + read entirely; on a miss we fall through to disk and
  // populate the mirror with the bytes we just read so that the *next*
  // Program hits it.
  auto deserialize_from = [&](std::istream &is) -> std::unique_ptr<CompiledKernelData> {
    CompiledKernelData::Err err;
    auto ckd = CompiledKernelData::load(is, &err);
    if (err != CompiledKernelData::Err::kNoError) {
      return nullptr;
    }
    if (auto cerr = ckd->check(); cerr != CompiledKernelData::Err::kNoError) {
      return nullptr;
    }
    return ckd;
  };

  if (auto cached_bytes = InprocDiskMirror::get(kernel_key)) {
    std::istringstream iss(*cached_bytes, std::ios::in | std::ios::binary);
    if (auto ckd = deserialize_from(iss)) {
      return ckd;
    }
    // If the mirrored bytes are somehow corrupt, fall through to disk —
    // don't propagate the error as a hard miss. The disk copy (if any)
    // will also re-populate the mirror below.
  }

  const auto filename = make_filename(kernel_key);
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return nullptr;
  }
  // Read the whole file into a string so we can (a) deserialize from it
  // and (b) stash it into the mirror without re-reading from disk.
  std::string bytes{std::istreambuf_iterator<char>(ifs),
                    std::istreambuf_iterator<char>()};
  ifs.close();

  std::istringstream iss(bytes, std::ios::in | std::ios::binary);
  auto ckd = deserialize_from(iss);
  if (ckd == nullptr) {
    TI_DEBUG("Load cache file {} failed or is corrupt", filename);
    return nullptr;
  }
  // Only cache well-formed bytes so that mirror hits are always valid.
  InprocDiskMirror::put(kernel_key, std::move(bytes));
  return ckd;
}

CacheData::CacheMode KernelCompilationManager::get_cache_mode(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) {
  return compile_config.offline_cache && kernel_def.ir_is_ast()
             ? CacheData::MemAndDiskCache
             : CacheData::MemCache;
}

}  // namespace taichi::lang
