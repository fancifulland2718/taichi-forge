#pragma once

#include <ctime>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/runtime/llvm/snode_tree_layout.h"
#include "taichi/program/kernel.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"

namespace taichi::lang {

// LLVM AOT deployment metadata and its reader/writer. The historical cache
// names remain, but JIT disk caching is owned by KernelCompilationManager.
// Program keeps tree-layout metadata in this container for AOT projection;
// the shared SNodeTreeLayout itself is independent of the container.

struct LlvmOfflineCache {
  using Version = uint16[3];  // {MAJOR, MINOR, PATCH}

  enum Format {
    LL = 0x01,
    BC = 0x10,
    CUDAArtifact = 0x100,
    PTX = CUDAArtifact,
  };

  struct KernelCacheData : LLVM::KernelCallContract {
    std::string kernel_key;
    LLVMCompiledKernel compiled_data;

    // Existing wire fields used by artifact size accounting and metadata
    // merges.
    std::size_t size{0};          // byte
    std::time_t created_at{0};    // seconds since epoch
    std::time_t last_used_at{0};  // seconds since epoch

    KernelCacheData() = default;
    explicit KernelCacheData(LLVM::CompiledKernelData::InternalData data)
        : LLVM::KernelCallContract(std::move(data)),
          compiled_data(std::move(data.compiled_data)) {
    }
    KernelCacheData(KernelCacheData &&) = default;
    KernelCacheData &operator=(KernelCacheData &&) = default;
    ~KernelCacheData() = default;

    KernelCacheData clone() const;
    KernelCacheData clone_metadata() const;
    LLVM::CompiledKernelData::InternalData convert_to_llvm_ckd_data() const;
    // Consume an owned cache result without cloning its context-bound module.
    LLVM::CompiledKernelData::InternalData take_llvm_ckd_data() &&;

    TI_IO_DEF(kernel_key,
              args,
              compiled_data,
              used_snode_tree_ids,
              graph_metadata,
              size,
              created_at,
              last_used_at,
              rets,
              ret_type,
              ret_size,
              args_type,
              args_size);
  };

  Version version{};
  std::size_t size{0};  // byte

  // TODO(zhanlue): we need a better identifier for each LLVM::SNodeTreeLayout
  // (SNodeTree) Given that snode_tree_id is not continuous, it is ridiculous to
  // ask the users to remember each of the snode_tree_ids
  // ** Find a way to name each SNodeTree **
  std::unordered_map<int, LLVM::SNodeTreeLayout> fields;  // key = snode_tree_id

  std::unordered_map<std::string, KernelCacheData>
      kernels;  // key = kernel_name

  // NOTE: The "version" must be the first field to be serialized
  TI_IO_DEF(version, size, fields, kernels);
};

class LlvmOfflineCacheFileReader {
 public:
  bool get_kernel_cache(LlvmOfflineCache::KernelCacheData &res,
                        const std::string &key,
                        llvm::LLVMContext &llvm_ctx);

  // AOT module lookup already caches the registered executable. Transfer the
  // reader's module once instead of retaining an original and cloning it.
  // This path never removes or rewrites artifact files on validation failure.
  bool take_kernel_cache(LlvmOfflineCache::KernelCacheData &res,
                         const std::string &key,
                         llvm::LLVMContext &llvm_ctx);
  bool has_kernel(const std::string &key) const;
  // Cold entry validation before take_kernel_* consumes the physical payload.
  const LlvmOfflineCache::KernelCacheData *kernel_metadata(
      const std::string &key) const;

  // PTX carries executable code separately; retain the same task and ABI data
  // without parsing or manufacturing an LLVM module.
  bool take_kernel_metadata(LlvmOfflineCache::KernelCacheData &res,
                            const std::string &key);

  bool get_field_cache(LLVM::SNodeTreeLayout &res, int snode_tree_id);

  size_t get_num_snode_trees();
  std::vector<int> get_snode_tree_ids() const;
  std::map<std::string, std::vector<int>> get_kernel_tree_dependencies() const;
  // Query the complete cold closure before take_kernel_* consumes entries.
  std::unordered_set<int> get_element_list_tree_ids(bool exact) const;

  static std::unique_ptr<LlvmOfflineCacheFileReader> make(
      const std::string &path,
      LlvmOfflineCache::Format format = LlvmOfflineCache::Format::LL);

  static std::unique_ptr<LlvmOfflineCacheFileReader> make(
      std::shared_ptr<const io::VirtualDir> directory,
      LlvmOfflineCache::Format format);

  static bool load_meta_data(LlvmOfflineCache &data,
                             const std::string &cache_file_path,
                             bool with_lock = true);

 private:
  LlvmOfflineCacheFileReader(std::shared_ptr<const io::VirtualDir> directory,
                             LlvmOfflineCache &&data,
                             LlvmOfflineCache::Format format);

  std::unique_ptr<llvm::Module> load_module(const std::string &key,
                                            llvm::LLVMContext &llvm_ctx) const;

  std::shared_ptr<const io::VirtualDir> directory_;
  LlvmOfflineCache data_;
  LlvmOfflineCache::Format format_;
  std::unordered_set<std::string> failed_kernels_;
};

class LlvmOfflineCacheFileWriter {
 public:
  void set_data(LlvmOfflineCache &&data) {
    this->mangled_ = false;
    this->data_ = std::move(data);
  }

  void set_data(std::unique_ptr<LlvmOfflineCache> &&data_ptr) {
    set_data(std::move(*data_ptr.get()));
  }

  void add_kernel_cache(const std::string &key,
                        LlvmOfflineCache::KernelCacheData &&kernel_cache) {
    data_.kernels[key] = std::move(kernel_cache);
  }

  void dump(const std::string &path,
            LlvmOfflineCache::Format format = LlvmOfflineCache::Format::LL,
            bool merge_with_old = false,
            std::function<size_t(const std::string &,
                                 LlvmOfflineCache::KernelCacheData &)>
                emit_device_artifact = {},
            std::function<std::unique_ptr<llvm::Module>(
                LlvmOfflineCache::KernelCacheData &)> prepare_module = {});

  void set_no_mangle() {
    mangled_ = true;
  }

 private:
  void merge_with(LlvmOfflineCache &&data);

  void mangle_offloaded_task_name(const std::string &kernel_key,
                                  LLVMCompiledKernel &compiled_data);

  LlvmOfflineCache data_;
  bool mangled_{false};
};

}  // namespace taichi::lang
#endif  // TI_WITH_LLVM
