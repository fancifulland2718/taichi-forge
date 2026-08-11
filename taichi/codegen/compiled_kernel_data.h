#pragma once

#include <string>
#include <memory>
#include <optional>
#include <algorithm>
#include <vector>

#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/codegen/offloaded_task_manifest.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

class KernelLaunchHandle {
 public:
  void set_launch_id(int id) {
    launch_id_ = id;
  }

  int get_launch_id() const {
    return launch_id_;
  }

 private:
  int launch_id_{-1};
};

class CompiledKernelDataFile {
 public:
  static constexpr char kHeadStr[] = "TIC";
  static constexpr std::size_t kHeadSize = std::size(kHeadStr);
  static constexpr std::size_t kHashSize = 64;
  enum class Err {
    kNoError,
    kNotTicFile,
    kCorruptedFile,
    kOutOfMemory,
    kIOStreamError,
  };

  Err dump(std::ostream &os);
  Err load(std::istream &is);

  CompiledKernelDataFile() {
    std::copy(kHeadStr, kHeadStr + kHeadSize, head_);
  }

  void set_arch(Arch arch) {
    arch_ = arch;
  }

  void set_metadata(std::string metadata) {
    metadata_ = std::move(metadata);
  }

  void set_src_code(std::string src) {
    src_code_ = std::move(src);
  }

  const Arch &arch() const {
    return arch_;
  }

  const std::string &metadata() const {
    return metadata_;
  }

  const std::string &src_code() const {
    return src_code_;
  }

 private:
  bool update_hash();

  char head_[kHeadSize];
  Arch arch_;
  std::string metadata_;
  std::string src_code_;
  std::string hash_;
};

class CompiledKernelData {
 public:
  enum class Err {
    kNoError = 0,
    kNotTicFile,
    kCorruptedFile,
    kParseMetadataFailed,
    kParseSrcCodeFailed,
    kArchNotMatched,
    kSerMetadataFailed,
    kSerSrcCodeFailed,
    kIOStreamError,
    kOutOfMemory,
    kTiWithoutLLVM,
    kTiWithoutSpirv,
    kCompiledKernelDataBroken,
    kUnknown,
  };

  CompiledKernelData() = default;
  CompiledKernelData(const CompiledKernelData &) = delete;
  CompiledKernelData &operator=(const CompiledKernelData &) = delete;
  virtual ~CompiledKernelData() = default;

  virtual Arch arch() const = 0;

  Err load(std::istream &is);
  Err dump(std::ostream &os) const;

  virtual std::unique_ptr<CompiledKernelData> clone() const = 0;

  // Sorted, unique Program SNodeTree ids actually referenced by the lowered
  // kernel. Graph lifecycle tracking consumes this backend-neutral view.
  virtual std::vector<int> snode_tree_ids() const = 0;

  // Allocation-free hot-path query. Ordinary launch uses this after the
  // first compilation to avoid taking the global SNode lifecycle guard for
  // kernels that cannot observe an SNodeTree.
  virtual bool has_snode_tree_dependencies() const noexcept = 0;

  // Number of backend tasks generated for this kernel. Graph diagnostics use
  // this metadata after compilation; it does not trigger compilation itself.
  virtual std::size_t task_count() const = 0;

  // Host-only task metadata. Calling this function must not register a
  // backend launch handle, allocate device memory, or enqueue work.
  virtual std::vector<OffloadedTaskManifest> task_manifest() const = 0;

  // The compilation cache key is the stable specialization identity: it
  // includes source/IR, compile configuration, device capabilities, and the
  // active backend. It is host-only and intentionally excluded from .tic
  // payloads; the cache manager restores it after either compile or load.
  void set_kernel_identity(std::string identity) {
    kernel_identity_ = std::move(identity);
    refresh_task_identities();
  }

  const std::string &kernel_identity() const {
    return kernel_identity_;
  }

  virtual const GraphKernelMetadata &graph_metadata() const = 0;
  virtual void set_graph_metadata(GraphKernelMetadata metadata) = 0;

  virtual Err debug_print(std::ostream &os) const {
    return dump(os);
  }

  virtual Err check() const {
    return Err::kNoError;
  }

  void set_handle(const KernelLaunchHandle &handle) const {
    kernel_launch_handle_ = handle;
  }

  const std::optional<KernelLaunchHandle> &get_handle() const {
    return kernel_launch_handle_;
  }

  void set_graph_masked_handle(const KernelLaunchHandle &handle) const {
    graph_masked_launch_handle_ = handle;
  }

  const std::optional<KernelLaunchHandle> &get_graph_masked_handle() const {
    return graph_masked_launch_handle_;
  }

  static std::unique_ptr<CompiledKernelData> load(std::istream &is, Err *p_err);

  static std::string get_err_msg(Err err);

 protected:
  std::string make_task_identity(std::size_t task_index,
                                 OffloadedTaskType task_type) const;

  virtual void refresh_task_identities() = 0;

  virtual Err load_impl(const CompiledKernelDataFile &file) = 0;
  virtual Err dump_impl(CompiledKernelDataFile &file) const = 0;

 private:
  using Creator = std::unique_ptr<CompiledKernelData>();
  static Creator *const llvm_creator;
  static Creator *const spriv_creator;

  static std::unique_ptr<CompiledKernelData> create(Arch arch, Err &err);

  mutable std::optional<KernelLaunchHandle> kernel_launch_handle_;
  // Internal CUDA Graph masking uses a separately compiled entry-gated
  // variant. Its handle belongs to the same compiled-kernel lifetime and must
  // not be cached by a reusable raw object address.
  mutable std::optional<KernelLaunchHandle> graph_masked_launch_handle_;
  std::string kernel_identity_;
};

}  // namespace taichi::lang
