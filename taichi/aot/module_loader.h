#pragma once

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "taichi/aot/module_data.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/snode.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/initial_root_data.h"
#include "taichi/aot/root_storage.h"
#include "taichi/aot/struct_layout.h"
#include "taichi/aot/value_parameters.h"

namespace taichi::io {
struct VirtualDir;
}

namespace taichi::lang {

struct RuntimeContext;
class Graph;
namespace aot {

// Payload errors may occur after metadata loading for an on-demand artifact.
// Keep these distinguishable from a missing export or a device/driver failure.
class ArtifactLoadError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TI_DLL_EXPORT Field {
 public:
  // Rule of 5 to make MSVC happy
  Field() = default;
  virtual ~Field() = default;
  Field(const Field &) = delete;
  Field &operator=(const Field &) = delete;
  Field(Field &&) = default;
  Field &operator=(Field &&) = default;
};

class TI_DLL_EXPORT KernelTemplateArg {
 public:
  using ArgUnion = std::variant<bool, int64_t, uint64_t, const Field *>;
  template <typename T>
  KernelTemplateArg(const std::string &name, T &&arg)
      : name_(name), targ_(std::forward<T>(arg)) {
  }

 private:
  std::string name_;
  /**
   * @brief Template arg
   *
   */
  ArgUnion targ_;
};

class TI_DLL_EXPORT KernelTemplate {
 public:
  // Rule of 5 to make MSVC happy
  KernelTemplate() = default;
  virtual ~KernelTemplate() = default;
  KernelTemplate(const KernelTemplate &) = delete;
  KernelTemplate &operator=(const KernelTemplate &) = delete;
  KernelTemplate(KernelTemplate &&) = default;
  KernelTemplate &operator=(KernelTemplate &&) = default;

  Kernel *get_kernel(const std::vector<KernelTemplateArg> &template_args);

 protected:
  virtual std::unique_ptr<Kernel> make_new_kernel(
      const std::vector<KernelTemplateArg> &template_args) = 0;

 private:
  std::unordered_map<std::string, std::unique_ptr<Kernel>> loaded_kernels_;
};

class TI_DLL_EXPORT Module {
 public:
  // Rule of 5 to make MSVC happy
  Module() = default;
  virtual ~Module() = default;
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  Module(Module &&) = default;
  Module &operator=(Module &&) = default;

  static std::unique_ptr<Module> load(Arch arch, std::any mod_params);

  // Module metadata
  // TODO: Instead of virtualize these simple properties, just store them as
  // member variables.
  virtual Arch arch() const = 0;
  virtual uint64_t version() const = 0;
  virtual size_t get_root_size() const = 0;
  virtual std::vector<size_t> get_root_sizes() const {
    return {get_root_size()};
  }

  struct RootStorage {
    DevicePtr pointer;
    std::uint64_t bytes;
  };
  struct BoundRootStorageView {
    RootStorageView layout;
    // Window origin; affine coordinates are relative to this pointer.
    DevicePtr pointer;
    std::uint64_t bytes;
    std::uint64_t byte_offset;
    bool packed;
  };
  // Cold, non-owning alias. The module must outlive its prepared executions;
  // its owner retires all recordings before freeing any root allocation.
  BoundRootStorageView get_root_storage_view(const std::string &name);
  DataType get_ndarray_layout(const std::string &name);

  Kernel *get_kernel(const std::string &name);
  KernelTemplate *get_kernel_template(const std::string &name);
  Field *get_snode_tree(const std::string &name);

  // Read and verify only a selected immutable data payload. No expanded host
  // payload cache is retained by the module; consumers own the returned bytes.
  std::vector<std::uint8_t> load_data(const std::string &sha256,
                                      std::uint64_t expected_bytes) const;

  virtual std::unique_ptr<aot::CompiledGraph> get_graph(
      const std::string &name) {
    TI_NOT_IMPLEMENTED;
  }

  virtual const DeviceCapabilityConfig &get_required_caps() const {
    static DeviceCapabilityConfig default_cfg;
    return default_cfg;
  }

  inline bool is_corrupted() const {
    return is_corrupted_;
  }

 protected:
  virtual RootStorage get_root_storage(int tree_id) {
    throw std::invalid_argument("This AOT backend has no root storage views");
  }
  RootStorageViews root_storage_views_;
  NamedStructLayouts ndarray_layouts_;
  bool ndarray_layouts_loaded_{false};
  NamedValueParameters value_parameters_;
  bool value_parameters_loaded_{false};
  virtual const io::VirtualDir *data_source() const {
    return nullptr;
  }
  static std::vector<std::uint8_t> load_data_from(const io::VirtualDir *source,
                                                  const std::string &sha256,
                                                  std::uint64_t expected_bytes);
  // Synchronous cold upload, after allocation and before executable
  // publication. Borrowed artifact sources may eagerly retain verified bytes
  // until this call.
  void initialize_root_data(
      const InitialRootData &entry,
      DevicePtr destination,
      const std::vector<std::uint8_t> *preloaded = nullptr) const;
  virtual std::unique_ptr<Kernel> make_new_kernel(const std::string &name) = 0;
  virtual std::unique_ptr<KernelTemplate> make_new_kernel_template(
      const std::string &name) = 0;
  virtual std::unique_ptr<Field> make_new_field(const std::string &name) = 0;
  inline void mark_corrupted() {
    is_corrupted_ = true;
  }
  std::unordered_map<std::string, CompiledGraph> graphs_;

 private:
  bool is_corrupted_{false};
  std::unordered_map<std::string, std::unique_ptr<Kernel>> loaded_kernels_;
  std::unordered_map<std::string, std::unique_ptr<KernelTemplate>>
      loaded_kernel_templates_;
  std::unordered_map<std::string, std::unique_ptr<Field>> loaded_fields_;
};

}  // namespace aot
}  // namespace taichi::lang
