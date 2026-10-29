#pragma once

#include <any>
#include <string>
#include <vector>

#include "taichi/runtime/gfx/aot_utils.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/aot/module_builder.h"
#include "taichi/aot/module_loader.h"
#include "taichi/common/virtual_dir.h"

namespace taichi::lang {
namespace gfx {

struct TI_DLL_EXPORT AotModuleParams {
  std::string module_path{};
  const io::VirtualDir *dir{nullptr};
  // Owned sources support deferred shader reads. The legacy borrowed `dir`
  // path remains eager, so its lifetime need only cover module construction.
  std::shared_ptr<const io::VirtualDir> source;
  GfxRuntime *runtime{nullptr};

  AotModuleParams() = default;

  [[deprecated]] AotModuleParams(const std::string &path, GfxRuntime *rt)
      : module_path(path), runtime(rt) {
  }
  AotModuleParams(const io::VirtualDir *dir, GfxRuntime *rt)
      : dir(dir), runtime(rt) {
  }
};

TI_DLL_EXPORT std::unique_ptr<aot::Module> make_aot_module(
    std::any mod_params,
    Arch device_api_backend);

// Called after capability admission and before looking up any kernel/Graph.
// The module then owns independent roots and its execution registrations.
TI_DLL_EXPORT void allocate_aot_root_buffers(aot::Module *module);

// Empty, private state reconstructed from the loaded module's validated tree
// recipes. Kernel views retain its executable pipelines and have immutable
// instance-local resource bindings; they never rewrite module bindings.
// The Runtime must outlive the module and all its instances. Retirement uses
// the Runtime's existing cold synchronized resource release.
class TI_DLL_EXPORT AotSNodeInstance {
 public:
  virtual ~AotSNodeInstance() = default;
  virtual KernelImpl *get_kernel(const std::string &name) const = 0;
  // Artifact-local tree -> runtime tree; -1 for trees outside the closure.
  virtual const std::vector<int> &root_bindings() const = 0;
};

TI_DLL_EXPORT std::unique_ptr<AotSNodeInstance> make_aot_snode_instance(
    std::shared_ptr<aot::Module> code,
    const std::vector<std::string> &entries);

}  // namespace gfx
}  // namespace taichi::lang
