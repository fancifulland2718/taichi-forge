#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "taichi/common/core.h"
#include "taichi/runtime/runtime_client.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {
class Program;
class Ndarray;
class Texture;
class RuntimeCompletion;
namespace cuda {
class ExecutionDomain;
}
namespace storage {
class DenseStorageDescriptor;
}
namespace aot {
struct CompiledGraph;
}

// Private split-runtime bridge. Consumers use the existing C API only while
// holding this client's scope. The Program owns the backend, never this client.
class AotRuntimeClient final : public RuntimeClient {
 public:
  static std::shared_ptr<AotRuntimeClient> create(Program &program);
  ~AotRuntimeClient() override;
  void enter();
  void leave();
  void close();
  void retire() override;
  bool closed() const;
  std::uintptr_t handle() const;
  std::uintptr_t device_identity() const;
  Arch arch() const;
  // Opaque memory token, byte extent, scalar dtype, index/element shapes.
  // The cold registration retains a generation-qualified storage lease.
  using NdarrayImport = std::tuple<std::uintptr_t,
                                   std::uint64_t,
                                   std::string,
                                   std::vector<int>,
                                   std::vector<int>>;
  NdarrayImport import_ndarray(const Ndarray *array);
  NdarrayImport import_dense_view(
      const storage::DenseStorageDescriptor &descriptor);
  // Opaque native-resource token; retains the Program's texture storage lease.
  std::uintptr_t import_texture(const Texture *texture);
  // Name, selected physical Graph, dispatch index, graph-gated launch ABI.
  // This borrows existing compiled code and Field roots, never an artifact
  // reload or a new frontend compilation request.
  using JitKernelImport =
      std::tuple<std::string, const aot::CompiledGraph *, std::uint32_t, bool>;
  std::uintptr_t import_jit_kernels(
      const std::vector<JitKernelImport> &kernels);
  // Cold GGUI bridge. Hot drawing never resolves an opaque domain handle.
  std::shared_ptr<cuda::ExecutionDomain> execution_domain(
      std::uintptr_t handle);

 private:
  struct State;
  explicit AotRuntimeClient(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};
// Pin an exact public ticket without entering the borrower's submission scope.
TI_DLL_EXPORT RuntimeCompletion borrow_aot_completion(std::uintptr_t handle);
// Publish a shared exact prefix as a host-only AOT submission ticket. Its
// caller releases it with ti_destroy_submission_ext after consumer snapshot.
TI_DLL_EXPORT std::uintptr_t publish_aot_completion(
    const RuntimeCompletion &completion);
}  // namespace taichi::lang
