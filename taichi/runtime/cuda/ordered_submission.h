#pragma once

#include <cstddef>
#include <memory>

namespace taichi::lang {
class NativeOperation;
class RuntimeFaultDomain;
class BackendRuntimeError;
namespace cuda {

// Completion ownership for host-invoked, stream-ordered native operations.
// Caller holds the CUDA submission/context guards. No provider or Program API.
class OrderedSubmissions {
 public:
  OrderedSubmissions(std::size_t operations,
                     std::shared_ptr<RuntimeFaultDomain> fault_domain);
  ~OrderedSubmissions();
  void begin();
  void run(NativeOperation &operation);
  void finish();
  void finish_after_failure(
      const BackendRuntimeError *error = nullptr) noexcept;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace cuda
}  // namespace taichi::lang
