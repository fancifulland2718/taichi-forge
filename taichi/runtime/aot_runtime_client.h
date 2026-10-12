#pragma once

#include <cstdint>
#include <memory>
#include "taichi/runtime/runtime_client.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {
class Program;

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

 private:
  struct State;
  explicit AotRuntimeClient(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang
