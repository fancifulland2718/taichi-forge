#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "taichi/rhi/cuda/primitives/graph_ptx.h"

namespace taichi::lang::cuda {

// Flat physical dispatch boundaries. The initial condition is followed by
// ordered leaf regions and a trailing outer condition; payloads already contain
// their selected tail conditions. Predicates/gates are borrowed device storage.
struct NestedPredicateGraphLayout {
  struct Inner {
    std::uintptr_t predicate{0};
    std::uintptr_t gate{0};
    std::uint64_t iterations{0};
    std::size_t condition_begin{0};
    std::size_t body_begin{0};
    std::size_t end{0};
  };
  std::uintptr_t predicate{0};
  std::uintptr_t gate{0};
  std::uint64_t iterations{0};
  std::size_t initial_dispatches{0};
  std::size_t dispatches{0};
  std::vector<Inner> inners;
};

// Program-independent cold construction, with no JIT compilation or submission
// ownership. The enclosing owner supplies its capture lock and stream, prepares
// immutable dispatch packets, and retains borrowed storage through completion.
// It validates the alias/effect contracts of those predicates and gates.
// Retire the executable before release(); release(false) abandons unsafe device
// resources. The destructor never calls the backend.
class NestedPredicateGraphControls {
 public:
  // A null node list requests an ordinary launch; otherwise capture every
  // payload task as device-updatable and return its device node handle.
  using Capture =
      std::function<std::uint32_t(std::size_t, std::vector<void *> *)>;

  NestedPredicateGraphControls();
  ~NestedPredicateGraphControls();
  NestedPredicateGraphControls(const NestedPredicateGraphControls &) = delete;
  NestedPredicateGraphControls &operator=(const NestedPredicateGraphControls &) =
      delete;
  std::uint32_t initialize(NestedPredicateGraphLayout layout,
                           bool diagnostics_enabled);
  std::uint32_t capture(void *stream, const Capture &payload);
  std::uint32_t finalize_nodes();
  // Only after instantiating and uploading the captured graph: device node
  // handles and child-control addresses are not portable artifacts.
  // Fixed execution can discard cold tables; JIT retains them for telemetry.
  std::uint32_t upload(bool retain_host_tables = true);
  void release(bool backend_safe);
  std::uint64_t device_bytes() const;

  // Cold binding data and opt-in telemetry, never interpreted during replay.
  CudaGraphPredicateGroupControl *device_controls{nullptr};
  std::uintptr_t *device_nodes{nullptr};
  std::vector<CudaGraphPredicateGroupControl> host_controls;
  std::vector<std::uintptr_t> host_nodes;

 private:
  struct BuildState;
  enum class Phase {
    empty, initialized, capturing, captured, finalized, uploaded
  };
  std::unique_ptr<BuildState> build_;
  std::uint64_t device_bytes_{0};
  Phase phase_{Phase::empty};
};

}  // namespace taichi::lang::cuda
