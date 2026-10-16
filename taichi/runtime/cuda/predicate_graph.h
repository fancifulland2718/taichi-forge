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

// Stateless masked recording over the same physical layout. Empty inners
// describe a flat while; its initial condition may be owned by the caller.
// Initial dispatches are ordinary code, outer payloads use layout.gate, and
// inner bodies use their own gate. Nested inner predicates continue if nonzero.
// The caller prepares the latch module before capture, owns the active capture
// transaction and retains all gates/packets until its executable retires.
// These functions never allocate device storage, instantiate or submit graphs.
using MaskedGraphCapture = std::function<void(std::size_t)>;
void capture_masked_while_graph(const NestedPredicateGraphLayout &layout,
                                bool continue_while_nonzero,
                                void *stream,
                                const MaskedGraphCapture &payload);

struct MaskedBranchGraphLayout {
  std::uintptr_t predicate{0};
  std::uintptr_t gate{0};
  std::uint32_t conditional_type{0};  // 0: if, 2: switch.
  std::uint32_t branch_count{0};
  std::uint32_t default_branch{~std::uint32_t{0}};
  std::size_t dispatches{0};
};

// Branch packets test their one-based branch ordinal against the latched gate.
// The owner validates branch-to-packet mapping and predicate alias/effects.
void capture_masked_branch_graph(const MaskedBranchGraphLayout &layout,
                                 void *stream,
                                 const MaskedGraphCapture &payload);

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
