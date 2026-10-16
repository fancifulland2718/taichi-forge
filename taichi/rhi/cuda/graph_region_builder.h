#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>
#include "taichi/rhi/cuda/cuda_types.h"

namespace taichi::lang::cuda {

class GraphBuildError : public std::runtime_error {
 public:
  GraphBuildError(std::uint32_t code, const char *stage);
  std::uint32_t code() const {
    return code_;
  }
  const char *stage() const {
    return stage_;
  }

 private:
  std::uint32_t code_;
  const char *stage_;
};

// Cold assembly shared by live JIT and loaded-code consumers. The caller owns
// the root graph, stream, arguments and device storage, holds the CUDA context
// and submission guards, and discards the root if assembly fails. Conditional
// bodies are borrowed from the root; they must never be destroyed separately.
// No kernel compilation, allocation, execution or graph instantiation occurs.
class GraphRegionBuilder {
 public:
  using Dependencies = std::vector<void *>;
  using Recorder = std::function<void(void *)>;
  struct While {
    void *node;
    CUgraph body;
  };
  enum class BranchKind : std::uint32_t { if_else = 0, indexed = 2 };
  struct Branch {
    void *node;
    // Borrowed from the root, like While::body. Never destroy individually.
    std::vector<CUgraph> bodies;
  };

  static bool available();
  GraphRegionBuilder(CUgraph root, void *stream);

  // Append commands to an existing graph (including a conditional body) and
  // return the stream's dependency frontier, rather than assuming one node.
  // On failure, stream capture ends but graph ownership stays with the caller.
  Dependencies capture(CUgraph destination,
                       const Dependencies &dependencies,
                       const Recorder &record) const;
  std::uint64_t conditional_handle() const;
  While add_while(CUgraph destination,
                  const Dependencies &dependencies,
                  std::uint64_t handle) const;
  Branch add_branch(CUgraph destination,
                    const Dependencies &dependencies,
                    std::uint64_t handle,
                    BranchKind kind,
                    std::size_t body_count) const;

 private:
  Branch add_conditional(CUgraph destination,
                         const Dependencies &dependencies,
                         std::uint64_t handle,
                         std::uint32_t type,
                         std::uint32_t body_count) const;
  CUgraph root_;
  void *stream_;
  void *context_{nullptr};
};
}  // namespace taichi::lang::cuda
