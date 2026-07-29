#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "taichi/common/serialization.h"

namespace taichi::lang {

class IRNode;
class Kernel;

struct GraphKernelResourceEffect {
  // "argument", "snode", or "opaque". Argument identities remain numeric
  // until a Graph dispatch maps them to its symbolic runtime names.
  std::string resource_kind{"opaque"};
  std::vector<int> arg_id;
  int snode_tree_id{-1};
  int snode_id{-1};
  bool is_grad{false};
  // "read", "write", "read_write", "atomic", or "opaque".
  std::string access{"opaque"};

  TI_IO_DEF(resource_kind,
            arg_id,
            snode_tree_id,
            snode_id,
            is_grad,
            access);
};

struct GraphKernelIterationDomain {
  // "unknown", "constant_range", "external_tensor", or "scalar_argument".
  std::string kind{"unknown"};
  std::vector<int> arg_id;
  int axis{-1};
  std::int64_t begin{0};
  std::int64_t end{0};

  TI_IO_DEF(kind, arg_id, axis, begin, end);
};

struct GraphKernelMetadata {
  static constexpr std::uint32_t kVersion = 1;

  std::uint32_t version{kVersion};
  bool available{false};
  bool opaque{true};
  bool elementwise{false};
  bool synchronization{false};
  GraphKernelIterationDomain iteration_domain;
  std::vector<GraphKernelResourceEffect> effects;
  std::vector<std::string> side_effects;
  std::string blocker{"metadata_unavailable"};

  TI_IO_DEF(version,
            available,
            opaque,
            elementwise,
            synchronization,
            iteration_domain,
            effects,
            side_effects,
            blocker);
};

// Conservatively describes a kernel at the last pre-offload compiler stage.
// The result is safe to serialize with compiled-kernel cache data, but is not
// part of AOT CGraph v1.
GraphKernelMetadata analyze_graph_kernel_metadata(IRNode *root,
                                                  const Kernel *kernel);

}  // namespace taichi::lang
