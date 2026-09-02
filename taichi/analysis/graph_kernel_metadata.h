#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "taichi/common/serialization.h"

namespace taichi::lang {

class IRNode;
class Kernel;

struct GraphKernelAccessFootprint {
  // "exact_pointwise", "affine", "stencil", or "opaque".
  std::string pattern{"opaque"};
  int iteration_rank{0};
  std::vector<std::vector<std::int64_t>> affine_coefficients;
  std::vector<std::int64_t> affine_offsets;
  std::vector<std::vector<std::int64_t>> halo;
  int contiguous_axis{-1};
  std::string reuse_class{"unknown"};

  TI_IO_DEF(pattern,
            iteration_rank,
            affine_coefficients,
            affine_offsets,
            halo,
            contiguous_axis,
            reuse_class);
};

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
  GraphKernelAccessFootprint footprint;

  TI_IO_DEF(resource_kind,
            arg_id,
            snode_tree_id,
            snode_id,
            is_grad,
            access,
            footprint);
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
  static constexpr std::uint32_t kVersion = 3;

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
