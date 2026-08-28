#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {

class CompileConfig;
class Kernel;

struct GraphMapSource {
  Kernel *kernel{nullptr};
  const std::vector<aot::Arg> *symbolic_args{nullptr};
  const GraphKernelMetadata *metadata{nullptr};
};

struct GraphMapComposition {
  std::unique_ptr<Kernel> kernel;
  std::vector<aot::Arg> symbolic_args;
};

// Compose between two and four compiler-proven pointwise maps into one
// pre-offload range-for kernel. The contract is deliberately internal and
// fail-closed: unsupported argument paths, loop layouts, or effects return
// std::nullopt.
std::optional<GraphMapComposition> compose_graph_map_kernels(
    const CompileConfig &config,
    const std::vector<GraphMapSource> &sources);

}  // namespace taichi::lang
