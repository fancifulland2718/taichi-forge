#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/aot/graph_data.h"

namespace taichi::lang {

class CompileConfig;
class Kernel;

struct GraphTwoMapSource {
  Kernel *kernel{nullptr};
  const std::vector<aot::Arg> *symbolic_args{nullptr};
  const GraphKernelMetadata *metadata{nullptr};
};

struct GraphTwoMapComposition {
  std::unique_ptr<Kernel> kernel;
  std::vector<aot::Arg> symbolic_args;
};

// Compose exactly two compiler-proven pointwise maps into one pre-offload
// range-for kernel. The contract is deliberately internal and fail-closed:
// unsupported argument paths, loop layouts, or effects return std::nullopt.
std::optional<GraphTwoMapComposition> compose_graph_two_map_kernel(
    const CompileConfig &config,
    const GraphTwoMapSource &first,
    const GraphTwoMapSource &second);

}  // namespace taichi::lang
