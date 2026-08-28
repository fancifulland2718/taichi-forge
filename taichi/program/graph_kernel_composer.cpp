#include "taichi/program/graph_kernel_composer.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/kernel.h"

namespace taichi::lang {
namespace {

bool qualified_metadata(const GraphKernelMetadata &metadata) {
  if (!metadata.available || metadata.opaque || !metadata.elementwise ||
      metadata.synchronization || !metadata.side_effects.empty()) {
    return false;
  }
  for (const auto &effect : metadata.effects) {
    if (effect.access == "atomic" || effect.access == "opaque" ||
        effect.resource_kind == "opaque") {
      return false;
    }
  }
  return true;
}

std::optional<std::string> canonical_domain(
    const GraphKernelMetadata &metadata,
    const std::vector<aot::Arg> &args) {
  const auto &domain = metadata.iteration_domain;
  if (domain.kind == "constant_range") {
    return fmt::format("constant:{}:{}", domain.begin, domain.end);
  }
  if (domain.kind != "external_tensor" &&
      domain.kind != "scalar_argument") {
    return std::nullopt;
  }
  if (domain.arg_id.size() != 1 || domain.arg_id[0] < 0 ||
      domain.arg_id[0] >= static_cast<int>(args.size())) {
    return std::nullopt;
  }
  return fmt::format("{}:{}:{}", domain.kind, args[domain.arg_id[0]].name,
                     domain.axis);
}

bool qualified_source(const GraphMapSource &source) {
  if (source.kernel == nullptr || source.symbolic_args == nullptr ||
      source.metadata == nullptr || source.kernel->definition_retired() ||
      source.kernel->autodiff_mode != AutodiffMode::kNone ||
      !source.kernel->rets.empty() ||
      !qualified_metadata(*source.metadata)) {
    return false;
  }
  const auto &args = *source.symbolic_args;
  if (args.size() != source.kernel->parameter_list.size()) {
    return false;
  }
  for (std::size_t index = 0; index < args.size(); ++index) {
    if (args[index].tag != aot::ArgKind::kScalar &&
        args[index].tag != aot::ArgKind::kNdarray) {
      return false;
    }
    const auto parameter = source.kernel->nested_parameters.find(
        std::vector<int>{static_cast<int>(index)});
    if (parameter == source.kernel->nested_parameters.end() ||
        parameter->second.is_argpack) {
      return false;
    }
  }
  return source.kernel->nested_parameters.size() == args.size();
}

bool build_argument_union(
    const std::vector<GraphMapSource> &sources,
    std::vector<aot::Arg> *combined_args,
    std::vector<Callable::Parameter> *combined_parameters,
    std::vector<std::vector<int>> *remaps) {
  std::unordered_map<std::string, int> indices;
  auto append = [&](const GraphMapSource &source,
                    std::vector<int> *remap) {
    const auto &args = *source.symbolic_args;
    remap->reserve(args.size());
    for (std::size_t index = 0; index < args.size(); ++index) {
      const auto &arg = args[index];
      const auto &parameter = source.kernel->parameter_list[index];
      const auto found = indices.find(arg.name);
      if (found != indices.end()) {
        const int combined_index = found->second;
        if ((*combined_args)[combined_index] != arg ||
            !((*combined_parameters)[combined_index] == parameter)) {
          return false;
        }
        remap->push_back(combined_index);
        continue;
      }
      const int combined_index = static_cast<int>(combined_args->size());
      indices.emplace(arg.name, combined_index);
      combined_args->push_back(arg);
      combined_parameters->push_back(parameter);
      remap->push_back(combined_index);
    }
    return true;
  };
  remaps->reserve(sources.size());
  for (const auto &source : sources) {
    remaps->emplace_back();
    if (!append(source, &remaps->back())) {
      return false;
    }
  }
  return true;
}

class FlatArgumentRemapper final : public BasicStmtVisitor {
 public:
  explicit FlatArgumentRemapper(const std::vector<int> &remap)
      : remap_(remap) {
    allow_undefined_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(ArgLoadStmt *stmt) override {
    remap_path(&stmt->arg_id);
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    remap_path(&stmt->arg_id);
  }

  void visit(ExternalTensorBasePtrStmt *stmt) override {
    remap_path(&stmt->arg_id);
  }

  bool valid() const {
    return valid_;
  }

 private:
  void remap_path(std::vector<int> *path) {
    if (path->size() != 1 || (*path)[0] < 0 ||
        (*path)[0] >= static_cast<int>(remap_.size())) {
      valid_ = false;
      return;
    }
    (*path)[0] = remap_[(*path)[0]];
  }

  const std::vector<int> &remap_;
  bool valid_{true};
};

std::unique_ptr<IRNode> lower_to_preoffload(const CompileConfig &config,
                                            Kernel *kernel,
                                            const std::vector<int> &remap) {
  auto ir = irpass::analysis::clone(kernel->ir.get());
  irpass::compile_to_offloads(ir.get(), config, kernel,
                              /*verbose=*/false,
                              /*autodiff_mode=*/AutodiffMode::kNone,
                              /*ad_use_stack=*/true,
                              /*start_from_ast=*/kernel->ir_is_ast(),
                              /*graph_metadata=*/nullptr,
                              /*stop_before_offload=*/true);
  FlatArgumentRemapper remapper(remap);
  ir->accept(&remapper);
  return remapper.valid() ? std::move(ir) : nullptr;
}

struct SingleLoopIR {
  VecStatement setup;
  std::unique_ptr<RangeForStmt> loop;
};

std::optional<SingleLoopIR> split_single_loop(std::unique_ptr<IRNode> ir) {
  auto *block = ir ? ir->cast<Block>() : nullptr;
  if (block == nullptr || block->statements.empty() ||
      !block->statements.back()->is<RangeForStmt>()) {
    return std::nullopt;
  }
  for (std::size_t index = 0; index + 1 < block->statements.size(); ++index) {
    if (block->statements[index]->is_container_statement() ||
        block->statements[index]->has_global_side_effect()) {
      return std::nullopt;
    }
  }
  SingleLoopIR result;
  while (block->statements.size() > 1) {
    result.setup.push_back(block->extract(0));
  }
  auto loop = block->extract(0);
  result.loop.reset(loop.release()->as<RangeForStmt>());
  return result;
}

bool compatible_loops(const RangeForStmt &first, const RangeForStmt &second) {
  return !first.reversed && !second.reversed &&
         first.is_bit_vectorized == second.is_bit_vectorized &&
         first.num_cpu_threads == second.num_cpu_threads &&
         first.block_dim == second.block_dim &&
         first.strictly_serialized == second.strictly_serialized;
}

class LoopIndexRebaser final : public BasicStmtVisitor {
 public:
  LoopIndexRebaser(RangeForStmt *source, RangeForStmt *destination)
      : source_(source), destination_(destination) {
    allow_undefined_visitor = true;
  }

  using BasicStmtVisitor::visit;

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->loop == source_) {
      stmt->loop = destination_;
    }
  }

 private:
  RangeForStmt *source_{nullptr};
  RangeForStmt *destination_{nullptr};
};

std::unique_ptr<IRNode> fuse_preoffload_loops(
    std::vector<SingleLoopIR> loops) {
  if (loops.size() < 2 || loops.size() > 4) {
    return nullptr;
  }
  for (std::size_t index = 1; index < loops.size(); ++index) {
    if (!compatible_loops(*loops[0].loop, *loops[index].loop)) {
      return nullptr;
    }
  }
  for (std::size_t index = 1; index < loops.size(); ++index) {
    LoopIndexRebaser rebaser(loops[index].loop.get(), loops[0].loop.get());
    loops[index].loop->body->accept(&rebaser);
    while (!loops[index].loop->body->statements.empty()) {
      loops[0].loop->body->insert(loops[index].loop->body->extract(0));
    }
  }

  auto root = std::make_unique<Block>();
  for (auto &loop : loops) {
    root->insert(std::move(loop.setup));
  }
  root->insert(std::move(loops[0].loop));
  return root;
}

}  // namespace

std::optional<GraphMapComposition> compose_graph_map_kernels(
    const CompileConfig &config,
    const std::vector<GraphMapSource> &sources) {
  if (sources.size() < 2 || sources.size() > 4 || config.debug) {
    return std::nullopt;
  }
  const auto &first = sources.front();
  if (!qualified_source(first)) {
    return std::nullopt;
  }
  const auto first_domain =
      canonical_domain(*first.metadata, *first.symbolic_args);
  if (!first_domain) {
    return std::nullopt;
  }
  for (std::size_t index = 1; index < sources.size(); ++index) {
    const auto &source = sources[index];
    if (!qualified_source(source) ||
        first.kernel->program != source.kernel->program ||
        canonical_domain(*source.metadata, *source.symbolic_args) !=
            first_domain) {
      return std::nullopt;
    }
  }

  GraphMapComposition result;
  std::vector<Callable::Parameter> parameters;
  std::vector<std::vector<int>> remaps;
  if (!build_argument_union(sources, &result.symbolic_args, &parameters,
                            &remaps)) {
    return std::nullopt;
  }

  std::vector<SingleLoopIR> loops;
  loops.reserve(sources.size());
  for (std::size_t index = 0; index < sources.size(); ++index) {
    auto ir = lower_to_preoffload(config, sources[index].kernel, remaps[index]);
    auto loop = split_single_loop(std::move(ir));
    if (!loop) {
      return std::nullopt;
    }
    loops.push_back(std::move(*loop));
  }
  auto fused_ir = fuse_preoffload_loops(std::move(loops));
  if (!fused_ir) {
    return std::nullopt;
  }

  result.kernel = std::make_unique<Kernel>(
      *first.kernel->program, std::move(fused_ir),
      fmt::format("{}__graph_fused_map{}", first.kernel->get_name(),
                  sources.size()));
  result.kernel->parameter_list = std::move(parameters);
  for (std::size_t index = 0; index < result.kernel->parameter_list.size();
       ++index) {
    result.kernel->nested_parameters[{static_cast<int>(index)}] =
        result.kernel->parameter_list[index];
  }
  result.kernel->finalize_params();
  return result;
}

}  // namespace taichi::lang
