#include "taichi/program/graph_builder.h"
#include "taichi/program/graph_kernel_composer.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi::lang {
aot::CompiledDispatch Dispatch::compile_dispatch() const {
  aot::CompiledDispatch dispatch;
  dispatch.kernel_name = kernel_->get_name();
  dispatch.symbolic_args = symbolic_args_;
  dispatch.indirect_dispatch_arg = indirect_dispatch_arg_;
  dispatch.ti_kernel = kernel_;
  dispatch.compiled_kernel = nullptr;
  const auto &compiled = kernel_->program->compile_kernel(
      kernel_->program->compile_config(),
      kernel_->program->get_device_caps(), *kernel_);
  dispatch.graph_metadata = compiled.graph_metadata();
  dispatch.compiled_task_count = static_cast<std::uint32_t>(
      std::min<std::size_t>(compiled.task_count(),
                            std::numeric_limits<std::uint32_t>::max()));
  TI_ERROR_IF(
      indirect_dispatch_arg_.has_value() &&
          dispatch.compiled_task_count != 1,
      "Graph indirect dispatch kernel {} must compile to exactly one task, "
      "but compiled to {} tasks",
      dispatch.kernel_name, dispatch.compiled_task_count);
  dispatch.source_dispatches.push_back(
      {dispatch.kernel_name, dispatch.symbolic_args, dispatch.graph_metadata});
  dispatch.snode_tree_dependencies =
      kernel_->program->snapshot_snode_tree_dependencies(
          compiled.snode_tree_ids());
  return dispatch;
}

void Dispatch::compile(
    std::vector<aot::CompiledDispatch> &compiled_dispatches) {
  compiled_dispatches.push_back(compile_dispatch());
}

void Sequential::compile(
    std::vector<aot::CompiledDispatch> &compiled_dispatches) {
  Dispatch *pending_dispatch = nullptr;
  std::optional<aot::CompiledDispatch> pending_compiled;
  auto flush_pending = [&]() {
    if (pending_compiled) {
      compiled_dispatches.push_back(std::move(*pending_compiled));
      pending_compiled.reset();
      pending_dispatch = nullptr;
    }
  };

  for (Node *n : sequence_) {
    auto *dispatch = dynamic_cast<Dispatch *>(n);
    if (dispatch == nullptr) {
      flush_pending();
      n->compile(compiled_dispatches);
      continue;
    }
    auto compiled = dispatch->compile_dispatch();
    if (dispatch->is_indirect()) {
      flush_pending();
      compiled_dispatches.push_back(std::move(compiled));
      continue;
    }
    if (pending_dispatch != nullptr) {
      if (owning_graph_->two_map_composer_enabled()) {
        auto composed = owning_graph_->try_compose_two_maps(
            *pending_dispatch, *pending_compiled, *dispatch, compiled);
        if (composed) {
          compiled_dispatches.push_back(std::move(*composed));
          pending_dispatch = nullptr;
          pending_compiled.reset();
          continue;
        }
      }
      flush_pending();
    }
    pending_dispatch = dispatch;
    pending_compiled = std::move(compiled);
  }
  flush_pending();
}

void Sequential::append(Node *node) {
  sequence_.push_back(node);
}

void Sequential::dispatch(Kernel *kernel, const std::vector<aot::Arg> &args) {
  Node *n = owning_graph_->new_dispatch_node(kernel, args);
  sequence_.push_back(n);
}

void Sequential::dispatch_indirect(Kernel *kernel,
                                   const std::vector<aot::Arg> &args,
                                   const aot::Arg &dispatch_packet) {
  Node *n = owning_graph_->new_indirect_dispatch_node(
      kernel, args, dispatch_packet);
  sequence_.push_back(n);
}

GraphBuilder::GraphBuilder() {
  seq_ = std::make_unique<Sequential>(this);
}

GraphBuilder::~GraphBuilder() = default;

Node *GraphBuilder::new_dispatch_node(Kernel *kernel,
                                      const std::vector<aot::Arg> &args) {
  for (const auto &arg : args) {
    register_arg(arg);
  }
  all_nodes_.push_back(std::make_unique<Dispatch>(kernel, args));
  return all_nodes_.back().get();
}

Node *GraphBuilder::new_indirect_dispatch_node(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &dispatch_packet) {
  TI_ERROR_IF(dispatch_packet.tag != aot::ArgKind::kNdarray ||
                  dispatch_packet.dtype_id != PrimitiveTypeID::u32 ||
                  dispatch_packet.field_dim != 1 ||
                  !dispatch_packet.element_shape.empty(),
              "Graph indirect dispatch packet {} must be a one-dimensional "
              "scalar u32 ndarray",
              dispatch_packet.name);
  for (const auto &arg : args) {
    register_arg(arg);
  }
  register_arg(dispatch_packet);
  all_nodes_.push_back(
      std::make_unique<Dispatch>(kernel, args, dispatch_packet));
  return all_nodes_.back().get();
}

void GraphBuilder::register_arg(const aot::Arg &arg) {
  if (all_args_.find(arg.name) != all_args_.end()) {
    TI_ERROR_IF(all_args_[arg.name] != arg,
                "An arg with name {} already exists!", arg.name);
  } else {
    all_args_[arg.name] = arg;
  }
}

Sequential *GraphBuilder::new_sequential_node() {
  all_nodes_.push_back(std::make_unique<Sequential>(this));
  return static_cast<Sequential *>(all_nodes_.back().get());
}

std::unique_ptr<aot::CompiledGraph> GraphBuilder::compile() {
  std::vector<aot::CompiledDispatch> dispatches;
  seq()->compile(dispatches);
  aot::CompiledGraph graph{dispatches, all_args_};
  for (auto &kernel : composed_kernels_) {
    graph.owned_jit_kernels.emplace_back(std::move(kernel));
  }
  return std::make_unique<aot::CompiledGraph>(std::move(graph));
}

Sequential *GraphBuilder::seq() const {
  return seq_.get();
}

void GraphBuilder::dispatch(Kernel *kernel, const std::vector<aot::Arg> &args) {
  seq()->dispatch(kernel, args);
}

void GraphBuilder::dispatch_indirect(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &dispatch_packet) {
  seq()->dispatch_indirect(kernel, args, dispatch_packet);
}

std::optional<aot::CompiledDispatch> GraphBuilder::try_compose_two_maps(
    const Dispatch &first,
    const aot::CompiledDispatch &first_compiled,
    const Dispatch &second,
    const aot::CompiledDispatch &second_compiled) {
  auto composition = compose_graph_two_map_kernel(
      first.kernel()->program->compile_config(),
      {first.kernel(), &first.symbolic_args(), &first_compiled.graph_metadata},
      {second.kernel(), &second.symbolic_args(),
       &second_compiled.graph_metadata});
  if (!composition) {
    return std::nullopt;
  }

  Dispatch composed_dispatch(composition->kernel.get(),
                             composition->symbolic_args);
  auto compiled = composed_dispatch.compile_dispatch();
  const auto source_task_count =
      static_cast<std::uint64_t>(first_compiled.compiled_task_count) +
      static_cast<std::uint64_t>(second_compiled.compiled_task_count);
  if (compiled.compiled_task_count >= source_task_count ||
      compiled.graph_metadata.opaque || !compiled.graph_metadata.elementwise) {
    return std::nullopt;
  }
  compiled.source_dispatches.clear();
  compiled.source_dispatches.insert(compiled.source_dispatches.end(),
                                    first_compiled.source_dispatches.begin(),
                                    first_compiled.source_dispatches.end());
  compiled.source_dispatches.insert(compiled.source_dispatches.end(),
                                    second_compiled.source_dispatches.begin(),
                                    second_compiled.source_dispatches.end());
  composed_kernels_.push_back(std::move(composition->kernel));
  return compiled;
}

}  // namespace taichi::lang
