#include "taichi/program/graph_builder.h"
#include "taichi/program/graph_kernel_composer.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {

class CudaSparseSpmvDispatch final : public Node {
 public:
  CudaSparseSpmvDispatch(SparseMatrix *matrix,
                         Program *program,
                         aot::Arg input,
                         aot::Arg output)
      : matrix_(matrix),
        program_(program),
        input_(std::move(input)),
        output_(std::move(output)) {
  }

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override {
    aot::CompiledDispatch dispatch;
    dispatch.kernel_name = "cuda_cusparse_spmv_f32";
    dispatch.symbolic_args = {input_, output_};
    dispatch.cuda_sparse_spmv_dispatch =
        aot::CudaSparseSpmvDispatchMetadata{matrix_, program_, input_, output_};
    // Preserve one logical source item for Forge pipeline attribution. The
    // default metadata is intentionally opaque: this provider command cannot
    // participate in kernel composition or AOT serialization.
    dispatch.source_dispatches.push_back(
        {dispatch.kernel_name, "", dispatch.symbolic_args, {}});
    dispatch.compiled_task_count = 0;
    compiled_dispatches.push_back(std::move(dispatch));
  }

 private:
  SparseMatrix *matrix_{nullptr};
  Program *program_{nullptr};
  aot::Arg input_;
  aot::Arg output_;
};

void validate_cuda_sparse_spmv_args(SparseMatrix *matrix,
                                    Program *program,
                                    const aot::Arg &input,
                                    const aot::Arg &output) {
  TI_ERROR_IF(matrix == nullptr || program == nullptr,
              "CUDA sparse SpMV Graph proof requires a live matrix and Program");
  TI_ERROR_IF(input.tag != aot::ArgKind::kNdarray ||
                  output.tag != aot::ArgKind::kNdarray ||
                  input.dtype_id != PrimitiveTypeID::f32 ||
                  output.dtype_id != PrimitiveTypeID::f32 ||
                  input.field_dim != 1 || output.field_dim != 1 ||
                  !input.element_shape.empty() || !output.element_shape.empty(),
              "CUDA sparse SpMV Graph proof requires scalar f32 rank-1 "
              "ndarray bindings");
  TI_ERROR_IF(input.name == output.name,
              "CUDA sparse SpMV Graph proof input and output must differ");
  TI_ERROR_IF(dynamic_cast<CuSparseMatrix *>(matrix) == nullptr &&
                  dynamic_cast<CuSparseBsrMatrix *>(matrix) == nullptr,
              "CUDA sparse SpMV Graph proof supports cuSPARSE CSR/BSR only");
  TI_ERROR_IF(matrix->get_data_type() != PrimitiveType::f32,
              "CUDA sparse SpMV Graph proof supports f32 matrices only");
}

}  // namespace

aot::CompiledDispatch Dispatch::compile_dispatch() const {
  aot::CompiledDispatch dispatch;
  dispatch.kernel_name = kernel_->get_name();
  dispatch.dispatch_label = dispatch_label_;
  dispatch.symbolic_args = symbolic_args_;
  dispatch.indirect_dispatch_arg = indirect_dispatch_arg_;
  dispatch.cuda_bounded_dispatch = cuda_bounded_dispatch_;
  dispatch.cpu_bounded_dispatch = cpu_bounded_dispatch_;
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
  TI_ERROR_IF(cuda_bounded_dispatch_.has_value() &&
                  dispatch.compiled_task_count != 1,
              "CUDA bounded Graph kernel {} must compile to exactly one task, "
              "but compiled to {} tasks",
              dispatch.kernel_name, dispatch.compiled_task_count);
  TI_ERROR_IF(cpu_bounded_dispatch_.has_value() &&
                  dispatch.compiled_task_count != 1,
              "CPU bounded Graph kernel {} must compile to exactly one task, "
              "but compiled to {} tasks",
              dispatch.kernel_name, dispatch.compiled_task_count);
  dispatch.source_dispatches.push_back(
      {dispatch.kernel_name, dispatch.dispatch_label, dispatch.symbolic_args,
       dispatch.graph_metadata});
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
    if (dispatch->is_indirect() || dispatch->is_cuda_bounded() ||
        dispatch->is_cpu_bounded()) {
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

void Sequential::dispatch(Kernel *kernel,
                          const std::vector<aot::Arg> &args,
                          const std::string &dispatch_label) {
  Node *n =
      owning_graph_->new_dispatch_node(kernel, args, dispatch_label);
  sequence_.push_back(n);
}

void Sequential::dispatch_indirect(Kernel *kernel,
                                   const std::vector<aot::Arg> &args,
                                   const aot::Arg &dispatch_packet,
                                   const std::string &dispatch_label) {
  Node *n = owning_graph_->new_indirect_dispatch_node(
      kernel, args, dispatch_packet, dispatch_label);
  sequence_.push_back(n);
}

void Sequential::dispatch_cuda_bounded(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &extent,
    std::uint32_t capacity,
    std::uint32_t block_dim,
    bool adaptive_grid,
    bool grouped_update,
    const std::string &dispatch_label) {
  Node *n = owning_graph_->new_cuda_bounded_dispatch_node(
      kernel, args, extent, capacity, block_dim, adaptive_grid, grouped_update,
      dispatch_label);
  sequence_.push_back(n);
}

void Sequential::dispatch_cpu_bounded(Kernel *kernel,
                                      const std::vector<aot::Arg> &args,
                                      const aot::Arg &extent,
                                      std::uint32_t capacity,
                                      const std::string &dispatch_label) {
  Node *n = owning_graph_->new_cpu_bounded_dispatch_node(
      kernel, args, extent, capacity, dispatch_label);
  sequence_.push_back(n);
}

GraphBuilder::GraphBuilder() {
  seq_ = std::make_unique<Sequential>(this);
}

GraphBuilder::~GraphBuilder() = default;

Node *GraphBuilder::new_dispatch_node(Kernel *kernel,
                                      const std::vector<aot::Arg> &args,
                                      const std::string &dispatch_label) {
  validate_dispatch_label(dispatch_label);
  for (const auto &arg : args) {
    register_arg(arg);
  }
  all_nodes_.push_back(
      std::make_unique<Dispatch>(kernel, args, std::nullopt, std::nullopt,
                                 dispatch_label));
  return all_nodes_.back().get();
}

Node *GraphBuilder::new_indirect_dispatch_node(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &dispatch_packet,
    const std::string &dispatch_label) {
  validate_dispatch_label(dispatch_label);
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
      std::make_unique<Dispatch>(kernel, args, dispatch_packet, std::nullopt,
                                 dispatch_label));
  return all_nodes_.back().get();
}

Node *GraphBuilder::new_cuda_bounded_dispatch_node(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &extent,
    std::uint32_t capacity,
    std::uint32_t block_dim,
    bool adaptive_grid,
    bool grouped_update,
    const std::string &dispatch_label) {
  validate_dispatch_label(dispatch_label);
  TI_ERROR_IF(extent.tag != aot::ArgKind::kNdarray ||
                  extent.dtype_id != PrimitiveTypeID::i32 ||
                  extent.field_dim != 1 || !extent.element_shape.empty(),
              "CUDA bounded Graph extent {} must be a one-dimensional "
              "scalar i32 ndarray",
              extent.name);
  TI_ERROR_IF(capacity == 0 || capacity > 0x7fffffffu || block_dim == 0 ||
                  block_dim > 1024,
              "CUDA bounded Graph capacity/block are out of range");
  TI_ERROR_IF(grouped_update && !adaptive_grid,
              "CUDA grouped bounded update requires adaptive grid control");
  TI_ERROR_IF(std::find(args.begin(), args.end(), extent) == args.end(),
              "CUDA bounded Graph extent {} must also be a payload argument",
              extent.name);
  for (const auto &arg : args) {
    register_arg(arg);
  }
  aot::CudaBoundedDispatchMetadata metadata{
      extent, capacity, block_dim, adaptive_grid, grouped_update};
  all_nodes_.push_back(std::make_unique<Dispatch>(
      kernel, args, std::nullopt, std::move(metadata), dispatch_label));
  return all_nodes_.back().get();
}

Node *GraphBuilder::new_cpu_bounded_dispatch_node(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &extent,
    std::uint32_t capacity,
    const std::string &dispatch_label) {
  validate_dispatch_label(dispatch_label);
  TI_ERROR_IF(extent.tag != aot::ArgKind::kNdarray ||
                  extent.dtype_id != PrimitiveTypeID::i32 ||
                  extent.field_dim != 1 || !extent.element_shape.empty(),
              "CPU bounded Graph extent {} must be a one-dimensional "
              "scalar i32 ndarray",
              extent.name);
  TI_ERROR_IF(capacity == 0 || capacity > 0x7fffffffu,
              "CPU bounded Graph capacity is out of range");
  TI_ERROR_IF(std::find(args.begin(), args.end(), extent) == args.end(),
              "CPU bounded Graph extent {} must also be a payload argument",
              extent.name);
  for (const auto &arg : args) {
    register_arg(arg);
  }
  auto dispatch = std::make_unique<Dispatch>(
      kernel, args, std::nullopt, std::nullopt, dispatch_label);
  dispatch->set_cpu_bounded_dispatch({extent, capacity});
  all_nodes_.push_back(std::move(dispatch));
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

void GraphBuilder::dispatch(Kernel *kernel,
                            const std::vector<aot::Arg> &args,
                            const std::string &dispatch_label) {
  seq()->dispatch(kernel, args, dispatch_label);
}

void GraphBuilder::dispatch_indirect(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &dispatch_packet,
    const std::string &dispatch_label) {
  seq()->dispatch_indirect(kernel, args, dispatch_packet, dispatch_label);
}

void GraphBuilder::dispatch_cuda_bounded(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &extent,
    std::uint32_t capacity,
    std::uint32_t block_dim,
    bool adaptive_grid,
    bool grouped_update,
    const std::string &dispatch_label) {
  seq()->dispatch_cuda_bounded(kernel, args, extent, capacity, block_dim,
                               adaptive_grid, grouped_update, dispatch_label);
}

void GraphBuilder::dispatch_cpu_bounded(
    Kernel *kernel,
    const std::vector<aot::Arg> &args,
    const aot::Arg &extent,
    std::uint32_t capacity,
    const std::string &dispatch_label) {
  seq()->dispatch_cpu_bounded(kernel, args, extent, capacity,
                              dispatch_label);
}

void GraphBuilder::dispatch_cuda_sparse_spmv(SparseMatrix *matrix,
                                             Program *program,
                                             const aot::Arg &input,
                                             const aot::Arg &output) {
  validate_cuda_sparse_spmv_args(matrix, program, input, output);
  register_arg(input);
  register_arg(output);
  all_nodes_.push_back(std::make_unique<CudaSparseSpmvDispatch>(
      matrix, program, input, output));
  seq()->append(all_nodes_.back().get());
}

std::optional<aot::CompiledDispatch> GraphBuilder::try_compose_two_maps(
    const Dispatch &first,
    const aot::CompiledDispatch &first_compiled,
    const Dispatch &second,
    const aot::CompiledDispatch &second_compiled) {
  if (!first.dispatch_label().empty() ||
      !second.dispatch_label().empty()) {
    return std::nullopt;
  }
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
