#include "taichi/program/graph_builder.h"
#include "taichi/program/graph_kernel_composer.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {

const Ndarray *cuda_sparse_spmv_ndarray(
    const aot::Arg &symbol,
    const std::unordered_map<std::string, aot::IValue> &args,
    int expected_size) {
  const auto value_it = args.find(symbol.name);
  if (value_it == args.end() || value_it->second.tag != aot::ArgKind::kNdarray) {
    return nullptr;
  }
  auto *array = reinterpret_cast<Ndarray *>(value_it->second.val);
  if (array == nullptr ||
      array->get_element_data_type() != PrimitiveType::f32 ||
      !array->get_element_shape().empty() ||
      array->get_nelement() != static_cast<std::size_t>(expected_size)) {
    return nullptr;
  }
  return array;
}

const Ndarray *cuda_sparse_spmm_ndarray(
    const aot::Arg &symbol,
    const std::unordered_map<std::string, aot::IValue> &args,
    Program &program,
    int expected_rows,
    int expected_columns) {
  const auto value_it = args.find(symbol.name);
  if (value_it == args.end() || value_it->second.tag != aot::ArgKind::kNdarray) {
    return nullptr;
  }
  auto *array = reinterpret_cast<Ndarray *>(value_it->second.val);
  if (array == nullptr || array->owning_program() != &program ||
      array->get_element_data_type() != PrimitiveType::f32 ||
      !array->get_element_shape().empty() ||
      array->shape != std::vector<int>({expected_rows, expected_columns})) {
    return nullptr;
  }
  return array;
}

Ndarray *cuda_cufft_ndarray(
    const aot::Arg &symbol,
    const std::unordered_map<std::string, aot::IValue> &args,
    Program &program,
    std::size_t expected_scalars) {
  const auto value_it = args.find(symbol.name);
  if (value_it == args.end() || value_it->second.tag != aot::ArgKind::kNdarray) {
    return nullptr;
  }
  auto *array = reinterpret_cast<Ndarray *>(value_it->second.val);
  if (array == nullptr || array->owning_program() != &program ||
      array->get_element_data_type() != PrimitiveType::f32 ||
      !array->get_element_shape().empty() ||
      array->get_nelement() != expected_scalars) {
    return nullptr;
  }
  return array;
}

class CudaSparseSpmvCaptureCommand final
    : public aot::CudaGraphCaptureCommand {
 public:
  CudaSparseSpmvCaptureCommand(SparseMatrix *matrix,
                               Program *program,
                               aot::Arg input,
                               aot::Arg output)
      : matrix_(matrix),
        program_(program),
        input_(std::move(input)),
        output_(std::move(output)) {
  }

  const char *kind() const override {
    return "cusparse_spmv_f32";
  }

  Program *program() const override {
    return program_;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (matrix_ == nullptr || program_ != &program ||
        matrix_->get_data_type() != PrimitiveType::f32 ||
        (dynamic_cast<CuSparseMatrix *>(matrix_) == nullptr &&
         dynamic_cast<CuSparseBsrMatrix *>(matrix_) == nullptr)) {
      return false;
    }
    const auto *input =
        cuda_sparse_spmv_ndarray(input_, args, matrix_->num_cols());
    const auto *output =
        cuda_sparse_spmv_ndarray(output_, args, matrix_->num_rows());
    return input != nullptr && output != nullptr &&
           input->get_device_allocation() != output->get_device_allocation();
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    record(args, program, nullptr);
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(matrix_ == nullptr || program_ != &program,
                "CUDA capture recipe provider generation is stale");
    TI_ERROR_IF(dynamic_cast<CuSparseMatrix *>(matrix_) == nullptr &&
                    dynamic_cast<CuSparseBsrMatrix *>(matrix_) == nullptr,
                "CUDA capture recipe provider is not cuSPARSE CSR/BSR");
    const auto *input =
        cuda_sparse_spmv_ndarray(input_, args, matrix_->num_cols());
    const auto *output =
        cuda_sparse_spmv_ndarray(output_, args, matrix_->num_rows());
    TI_ERROR_IF(input == nullptr,
                "CUDA cuSPARSE capture input {} must be an owning f32 ndarray "
                "of shape ({}) in the active Program",
                input_.name, matrix_->num_cols());
    TI_ERROR_IF(output == nullptr,
                "CUDA cuSPARSE capture output {} must be an owning f32 ndarray "
                "of shape ({}) in the active Program",
                output_.name, matrix_->num_rows());
    TI_ERROR_IF(input->get_device_allocation() ==
                    output->get_device_allocation(),
                "CUDA cuSPARSE capture input/output alias");
    const auto input_address =
        static_cast<std::size_t>(program.get_ndarray_data_ptr_as_int(input));
    const auto output_address =
        static_cast<std::size_t>(program.get_ndarray_data_ptr_as_int(output));
#if defined(TI_WITH_CUDA)
    auto capture_stream = reinterpret_cast<CUstream>(stream);
    if (auto *csr = dynamic_cast<CuSparseMatrix *>(matrix_)) {
      csr->spmv(input_address, output_address, capture_stream);
      return;
    }
    auto *bsr = dynamic_cast<CuSparseBsrMatrix *>(matrix_);
    TI_ASSERT(bsr != nullptr);
    bsr->spmv(input_address, output_address, capture_stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

 private:
  SparseMatrix *matrix_{nullptr};
  Program *program_{nullptr};
  aot::Arg input_;
  aot::Arg output_;
};

class CudaSparseSpmmCaptureCommand final
    : public aot::CudaGraphCaptureCommand {
 public:
  CudaSparseSpmmCaptureCommand(CuSparseMatrix *matrix,
                               Program *program,
                               aot::Arg input,
                               aot::Arg output,
                               int rhs_count,
                               int algorithm)
      : matrix_(matrix),
        program_(program),
        input_(std::move(input)),
        output_(std::move(output)),
        rhs_count_(rhs_count),
        algorithm_(algorithm) {
  }

  const char *kind() const override {
    return "cusparse_spmm_f32";
  }

  Program *program() const override {
    return program_;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (matrix_ == nullptr || program_ != &program || rhs_count_ < 2 ||
        (algorithm_ != 0 && algorithm_ != 1)) {
      return false;
    }
    const auto *input = cuda_sparse_spmm_ndarray(
        input_, args, program, matrix_->num_cols(), rhs_count_);
    const auto *output = cuda_sparse_spmm_ndarray(
        output_, args, program, matrix_->num_rows(), rhs_count_);
    return input != nullptr && output != nullptr &&
           input->get_device_allocation() != output->get_device_allocation();
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    record(args, program, nullptr);
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(matrix_ == nullptr || program_ != &program,
                "CUDA cuSPARSE SpMM capture recipe generation is stale");
    const auto *input = cuda_sparse_spmm_ndarray(
        input_, args, program, matrix_->num_cols(), rhs_count_);
    const auto *output = cuda_sparse_spmm_ndarray(
        output_, args, program, matrix_->num_rows(), rhs_count_);
    TI_ERROR_IF(input == nullptr || output == nullptr,
                "CUDA cuSPARSE SpMM capture requires owning compact f32 "
                "arrays with shapes ({}, {}) and ({}, {})",
                matrix_->num_cols(), rhs_count_, matrix_->num_rows(),
                rhs_count_);
    TI_ERROR_IF(input->get_device_allocation() ==
                    output->get_device_allocation(),
                "CUDA cuSPARSE SpMM capture input/output alias");
    matrix_->spmm(
        static_cast<std::size_t>(program.get_ndarray_data_ptr_as_int(input)),
        static_cast<std::size_t>(program.get_ndarray_data_ptr_as_int(output)),
        rhs_count_, algorithm_, reinterpret_cast<CUstream>(stream));
  }

 private:
  CuSparseMatrix *matrix_{nullptr};
  Program *program_{nullptr};
  aot::Arg input_;
  aot::Arg output_;
  int rhs_count_{0};
  int algorithm_{0};
};

class CudaSparseTriangularCaptureCommand final
    : public aot::CudaGraphCaptureCommand {
 public:
  CudaSparseTriangularCaptureCommand(CuSparseMatrix *matrix,
                                     Program *program,
                                     aot::Arg input,
                                     aot::Arg output,
                                     int rhs_count,
                                     int fill_mode,
                                     bool unit_diagonal,
                                     bool transpose)
      : matrix_(matrix),
        program_(program),
        input_(std::move(input)),
        output_(std::move(output)),
        rhs_count_(rhs_count),
        fill_mode_(fill_mode),
        unit_diagonal_(unit_diagonal),
        transpose_(transpose) {
  }

  const char *kind() const override {
    return rhs_count_ == 1 ? "cusparse_spsv_f32" : "cusparse_spsm_f32";
  }

  Program *program() const override {
    return program_;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (matrix_ == nullptr || program_ != &program ||
        matrix_->num_rows() != matrix_->num_cols() || rhs_count_ < 1 ||
        (fill_mode_ != 0 && fill_mode_ != 1)) {
      return false;
    }
    const Ndarray *input = nullptr;
    const Ndarray *output = nullptr;
    if (rhs_count_ == 1) {
      input = cuda_sparse_spmv_ndarray(input_, args, matrix_->num_rows());
      output = cuda_sparse_spmv_ndarray(output_, args, matrix_->num_rows());
    } else {
      input = cuda_sparse_spmm_ndarray(input_, args, program,
                                       matrix_->num_rows(), rhs_count_);
      output = cuda_sparse_spmm_ndarray(output_, args, program,
                                        matrix_->num_rows(), rhs_count_);
    }
    return input != nullptr && output != nullptr &&
           input->get_device_allocation() != output->get_device_allocation();
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    record(args, program, nullptr);
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(matrix_ == nullptr || program_ != &program,
                "CUDA cuSPARSE triangular capture recipe generation is "
                "stale");
    const Ndarray *input = nullptr;
    const Ndarray *output = nullptr;
    if (rhs_count_ == 1) {
      input = cuda_sparse_spmv_ndarray(input_, args, matrix_->num_rows());
      output = cuda_sparse_spmv_ndarray(output_, args, matrix_->num_rows());
    } else {
      input = cuda_sparse_spmm_ndarray(input_, args, program,
                                       matrix_->num_rows(), rhs_count_);
      output = cuda_sparse_spmm_ndarray(output_, args, program,
                                        matrix_->num_rows(), rhs_count_);
    }
    TI_ERROR_IF(input == nullptr || output == nullptr,
                "CUDA cuSPARSE triangular capture requires owning compact "
                "f32 input/output arrays for {} right-hand side(s)",
                rhs_count_);
    TI_ERROR_IF(input->get_device_allocation() ==
                    output->get_device_allocation(),
                "CUDA cuSPARSE triangular capture input/output alias");
    const auto input_address =
        static_cast<std::size_t>(program.get_ndarray_data_ptr_as_int(input));
    const auto output_address =
        static_cast<std::size_t>(program.get_ndarray_data_ptr_as_int(output));
    auto capture_stream = reinterpret_cast<CUstream>(stream);
    if (rhs_count_ == 1) {
      matrix_->spsv(input_address, output_address, fill_mode_, unit_diagonal_,
                    transpose_, capture_stream);
    } else {
      matrix_->spsm(input_address, output_address, rhs_count_, fill_mode_,
                    unit_diagonal_, transpose_, capture_stream);
    }
  }

 private:
  CuSparseMatrix *matrix_{nullptr};
  Program *program_{nullptr};
  aot::Arg input_;
  aot::Arg output_;
  int rhs_count_{0};
  int fill_mode_{0};
  bool unit_diagonal_{false};
  bool transpose_{false};
};

class CudaCufftCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaCufftCaptureCommand(std::uint64_t plan_handle,
                          Program *program,
                          aot::Arg input,
                          aot::Arg output,
                          int direction,
                          std::size_t input_scalars,
                          std::size_t output_scalars)
      : plan_handle_(plan_handle),
        program_(program),
        input_(std::move(input)),
        output_(std::move(output)),
        direction_(direction),
        input_scalars_(input_scalars),
        output_scalars_(output_scalars) {
  }

  const char *kind() const override {
    return "cufft_fixed_plan";
  }

  Program *program() const override {
    return program_;
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (program_ != &program || plan_handle_ == 0 ||
        !program.cuda_cufft_capture_plan_available(plan_handle_)) {
      return false;
    }
    auto *input = cuda_cufft_ndarray(input_, args, program, input_scalars_);
    auto *output = cuda_cufft_ndarray(output_, args, program, output_scalars_);
    return input != nullptr && output != nullptr &&
           input->get_device_allocation() != output->get_device_allocation();
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    record(args, program, nullptr);
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(program_ != &program || plan_handle_ == 0,
                "CUDA cuFFT capture recipe provider generation is stale");
    auto *input = cuda_cufft_ndarray(input_, args, program, input_scalars_);
    auto *output = cuda_cufft_ndarray(output_, args, program, output_scalars_);
    TI_ERROR_IF(input == nullptr,
                "CUDA cuFFT capture input {} must be an owning compact f32 "
                "ndarray with {} scalar elements in the active Program",
                input_.name, input_scalars_);
    TI_ERROR_IF(output == nullptr,
                "CUDA cuFFT capture output {} must be an owning compact f32 "
                "ndarray with {} scalar elements in the active Program",
                output_.name, output_scalars_);
    TI_ERROR_IF(input->get_device_allocation() ==
                    output->get_device_allocation(),
                "CUDA cuFFT capture input/output alias");
    program.cuda_cufft_capture_record(plan_handle_, input, output, direction_,
                                      stream);
  }

 private:
  std::uint64_t plan_handle_{0};
  Program *program_{nullptr};
  aot::Arg input_;
  aot::Arg output_;
  int direction_{0};
  std::size_t input_scalars_{0};
  std::size_t output_scalars_{0};
};

class CudaCaptureCommandDispatch final : public Node {
 public:
  CudaCaptureCommandDispatch(
      std::shared_ptr<aot::CudaGraphCaptureCommand> command,
      std::vector<aot::Arg> symbolic_args)
      : command_(std::move(command)), symbolic_args_(std::move(symbolic_args)) {
  }

  void compile(
      std::vector<aot::CompiledDispatch> &compiled_dispatches) override {
    TI_ASSERT(command_ != nullptr);
    aot::CompiledDispatch dispatch;
    dispatch.kernel_name = "cuda_capture_" + std::string(command_->kind());
    dispatch.symbolic_args = symbolic_args_;
    dispatch.cuda_capture_command = command_;
    // Preserve one logical source item for Forge pipeline attribution. The
    // provider recipe cannot participate in kernel composition or AOT
    // serialization.
    dispatch.source_dispatches.push_back(
        {dispatch.kernel_name, "", dispatch.symbolic_args, {}});
    dispatch.compiled_task_count = 0;
    compiled_dispatches.push_back(std::move(dispatch));
  }

 private:
  std::shared_ptr<aot::CudaGraphCaptureCommand> command_;
  std::vector<aot::Arg> symbolic_args_;
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

void validate_cuda_sparse_spmm_args(CuSparseMatrix *matrix,
                                    Program *program,
                                    const aot::Arg &input,
                                    const aot::Arg &output,
                                    int rhs_count,
                                    int algorithm) {
  TI_ERROR_IF(matrix == nullptr || program == nullptr,
              "CUDA sparse SpMM Graph proof requires a live CSR matrix and "
              "Program");
  TI_ERROR_IF(input.tag != aot::ArgKind::kNdarray ||
                  output.tag != aot::ArgKind::kNdarray ||
                  input.dtype_id != PrimitiveTypeID::f32 ||
                  output.dtype_id != PrimitiveTypeID::f32 ||
                  input.field_dim != 2 || output.field_dim != 2 ||
                  !input.element_shape.empty() || !output.element_shape.empty(),
              "CUDA sparse SpMM Graph proof requires scalar f32 rank-2 "
              "ndarray bindings");
  TI_ERROR_IF(input.name == output.name,
              "CUDA sparse SpMM Graph proof input and output must differ");
  TI_ERROR_IF(rhs_count < 2,
              "CUDA sparse SpMM Graph proof requires at least two "
              "right-hand sides");
  TI_ERROR_IF(algorithm != 0 && algorithm != 1,
              "CUDA sparse SpMM Graph proof algorithm is invalid");
}

void validate_cuda_sparse_triangular_args(CuSparseMatrix *matrix,
                                          Program *program,
                                          const aot::Arg &input,
                                          const aot::Arg &output,
                                          int rhs_count,
                                          int fill_mode) {
  TI_ERROR_IF(matrix == nullptr || program == nullptr,
              "CUDA sparse triangular Graph command requires a live CSR "
              "matrix and Program");
  const int expected_rank = rhs_count == 1 ? 1 : 2;
  TI_ERROR_IF(input.tag != aot::ArgKind::kNdarray ||
                  output.tag != aot::ArgKind::kNdarray ||
                  input.dtype_id != PrimitiveTypeID::f32 ||
                  output.dtype_id != PrimitiveTypeID::f32 ||
                  input.field_dim != expected_rank ||
                  output.field_dim != expected_rank ||
                  !input.element_shape.empty() || !output.element_shape.empty(),
              "CUDA sparse triangular Graph command requires scalar f32 "
              "rank-{} ndarray bindings",
              expected_rank);
  TI_ERROR_IF(input.name == output.name,
              "CUDA sparse triangular Graph input and output must differ");
  TI_ERROR_IF(matrix->num_rows() != matrix->num_cols(),
              "CUDA sparse triangular Graph command requires a square "
              "matrix");
  TI_ERROR_IF(rhs_count < 1,
              "CUDA sparse triangular Graph right-hand-side count must be "
              "positive");
  TI_ERROR_IF(fill_mode != 0 && fill_mode != 1,
              "CUDA sparse triangular Graph fill mode is invalid");
}

void validate_cuda_cufft_args(std::uint64_t plan_handle,
                              Program *program,
                              const aot::Arg &input,
                              const aot::Arg &output,
                              int direction,
                              std::size_t input_scalars,
                              std::size_t output_scalars) {
  TI_ERROR_IF(plan_handle == 0 || program == nullptr,
              "CUDA cuFFT Graph proof requires a live plan and Program");
  TI_ERROR_IF(input.tag != aot::ArgKind::kNdarray ||
                  output.tag != aot::ArgKind::kNdarray ||
                  input.dtype_id != PrimitiveTypeID::f32 ||
                  output.dtype_id != PrimitiveTypeID::f32 ||
                  input.field_dim == 0 || output.field_dim == 0 ||
                  !input.element_shape.empty() || !output.element_shape.empty(),
              "CUDA cuFFT Graph proof requires scalar f32 ndarray bindings");
  TI_ERROR_IF(input.name == output.name,
              "CUDA cuFFT Graph proof input and output must differ");
  TI_ERROR_IF(direction != -1 && direction != 1,
              "CUDA cuFFT Graph proof direction must be -1 or 1");
  TI_ERROR_IF(input_scalars == 0 || output_scalars == 0,
              "CUDA cuFFT Graph proof scalar counts must be positive");
  TI_ERROR_IF(!program->cuda_cufft_capture_plan_available(plan_handle),
              "CUDA cuFFT Graph proof plan is stale or closed");
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
       dispatch.graph_metadata, logical_dispatch_id_});
  dispatch.source_dispatches.back().logical_kernel_identity =
      compiled.logical_kernel_identity();
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
  struct PendingDispatch {
    Dispatch *source{nullptr};
    aot::CompiledDispatch compiled;
  };
  std::vector<PendingDispatch> pending;
  auto flush_pending = [&]() {
    std::size_t offset = 0;
    while (offset < pending.size()) {
      const auto remaining = pending.size() - offset;
      const auto max_group_size = std::min<std::size_t>(
          owning_graph_->map_composer_max_group_size(), remaining);
      bool composed = false;
      for (std::size_t group_size = max_group_size;
           group_size >= 2; --group_size) {
        std::vector<const Dispatch *> sources;
        std::vector<const aot::CompiledDispatch *> compiled_sources;
        sources.reserve(group_size);
        compiled_sources.reserve(group_size);
        for (std::size_t index = 0; index < group_size; ++index) {
          sources.push_back(pending[offset + index].source);
          compiled_sources.push_back(&pending[offset + index].compiled);
        }
        auto candidate = owning_graph_->try_compose_maps(
            sources, compiled_sources);
        if (candidate) {
          compiled_dispatches.push_back(std::move(*candidate));
          offset += group_size;
          composed = true;
          break;
        }
      }
      if (!composed) {
        compiled_dispatches.push_back(
            std::move(pending[offset].compiled));
        ++offset;
      }
    }
    pending.clear();
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
    pending.push_back({dispatch, std::move(compiled)});
    if (!owning_graph_->two_map_composer_enabled()) {
      flush_pending();
    }
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
                                 dispatch_label,
                                 next_logical_dispatch_id_++));
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
                                 dispatch_label,
                                 next_logical_dispatch_id_++));
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
      kernel, args, std::nullopt, std::move(metadata), dispatch_label,
      next_logical_dispatch_id_++));
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
      kernel, args, std::nullopt, std::nullopt, dispatch_label,
      next_logical_dispatch_id_++);
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

void GraphBuilder::dispatch_cuda_capture_cusparse_spmv(
    SparseMatrix *matrix,
    Program *program,
    const aot::Arg &input,
    const aot::Arg &output) {
  validate_cuda_sparse_spmv_args(matrix, program, input, output);
  register_arg(input);
  register_arg(output);
  auto command = std::make_shared<CudaSparseSpmvCaptureCommand>(
      matrix, program, input, output);
  all_nodes_.push_back(std::make_unique<CudaCaptureCommandDispatch>(
      std::move(command), std::vector<aot::Arg>{input, output}));
  seq()->append(all_nodes_.back().get());
}

void GraphBuilder::dispatch_cuda_capture_cusparse_spmm(
    CuSparseMatrix *matrix,
    Program *program,
    const aot::Arg &input,
    const aot::Arg &output,
    int rhs_count,
    int algorithm) {
  validate_cuda_sparse_spmm_args(matrix, program, input, output, rhs_count,
                                 algorithm);
  register_arg(input);
  register_arg(output);
  auto command = std::make_shared<CudaSparseSpmmCaptureCommand>(
      matrix, program, input, output, rhs_count, algorithm);
  all_nodes_.push_back(std::make_unique<CudaCaptureCommandDispatch>(
      std::move(command), std::vector<aot::Arg>{input, output}));
  seq()->append(all_nodes_.back().get());
}

void GraphBuilder::dispatch_cuda_capture_cusparse_triangular(
    CuSparseMatrix *matrix,
    Program *program,
    const aot::Arg &input,
    const aot::Arg &output,
    int rhs_count,
    int fill_mode,
    bool unit_diagonal,
    bool transpose) {
  validate_cuda_sparse_triangular_args(matrix, program, input, output,
                                       rhs_count, fill_mode);
  register_arg(input);
  register_arg(output);
  auto command = std::make_shared<CudaSparseTriangularCaptureCommand>(
      matrix, program, input, output, rhs_count, fill_mode, unit_diagonal,
      transpose);
  all_nodes_.push_back(std::make_unique<CudaCaptureCommandDispatch>(
      std::move(command), std::vector<aot::Arg>{input, output}));
  seq()->append(all_nodes_.back().get());
}

void GraphBuilder::dispatch_cuda_capture_cufft(std::uint64_t plan_handle,
                                               Program *program,
                                               const aot::Arg &input,
                                               const aot::Arg &output,
                                               int direction,
                                               std::size_t input_scalars,
                                               std::size_t output_scalars) {
  validate_cuda_cufft_args(plan_handle, program, input, output, direction,
                           input_scalars, output_scalars);
  register_arg(input);
  register_arg(output);
  auto command = std::make_shared<CudaCufftCaptureCommand>(
      plan_handle, program, input, output, direction, input_scalars,
      output_scalars);
  all_nodes_.push_back(std::make_unique<CudaCaptureCommandDispatch>(
      std::move(command), std::vector<aot::Arg>{input, output}));
  seq()->append(all_nodes_.back().get());
}

void GraphBuilder::set_map_composer_max_group_size(
    std::uint32_t max_group_size) {
  TI_ERROR_IF(max_group_size < 1 || max_group_size > 4,
              "Graph map composer group size must be in [1, 4]");
  map_composer_max_group_size_ = max_group_size;
}

std::optional<aot::CompiledDispatch> GraphBuilder::try_compose_maps(
    const std::vector<const Dispatch *> &sources,
    const std::vector<const aot::CompiledDispatch *> &compiled_sources) {
  if (sources.size() < 2 || sources.size() > 4 ||
      compiled_sources.size() != sources.size()) {
    return std::nullopt;
  }
  std::vector<GraphMapSource> map_sources;
  map_sources.reserve(sources.size());
  for (std::size_t index = 0; index < sources.size(); ++index) {
    if (sources[index] == nullptr || compiled_sources[index] == nullptr ||
        !sources[index]->dispatch_label().empty()) {
      return std::nullopt;
    }
    map_sources.push_back(
        {sources[index]->kernel(), &sources[index]->symbolic_args(),
         &compiled_sources[index]->graph_metadata});
  }
  auto composition = compose_graph_map_kernels(
      sources.front()->kernel()->program->compile_config(), map_sources);
  if (!composition) {
    return std::nullopt;
  }

  Dispatch composed_dispatch(composition->kernel.get(),
                             composition->symbolic_args);
  auto compiled = composed_dispatch.compile_dispatch();
  std::uint64_t source_task_count = 0;
  for (const auto *source : compiled_sources) {
    source_task_count += source->compiled_task_count;
  }
  if (compiled.compiled_task_count >= source_task_count ||
      compiled.graph_metadata.opaque || !compiled.graph_metadata.elementwise) {
    return std::nullopt;
  }
  compiled.source_dispatches.clear();
  for (const auto *source : compiled_sources) {
    compiled.source_dispatches.insert(compiled.source_dispatches.end(),
                                      source->source_dispatches.begin(),
                                      source->source_dispatches.end());
  }
  composed_kernels_.push_back(std::move(composition->kernel));
  return compiled;
}

}  // namespace taichi::lang
