#include "gtest/gtest.h"

#include <array>
#include <vector>

#include "taichi/program/sparse_solver.h"
#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

namespace taichi::lang {
namespace {

template <typename Scalar>
std::unique_ptr<SparseMatrix> make_test_matrix(
    const std::array<Scalar, 9> &values,
    DataType dtype) {
  auto matrix = make_sparse_matrix(3, 3, dtype, "col_major");
  std::vector<Eigen::Triplet<Scalar>> triplets;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (values[row * 3 + col] != 0)
        triplets.emplace_back(row, col, values[row * 3 + col]);
    }
  }
  matrix->build_triplets(&triplets);
  return matrix;
}

template <typename Scalar>
void check_raw_families(DataType dtype) {
  const std::array<std::array<Scalar, 9>, 3> matrices{{
      {4, 1, 0, 1, 3, 1, 0, 1, 2},
      {4, 1, 0, 1, -3, 1, 0, 1, 2},
      {4, 2, 0, 1, 3, 1, 0, 1, 2},
  }};
  const std::array<const char *, 3> methods{"LLT", "LDLT", "LU"};
  for (int family = 0; family < 3; ++family) {
    for (const char *ordering : {"AMD", "COLAMD"}) {
      SCOPED_TRACE(std::string(methods[family]) + "/" + ordering);
      auto matrix = make_test_matrix(matrices[family], dtype);
      auto solver = make_sparse_solver(dtype, methods[family], ordering);
      ASSERT_TRUE(solver->compute(*matrix));
      // No Program, Ndarray, JIT, or runtime executor is constructed.
      for (int pass = 1; pass <= 2; ++pass) {
        std::array<Scalar, 3> expected{Scalar(pass), Scalar(-2 * pass),
                                     Scalar(3 * pass)};
        std::array<Scalar, 3> rhs{};
        std::array<Scalar, 3> output{};
        for (int row = 0; row < 3; ++row)
          for (int col = 0; col < 3; ++col)
            rhs[row] += matrices[family][row * 3 + col] * expected[col];
        solver->solve_dense_vector(*matrix, {rhs.data(), dtype, rhs.size()},
                                   {output.data(), dtype, output.size()});
        for (int row = 0; row < 3; ++row)
          EXPECT_NEAR(output[row], expected[row],
                      sizeof(Scalar) == 4 ? 1e-5 : 1e-12);
      }
    }
  }
}

TEST(CpuSparseSolverCore, FloatFamiliesWithoutProgram) {
  check_raw_families<float>(PrimitiveType::f32);
}

TEST(CpuSparseSolverCore, DoubleFamiliesWithoutProgram) {
  check_raw_families<double>(PrimitiveType::f64);
}

TEST(CpuSparseSolverCore, InvalidBindingDoesNotWriteOutput) {
  const auto dtype = DataType(PrimitiveType::f32);
  auto matrix =
      make_test_matrix<float>({4, 1, 0, 1, 3, 1, 0, 1, 2}, dtype);
  auto solver = make_sparse_solver(dtype, "LLT", "AMD");
  ASSERT_TRUE(solver->compute(*matrix));
  std::array<float, 4> rhs{1, 2, 3, 4};
  std::array<float, 3> output{-17, -17, -17};
  const SparseSolverConstVectorView valid_rhs{rhs.data(), dtype, 3};
  const SparseSolverVectorView valid_output{output.data(), dtype, 3};
  EXPECT_ANY_THROW(solver->solve_dense_vector(
      *matrix, {rhs.data(), PrimitiveType::f64, 3}, valid_output));
  EXPECT_ANY_THROW(solver->solve_dense_vector(
      *matrix, {rhs.data(), dtype, 2}, valid_output));
  EXPECT_ANY_THROW(solver->solve_dense_vector(
      *matrix, {nullptr, dtype, 3}, valid_output));
  EXPECT_ANY_THROW(solver->solve_dense_vector(
      *matrix, valid_rhs, {output.data(), PrimitiveType::f64, 3}));
  EXPECT_ANY_THROW(solver->solve_dense_vector(
      *matrix, valid_rhs, {output.data(), dtype, 2}));
  EXPECT_ANY_THROW(solver->solve_dense_vector(
      *matrix, valid_rhs, {rhs.data() + 1, dtype, 3}));
  EXPECT_EQ(output, (std::array<float, 3>{-17, -17, -17}));
  EXPECT_EQ(rhs, (std::array<float, 4>{1, 2, 3, 4}));
}

TEST(CpuSparseSolverCore, StaleOrFailedFactorCannotSolveAndCanRecover) {
  const auto dtype = DataType(PrimitiveType::f64);
  auto matrix =
      make_test_matrix<double>({4, 1, 0, 1, 3, 1, 0, 1, 2}, dtype);
  auto other =
      make_test_matrix<double>({4, 1, 0, 1, 3, 1, 0, 1, 2}, dtype);
  auto solver = make_sparse_solver(dtype, "LLT", "COLAMD");
  std::array<double, 3> rhs{6, 10, 8};
  std::array<double, 3> output{-17, -17, -17};
  auto solve = [&](const SparseMatrix &source) {
    solver->solve_dense_vector(source, {rhs.data(), dtype, 3},
                                {output.data(), dtype, 3});
  };
  EXPECT_ANY_THROW(solve(*matrix));
  ASSERT_TRUE(solver->compute(*matrix));
  EXPECT_ANY_THROW(solve(*other));  // Same contents, different physical owner.
  auto &eigen_matrix =
      static_cast<EigenSparseMatrix<Eigen::SparseMatrix<double>> &>(*matrix);
  eigen_matrix.set_element(0, 0, -4.0);
  EXPECT_ANY_THROW(solve(*matrix));  // Same pattern, changed numeric generation.
  solver->factorize(*matrix);
  EXPECT_FALSE(solver->info());
  EXPECT_ANY_THROW(solve(*matrix));
  EXPECT_EQ(output, (std::array<double, 3>{-17, -17, -17}));
  eigen_matrix.set_element(0, 0, 4.0);
  solver->factorize(*matrix);
  ASSERT_TRUE(solver->info());
  solve(*matrix);
  for (int row = 0; row < 3; ++row)
    EXPECT_NEAR(output[row], row + 1, 1e-12);
}

#if defined(TI_WITH_CUDA)
TEST(CudaSparseSolverCore, RawVectorsWithoutProgramRestoreContextAndRetire) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto &cuda = CUDADriver::get_instance();
  auto &sparse = CUSPARSEDriver::get_instance();
  auto &solver_api = CUSOLVERDriver::get_instance();
  if ((!sparse.is_loaded() && !sparse.load_cusparse()) ||
      (!solver_api.is_loaded() && !solver_api.load_cusolver()))
    GTEST_SKIP();
  // This test deliberately constructs no Program or Ndarray. A runtime-only
  // resource can own the same compressed storage and call the same core.
  auto upload = [&](auto &values) {
    void *result = nullptr;
    cuda.malloc(&result, sizeof(values));
    cuda.memcpy_host_to_device(result, values.data(), sizeof(values));
    return result;
  };
  std::array<int, 4> rows{0, 2, 5, 7};
  std::array<int, 7> columns{0, 1, 0, 1, 2, 1, 2};
  std::array<float, 7> values{4, 1, 1, 3, 1, 1, 2};
  auto *device_rows = upload(rows);
  auto *device_columns = upload(columns);
  auto *device_values = upload(values);
  cusparseSpMatDescr_t descriptor = nullptr;
  sparse.cpCreateCsr(&descriptor, 3, 3, 7, device_rows, device_columns,
                     device_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CuSparseMatrix matrix(descriptor, 3, 3, PrimitiveType::f32, device_rows,
                        device_columns, device_values, 7);
  std::array<float, 3> rhs{6, 10, 8}, output{-17, -17, -17};
  auto *device_rhs = upload(rhs);
  auto *device_output = upload(output);
  void *owned_context = nullptr;
  cuda.context_get_current(&owned_context);
  for (const char *method : {"LLT", "LU"}) {
    for (const char *ordering : {"AMD", "COLAMD"}) {
      auto solver = make_cusparse_solver(PrimitiveType::f32, method, ordering);
      cuda.context_set_current(nullptr);
      ASSERT_TRUE(solver->compute(matrix));
      solver->solve_dense_vector(
          matrix, {device_rhs, PrimitiveType::f32, 3},
          {device_output, PrimitiveType::f32, 3});
      EXPECT_ANY_THROW(solver->solve_dense_vector(
          matrix, {device_rhs, PrimitiveType::f64, 3},
          {device_output, PrimitiveType::f32, 3}));
      if (std::string(method) == "LLT") {
        cuda.context_set_current(owned_context);
        cuda.stream_synchronize(nullptr);
        void *stream = nullptr;
        cuda.stream_create(&stream, CU_STREAM_NON_BLOCKING);
        CUgraph graph = nullptr;
        cuda.stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        try {
          static_cast<CuSparseSolver &>(*solver).record_dense_vector(
              matrix, {device_rhs, PrimitiveType::f32, 3},
              {device_output, PrimitiveType::f32, 3}, stream);
        } catch (...) {
          cuda.stream_end_capture.call(stream, &graph);
          if (graph)
            cuda.graph_destroy.call(graph);
          cuda.stream_destroy.call(stream);
          throw;
        }
        cuda.stream_end_capture(stream, &graph);
        CUgraphExec executable = nullptr;
        cuda.graph_instantiate_with_flags(&executable, graph, 0);
        std::array<float, 3> sentinel{-17, -17, -17};
        for (float scale : {2.0f, 3.0f}) {
          std::array<float, 3> input{6 * scale, 10 * scale, 8 * scale};
          cuda.memcpy_host_to_device(device_rhs, input.data(), sizeof(input));
          cuda.memcpy_host_to_device(device_output, sentinel.data(),
                                     sizeof(sentinel));
          cuda.graph_launch(executable, stream);
          cuda.stream_synchronize(stream);
          cuda.memcpy_device_to_host(output.data(), device_output, sizeof(output));
          for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(output[i], scale * (i + 1), 1e-5);
        }
        cuda.graph_exec_destroy(executable);
        cuda.graph_destroy(graph);
        cuda.stream_destroy(stream);
        cuda.memcpy_host_to_device(device_rhs, rhs.data(), sizeof(rhs));
        solver->solve_dense_vector(
            matrix, {device_rhs, PrimitiveType::f32, 3},
            {device_output, PrimitiveType::f32, 3});
        cuda.context_set_current(nullptr);
      }
      // Destruction must retire queued solves with its own context.
      solver.reset();
      void *restored = owned_context;
      cuda.context_get_current(&restored);
      EXPECT_EQ(restored, nullptr);
      cuda.context_set_current(owned_context);
      cuda.memcpy_device_to_host(output.data(), device_output, sizeof(output));
      for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(output[i], i + 1, 1e-5);
    }
  }
  cuda.mem_free(device_rhs);
  cuda.mem_free(device_output);
}
#endif

}  // namespace
}  // namespace taichi::lang
