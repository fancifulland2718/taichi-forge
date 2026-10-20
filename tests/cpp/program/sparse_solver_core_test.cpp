#include "gtest/gtest.h"

#include <array>
#include <vector>

#include "taichi/program/sparse_solver.h"

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

}  // namespace
}  // namespace taichi::lang
