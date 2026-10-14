#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>

#include "taichi/program/linear_operator.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/sparse_matrix_export.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#endif

namespace taichi::lang {
namespace {

#ifdef TI_WITH_LLVM

using namespace std::chrono_literals;

template <typename Operation>
void expect_blocked_by_resource_lease(OperatorResourceLease transaction,
                                      Operation operation) {
  std::promise<void> started_promise;
  auto started = started_promise.get_future();
  std::packaged_task<void()> task(
      [&started_promise, operation = std::move(operation)]() mutable {
        started_promise.set_value();
        operation();
      });
  auto completed = task.get_future();
  std::thread worker(std::move(task));

  EXPECT_EQ(started.wait_for(2s), std::future_status::ready);
  EXPECT_EQ(completed.wait_for(50ms), std::future_status::timeout);
  transaction = OperatorResourceLease{};
  EXPECT_EQ(completed.wait_for(2s), std::future_status::ready);
  worker.join();
  completed.get();
}

TEST(CpuSparseNumericTransaction, CsrPinsValueUpdateAndRawApply) {
  Program owned_program(Arch::x64);
  Program *program = &owned_program;

  Ndarray *row_offsets =
      program->create_ndarray(PrimitiveType::i32, {3});
  row_offsets->write_int({0}, 0);
  row_offsets->write_int({1}, 1);
  row_offsets->write_int({2}, 2);
  Ndarray *column_indices =
      program->create_ndarray(PrimitiveType::i32, {2});
  column_indices->write_int({0}, 0);
  column_indices->write_int({1}, 1);
  Ndarray *values = program->create_ndarray(PrimitiveType::f32, {2});
  values->write_float({0}, 2.0);
  values->write_float({1}, 3.0);
  Ndarray *replacement =
      program->create_ndarray(PrimitiveType::f32, {2});
  replacement->write_float({0}, 5.0);
  replacement->write_float({1}, 7.0);

  auto pattern = std::make_shared<SparseCsrPattern>(
      program, 2, 2, *row_offsets, *column_indices);
  CpuSparseCsrMatrix matrix(std::move(pattern), *values);
  const auto numeric_version = matrix.numeric_version();
  auto binding = make_cpu_csr_operator_binding(program, matrix);
  expect_blocked_by_resource_lease(
      binding.acquire_resource_lease(),
      [&] { matrix.update_values(program, *replacement); });
  EXPECT_EQ(matrix.numeric_version(), numeric_version + 1);

  std::array<float, 2> input{2.0f, 3.0f};
  std::array<float, 2> output{};
  expect_blocked_by_resource_lease(
      OperatorResourceLease::hold(matrix.acquire_numeric_access_guard()),
      [&] {
        matrix.spmv_cpu_raw(
            program, reinterpret_cast<std::uintptr_t>(input.data()),
            reinterpret_cast<std::uintptr_t>(output.data()));
      });
  EXPECT_FLOAT_EQ(output[0], 10.0f);
  EXPECT_FLOAT_EQ(output[1], 21.0f);
}

TEST(CpuSparseNumericTransaction, BsrPinsValueUpdateAndRawApply) {
  Program owned_program(Arch::x64);
  Program *program = &owned_program;

  Ndarray *row_offsets =
      program->create_ndarray(PrimitiveType::i32, {2});
  row_offsets->write_int({0}, 0);
  row_offsets->write_int({1}, 1);
  Ndarray *column_indices =
      program->create_ndarray(PrimitiveType::i32, {1});
  column_indices->write_int({0}, 0);
  Ndarray *values = program->create_ndarray(PrimitiveType::f32, {4});
  values->write_float({0}, 1.0);
  values->write_float({1}, 0.0);
  values->write_float({2}, 0.0);
  values->write_float({3}, 1.0);
  Ndarray *replacement =
      program->create_ndarray(PrimitiveType::f32, {4});
  replacement->write_float({0}, 2.0);
  replacement->write_float({1}, 1.0);
  replacement->write_float({2}, 3.0);
  replacement->write_float({3}, 4.0);

  CpuSparseBsrMatrix matrix(program, 1, 1, 2, *row_offsets,
                            *column_indices, *values);
  const auto numeric_version = matrix.numeric_version();
  auto binding = make_cpu_bsr_operator_binding(program, matrix);
  expect_blocked_by_resource_lease(
      binding.acquire_resource_lease(),
      [&] { matrix.update_values(program, *replacement); });
  EXPECT_EQ(matrix.numeric_version(), numeric_version + 1);

  std::array<float, 2> input{2.0f, 3.0f};
  std::array<float, 2> output{};
  expect_blocked_by_resource_lease(
      OperatorResourceLease::hold(matrix.acquire_numeric_access_guard()),
      [&] {
        matrix.spmv_cpu_raw(
            program, reinterpret_cast<std::uintptr_t>(input.data()),
            reinterpret_cast<std::uintptr_t>(output.data()));
      });
  EXPECT_FLOAT_EQ(output[0], 7.0f);
  EXPECT_FLOAT_EQ(output[1], 18.0f);
}

#endif  // TI_WITH_LLVM

#if defined(TI_WITH_CUDA) && defined(TI_WITH_LLVM)
TEST(CudaSparseNumericTransaction, ExportDoesNotInvertTheUpdateGate) {
  if (!CUDADriver::get_instance_without_context().nvidia_extensions_available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  auto &driver = CUSPARSEDriver::get_instance();
  if (!driver.is_loaded() && !driver.load_cusparse())
    GTEST_SKIP();
  program.materialize_runtime();
  auto upload = [&](DataType dtype, auto data) {
    auto *result = program.create_ndarray(dtype, {int(data.size())});
    program.copy_ndarray_from_host(result, data.data(),
                                   data.size() * sizeof(data[0]));
    return result;
  };
  auto *rows = upload(PrimitiveType::i32, std::array<int, 3>{0, 1, 2});
  auto *cols = upload(PrimitiveType::i32, std::array<int, 2>{0, 1});
  auto *values = upload(PrimitiveType::f32, std::array<float, 2>{2, 3});
  auto *replacement = upload(PrimitiveType::f32, std::array<float, 2>{5, 7});
  auto pattern =
      std::make_shared<SparseCsrPattern>(&program, 2, 2, *rows, *cols);
  CuSparseMatrix matrix(pattern, *values);
  auto numeric = matrix.acquire_numeric_access_guard();
  std::promise<void> started;
  auto started_future = started.get_future();
  auto snapshot = std::async(std::launch::async, [&] {
    started.set_value();
    return snapshot_cuda_csr(program, matrix);
  });
  started_future.wait();
  EXPECT_EQ(snapshot.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  auto submission = std::async(std::launch::async, [&] {
    auto guard = program.acquire_runtime_resource_submission_guard();
  });
  const auto ready = submission.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(ready, std::future_status::ready);
  if (ready == std::future_status::ready)
    matrix.update_values(&program, *replacement);
  // Even a regression releases the gate before joining either worker.
  numeric.unlock();
  submission.get();
  auto result = snapshot.get();
  EXPECT_EQ(result.row_offsets, (std::vector<std::int32_t>{0, 1, 2}));
  EXPECT_EQ(result.column_indices, (std::vector<std::int32_t>{0, 1}));
  EXPECT_EQ(result.values, (std::vector<float>{5, 7}));
}
#endif

}  // namespace
}  // namespace taichi::lang
