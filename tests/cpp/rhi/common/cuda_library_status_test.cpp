#include "gtest/gtest.h"

#include <limits>
#include <mutex>
#include <string>
#include <utility>

#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

#ifdef TI_WITH_CUDA
TEST(CudaLibraryStatus, EqualNumbersRetainDifferentLibraryMeanings) {
  CUDADriverFunction<> function;
  for (const auto &[symbol, expected] :
       {std::pair{"cusolverSpScsrcholBufferInfo",
                  "CUSOLVER_STATUS_INVALID_VALUE"},
        std::pair{"cusparseSpSV_bufferSize", "CUSPARSE_STATUS_INVALID_VALUE"},
        std::pair{"cublasSgemm_v2", "CUBLAS_STATUS_ALLOC_FAILED"},
        std::pair{"cufftMakePlanMany", "CUFFT_INVALID_TYPE"}}) {
    function.set_names("public_alias", symbol);
    const auto message = function.get_error_message(3);
    EXPECT_NE(message.find(expected), std::string::npos);
    EXPECT_NE(message.find("code=3"), std::string::npos);
    EXPECT_NE(message.find(symbol), std::string::npos);
    EXPECT_NE(message.find("public_alias"), std::string::npos);
    EXPECT_EQ(message.find("driver error"), std::string::npos);
    const auto unknown =
        function.get_error_message(std::numeric_limits<uint32>::max());
    EXPECT_NE(unknown.find("UNKNOWN_STATUS"), std::string::npos);
    EXPECT_NE(unknown.find("4294967295"), std::string::npos);
    EXPECT_EQ(unknown.find("driver error"), std::string::npos);
  }
  // Newer / sparse status numbers must not index a dense common-domain table.
  function.set_names("factor", "cusolverDnXgetrf");
  EXPECT_NE(function.get_error_message(31).find("INVALID_WORKSPACE"),
            std::string::npos);
  EXPECT_NE(function.get_error_message(17).find("UNKNOWN_STATUS"),
            std::string::npos);
  function.set_names("plan", "cufftMakePlanMany");
  EXPECT_NE(function.get_error_message(19).find("NVJITLINK_FAILURE"),
            std::string::npos);
}

uint32 invalid_solver_call(int *calls) {
  ++*calls;
  return 3;
}

TEST(CudaLibraryStatus, ReportingDoesNotRepeatTheFailedOperation) {
  CUDADriverFunction<int *> function;
  std::mutex mutex;
  function.set(reinterpret_cast<void *>(&invalid_solver_call));
  function.set_lock(&mutex);
  function.set_names("factor", "cusolverSpScsrcholFactor");
  int calls = 0;
  EXPECT_EQ(function.call(&calls), 3u);
  EXPECT_EQ(calls, 1);
  try {
    function(&calls);
    FAIL() << "expected the original vendor error";
  } catch (const std::string &message) {
    // Legacy vendor wrappers retain Logger::error's string exception contract.
    EXPECT_NE(message.find("CUSOLVER_STATUS_INVALID_VALUE"), std::string::npos);
    EXPECT_NE(message.find("code=3"), std::string::npos);
  }
  EXPECT_EQ(calls, 2);
}

TEST(CudaLibraryStatus, UnknownDriverStatusHasNumericFallback) {
  auto &driver = CUDADriver::get_instance_without_context();
  const auto message =
      get_cuda_error_message(std::numeric_limits<uint32>::max());
  EXPECT_NE(message.find("driver error"), std::string::npos);
  EXPECT_NE(message.find("4294967295"), std::string::npos);
  if (driver.detected()) {
    EXPECT_NE(get_cuda_error_message(1).find("INVALID_VALUE"),
              std::string::npos);
  }
}
#endif

}  // namespace
}  // namespace taichi::lang
