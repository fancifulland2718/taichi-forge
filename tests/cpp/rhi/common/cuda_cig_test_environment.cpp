#include <gtest/gtest.h>
#include <cstdlib>
#include "c_api/src/taichi_core_impl.h"
#include "taichi/taichi_cuda.h"
#include "taichi/platform/windows/windows.h"

namespace {
class CigEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!std::getenv("TI_TEST_CIG"))
      return;
    const char *path = std::getenv("TI_TEST_CIG_FIXTURE");
    ASSERT_NE(path, nullptr);
    // The process-lived context retains a callback into this module.
    library_ = LoadLibraryA(path);
    ASSERT_NE(library_, nullptr);
    const auto create = reinterpret_cast<const TiCudaContextConfig *(*)()>(
        GetProcAddress(library_, "cig_fixture_create"));
    release_ = reinterpret_cast<void (*)(const TiCudaContextConfig *)>(
        GetProcAddress(library_, "cig_fixture_release"));
    ASSERT_NE(create, nullptr);
    ASSERT_NE(release_, nullptr);
    binding_ = create();
    ASSERT_NE(binding_, nullptr);
    ti_configure_cuda_context_ext(binding_);
    char message[512]{};
    uint64_t size = sizeof(message);
    ASSERT_EQ(ti_get_last_error(&size, message), TI_ERROR_SUCCESS) << message;
    TiCudaContextInfo info{};
    info.struct_size = sizeof(info);
    info.schema_version = 1;
    ti_get_cuda_context_info_ext(&info);
    ASSERT_TRUE(info.vulkan_cig);
    ASSERT_EQ(info.allocation_checks, 19u);
  }
  void TearDown() override {
    if (binding_)
      release_(binding_);
  }

 private:
  HMODULE library_{};
  const TiCudaContextConfig *binding_{};
  void (*release_)(const TiCudaContextConfig *){};
};
auto *const environment =
    ::testing::AddGlobalTestEnvironment(new CigEnvironment);
}  // namespace
