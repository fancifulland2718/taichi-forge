#include <chrono>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"
#include "c_api_test_utils.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"

TEST_F(CapiTest, AotVulkanCapabilityRequirements) {
  if (!ti::is_arch_available(TI_ARCH_VULKAN)) GTEST_SKIP();
  ti::Runtime runtime(TI_ARCH_VULKAN);
  const auto available = runtime.get_capabilities().get(TI_CAPABILITY_SPIRV_VERSION);
  ASSERT_GE(available, 0x10000);
  auto path = std::filesystem::temp_directory_path() /
      ("forge-aot-caps-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(path);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::filesystem::remove(path / "metadata.json");
      std::filesystem::remove(path / "graphs.json");
      std::filesystem::remove(path);
    }
  } cleanup{path};
  std::ofstream(path / "graphs.json") << "[]";
  for (uint32_t required : {uint32_t(0x10000), available, available + 0x100}) {
    std::ofstream(path / "metadata.json")
        << "{\"metadata_version\":1,\"kernels\":[],\"kernel_metadata\":[],"
           "\"fields\":[],\"required_caps\":[{\"key\":\"spirv_version\",\"value\":"
        << required << "}],\"root_buffer_size\":0,\"root_buffer_sizes\":[]}";
    auto module = ti_load_aot_module(runtime, path.string().c_str());
    if (required <= available) {
      ASSERT_NE(module, nullptr);
      ASSERT_TAICHI_SUCCESS();
      ti_destroy_aot_module(module);
    } else {
      EXPECT_EQ(module, nullptr);
      EXPECT_TAICHI_ERROR(TI_ERROR_INCOMPATIBLE_MODULE, "spirv_version");
      if (module) ti_destroy_aot_module(module);
    }
  }
}

// Uses ndarray_autodiff.py with the selected format and no owned fields.
static void check_cuda_artifact_integrity(CapiTest &test,
                                          const std::string &extension) {
  if (!ti::is_arch_available(TI_ARCH_CUDA)) GTEST_SKIP();
  const char *source = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(source, nullptr);
  auto path = std::filesystem::temp_directory_path() /
      ("forge-aot-device-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::copy(source, path, std::filesystem::copy_options::recursive);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{path};
  auto read = [](const std::filesystem::path &file) {
    std::ifstream stream(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), {});
  };
  auto write = [](const std::filesystem::path &file, const std::string &bytes) {
    std::ofstream stream(file, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), bytes.size());
  };
  const auto ptx_path = path / ("forward_scalar." + extension);
  const auto original = read(ptx_path);
  ASSERT_FALSE(original.empty());
  ti::Runtime runtime(TI_ARCH_CUDA);
  for (bool missing : {false, true}) {
    if (missing) {
      std::filesystem::remove(ptx_path);
      // A stale representation must never be read as fallback.
      write(path / "forward_scalar.bc", "stale bitcode");
    } else {
      auto corrupt = original;
      corrupt.back() ^= 1;
      write(ptx_path, corrupt);
    }
    auto module = runtime.load_aot_module(path.string().c_str());
    ASSERT_TRUE(module.is_valid());
    test.ASSERT_TAICHI_SUCCESS();
    auto kernel = module.get_kernel("forward_scalar");
    EXPECT_FALSE(kernel.is_valid());
    test.EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA,
                        missing ? "Cannot read" : "integrity");
    write(ptx_path, original);
    // Failed lookup is stable; a file replacement does not silently retry.
    kernel = module.get_kernel("forward_scalar");
    EXPECT_FALSE(kernel.is_valid());
    test.EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA,
                        missing ? "Cannot read" : "integrity");
  }
  const auto abi_path = path / "metadata.tcb";
  auto abi = read(abi_path);
  ASSERT_FALSE(abi.empty());
  abi.back() ^= 1;
  write(abi_path, abi);
  auto invalid = runtime.load_aot_module(path.string().c_str());
  EXPECT_FALSE(invalid.is_valid());
  test.EXPECT_TAICHI_ERROR(TI_ERROR_INVALID_STATE, "ABI checksum mismatch");
}

TEST_F(CapiTest, AotCudaPtxIntegrity) {
  check_cuda_artifact_integrity(*this, "ptx");
}

TEST_F(CapiTest, AotCudaCubinIntegrity) {
  check_cuda_artifact_integrity(*this, "cubin");
}

// Uses field_autodiff.py cuda --artifact-format ptx. A torn relocation table
// must fail before publishing a kernel, without affecting a live peer module.
TEST_F(CapiTest, AotCudaDenseRelocationIntegrity) {
  if (!ti::is_arch_available(TI_ARCH_CUDA))
    GTEST_SKIP();
  const char *source = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(source, nullptr);
  auto path =
      std::filesystem::temp_directory_path() /
      ("forge-aot-relocation-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::copy(source, path, std::filesystem::copy_options::recursive);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{path};
  const auto metadata_path = path / "aot_metadata.json";
  std::ifstream stream(metadata_path, std::ios::binary);
  const std::string original(std::istreambuf_iterator<char>(stream), {});
  stream.close();
  auto write = [&](const std::string &bytes) {
    std::ofstream output(metadata_path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), bytes.size());
  };
  const auto entry = original.find("\"key\":\"get_scalar\"");
  ASSERT_NE(entry, std::string::npos);
  const auto table = original.find("\"tree_relocations\":[", entry);
  ASSERT_NE(table, std::string::npos);
  const auto begin = original.find('[', table);
  const auto end = original.find(']', begin);
  ASSERT_NE(end, std::string::npos);
  ASSERT_GT(end, begin + 1);

  ti::Runtime runtime(TI_ARCH_CUDA);
  auto peer = runtime.load_aot_module(source);
  ASSERT_TRUE(peer.is_valid());
  auto output = runtime.allocate_ndarray<float>({8}, {}, true);
  auto graph = peer.get_compute_graph("differentiate");
  graph["out"] = output;
  graph["seed"] = 1.0f;
  graph.launch();
  runtime.wait();
  ASSERT_TAICHI_SUCCESS();
  std::vector<float> expected(8), actual(8);
  output.read(expected);

  for (const auto *replacement : {"[]", "[999]"}) {
    auto corrupt = original;
    corrupt.replace(begin, end - begin + 1, replacement);
    write(corrupt);
    auto broken = runtime.load_aot_module(path.string().c_str());
    ASSERT_TRUE(broken.is_valid());
    auto kernel = broken.get_kernel("get_scalar");
    EXPECT_FALSE(kernel.is_valid());
    EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA, "resource contract checksum");
    write(original);
    kernel = broken.get_kernel("get_scalar");
    EXPECT_FALSE(kernel.is_valid());
    EXPECT_TAICHI_ERROR(TI_ERROR_CORRUPTED_DATA, "resource contract checksum");
    broken.destroy();
    // Freed IDs must be reusable after failed lazy materialization.
    auto fresh = runtime.load_aot_module(path.string().c_str());
    ASSERT_TRUE(fresh.is_valid());
    ASSERT_TRUE(fresh.get_kernel("get_scalar").is_valid());
    fresh.destroy();
    graph.launch();
    runtime.wait();
    ASSERT_TAICHI_SUCCESS();
    output.read(actual);
    EXPECT_EQ(actual, expected);
  }
}
