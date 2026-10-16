#include "gtest/gtest.h"
#include <algorithm>
#include <limits>
#include <utility>
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/common/operation_json.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/graph_region_builder.h"
#endif

namespace taichi::lang {
TEST(NativeOperationRegistry, CubSourceFreezesIdentityWithoutLoadingCode) {
  const auto &registry = native_operation_registry();
  NativeOperationDescription plan;
  plan.provider = "cub_source";
  plan.operation = "plan";
  plan.parameters = liong::json::parse_complete(
      R"({"operation":"segmented_inclusive_scan_u32","num_items":2147483647,
      "dtype":"i32","workspace_bytes":0,"cub_version":300200,"cuda_runtime_version":13020,
      "binary_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
      "target_code":["sm_120","compute_120"],"minimum_driver_api_version":13000,
      "ptx_driver_api_version":13020})");
  EXPECT_EQ(registry.decode(plan, NativeObjectKind::resource).kind(), "cub_source_plan");
  for (const auto &[field, value] : std::vector<std::pair<std::string, std::string>>{
           {"dtype", R"("f32")"}, {"num_items", "2147483648"}, {"num_items", "true"},
           {"binary_sha256", R"("short")"}, {"target_code", "[]"},
           {"target_code", R"(["sm_42949672960"])"}, {"cub_version", "0"},
           {"unknown_field", "1"}}) {
    auto invalid = plan;
    invalid.parameters.obj.inner[field] = liong::json::parse_complete(value);
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource), std::invalid_argument);
  }
  EXPECT_EQ(registry.find("cub_source_execute", 1, Arch::x64), nullptr);
  EXPECT_EQ(registry.find("cub_source_execute", 1, Arch::vulkan), nullptr);
}

TEST(NativeOperationRegistry, MatrixMmaUsesExistingDriverBatchAbi) {
  const auto &registry = native_operation_registry();
  auto op = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cuda_driver","operation":"matrix_mma_f16_f32","version":1,
      "parameters":{"batch_count":2147483647},"bindings":{"a":"a","b":"b","output":"c"}})"));
  EXPECT_EQ(registry.decode(op).kind(), "cuda_matrix_mma_f16_f32");
  for (const char *parameters : {R"({"batch_count":0})", R"({"batch_count":-1})",
                                R"({"batch_count":2147483648})",
                                R"({"batch_count":true})",
                                R"({"batch_count":1,"transpose":true})"}) {
    auto bad = op;
    bad.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(bad), std::invalid_argument);
  }
  EXPECT_EQ(registry.find("cuda_matrix_mma_f16_f32", 1, Arch::x64), nullptr);
  EXPECT_EQ(registry.find("cuda_matrix_mma_f16_f32", 1, Arch::vulkan), nullptr);
}

namespace {
TEST(NativeOperationRegistry, RetainedScanPreservesItsSelectedCountAndType) {
  const auto &registry = native_operation_registry();
  auto op = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cuda_driver","operation":"retained_scan","version":1,
      "parameters":{"size":2147483647,"dtype":"u32"},"bindings":{"values":"v","scratch":"s"}})"));
  EXPECT_EQ(registry.decode(op).kind(), "cuda_retained_scan");
  for (const char *parameters :
       {R"({"size":-1,"dtype":"u32"})", R"({"size":2147483648,"dtype":"u32"})",
        R"({"size":true,"dtype":"i32"})", R"({"size":4,"dtype":"f32"})",
        R"({"size":4,"dtype":"i32","reverse":true})"}) {
    auto bad = op;
    bad.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(bad), std::invalid_argument);
  }
  op.parameters = liong::json::parse_complete(R"({"size":0,"dtype":"i32"})");
  EXPECT_EQ(registry.decode(op).kind(), "cuda_retained_scan");
  EXPECT_EQ(registry.find("cuda_retained_scan", 1, Arch::x64), nullptr);
  EXPECT_EQ(registry.find("cuda_retained_scan", 1, Arch::vulkan), nullptr);
}

TEST(NativeOperationRegistry, VkfftPlansDecodeWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto plan = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"vkfft","operation":"plan_f32","version":1,
      "parameters":{"dimensions":[8],"batch_count":3,"batch_tile":2,
      "direction":"inverse","normalization":"inverse"},"bindings":{"data":"x"}})"));
  EXPECT_EQ(registry.decode(plan, NativeObjectKind::resource).kind(),
            "vkfft_plan_f32");
  for (const char *parameters :
       {R"({"dimensions":[]})", R"({"dimensions":[0]})",
        R"({"dimensions":[17]})", R"({"dimensions":[2,2,2,2]})",
        R"({"dimensions":[8],"batch_count":0})",
        R"({"dimensions":[8],"batch_count":true})",
        R"({"dimensions":[8],"batch_tile":2})",
        R"({"dimensions":[8],"direction":"backward"})",
        R"({"dimensions":[8],"normalization":"ortho"})",
        R"({"dimensions":[1073741824,1073741824,8]})",
        R"({"dimensions":[8],"adapter_path":"ignored.dll"})"}) {
    auto bad = plan;
    bad.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(bad, NativeObjectKind::resource),
                 std::invalid_argument)
        << parameters;
  }
  plan.bindings.clear();
  EXPECT_THROW(registry.decode(plan, NativeObjectKind::resource),
               std::invalid_argument);
  auto execute = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"vkfft","operation":"execute_f32","version":1,
      "parameters":{},"bindings":{"plan":"p"}})"));
  EXPECT_EQ(registry.decode(execute).kind(), "vkfft_execute_f32");
  EXPECT_THROW(registry.decode(execute, NativeObjectKind::resource),
               std::invalid_argument);
  execute.parameters =
      liong::json::parse_complete(R"({"direction":"inverse"})");
  EXPECT_THROW(registry.decode(execute), std::invalid_argument);
}

struct TestDescriptor {
  static constexpr const char *kind = "test.operation";
  static constexpr std::uint32_t version = 1;
  bool valid{true};
};
struct ConflictingDescriptor {
  static constexpr const char *kind = TestDescriptor::kind;
  static constexpr std::uint32_t version = TestDescriptor::version;
};

TEST(NativeOperationRegistry, BindingGroupsValidateTheCompleteInventory) {
  NativeOperationRegistry registry;
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
  int decoded = 0;
  registry.register_schema<TestDescriptor>(
      "test", "group", {"plan"},
      [&](const liong::json::JsonValue &parameters) {
        operation_json::fields(parameters, {"rhs_count"});
        ++decoded;
        return [](const NativeOperationBindings &) { return TestDescriptor{}; };
      },
      {}, {"values"},
      NativeOperationRegistry::BindingGroup{"rhs_count", {"rhs", "output"}});
  auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"test","operation":"group","version":1,
      "parameters":{"rhs_count":64},"bindings":{"plan":"p","values":"v"}})"));
  for (int i = 0; i < 64; ++i) {
    description.bindings.emplace("rhs_" + std::to_string(i), "input");
    description.bindings.emplace("output_" + std::to_string(i),
                                 "solution_" + std::to_string(i));
  }
  EXPECT_NO_THROW(registry.decode(description));
  EXPECT_EQ(decoded, 1);
  description.bindings.erase("values");
  EXPECT_NO_THROW(registry.decode(description));
  EXPECT_EQ(decoded, 2);
  for (const char *parameters :
       {"{}", "null", "[]", R"({"rhs_count":true})", R"({"rhs_count":-1})",
        R"({"rhs_count":1.0})", R"({"rhs_count":9223372036854775807})",
        R"({"rhs_count":63})", R"({"rhs_count":65})"}) {
    auto bad = description;
    bad.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(bad), std::invalid_argument) << parameters;
  }
  EXPECT_EQ(decoded,
            2);  // Reject malformed inventory before its typed decoder.
  for (const auto *port : {"plan", "rhs_0", "output_63"}) {
    auto bad = description;
    bad.bindings.erase(port);
    EXPECT_THROW(registry.decode(bad), std::invalid_argument);
  }
  for (const auto *port : {"rhs_00", "rhs_64", "outputs_0", "unused"}) {
    auto bad = description;
    bad.bindings.emplace(port, "input");
    EXPECT_THROW(registry.decode(bad), std::invalid_argument);
  }
  description.parameters = liong::json::parse_complete(R"({"rhs_count":0})");
  description.bindings = {{"plan", "p"}};
  EXPECT_NO_THROW(registry.decode(description));
  EXPECT_EQ(decoded, 3);  // Individual operations may require a nonempty group.
}

TEST(NativeOperationRegistry, BindingGroupsRejectAmbiguousRegistration) {
  NativeOperationRegistry registry;
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
  auto decode = [](const liong::json::JsonValue &) {
    return [](const NativeOperationBindings &) { return TestDescriptor{}; };
  };
  using Group = NativeOperationRegistry::BindingGroup;
  for (const auto &group :
       {Group{"", {"rhs"}}, Group{"count", {}}, Group{"count", {""}},
        Group{"count", {"rhs", "rhs"}}, Group{"count", {"fixed"}},
        Group{"count", {"optional"}}}) {
    EXPECT_THROW(
        registry.register_schema<TestDescriptor>(
            "test", "group", {"fixed_0"}, decode, {}, {"optional_1"}, group),
        std::invalid_argument);
  }
  EXPECT_THROW(registry.register_schema<TestDescriptor>(
                   "test", "group", {}, decode, "legacy", {},
                   Group{"count", {"rhs"}}),
               std::invalid_argument);
  // Failed registration must not reserve the schema key.
  EXPECT_NO_THROW(registry.register_schema<TestDescriptor>(
      "test", "group", {"fixed_0"}, decode, {}, {"optional_1"},
      Group{"count", {"rhs", "output"}}));
}

TEST(NativeOperationRegistry, CsrDescriptionsDecodeWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto matrix = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cusparse","operation":"csr_f32","version":1,
      "parameters":{"rows":8589934609,"columns":17},
      "bindings":{"row_offsets":"r","column_indices":"c","values":"v"}})"));
  EXPECT_EQ(registry.decode(matrix, NativeObjectKind::resource).kind(),
            "cusparse_csr_f32");
  EXPECT_THROW(registry.decode(matrix), std::invalid_argument);
  for (const char *parameters :
       {R"({"rows":-1,"columns":2})", R"({"rows":true,"columns":2})",
        R"({"rows":1,"columns":2,"base":1})"}) {
    matrix.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(matrix, NativeObjectKind::resource),
                 std::invalid_argument);
  }
  auto spmv = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cusparse","operation":"spmv_f32","version":1,
      "parameters":{"transpose":true,"preprocess":false,"algorithm":"csr_alg2","alpha":-2,"beta":0.5},
      "bindings":{"matrix":"a","input":"x","output":"y"}})"));
  EXPECT_EQ(registry.decode(spmv).kind(), "cusparse_spmv_f32");
  for (const char *parameters :
       {R"({"transpose":1})", R"({"preprocess":0})", R"({"algorithm":"auto"})",
        R"({"alpha":true})", R"({"beta":1e300})", R"({"stride":1})"}) {
    spmv.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(spmv), std::invalid_argument);
  }
  EXPECT_EQ(registry.find("cusparse_spmv_f32", 1, Arch::x64), nullptr);
  EXPECT_EQ(registry.find("cusparse_spmv_f32", 1, Arch::vulkan), nullptr);
}

TEST(NativeOperationRegistry,
     BsrDescriptionsPreserveBlocksWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto matrix = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cusparse","operation":"bsr_f32","version":1,
      "parameters":{"block_rows":8589934609,"block_columns":17,"block_size":4},
      "bindings":{"row_offsets":"r","column_indices":"c","values":"v"}})"));
  EXPECT_EQ(registry.decode(matrix, NativeObjectKind::resource).kind(),
            "cusparse_bsr_f32");
  EXPECT_THROW(registry.decode(matrix), std::invalid_argument);
  for (
      const char *parameters :
      {R"({"block_rows":-1,"block_columns":2,"block_size":2})",
       R"({"block_rows":1,"block_columns":2,"block_size":0})",
       R"({"block_rows":1,"block_columns":2,"block_size":true})",
       R"({"block_rows":9223372036854775807,"block_columns":2,"block_size":2})",
       R"({"block_rows":1,"block_columns":2,"block_size":2,"base":1})"}) {
    matrix.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(matrix, NativeObjectKind::resource),
                 std::invalid_argument);
  }
  auto spmv = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cusparse","operation":"bsr_spmv_f32","version":1,
      "parameters":{"alpha":-2,"beta":0.5},
      "bindings":{"matrix":"a","input":"x","output":"y"}})"));
  EXPECT_EQ(registry.decode(spmv).kind(), "cusparse_bsr_spmv_f32");
  for (const char *parameters :
       {R"({"transpose":true})", R"({"algorithm":"csr_alg1"})",
        R"({"alpha":true})", R"({"beta":1e300})"}) {
    spmv.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(spmv), std::invalid_argument);
  }
  EXPECT_EQ(registry.find("cusparse_bsr_spmv_f32", 1, Arch::x64), nullptr);
  EXPECT_EQ(registry.find("cusparse_bsr_spmv_f32", 1, Arch::vulkan), nullptr);
}

TEST(NativeOperationRegistry, SpmmDescriptionsDecodeWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"cusparse","operation":"spmm_f32","version":1,
      "parameters":{"rhs_count":8589934609,"algorithm":"csr_alg3","preprocess":"required","expected_workspace_bytes":17},
      "bindings":{"matrix":"a","input":"x","output":"y"}})"));
  EXPECT_EQ(registry.decode(description).kind(), "cusparse_spmm_f32");
  for (const auto *parameters :
       {R"({"rhs_count":0})", R"({"rhs_count":true})", R"({"rhs_count":-1})",
        R"({"rhs_count":2,"preprocess":true})",
        R"({"rhs_count":2,"preprocess":"auto"})",
        R"({"rhs_count":2,"algorithm":"csr_alg1"})",
        R"({"rhs_count":2,"alpha":1e300})",
        R"({"rhs_count":2,"expected_workspace_bytes":-1})",
        R"({"rhs_count":2,"transpose":true})"}) {
    description.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(description), std::invalid_argument);
  }
  EXPECT_EQ(registry.find("cusparse_spmm_f32", 1, Arch::x64), nullptr);
  EXPECT_EQ(registry.find("cusparse_spmm_f32", 1, Arch::vulkan), nullptr);
}

TEST(NativeOperationRegistry,
     TriangularDescriptionsDecodeWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"cusparse","operation":"triangular_plan_f32","version":1,
      "parameters":{"rhs_count":2,"fill_mode":"upper","transpose":true,"unit_diagonal":true,"mutable_values":true},
      "bindings":{"matrix":"a","input":"x","output":"y"}})"));
  EXPECT_EQ(registry.decode(description, NativeObjectKind::resource).kind(),
            "cusparse_triangular_plan_f32");
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  for (const auto *parameters :
       {R"({"rhs_count":0})", R"({"rhs_count":true})", R"({"rhs_count":-1})",
        R"({"rhs_count":2,"fill_mode":"auto"})",
        R"({"rhs_count":1,"transpose":1})",
        R"({"rhs_count":1,"unit_diagonal":0})",
        R"({"rhs_count":1,"mutable_values":"yes"})",
        R"({"rhs_count":1,"algorithm":"auto"})"}) {
    description.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(description, NativeObjectKind::resource),
                 std::invalid_argument);
  }
  for (const auto *operation :
       {"triangular_solve_f32", "triangular_update_f32"}) {
    description.operation = operation;
    description.bindings = {{"plan", "p"}};
    description.parameters = liong::json::parse_complete("{}");
    const auto decoded = registry.decode(description);
    EXPECT_EQ(registry.find(decoded.kind(), 1, Arch::x64), nullptr);
    EXPECT_EQ(registry.find(decoded.kind(), 1, Arch::vulkan), nullptr);
    description.parameters =
        liong::json::parse_complete(R"({"transpose":true})");
    EXPECT_THROW(registry.decode(description), std::invalid_argument);
  }
}

TEST(NativeOperationRegistry, SelectedFftPlansDecodeWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"cufft","operation":"plan_f32","version":1,
      "parameters":{},"bindings":{}})"));
  for (const auto *strategy : {"whole_transform", "row_batch_column_inplace",
                               "row_batch_cross_batch_columns"}) {
    description.parameters = liong::json::parse_complete(
        R"({"dimensions":[5,7],"batch_count":2,"transform":"c2c","expected_workspace_bytes":0})");
    description.parameters.obj.inner["strategy"] =
        liong::json::JsonValue(strategy);
    EXPECT_EQ(registry.decode(description, NativeObjectKind::resource).kind(),
              "cufft_plan_f32");
  }
  for (
      const auto *parameters :
      {R"({"dimensions":[5,7],"transform":"c2c","strategy":"guess"})",
       R"({"dimensions":[5],"transform":"c2c","strategy":"row_batch_column_inplace"})",
       R"({"dimensions":[5,7],"transform":"r2c","strategy":"row_batch_cross_batch_columns"})",
       R"({"dimensions":[5,7],"transform":"c2c","strategy":"row_batch_column_inplace","input_layout":{"embed":[5,7],"stride":2,"batch_distance":70}})",
       R"({"dimensions":[5,7],"transform":"c2c","strategy":"row_batch_column_inplace","output_layout":{"embed":[5,8],"stride":1,"batch_distance":40}})",
       R"({"dimensions":[5,7],"transform":"c2c","strategy":"row_batch_column_inplace","input_layout":{"embed":[5,7],"stride":1,"batch_distance":36}})",
       R"({"dimensions":[2,1],"batch_count":2147483647,"transform":"c2c","strategy":"row_batch_cross_batch_columns"})",
       R"({"dimensions":[5,7],"transform":"c2c","expected_workspace_bytes":-1})",
       R"({"dimensions":[5,7],"transform":"c2c","expected_workspace_bytes":true})"}) {
    description.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(description, NativeObjectKind::resource),
                 std::invalid_argument)
        << parameters;
  }
}

TEST(NativeOperationRegistry, FftPlansDecodeWithoutLoadingProvider) {
  const auto &registry = native_operation_registry();
  auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cufft","operation":"plan_f32","version":1,
      "parameters":{"dimensions":[2,3,5],"batch_count":2,"transform":"r2c"},
      "bindings":{}})"));
  EXPECT_EQ(registry.decode(description, NativeObjectKind::resource).kind(),
            "cufft_plan_f32");
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  for (
      const auto *parameters :
      {R"({"dimensions":[],"transform":"c2c"})",
       R"({"dimensions":[1,1,1,1],"transform":"c2c"})",
       R"({"dimensions":[0],"transform":"c2c"})",
       R"({"dimensions":[2147483648],"transform":"c2c"})",
       R"({"dimensions":[65536,65536],"transform":"c2c"})",
       R"({"dimensions":[7],"batch_count":0,"transform":"c2c"})",
       R"({"dimensions":[7],"transform":"f64"})",
       R"({"dimensions":[7],"transform":"c2c","callback":"lost"})",
       R"({"dimensions":[7],"transform":"c2c","input_layout":{"embed":[6],"stride":1,"batch_distance":7}})",
       R"({"dimensions":[7],"transform":"c2c","input_layout":{"embed":[7,7],"stride":1,"batch_distance":49}})",
       R"({"dimensions":[7],"transform":"c2c","input_layout":{"embed":[7],"stride":0,"batch_distance":7}})"}) {
    description.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(description, NativeObjectKind::resource),
                 std::invalid_argument)
        << parameters;
  }
  description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cufft","operation":"execute_f32","version":1,
      "parameters":{"direction":"inverse"},
      "bindings":{"plan":"p","input":"x","output":"y"}})"));
  EXPECT_EQ(registry.decode(description).kind(), "cufft_execute_f32");
  description.parameters =
      liong::json::parse_complete(R"({"direction":"backward"})");
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
}

TEST(NativeOperationRegistry, FftCallbackSchemaRejectsUnsupportedContracts) {
  const auto &registry = native_operation_registry();
  auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"cufft","operation":"plan_f32","version":1,
      "parameters":{"dimensions":[8,8],"transform":"c2c"},"bindings":{"store_callback_ir":"code"}})"));
  auto &parameters = description.parameters.obj.inner;
  parameters["store_callback"] = liong::json::parse_complete(
      R"({"format":"cufft-store-scale-lto-v1","compiler_version":[13,2]})");
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
  parameters["transform"] = liong::json::JsonValue("r2c");
  EXPECT_THROW(registry.decode(description, NativeObjectKind::resource),
               std::invalid_argument);
  parameters["transform"] = liong::json::JsonValue("c2c");
  parameters["strategy"] = liong::json::JsonValue("row_batch_column_inplace");
  EXPECT_THROW(registry.decode(description, NativeObjectKind::resource),
               std::invalid_argument);
  parameters.erase("strategy");
  for (
      const auto *callback :
      {R"({"format":"other","compiler_version":[13,2]})",
       R"({"format":"cufft-store-scale-lto-v1","compiler_version":[13]})",
       R"({"format":"cufft-store-scale-lto-v1","compiler_version":[0,2]})",
       R"({"format":"cufft-store-scale-lto-v1","compiler_version":[13,-1]})",
       R"({"format":"cufft-store-scale-lto-v1","compiler_version":[13,true]})"}) {
    parameters["store_callback"] = liong::json::parse_complete(callback);
    EXPECT_THROW(registry.decode(description, NativeObjectKind::resource),
                 std::invalid_argument);
  }
}

#ifdef TI_WITH_CUDA
struct CaptureResourceCounts {
  int recorded{0};
  int finalized{0};
  int released{0};
};

class CapturedScalar final : public CudaGraphCaptureResources {
 public:
  CapturedScalar(uint32_t value, CaptureResourceCounts &counts, bool fail)
      : value_(value), counts_(counts), fail_(fail) {
  }
  ~CapturedScalar() override {
    release(true);
  }
  void record(void *output, void *stream) {
    auto &driver = CUDADriver::get_instance();
    CUgraph graph = nullptr;
    uint32_t status = 0;
    driver.stream_get_capture_info_v2(stream, &status, nullptr, &graph, nullptr,
                                      nullptr);
    std::size_t count = 0;
    driver.graph_get_nodes(graph, nullptr, &count);
    std::vector<void *> before(count);
    driver.graph_get_nodes(graph, before.data(), &count);
    driver.memcpy_host_to_device_async(output, &value_, sizeof(value_), stream);
    driver.graph_get_nodes(graph, nullptr, &count);
    std::vector<void *> after(count);
    driver.graph_get_nodes(graph, after.data(), &count);
    for (auto *node : after) {
      if (std::find(before.begin(), before.end(), node) == before.end()) {
        node_ = node;
        driver.graph_memcpy_node_get_params(node_, &params_);
      }
    }
    if (!node_)
      throw std::runtime_error("Missing captured parameter copy");
  }
  void finalize() override {
    ++counts_.finalized;
    auto &driver = CUDADriver::get_instance();
    driver.malloc(&device_, sizeof(value_));
    driver.memcpy_host_to_device(device_, &value_, sizeof(value_));
    if (fail_)
      throw std::runtime_error("Capture finalization failure after allocation");
    params_.srcMemoryType = static_cast<CUmemorytype>(CU_MEMORYTYPE_DEVICE);
    params_.srcHost = nullptr;
    params_.srcDevice = device_;
    params_.srcPitch = sizeof(value_);
    driver.graph_memcpy_node_set_params(node_, &params_);
  }
  uint64_t requested_device_bytes() const override {
    return device_ ? sizeof(value_) : 0;
  }
  void release(bool backend_safe) noexcept override {
    if (released_)
      return;
    released_ = true;
    ++counts_.released;
    if (device_ && backend_safe)
      CUDADriver::get_instance().mem_free.call(device_);
    device_ = nullptr;
  }

 private:
  uint32_t value_;
  CaptureResourceCounts &counts_;
  bool fail_, released_{false};
  void *node_{nullptr}, *device_{nullptr};
  CUDA_MEMCPY3D params_{};
};

class CaptureScalarOperation final : public NativeOperation {
 public:
  CaptureScalarOperation(Device &device,
                         void *output,
                         CaptureResourceCounts &counts)
      : device_(device), output_(output), counts_(counts) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void record_cuda(void *stream) override {
    ++counts_.recorded;
    pending_ = std::make_shared<CapturedScalar>(value, counts_, fail_finalize);
    last = pending_;
    pending_->record(output_, stream);
    if (fail_record)
      throw std::runtime_error("Provider failed during capture");
  }
  std::shared_ptr<CudaGraphCaptureResources> take_cuda_capture_resources()
      override {
    return std::exchange(pending_, {});
  }
  uint32_t value{7};
  bool fail_record{false}, fail_finalize{false};
  std::weak_ptr<CudaGraphCaptureResources> last;

 private:
  Device &device_;
  void *output_;
  CaptureResourceCounts &counts_;
  std::shared_ptr<CapturedScalar> pending_;
};

TEST(NativeOperationRegistry, CaptureResourcesArePerGraphAndRetireAfterLaunch) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto allocation = device.allocate_memory_unique({4});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto *output = device.get_memory_addr(*allocation.first);
  driver.memsetd32(output, 99, 1);
  auto read = [&] {
    uint32_t value = 0;
    driver.memcpy_device_to_host(&value, output, 4);
    return value;
  };
  CaptureResourceCounts counts;
  auto operation =
      std::make_shared<CaptureScalarOperation>(device, output, counts);
  const std::vector<FixedExecution::Operation> operations{operation};
  auto first =
      std::make_unique<FixedExecution>(Arch::cuda, device, nullptr, operations);
  auto first_resource = operation->last;
  operation->value = 11;
  auto second =
      std::make_unique<FixedExecution>(Arch::cuda, device, nullptr, operations);
  EXPECT_EQ(read(),
            99);  // Cold capture/finalization does not execute the math.
  EXPECT_EQ(first->workspace_bytes(), 4);
  EXPECT_EQ(second->workspace_bytes(), 4);
  for (int i = 0; i < 3; ++i) {
    first->run();
    EXPECT_EQ(read(), 7);
    second->run();
    EXPECT_EQ(read(), 11);
  }
  EXPECT_EQ(counts.recorded, 2);
  EXPECT_EQ(counts.finalized, 2);
  EXPECT_EQ(counts.released, 0);
  first->run();
  first.reset();  // Retirement itself must wait for the outstanding launch.
  EXPECT_TRUE(first_resource.expired());
  EXPECT_EQ(read(), 7);
  second.reset();
  EXPECT_EQ(counts.released, 2);
  operation->fail_record = true;
  EXPECT_THROW(FixedExecution(Arch::cuda, device, nullptr, operations),
               std::runtime_error);
  EXPECT_TRUE(operation->last.expired());
  EXPECT_EQ(counts.finalized, 2);
  EXPECT_EQ(counts.released, 3);
  operation->fail_record = false;
  operation->fail_finalize = true;
  EXPECT_THROW(FixedExecution(Arch::cuda, device, nullptr, operations),
               std::runtime_error);
  EXPECT_TRUE(operation->last.expired());
  EXPECT_EQ(counts.released, 4);
  operation->fail_finalize = false;
  {
    FixedExecution recovered(Arch::cuda, device, nullptr, operations);
    recovered.run();
  }
  EXPECT_EQ(read(), 11);
  EXPECT_EQ(counts.released, 5);
}

TEST(NativeOperationRegistry,
     CaptureResourcesFinalizeInsideConditionalRegions) {
  if (!CUDADriver::get_instance_without_context().detected() ||
      !cuda::GraphRegionBuilder::available())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto allocation = device.allocate_memory_unique({8});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto predicate = allocation.first->get_ptr();
  auto output_ptr = predicate;
  output_ptr.offset = 4;
  auto *output =
      static_cast<char *>(device.get_memory_addr(*allocation.first)) + 4;
  CaptureResourceCounts counts;
  auto operation =
      std::make_shared<CaptureScalarOperation>(device, output, counts);
  auto loop = std::make_shared<FixedExecution::While>();
  loop->predicate = predicate;
  loop->condition = {FixedExecution::BufferFill{predicate, 4, 1}};
  loop->body = {operation};
  for (auto budget : {0u, 2u}) {
    loop->max_iterations = budget;
    {
      FixedExecution graph(
          Arch::cuda, device, nullptr,
          {FixedExecution::BufferFill{output_ptr, 4, 99}, loop});
      EXPECT_GT(graph.workspace_bytes(), 4);
      graph.run();
    }
    uint32_t value = 0;
    driver.memcpy_device_to_host(&value, output, 4);
    EXPECT_EQ(value, budget ? 7 : 99);
    EXPECT_TRUE(operation->last.expired());
  }
  EXPECT_EQ(counts.recorded, 2);
  EXPECT_EQ(counts.finalized, 2);
  EXPECT_EQ(counts.released, 2);
}

TEST(NativeOperationRegistry,
     SynchronousCudaActionsRespectPrivateStreamsAndFailure) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  std::array<DeviceAllocationUnique, 3> arrays;
  std::array<void *, 3> pointers;
  for (int i = 0; i < 3; ++i) {
    auto allocation = device.allocate_memory_unique({16 * sizeof(uint32_t)});
    ASSERT_EQ(allocation.second, RhiResult::success);
    arrays[i] = std::move(allocation.first);
    pointers[i] = device.get_memory_addr(*arrays[i]);
    driver.memsetd32(pointers[i], 8, 16);
  }
  int calls = 0;
  class Copy final : public NativeOperation {
   public:
    Copy(Device &device,
         void *source,
         void *destination,
         int &calls,
         bool fail = false)
        : device_(device),
          source_(source),
          destination_(destination),
          calls_(calls),
          fail_(fail) {
      CUDADriver::get_instance().stream_create(&stream_,
                                               CU_STREAM_NON_BLOCKING);
    }
    ~Copy() override {
      CUDADriver::get_instance().stream_destroy(stream_);
    }
    Arch arch() const override {
      return Arch::cuda;
    }
    Device &device() const override {
      return device_;
    }
    std::size_t workspace_bytes() const override {
      return 0;
    }
    NativeOperationMode execution_mode() const override {
      return NativeOperationMode::cuda_synchronous;
    }
    void run_cuda_synchronous() override {
      ++calls_;
      CUDADriver::get_instance().memcpy_device_to_device_async(
          destination_, source_, 64, stream_);
      if (fail_)
        throw std::runtime_error("provider failed after submission");
    }

   private:
    Device &device_;
    void *source_, *destination_, *stream_{nullptr};
    int &calls_;
    bool fail_;
  };
  auto copy = std::make_shared<Copy>(device, pointers[0], pointers[1], calls);
  auto fill = [&](int index, uint32_t value) {
    return FixedExecution::BufferFill{arrays[index]->get_ptr(), 64, value};
  };
  auto read = [&](int index, uint32_t expected) {
    std::array<uint32_t, 16> values;
    driver.memcpy_device_to_host(values.data(), pointers[index], 64);
    for (auto value : values)
      EXPECT_EQ(value, expected);
  };
  FixedExecution mixed(Arch::cuda, device, nullptr,
                       {fill(0, 7), fill(2, 0), copy,
                        FixedExecution::BufferCopy{arrays[2]->get_ptr(),
                                                   arrays[1]->get_ptr(), 64}});
  EXPECT_EQ(mixed.execution_kind(), 3);
  EXPECT_EQ(calls, 0);
  read(0, 8);
  for (int i = 0; i < 3; ++i) {
    mixed.run();
    read(2, 7);
  }
  EXPECT_EQ(calls, 3);
  auto second = std::make_shared<Copy>(device, pointers[1], pointers[2], calls);
  FixedExecution only_actions(Arch::cuda, device, nullptr, {copy, second});
  driver.memsetd32(pointers[0], 13, 16);
  only_actions.run();
  read(2, 13);
  auto failing =
      std::make_shared<Copy>(device, pointers[0], pointers[1], calls, true);
  FixedExecution failure(Arch::cuda, device, nullptr,
                         {fill(0, 11), failing, fill(2, 99)});
  EXPECT_THROW(failure.run(), std::runtime_error);
  read(1, 11);
  read(2, 13);
  const auto after_failure = calls;
  EXPECT_THROW(failure.run(), std::runtime_error);
  EXPECT_EQ(calls, after_failure);
  NativeOperationRegistry registry;
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
  registry.register_factory<TestDescriptor>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [&](Arch, Device &, const TestDescriptor &) { return copy; });
  EXPECT_THROW(registry.prepare(Arch::cuda, device, TestDescriptor{}),
               std::invalid_argument);
}
#endif

TEST(NativeOperationRegistry, ReportsOnlyCompiledBackendComposition) {
  const auto &registry = native_operation_registry();
  for (const char *kind : {"compact", "sort"}) {
#ifdef TI_WITH_LLVM
    ASSERT_NE(registry.find(kind, 1, host_arch()), nullptr);
    EXPECT_EQ(registry.find(kind, 1, host_arch())->mode,
              NativeOperationMode::cpu);
#else
    EXPECT_EQ(registry.find(kind, 1, host_arch()), nullptr);
#endif
#ifdef TI_WITH_CUDA
    ASSERT_NE(registry.find(kind, 1, Arch::cuda), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::cuda)->mode,
              NativeOperationMode::cuda_capture);
#else
    EXPECT_EQ(registry.find(kind, 1, Arch::cuda), nullptr);
#endif
#ifdef TI_WITH_VULKAN
    ASSERT_NE(registry.find(kind, 1, Arch::vulkan), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::vulkan)->mode,
              NativeOperationMode::vulkan_inline);
#else
    EXPECT_EQ(registry.find(kind, 1, Arch::vulkan), nullptr);
#endif
    EXPECT_EQ(registry.find(kind, 2, host_arch()), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::metal), nullptr);
  }
  EXPECT_EQ(registry.find("missing.provider", 1, host_arch()), nullptr);
  for (const auto &capability : registry.capabilities()) {
    const auto *found =
        registry.find(capability.kind, capability.version, capability.arch);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(capability.mode, found->mode);
  }
}
TEST(NativeOperationRegistry,
     CublasLtFrozenDescriptionsValidateWithoutLoadingCode) {
  auto description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cublaslt","operation":"matmul_plan_f32","version":1,"bindings":{},"parameters":{"m":17,"n":9,"k":13,"batch_count":2,"layout":"row_major","epilogue":"identity","workspace_limit_bytes":4096,"provider_version":130300,"algorithm":{"workspace_bytes":0,"configuration":{"id":20,"tile":16,"split_k":1,"reduction":0,"swizzle":0,"custom":0,"stages":28,"inner_shape":null,"cluster_shape":null}}}})"));
  const auto &registry = native_operation_registry();
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  EXPECT_EQ(registry.find("cublaslt_matmul_f32", 1, host_arch()), nullptr);
  EXPECT_EQ(registry.find("cublaslt_matmul_f32", 1, Arch::vulkan), nullptr);
  for (const auto *patch :
       {R"({"m":0})", R"({"k":9223372036854775807})",
        R"({"batch_count":2147483648})", R"({"transpose_a":1})",
        R"({"alpha":1e100})", R"({"layout":"column_major"})",
        R"({"epilogue":"gelu"})", R"({"provider_version":0})",
        R"({"provider_version":4294967296})", R"({"unknown":1})"}) {
    auto invalid = description;
    for (const auto &[name, value] :
         liong::json::parse_complete(patch).obj.inner)
      invalid.parameters.obj.inner[name] = value;
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument)
        << patch;
  }
  for (const auto *patch :
       {R"({"tile":4294967296})", R"({"split_k":2147483648})",
        R"({"inner_shape":65536})", R"({"stages":1.0})", R"({"id":null})",
        R"({"unknown":1})"}) {
    auto invalid = description;
    for (const auto &[name, value] :
         liong::json::parse_complete(patch).obj.inner)
      invalid.parameters["algorithm"]["configuration"].obj.inner[name] = value;
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument)
        << patch;
  }
  auto absent = description;
  absent.parameters["algorithm"]["configuration"].obj.inner.erase(
      "inner_shape");
  EXPECT_NO_THROW(registry.decode(absent, NativeObjectKind::resource));
  absent.parameters["algorithm"]["configuration"].obj.inner.erase("id");
  EXPECT_THROW(registry.decode(absent, NativeObjectKind::resource),
               std::invalid_argument);
  auto too_large = description;
  too_large.parameters["algorithm"]["workspace_bytes"] =
      liong::json::JsonValue(4097);
  EXPECT_THROW(registry.decode(too_large, NativeObjectKind::resource),
               std::invalid_argument);
}

TEST(NativeOperationRegistry, CublasLtOperationsRequireExactPorts) {
  auto description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cublaslt","operation":"matmul_f32","version":1,"parameters":{},"bindings":{"plan":"p","a":"a","b":"b","output":"c"}})"));
  const auto &registry = native_operation_registry();
  EXPECT_NO_THROW(registry.decode(description));
  description.bindings.erase("plan");
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  description.bindings["plan"] = "p";
  description.parameters.obj.inner["alpha"] = liong::json::JsonValue(2);
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
}

TEST(NativeOperationRegistry, CutensorDescriptionsValidateWithoutLoadingCode) {
  auto description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cutensor","operation":"contraction_plan_f32","version":1,"bindings":{},"parameters":{"a":{"shape":[2,3],"modes":[0,2]},"b":{"shape":[3,4],"modes":[2,1]},"c":{"shape":[2,4],"modes":[0,1]},"d":{"shape":[2,4],"modes":[0,1]}}})"));
  const auto &registry = native_operation_registry();
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  for (const auto *value : {"0", "256"}) {
    auto frozen = description;
    frozen.parameters.obj.inner["expected_workspace_bytes"] =
        liong::json::parse_complete(value);
    EXPECT_NO_THROW(registry.decode(frozen, NativeObjectKind::resource));
  }
  for (const auto *value : {"-1", "1.0", "true", "null", "\"256\""}) {
    auto invalid = description;
    invalid.parameters.obj.inner["expected_workspace_bytes"] =
        liong::json::parse_complete(value);
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument);
  }
  for (const auto *tensor :
       {R"({"shape":[],"modes":[]})", R"({"shape":[0,3],"modes":[0,2]})",
        R"({"shape":[2,3],"modes":[0,0]})",
        R"({"shape":[2,3],"modes":[0,2147483648]})",
        R"({"shape":[2,3],"modes":[0.0,2]})",
        R"({"shape":[2,5],"modes":[0,2]})",
        R"({"shape":[2,9223372036854775807],"modes":[0,2]})"}) {
    auto invalid = description;
    invalid.parameters.obj.inner["a"] = liong::json::parse_complete(tensor);
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument);
  }
}

TEST(NativeOperationRegistry,
     CusparseLtDescriptionsValidateWithoutLoadingCode) {
  auto description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cusparselt","operation":"matmul_plan_f16","version":1,"bindings":{},"parameters":{"m":32,"n":48,"k":64}})"));
  const auto &registry = native_operation_registry();
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  for (const auto *patch :
       {R"({"m":0})", R"({"n":17})", R"({"k":-1})", R"({"alignment_bytes":3})",
        R"({"alignment_bytes":4294967296})", R"({"algorithm_id":-2})",
        R"({"split_k":0.0})", R"({"relu":1})", R"({"expected_resources":[]})",
        R"({"expected_resources":[1,2]})",
        R"({"expected_resources":[1,2,3,4]})", R"({"expected_resources":null})",
        R"({"expected_resources":[1,-1,0]})",
        R"({"expected_resources":[1,true,0]})",
        R"({"expected_resources":[1,0.0,0]})",
        R"({"m":9223372036854775792})"}) {
    auto invalid = description;
    for (const auto &[name, value] :
         liong::json::parse_complete(patch).obj.inner)
      invalid.parameters.obj.inner[name] = value;
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument);
  }
  description.parameters.obj.inner["expected_resources"] =
      liong::json::parse_complete("[1024,0,0]");
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
}

TEST(NativeOperationRegistry, RejectsAmbiguousDefinitionsBeforePublication) {
  NativeOperationRegistry registry;
  NativeOperationRegistry::Factory<TestDescriptor> factory =
      [](Arch, Device &, const TestDescriptor &) { return nullptr; };
  EXPECT_THROW(registry.register_factory<TestDescriptor>(
                   host_arch(), NativeOperationMode::cpu, factory),
               std::invalid_argument);
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
  EXPECT_THROW(
      registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {}),
      std::invalid_argument);
  EXPECT_THROW(registry.define<ConflictingDescriptor>(
                   [](Device &, const ConflictingDescriptor &) {}),
               std::invalid_argument);
  EXPECT_THROW(registry.register_factory<ConflictingDescriptor>(
                   host_arch(), NativeOperationMode::cpu,
                   [](Arch, Device &, const ConflictingDescriptor &) {
                     return nullptr;
                   }),
               std::invalid_argument);
  EXPECT_THROW(registry.register_factory<TestDescriptor>(
                   host_arch(), NativeOperationMode::cuda_capture, factory),
               std::invalid_argument);
  registry.register_factory<TestDescriptor>(host_arch(),
                                            NativeOperationMode::cpu, factory);
  EXPECT_THROW(registry.register_factory<TestDescriptor>(
                   host_arch(), NativeOperationMode::cpu, factory),
               std::invalid_argument);
  ASSERT_EQ(registry.capabilities().size(), 1);
}

TEST(NativeOperationRegistry, AmgxDescriptionsValidateWithoutLoadingCode) {
  const auto &registry = native_operation_registry();
  auto description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"amgx","operation":"solver_csr","version":1,"bindings":{},"parameters":{"rows":17,"nonzeros":49,"dtype":"f64","config":"config_version=2"}})"));
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
  for (const auto *patch :
       {R"({"rows":0})", R"({"nonzeros":-1})", R"({"rows":2147483648})",
        R"({"dtype":"f16"})", R"({"config":""})", R"({"compute_residual":1})",
        R"({"unknown":1})"}) {
    auto invalid = description;
    for (const auto &[key, value] :
         liong::json::parse_complete(patch).obj.inner)
      invalid.parameters.obj.inner[key] = value;
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument);
  }
  auto solve = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"amgx","operation":"solve","version":1,"bindings":{"solver":"s","rhs":"b","solution":"x","status":"status"},"parameters":{"initial_guess":"upload","retain_initial_guess":true}})"));
  EXPECT_NO_THROW(registry.decode(solve));
  for (const auto *patch :
       {R"({"initial_guess":"automatic"})", R"({"initial_guess":2})",
        R"({"retain_initial_guess":1})"}) {
    auto invalid = solve;
    for (const auto &[key, value] :
         liong::json::parse_complete(patch).obj.inner)
      invalid.parameters.obj.inner[key] = value;
    EXPECT_THROW(registry.decode(invalid), std::invalid_argument);
  }
  for (const auto *kind :
       {"amgx_setup_csr", "amgx_replace_coefficients", "amgx_solve"}) {
    EXPECT_EQ(registry.find(kind, 1, host_arch()), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::vulkan), nullptr);
#ifdef TI_WITH_CUDA
    ASSERT_NE(registry.find(kind, 1, Arch::cuda), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::cuda)->mode,
              NativeOperationMode::cuda_synchronous);
#endif
  }
}

TEST(NativeOperationRegistry, BlasVectorDescriptionsAndBackendRegistration) {
  const auto &registry = native_operation_registry();
  for (const auto *kind : {"axpy_f32", "scal_f32", "nrm2_f32"}) {
    auto description = NativeOperationDescription::parse(liong::json::parse_complete(
        R"({"kind":"native","provider":"cublas","operation":"scal_f32","version":1,"parameters":{"size":2147483648},"bindings":{"x":"x","context":"shared"}})"));
    description.operation = kind;
    if (description.operation == "axpy_f32")
      description.bindings.emplace("y", "y");
    if (description.operation == "nrm2_f32")
      description.bindings.emplace("output", "result");
    else
      description.bindings.emplace("alpha", "device_alpha");
    // Decoding is independent of device allocation or optional library load.
    EXPECT_NO_THROW(registry.decode(description));
    for (const auto *patch : {R"({"size":-1})", R"({"size":1.5})",
                              R"({"size":9223372036854775807})",
                              R"({"alpha":true})", R"({"unknown":1})"}) {
      auto invalid = description;
      for (const auto &[name, value] :
           liong::json::parse_complete(patch).obj.inner)
        invalid.parameters.obj.inner[name] = value;
      EXPECT_THROW(registry.decode(invalid), std::invalid_argument);
    }
    EXPECT_EQ(registry.find(kind, 1, host_arch()), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::vulkan), nullptr);
#ifdef TI_WITH_CUDA
    ASSERT_NE(registry.find(kind, 1, Arch::cuda), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::cuda)->mode,
              NativeOperationMode::cuda_capture);
#endif
  }
}

TEST(NativeOperationRegistry, GemmRejectsOverflowAndInvalidScalars) {
  GemmF32Operation op{{}, {}, {}, 17, 9, 13};
  EXPECT_EQ(gemm_f32_element_counts(op),
            (std::array<std::size_t, 3>{221, 117, 153}));
  op.rows = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(gemm_f32_element_counts(op), std::invalid_argument);
  op.rows = 0;
  EXPECT_THROW(gemm_f32_element_counts(op), std::invalid_argument);
  op.rows = 17;
  op.alpha = std::numeric_limits<float>::infinity();
  EXPECT_THROW(gemm_f32_element_counts(op), std::invalid_argument);
  op.alpha = 1;
  op.beta = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(gemm_f32_element_counts(op), std::invalid_argument);
}

#ifdef TI_WITH_LLVM
TEST(NativeOperationRegistry,
     FixedExecutionRetainsTypedFaultsAndRejectsReplay) {
  // Exercise error transport without damaging a real device/context. A CPU
  // callback injects the same typed errors that a backend driver would throw.
  cpu::CpuDevice device;
  class InjectedError final : public NativeOperation {
   public:
    explicit InjectedError(Device &device) : device_(device) {
    }
    Arch arch() const override {
      return host_arch();
    }
    Device &device() const override {
      return device_;
    }
    std::size_t workspace_bytes() const override {
      return 0;
    }
    void run_cpu() override {
      ++calls;
      if (code)
        throw BackendRuntimeError(Arch::cuda, code, "injected native call",
                                  "test error");
    }
    int calls{0}, code{1};

   private:
    Device &device_;
  };
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 101);
  auto operation = std::make_shared<InjectedError>(device);
  FixedExecution execution(host_arch(), device, nullptr, {operation}, nullptr,
                           domain);
  EXPECT_THROW(execution.run(), BackendRuntimeError);
  EXPECT_TRUE(
      domain->submission_allowed());  // Invalid value is operation-local.
  operation->code = 0;
  execution.run();
  EXPECT_EQ(operation->calls, 2);
  operation->code =
      700;  // CUDA illegal address, injected without a driver call.
  EXPECT_THROW(execution.run(), BackendRuntimeError);
  ASSERT_TRUE(domain->has_fatal_fault());
  EXPECT_EQ(domain->snapshot().first_fault->operation, "injected native call");
  operation->code = 0;
  EXPECT_THROW(execution.run(), TaichiRuntimeError);
  EXPECT_EQ(operation->calls, 3);
  EXPECT_THROW(FixedExecution(host_arch(), device, nullptr, {operation},
                              nullptr, domain),
               TaichiRuntimeError);
  auto independent = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 102);
  FixedExecution other(host_arch(), device, nullptr, {operation}, nullptr,
                       independent);
  other.run();
  EXPECT_EQ(operation->calls, 4);
  EXPECT_TRUE(independent->submission_allowed());
  independent->mark_finalized();
  EXPECT_THROW(other.run(), TaichiRuntimeError);
  EXPECT_EQ(operation->calls, 4);
}

class TestOperation final : public NativeOperation {
 public:
  TestOperation(Arch arch, Device &device, int &runs, int &retired)
      : arch_(arch), device_(device), runs_(runs), retired_(retired) {
  }
  ~TestOperation() override {
    ++retired_;
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void run_cpu() override {
    ++runs_;
  }

 private:
  Arch arch_;
  Device &device_;
  int &runs_, &retired_;
};

TEST(NativeOperationRegistry, RecordingInitializationIsLocalAndPrecedesWork) {
  cpu::CpuDevice device;
  int runs = 0, retired = 0;
  TestOperation marker(host_arch(), device, runs, retired);
  class SequenceOperation final : public NativeOperation {
   public:
    SequenceOperation(NativeResource &marker, bool write, int &runs)
        : marker_(marker), write_(write), runs_(runs) {
    }
    Arch arch() const override {
      return marker_.arch();
    }
    Device &device() const override {
      return marker_.device();
    }
    std::size_t workspace_bytes() const override {
      return 0;
    }
    void run_cpu() override {
      ++runs_;
    }
    void validate_recording(NativeRecordingState &state) const override {
      if (write_)
        state.initialized.insert(&marker_);
      else if (!state.initialized.count(&marker_))
        throw std::invalid_argument("read before initialization");
    }

   private:
    NativeResource &marker_;
    bool write_;
    int &runs_;
  };
  auto writer = std::make_shared<SequenceOperation>(marker, true, runs);
  auto reader = std::make_shared<SequenceOperation>(marker, false, runs);
  std::vector<FixedExecution::Operation> bad{reader, writer};
  EXPECT_THROW(FixedExecution(host_arch(), device, nullptr, bad),
               std::invalid_argument);
  EXPECT_EQ(runs, 0);
  FixedExecution good(host_arch(), device, nullptr, {writer, reader, reader});
  EXPECT_EQ(runs, 0);
  good.run();
  good.run();
  EXPECT_EQ(runs, 6);
  EXPECT_THROW(FixedExecution(host_arch(), device, nullptr, bad),
               std::invalid_argument);
  NativeRecordingState sequence;
  FixedExecution initialize(host_arch(), device, nullptr, {writer}, &sequence);
  FixedExecution replay(host_arch(), device, nullptr, {reader}, &sequence);
  EXPECT_EQ(runs, 6);
  initialize.run();
  replay.run();
  replay.run();
  EXPECT_EQ(runs, 9);
  NativeRecordingState independent;
  EXPECT_THROW(
      FixedExecution(host_arch(), device, nullptr, {reader}, &independent),
      std::invalid_argument);
  EXPECT_TRUE(independent.initialized.empty());
  struct SolverFacts {
    bool has_solution{false};
  };
  sequence.provider_state<SolverFacts>(&marker).has_solution = true;
  auto tentative = sequence;
  tentative.provider_state<SolverFacts>(&marker).has_solution = false;
  EXPECT_TRUE(sequence.provider_state<SolverFacts>(&marker).has_solution);
  EXPECT_FALSE(independent.provider_state<SolverFacts>(&marker).has_solution);
  auto conditional = sequence.control_branch();
  EXPECT_THROW(conditional.provider_state<SolverFacts>(&marker),
               std::invalid_argument);

  auto allocation = device.allocate_memory_unique({sizeof(std::uint32_t)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto predicate = std::move(allocation.first);
  auto loop = std::make_shared<FixedExecution::While>();
  loop->predicate = predicate->get_ptr();
  loop->max_iterations = 0;
  loop->condition = {FixedExecution::BufferFill{predicate->get_ptr(), 4, 0}};
  loop->body = {writer, reader};
  // A conditional initializer cannot satisfy a reader after a zero-trip loop.
  EXPECT_THROW(FixedExecution(host_arch(), device, nullptr, {loop, reader}),
               std::invalid_argument);
  EXPECT_EQ(runs, 9);
  // The initial condition is unconditional even with a zero iteration budget.
  loop->condition.emplace_back(writer);
  loop->body = {reader};
  FixedExecution zero_trip(host_arch(), device, nullptr, {loop, reader});
  EXPECT_EQ(runs, 9);
  zero_trip.run();
  EXPECT_EQ(runs, 11);
}

TEST(NativeOperationRegistry,
     QueriesAreColdAndPreparedObjectsOwnTheirLifetime) {
  cpu::CpuDevice device;
  int validations = 0, factories = 0, runs = 0, retired = 0;
  std::shared_ptr<NativeOperation> operation;
  {
    NativeOperationRegistry registry;
    registry.define<TestDescriptor>([&](Device &, const TestDescriptor &value) {
      ++validations;
      if (!value.valid)
        throw std::invalid_argument("invalid descriptor");
    });
    registry.register_factory<TestDescriptor>(
        host_arch(), NativeOperationMode::cpu,
        [&](Arch arch, Device &device, const TestDescriptor &) {
          ++factories;
          return std::make_shared<TestOperation>(arch, device, runs, retired);
        });
    ASSERT_NE(registry.find(TestDescriptor::kind, 1, host_arch()), nullptr);
    EXPECT_EQ(validations, 0);
    EXPECT_EQ(factories, 0);
    EXPECT_THROW(registry.prepare(host_arch(), device, TestDescriptor{false}),
                 std::invalid_argument);
    EXPECT_THROW(registry.prepare(host_arch(), device, ConflictingDescriptor{}),
                 std::invalid_argument);
    EXPECT_THROW(registry.prepare(Arch::cuda, device, TestDescriptor{}),
                 std::invalid_argument);
    EXPECT_EQ(validations, 1);
    EXPECT_EQ(factories, 0);
    operation = registry.prepare(host_arch(), device, TestDescriptor{});
    EXPECT_EQ(factories, 1);
    EXPECT_EQ(runs, 0);
    EXPECT_EQ(retired, 0);
  }
  operation->run_cpu();
  EXPECT_EQ(runs, 1);
  EXPECT_EQ(retired, 0);
  operation.reset();
  EXPECT_EQ(retired, 1);
}

TEST(NativeOperationRegistry, RejectsNullOrForeignFactoryResults) {
  cpu::CpuDevice device, other;
  int runs = 0, retired = 0;
  for (int mode = 0; mode < 3; ++mode) {
    NativeOperationRegistry registry;
    registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
    registry.register_factory<TestDescriptor>(
        host_arch(), NativeOperationMode::cpu,
        [&](Arch arch, Device &device,
            const TestDescriptor &) -> std::shared_ptr<NativeOperation> {
          if (!mode)
            return nullptr;
          return std::make_shared<TestOperation>(mode == 1 ? Arch::cuda : arch,
                                                 mode == 2 ? other : device,
                                                 runs, retired);
        });
    EXPECT_THROW(registry.prepare(host_arch(), device, TestDescriptor{}),
                 std::invalid_argument);
  }
  EXPECT_EQ(runs, 0);
  EXPECT_EQ(retired, 2);
}

TEST(NativeOperationRegistry, SchemaDecodesColdAndUsesTheExistingTypedFactory) {
  cpu::CpuDevice device;
  NativeOperationRegistry registry;
  int decodes = 0, probes = 0, factories = 0, runs = 0, retired = 0;
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &value) {
    if (!value.valid)
      throw std::invalid_argument("Invalid test storage");
  });
  registry.register_factory<TestDescriptor>(
      host_arch(), NativeOperationMode::cpu,
      [&](Arch arch, Device &device, const TestDescriptor &) {
        ++factories;
        return std::make_shared<TestOperation>(arch, device, runs, retired);
      },
      [&] {
        ++probes;
        return NativeOperationAvailability{true, 1};
      });
  NativeOperationRegistry::Decoder<TestDescriptor> decoder =
      [&](const liong::json::JsonValue &parameters)
      -> NativeOperationRegistry::Binder<TestDescriptor> {
    ++decodes;
    operation_json::fields(parameters, {"bytes"});
    const auto bytes = operation_json::integer(parameters["bytes"]);
    return [bytes](const NativeOperationBindings &bindings) {
      return TestDescriptor{bindings.at("input").array().bytes == bytes};
    };
  };
  registry.register_schema<TestDescriptor>(
      "test.provider", "consume", {"input"}, decoder, "legacy.consume");
  const auto description =
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"test.provider","operation":"consume","version":1,
          "parameters":{"bytes":12},"bindings":{"input":"caller_name"}})"));
  const auto decoded = registry.decode(description);
  EXPECT_EQ(decodes, 1);
  EXPECT_EQ(probes, 0);
  EXPECT_EQ(factories, 0);
  NativeOperationBindings bindings{
      {"input", NativeArrayBinding{{}, 12, PrimitiveType::f32, {3}, {}}}};
  auto op = registry.prepare(host_arch(), device, decoded, bindings);
  EXPECT_EQ(factories, 1);
  EXPECT_EQ(probes, 1);
  for (int i = 0; i < 3; ++i)
    op->run_cpu();
  EXPECT_EQ(runs, 3);
  EXPECT_EQ(decodes, 1);
  EXPECT_EQ(probes, 1);
  bindings.at("input") = NativeArrayBinding{{}, 8, PrimitiveType::f32, {2}, {}};
  EXPECT_THROW(registry.prepare(host_arch(), device, decoded, bindings),
               std::invalid_argument);
  EXPECT_EQ(factories, 1);
  EXPECT_THROW(registry.prepare(host_arch(), device, decoded, {}),
               std::invalid_argument);
  auto invalid = description;
  invalid.bindings.emplace("extra", "x");
  EXPECT_THROW(registry.decode(invalid), std::invalid_argument);
  invalid = description;
  invalid.provider = "other.provider";
  EXPECT_THROW(registry.decode(invalid), std::invalid_argument);
  EXPECT_THROW(registry.register_schema<TestDescriptor>(
                   "test.provider", "consume", {"input"}, decoder),
               std::invalid_argument);
  auto legacy = registry.lower_legacy(liong::json::parse_complete(
      R"({"kind":"legacy.consume","input":"caller_name","bytes":12})"));
  ASSERT_TRUE(legacy.has_value());
  EXPECT_EQ(legacy->key(), description.key());
  EXPECT_EQ(legacy->bindings, description.bindings);
  EXPECT_NO_THROW(registry.decode(*legacy));
  op.reset();
  EXPECT_EQ(retired, 1);
}

TEST(NativeOperationRegistry, OptionalProviderProbeIsColdAndPrecedesFactory) {
  cpu::CpuDevice device;
  NativeOperationRegistry registry;
  bool available = false;
  int probes = 0, factories = 0, runs = 0, retired = 0;
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
  registry.register_factory<TestDescriptor>(
      host_arch(), NativeOperationMode::cpu,
      [&](Arch arch, Device &device, const TestDescriptor &) {
        ++factories;
        return std::make_shared<TestOperation>(arch, device, runs, retired);
      },
      [&] {
        ++probes;
        return NativeOperationAvailability{available, available ? 12001u : 0u};
      });
  EXPECT_NE(registry.find(TestDescriptor::kind, 1, host_arch()), nullptr);
  EXPECT_EQ(registry.capabilities().size(), 1);
  EXPECT_EQ(probes, 0);
  EXPECT_FALSE(registry.probe("unrelated", 1, host_arch()).available);
  EXPECT_EQ(probes, 0);
  EXPECT_FALSE(registry.probe(TestDescriptor::kind, 1, host_arch()).available);
  EXPECT_THROW(registry.prepare(host_arch(), device, TestDescriptor{}),
               std::invalid_argument);
  EXPECT_EQ(factories, 0);
  available = true;
  EXPECT_EQ(
      registry.probe(TestDescriptor::kind, 1, host_arch()).library_version,
      12001);
  auto prepared = registry.prepare(host_arch(), device, TestDescriptor{});
  EXPECT_EQ(factories, 1);
  const auto cold_probes = probes;
  prepared->run_cpu();
  prepared->run_cpu();
  EXPECT_EQ(probes, cold_probes);
  EXPECT_EQ(runs, 2);
}
TEST(NativeOperationRegistry, ContextFactoryReceivesThePreparationSession) {
  cpu::CpuDevice device;
  NativeOperationRegistry registry;
  int runs = 0, retired = 0;
  NativeProviderSession *probed = nullptr;
  registry.define<TestDescriptor>([](Device &, const TestDescriptor &) {});
  registry.register_context_factory<TestDescriptor>(
      host_arch(), NativeOperationMode::cpu,
      [&](NativePreparationContext &context, const TestDescriptor &) {
        EXPECT_EQ(probed, &context.providers);
        return std::make_shared<TestOperation>(context.arch, context.device,
                                               runs, retired);
      },
      [&](NativeProviderSession &providers) {
        probed = &providers;
        return NativeOperationAvailability{true, 0};
      });
  NativePreparationContext first(host_arch(), device);
  auto operation = registry.prepare(first, TestDescriptor{});
  EXPECT_EQ(probed, &first.providers);
  NativePreparationContext second(host_arch(), device);
  auto another = registry.prepare(second, TestDescriptor{});
  EXPECT_EQ(probed, &second.providers);
  operation->run_cpu();
  another->run_cpu();
  EXPECT_EQ(runs, 2);
}

TEST(NativeOperationRegistry, HostBlobsAreSharedImmutableResourceBindings) {
  cpu::CpuDevice device;
  auto bytes = std::make_shared<const NativeHostBlobResource::Bytes>(
      NativeHostBlobResource::Bytes{0, 255, 3});
  std::weak_ptr<const NativeHostBlobResource::Bytes> retained = bytes;
  auto first =
      std::make_shared<NativeHostBlobResource>(host_arch(), device, bytes);
  auto second =
      std::make_shared<NativeHostBlobResource>(host_arch(), device, bytes);
  NativeOperationBindings bindings{{"code", NativeBinding(first)}};
  EXPECT_EQ(&first->bytes(), &second->bytes());
  EXPECT_EQ(first->workspace_bytes(), 0);
  EXPECT_EQ(&first->device(), &device);
  EXPECT_THROW(bindings.at("code").array(), std::invalid_argument);
  bytes.reset();
  first.reset();
  second.reset();
  EXPECT_FALSE(retained.expired());
  EXPECT_EQ(bindings.at("code").resource<NativeHostBlobResource>()->bytes()[1],
            255);
  bindings.clear();
  EXPECT_TRUE(retained.expired());
}

struct ResourceDescriptor {
  static constexpr const char *kind = "test.resource";
  static constexpr std::uint32_t version = 1;
  int id;
  std::shared_ptr<NativeResource> parent;
};

class TestResource final : public NativeResource {
 public:
  TestResource(Device &device,
               ResourceDescriptor value,
               std::vector<int> &retired)
      : device_(device), value_(std::move(value)), retired_(retired) {
  }
  ~TestResource() override {
    retired_.push_back(value_.id);
  }
  Arch arch() const override {
    return host_arch();
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 7;
  }

 private:
  Device &device_;
  ResourceDescriptor value_;
  std::vector<int> &retired_;
};

TEST(NativeOperationRegistry,
     ResourcesUseTheSameRegistryAndRetainDependencies) {
  cpu::CpuDevice device;
  std::vector<int> retired;
  int factories = 0;
  NativeOperationRegistry registry;
  registry.define<ResourceDescriptor>(
      [](Device &device, const ResourceDescriptor &value) {
        if (value.parent && &value.parent->device() != &device)
          throw std::invalid_argument("foreign resource");
      },
      NativeObjectKind::resource);
  registry.register_factory<ResourceDescriptor>(
      host_arch(), NativeOperationMode::cpu,
      [&](Arch, Device &device, const ResourceDescriptor &value) {
        ++factories;
        return std::make_shared<TestResource>(device, value, retired);
      });
  registry.register_schema<ResourceDescriptor>(
      "test", "resource", {},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<ResourceDescriptor> {
        const auto id = static_cast<int>(parameters["id"]);
        return [id](const NativeOperationBindings &bindings) {
          const auto parent = bindings.find("parent");
          return ResourceDescriptor{
              id, parent == bindings.end()
                      ? nullptr
                      : parent->second.resource<TestResource>()};
        };
      },
      {}, {"parent"});
  auto root = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"test","operation":"resource","version":1,
      "parameters":{"id":1},"bindings":{}})"));
  EXPECT_THROW(registry.decode(root), std::invalid_argument);
  EXPECT_TRUE(registry.capabilities().empty());
  EXPECT_EQ(registry.find(ResourceDescriptor::kind, 1, host_arch()), nullptr);
  EXPECT_FALSE(
      registry.probe(ResourceDescriptor::kind, 1, host_arch()).available);
  EXPECT_TRUE(registry
                  .probe(ResourceDescriptor::kind, 1, host_arch(),
                         NativeObjectKind::resource)
                  .available);
  const auto decoded = registry.decode(root, NativeObjectKind::resource);
  EXPECT_EQ(factories, 0);
  EXPECT_THROW(registry.prepare(host_arch(), device, decoded, {}),
               std::invalid_argument);
  auto parent = registry.prepare_resource(host_arch(), device, decoded, {});
  std::weak_ptr<NativeResource> weak = parent;
  NativeOperationBindings bindings{{"parent", parent}};
  EXPECT_THROW(bindings.at("parent").array(), std::invalid_argument);
  root.parameters = liong::json::parse_complete(R"({"id":2})");
  root.bindings.emplace("parent", "root_slot");
  const auto child_description =
      registry.decode(root, NativeObjectKind::resource);
  auto child = registry.prepare_resource(host_arch(), device, child_description,
                                         bindings);
  EXPECT_EQ(factories, 2);
  bindings.clear();
  parent.reset();
  EXPECT_FALSE(weak.expired());
  child.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(retired, (std::vector<int>{2, 1}));
}
#endif
}  // namespace
}  // namespace taichi::lang
