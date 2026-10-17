#include "gtest/gtest.h"

#include <cstdlib>

#include "taichi/common/dynamic_loader.h"
#include "taichi/optix/forge_optix_runtime_provider.h"
#include "taichi/runtime/native_provider.h"

namespace taichi::lang {
namespace {

const NativeProviderDefinition definition{
    "optix", TI_FORGE_OPTIX_RUNTIME_PROVIDER_QUERY_SYMBOL,
    TI_FORGE_OPTIX_RUNTIME_VERSION_FAMILY};
constexpr const char *missing_runtime = "forge-test-missing-optix-runtime";

class OptixCommonProvider : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *path = std::getenv("TAICHI_FORGE_TEST_OPTIX_PROVIDER");
    if (!path || !*path) {
      GTEST_SKIP() << "Set TAICHI_FORGE_TEST_OPTIX_PROVIDER for a real adapter";
    }
    library = std::make_unique<DynamicLoader>(path);
    ASSERT_TRUE(library->loaded());
    library->load_function(TI_FORGE_OPTIX_RUNTIME_PROVIDER_QUERY_SYMBOL, query);
    ASSERT_EQ(query(TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION, sizeof(api), &api),
              TI_FORGE_RUNTIME_PROVIDER_SUCCESS);
    module = NativeProviderModule::load(definition, path);
  }

  template <typename Fn>
  void expect_error(Fn &&fn, TiForgeRuntimeProviderResult expected) {
    try {
      fn();
      FAIL() << "provider call unexpectedly succeeded";
    } catch (const NativeProviderError &error) {
      EXPECT_EQ(error.result(), expected) << error.what();
    }
  }

  std::unique_ptr<DynamicLoader> library;
  decltype(&taichi_forge_optix_runtime_provider_query) query{};
  TiForgeRuntimeProviderApi api{};
  std::shared_ptr<NativeProviderModule> module;
};

TEST_F(OptixCommonProvider, ValidatesMetadataAndOutputsBeforeVendorAccess) {
  EXPECT_EQ(module->info().provider_id, "optix");
  EXPECT_EQ(module->info().required_symbol_count, 1);
  EXPECT_FALSE(module->info().features &
               TI_FORGE_RUNTIME_PROVIDER_FEATURE_EMBEDDED_RUNTIME);
  EXPECT_EQ(query(0, sizeof(api), &api),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  EXPECT_EQ(query(TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION, sizeof(api) - 1, &api),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(query(TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION, sizeof(api), nullptr),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  TiForgeRuntimeInfo facts{};
  TiForgeRuntimeProviderRuntime handle = reinterpret_cast<void *>(1);
  EXPECT_EQ(api.probe_runtime(missing_runtime, &facts),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(api.create_runtime(missing_runtime, &handle, &facts),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(handle, nullptr);
  facts.struct_size = sizeof(facts);
  EXPECT_EQ(api.create_runtime(missing_runtime, nullptr, &facts),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(api.destroy_runtime(nullptr),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  expect_error([&] { module->probe(missing_runtime); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE);
  expect_error([&] { module->create_runtime(missing_runtime); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE);
  char diagnostic[256]{};
  EXPECT_GT(api.get_last_error(diagnostic, sizeof(diagnostic)), 1);
  EXPECT_NE(std::string(diagnostic).find("OptiX"), std::string::npos);
}

TEST_F(OptixCommonProvider, SharesLegacyLoaderWithoutCreatingADeviceContext) {
  const auto facts = module->probe();
  EXPECT_EQ(facts.version_major, 0);
  EXPECT_EQ(facts.version_minor, 0);
  EXPECT_EQ(facts.version_patch, 0);
  EXPECT_EQ(facts.cuda_runtime_version, 0);
  EXPECT_TRUE(facts.build_version.empty());
  EXPECT_TRUE(facts.library_path.empty());
  auto first = module->create_runtime();
  auto second = module->create_runtime();
  const auto execution = first->execution_api<TiForgeOptixExecutionApi>();
  EXPECT_EQ(execution.provider.provider_abi_version,
            TI_FORGE_OPTIX_PROVIDER_ABI_VERSION);
  EXPECT_EQ(execution.provider.struct_size, sizeof(TiForgeOptixProviderApi));
  EXPECT_NE(execution.provider.trace_faces, nullptr);
  EXPECT_NE(execution.provider.trace_instance_occlusion, nullptr);
  EXPECT_NE(execution.provider.update_instance_scene, nullptr);
  EXPECT_NE(execution.provider.get_program_api, nullptr);

  decltype(&taichi_forge_optix_provider_query) legacy_query{};
  library->load_function(TI_FORGE_OPTIX_PROVIDER_QUERY_SYMBOL, legacy_query);
  TiForgeOptixProviderApi legacy{};
  ASSERT_EQ(legacy_query(TI_FORGE_OPTIX_PROVIDER_ABI_VERSION, sizeof(legacy),
                         &legacy),
            TI_FORGE_OPTIX_SUCCESS);
  EXPECT_EQ(legacy.info.features, execution.provider.info.features);
  EXPECT_EQ(legacy.create_context, execution.provider.create_context);
  EXPECT_EQ(legacy.trace_faces, execution.provider.trace_faces);
  EXPECT_EQ(legacy.probe_runtime(nullptr), TI_FORGE_OPTIX_SUCCESS);
  EXPECT_EQ(legacy.probe_runtime(missing_runtime), TI_FORGE_OPTIX_ERROR_LIFETIME);
  expect_error([&] { module->probe(missing_runtime); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME);
  expect_error([&] { module->create_runtime(missing_runtime); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME);

  expect_error([&] { first->execution_api<TiForgeOptixExecutionApi>(99); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  TiForgeOptixExecutionApi output{};
  EXPECT_EQ(api.query_execution_api(first->handle(), 1, sizeof(output) - 1,
                                     &output),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(api.query_execution_api(nullptr, 1, sizeof(output), &output),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(api.query_execution_api(first->handle(), 1, sizeof(output), nullptr),
            TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  first.reset();
  EXPECT_EQ(legacy.probe_runtime(missing_runtime), TI_FORGE_OPTIX_ERROR_LIFETIME);
  EXPECT_NE(second->execution_api<TiForgeOptixExecutionApi>().provider.trace,
            nullptr);
  second.reset();
  EXPECT_EQ(legacy.probe_runtime(missing_runtime),
            TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE);
  // Last-lease retirement permits a fresh session after a failed load.
  auto reloaded = module->create_runtime();
  EXPECT_NE(reloaded->execution_api<TiForgeOptixExecutionApi>().provider.trace,
            nullptr);
}

TEST_F(OptixCommonProvider, SessionAndResourcesRetainOneLazyRuntime) {
  const auto path = std::getenv("TAICHI_FORGE_TEST_OPTIX_PROVIDER");
  std::shared_ptr<NativeProviderRuntime> retained;
  {
    NativeProviderSession session({{"optix", {path, ""}},
                                   {"unused", {"absent-adapter", ""}}});
    retained = session.runtime(definition);
    EXPECT_EQ(retained, session.runtime(definition));
    EXPECT_EQ(session.probe(definition).version_major, 0);
  }
  EXPECT_NE(retained->execution_api<TiForgeOptixExecutionApi>().provider.trace,
            nullptr);
  retained.reset();
  auto runtime = module->create_runtime();
  std::weak_ptr<NativeProviderModule> weak = module;
  library.reset();
  module.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_NE(runtime->execution_api<TiForgeOptixExecutionApi>().provider.trace,
            nullptr);
  runtime.reset();
  EXPECT_TRUE(weak.expired());
}

TEST_F(OptixCommonProvider, OptionalFixedQueryTablePreservesVersionOne) {
  decltype(&taichi_forge_optix_fixed_query_api) get_queries{};
  library->load_function(TI_FORGE_OPTIX_FIXED_QUERY_API_SYMBOL, get_queries);
  ASSERT_NE(get_queries, nullptr);
  TiForgeOptixFixedQueryApi queries{};
  EXPECT_EQ(get_queries(0, sizeof(queries), &queries), TI_FORGE_OPTIX_ERROR_ABI_MISMATCH);
  EXPECT_EQ(get_queries(1, sizeof(queries) - 1, &queries), TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(get_queries(1, sizeof(queries), nullptr), TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(get_queries(1, sizeof(queries), &queries), TI_FORGE_OPTIX_SUCCESS);
  EXPECT_EQ(queries.revision, 1);
  TiForgeOptixPreparedQuery query_handle = reinterpret_cast<void *>(1);
  EXPECT_EQ(queries.prepare(nullptr, nullptr, &query_handle), TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(query_handle, nullptr);
  EXPECT_EQ(queries.launch(nullptr, 0), TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(queries.destroy(nullptr), TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(queries.memory(nullptr, nullptr),
            TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT);
  auto runtime = module->create_runtime();
  const auto legacy = runtime->execution_api<TiForgeOptixExecutionApi>();
  const auto extended = runtime->execution_api<TiForgeOptixExecutionApiV2>(2);
  EXPECT_EQ(legacy.struct_size, sizeof(legacy));
  EXPECT_EQ(extended.provider.trace_faces, legacy.provider.trace_faces);
  EXPECT_EQ(extended.queries.prepare, queries.prepare);
  EXPECT_TRUE(extended.provider.info.features &
              TI_FORGE_OPTIX_FEATURE_PREPARED_FIXED_QUERY);
  EXPECT_EQ(
      api.query_execution_api(runtime->handle(), 2, sizeof(legacy), &queries),
      TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT);
  // Version 1 must not write an optional suffix into an old client's storage.
  struct OldClient {
    TiForgeOptixExecutionApi table;
    uint64_t canary = 0x1234567812345678;
  } old{};
  EXPECT_EQ(api.query_execution_api(runtime->handle(), 1, sizeof(old.table),
                                    &old.table),
            TI_FORGE_RUNTIME_PROVIDER_SUCCESS);
  EXPECT_EQ(old.canary, 0x1234567812345678);
}

}  // namespace
}  // namespace taichi::lang
