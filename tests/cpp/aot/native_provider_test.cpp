#include "gtest/gtest.h"

#include "taichi/common/dynamic_loader.h"
#include "taichi/runtime/native_provider.h"
#include "taichi/external_runtime/forge_cublaslt_provider.h"
#include "taichi/external_runtime/forge_cub_source_provider.h"
#include "tests/native/runtime_provider_stub.h"

#ifdef _WIN32
#include "taichi/platform/windows/windows.h"
#else
#include <dlfcn.h>
#endif

namespace taichi::lang {
namespace {
using provider_stub::ExecutionApi;
using provider_stub::Mode;
const NativeProviderDefinition definition{"stub", "stub_query", "2.x"};

class NativeProvider : public ::testing::Test {
 protected:
  void SetUp() override {
    controller = std::make_unique<DynamicLoader>(TI_TEST_RUNTIME_PROVIDER_PATH);
    ASSERT_TRUE(controller->loaded());
    controller->load_function("stub_set_mode", set_mode);
    controller->load_function("stub_stats", stats);
    set_mode(Mode::good);
    baseline = stats();
    ASSERT_EQ(baseline.live, 0);
  }
  void TearDown() override {
    EXPECT_EQ(stats().live, 0);
  }
  std::shared_ptr<NativeProviderModule> load() {
    return NativeProviderModule::load(definition,
                                      TI_TEST_RUNTIME_PROVIDER_PATH);
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

  std::unique_ptr<DynamicLoader> controller;
  void (*set_mode)(Mode){nullptr};
  provider_stub::Stats (*stats)(){nullptr};
  provider_stub::Stats baseline{};
};

TEST_F(NativeProvider, RejectsMissingLocationsAndQuery) {
  EXPECT_THROW(NativeProviderModule::load({}, TI_TEST_RUNTIME_PROVIDER_PATH),
               std::invalid_argument);
  EXPECT_THROW(NativeProviderModule::load(definition, ""),
               std::invalid_argument);
  expect_error(
      [&] {
        NativeProviderModule::load(definition, "nonexistent-adapter.dll");
      },
      TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE);
  auto wrong = definition;
  wrong.query_symbol = "absent_query_symbol";
  expect_error(
      [&] { NativeProviderModule::load(wrong, TI_TEST_RUNTIME_PROVIDER_PATH); },
      TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
}

TEST_F(NativeProvider, AuxiliaryLibrariesAreLazySharedAndRetained) {
  std::shared_ptr<NativeLibrary> retained;
  std::weak_ptr<NativeLibrary> weak;
  {
    NativeLibraryLocations locations{{"used", TI_TEST_RUNTIME_PROVIDER_PATH},
                                     {"unused", "absent.dll"}};
    NativeProviderSession session({}, locations);
    locations["used"] = "changed.dll";
    retained = session.library("used");
    EXPECT_EQ(retained, session.library("used"));
    EXPECT_THROW(session.library("missing"), NativeProviderError);
    EXPECT_THROW(session.library("unused"), NativeProviderError);
    EXPECT_THROW(retained->symbol("missing_symbol"), NativeProviderError);
    weak = retained;
    EXPECT_EQ(stats().probes, baseline.probes);
    EXPECT_EQ(stats().creates, baseline.creates);
  }
  EXPECT_FALSE(weak.expired());
  auto read = reinterpret_cast<provider_stub::Stats (*)()>(
      retained->symbol("stub_stats"));
  EXPECT_EQ(read().creates, baseline.creates);
  retained.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_THROW(validate_native_library_location("", "path"),
               std::invalid_argument);
  EXPECT_THROW(validate_native_library_location("name", ""),
               std::invalid_argument);
  EXPECT_THROW(validate_native_library_location("name", std::string("a\0b", 3)),
               std::invalid_argument);
}

TEST_F(NativeProvider, RejectsMalformedAdapterBeforeVendorAccess) {
  for (auto mode :
       {Mode::short_api, Mode::api_version, Mode::short_info,
        Mode::info_version, Mode::wrong_id, Mode::wrong_family,
        Mode::missing_feature, Mode::empty_audit, Mode::missing_probe,
        Mode::missing_create, Mode::missing_destroy, Mode::missing_execution,
        Mode::missing_error}) {
    SCOPED_TRACE(static_cast<int>(mode));
    set_mode(mode);
    expect_error([&] { load(); }, TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  }
  set_mode(Mode::query_failure);
  expect_error([&] { load(); }, TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL);
  EXPECT_EQ(stats().probes, baseline.probes);
  EXPECT_EQ(stats().creates, baseline.creates);
}

TEST_F(NativeProvider, LoadsMetadataWithoutProbingAndCopiesProbeFacts) {
  auto module = load();
  EXPECT_EQ(module->info().provider_id, "stub");
  EXPECT_EQ(module->info().build_identity, "test-only");
  EXPECT_EQ(stats().probes, baseline.probes);
  EXPECT_EQ(stats().creates, baseline.creates);
  const auto first = module->probe("first-library");
  const auto second = module->probe();
  EXPECT_EQ(first.library_path, "first-library");
  EXPECT_EQ(second.library_path, "system-default");
  EXPECT_EQ(first.version_major, 2);
  EXPECT_EQ(first.version_minor, 7);
  EXPECT_EQ(first.cuda_runtime_version, 13000);
  EXPECT_EQ(first.build_version, "stub-build");
  EXPECT_EQ(stats().creates, baseline.creates);
  set_mode(Mode::short_runtime_info);
  expect_error([&] { module->probe(); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
}

TEST_F(NativeProvider, PreservesProviderFailureDetails) {
  auto module = load();
  set_mode(Mode::probe_failure);
  try {
    module->probe();
    FAIL();
  } catch (const NativeProviderError &error) {
    EXPECT_EQ(error.result(),
              TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE);
    EXPECT_NE(std::string(error.what()).find("stub: probe runtime"),
              std::string::npos);
    EXPECT_NE(std::string(error.what()).find("stub failure detail"),
              std::string::npos);
  }
}

TEST_F(NativeProvider, EmbeddedRuntimeNeedsNoExternalVendorSymbols) {
  set_mode(Mode::embedded_runtime);
  auto module = load();
  EXPECT_EQ(module->info().required_symbol_count, 0);
  EXPECT_TRUE(module->info().features &
              TI_FORGE_RUNTIME_PROVIDER_FEATURE_EMBEDDED_RUNTIME);
  EXPECT_EQ(stats().probes, baseline.probes);
  EXPECT_EQ(stats().creates, baseline.creates);
  std::weak_ptr<NativeProviderModule> retained = module;
  auto runtime = module->create_runtime();
  module.reset();
  EXPECT_FALSE(retained.expired());
  EXPECT_EQ(runtime->execution_api<ExecutionApi>().value(runtime->handle()), 7);
  runtime.reset();
  EXPECT_TRUE(retained.expired());
}

TEST_F(NativeProvider, RollsBackPartialCreationAndRejectsNullRuntime) {
  auto module = load();
  for (auto mode : {Mode::create_failure, Mode::create_failure_with_handle}) {
    set_mode(mode);
    expect_error([&] { module->create_runtime(); },
                 TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL);
    EXPECT_EQ(stats().live, 0);
  }
  EXPECT_EQ(stats().destroys - baseline.destroys, 1);
  set_mode(Mode::short_runtime_info);
  expect_error([&] { module->create_runtime(); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  EXPECT_EQ(stats().live, 0);
  EXPECT_EQ(stats().destroys - baseline.destroys, 2);
  set_mode(Mode::create_null_handle);
  expect_error([&] { module->create_runtime(); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL);
  EXPECT_EQ(stats().destroys - baseline.destroys, 2);
}

TEST_F(NativeProvider, RuntimeRetainsModuleAndExecutionTable) {
  auto module = load();
  std::weak_ptr<NativeProviderModule> weak = module;
  auto runtime = module->create_runtime("chosen-vendor");
  auto plan_lease = runtime;
  const auto api = runtime->execution_api<ExecutionApi>();
  EXPECT_EQ(runtime->info().library_path, "chosen-vendor");
  module.reset();
  runtime.reset();
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(api.value(plan_lease->handle()), 42);
  EXPECT_EQ(stats().live, 1);
  plan_lease.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(stats().destroys - baseline.destroys, 1);
}

TEST_F(NativeProvider, ChecksExecutionTableSizeVersionAndErrors) {
  auto runtime = load()->create_runtime();
  for (auto mode : {Mode::short_execution, Mode::execution_version}) {
    set_mode(mode);
    expect_error([&] { runtime->execution_api<ExecutionApi>(); },
                 TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  }
  set_mode(Mode::execution_failure);
  expect_error([&] { runtime->execution_api<ExecutionApi>(); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED);
  set_mode(Mode::good);
  expect_error([&] { runtime->execution_api<ExecutionApi>(2); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  EXPECT_EQ(runtime->execution_api<ExecutionApi>().value(runtime->handle()), 7);
}

TEST_F(NativeProvider, PreparationSnapshotsLocationsAndProbesOnce) {
  NativeProviderLocations locations{
      {"stub", {TI_TEST_RUNTIME_PROVIDER_PATH, "chosen-vendor"}},
      {"unused", {"missing-unused-adapter", "missing-unused-vendor"}}};
  std::shared_ptr<NativeProviderRuntime> retained;
  {
    NativeProviderSession first(locations);
    locations.at("stub").library_path = "another-vendor";
    NativeProviderSession second(locations);
    EXPECT_EQ(stats().probes, baseline.probes);
    EXPECT_EQ(first.probe(definition).library_path, "chosen-vendor");
    EXPECT_EQ(first.probe(definition).library_path, "chosen-vendor");
    EXPECT_EQ(stats().probes - baseline.probes, 1);
    EXPECT_EQ(second.probe(definition).library_path, "another-vendor");
    EXPECT_EQ(stats().probes - baseline.probes, 2);
    retained = first.runtime(definition);
    EXPECT_EQ(first.runtime(definition), retained);
    EXPECT_EQ(stats().creates - baseline.creates, 1);
    EXPECT_EQ(retained->info().library_path, "chosen-vendor");
    EXPECT_EQ(stats().probes - baseline.probes, 2);
    auto conflicting = definition;
    conflicting.supported_version_family = "wrong-family";
    expect_error([&] { first.probe(conflicting); },
                 TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH);
  }
  EXPECT_EQ(stats().live, 1);
  EXPECT_EQ(retained->execution_api<ExecutionApi>().value(retained->handle()),
            42);
  retained.reset();
  EXPECT_EQ(stats().live, 0);
  EXPECT_EQ(stats().destroys - baseline.destroys, 1);
}

TEST_F(NativeProvider, PreparationRejectsMissingOrMalformedLocations) {
  NativeProviderSession empty;
  expect_error([&] { empty.probe(definition); },
               TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE);
  EXPECT_THROW(validate_native_provider_location("", {"adapter", "vendor"}),
               std::invalid_argument);
  EXPECT_THROW(validate_native_provider_location("stub", {"", "vendor"}),
               std::invalid_argument);
  EXPECT_THROW(validate_native_provider_location(
                   "stub", {"adapter", std::string("bad\0path", 8)}),
               std::invalid_argument);
  EXPECT_EQ(stats().probes, baseline.probes);
}

TEST_F(NativeProvider, FailedDestructionRetainsAdapterInsteadOfThrowing) {
  auto runtime = load()->create_runtime();
  const auto handle = runtime->handle();
  // This injected provider failure deliberately retains an OS library
  // reference. The test controller then releases the stub allocation;
  // production providers must diagnose and fix their cleanup failure rather
  // than unload live code.
  EXPECT_EQ(stats().live, 1);
  set_mode(Mode::destroy_failure);
  EXPECT_NO_THROW(runtime.reset());
  EXPECT_EQ(stats().live, 1);
  EXPECT_EQ(stats().destroys - baseline.destroys, 1);
  set_mode(Mode::good);
  TiForgeRuntimeProviderQueryFn query;
  controller->load_function("stub_query", query);
  TiForgeRuntimeProviderApi api{};
  ASSERT_EQ(query(TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION, sizeof(api), &api),
            TI_FORGE_RUNTIME_PROVIDER_SUCCESS);
  EXPECT_EQ(api.destroy_runtime(handle), TI_FORGE_RUNTIME_PROVIDER_SUCCESS);
  controller.reset();
#ifdef _WIN32
  EXPECT_NE(GetModuleHandleA(TI_TEST_RUNTIME_PROVIDER_PATH), nullptr);
#else
  auto *retained =
      dlopen(TI_TEST_RUNTIME_PROVIDER_PATH, RTLD_NOLOAD | RTLD_LAZY);
  EXPECT_NE(retained, nullptr);
  if (retained) {
    dlclose(retained);
  }
#endif
  controller = std::make_unique<DynamicLoader>(TI_TEST_RUNTIME_PROVIDER_PATH);
  controller->load_function("stub_stats", stats);
}

#ifdef TI_TEST_CUTENSOR_PROVIDER_PATH
TEST(NativeProviderAdapters,
     RealAdaptersRejectMissingVendorThroughCommonOwner) {
  struct Adapter {
    NativeProviderDefinition definition;
    const char *path;
  };
  for (const auto &adapter :
       {Adapter{{"cutensor", TI_FORGE_CUTENSOR_PROVIDER_QUERY_SYMBOL,
                 "2.0.x-2.7.x"},
                TI_TEST_CUTENSOR_PROVIDER_PATH},
        Adapter{{"cusparselt", TI_FORGE_CUSPARSELT_PROVIDER_QUERY_SYMBOL,
                 "0.8.x-0.9.x"},
                TI_TEST_CUSPARSELT_PROVIDER_PATH},
        Adapter{{"amgx", TI_FORGE_AMGX_PROVIDER_QUERY_SYMBOL, "stable C API"},
                TI_TEST_AMGX_PROVIDER_PATH},
        Adapter{{"cublaslt", TI_FORGE_CUBLASLT_PROVIDER_QUERY_SYMBOL,
                 "documented f32 matmul configuration C API"},
                TI_TEST_CUBLASLT_PROVIDER_PATH},
        Adapter{{"cub_source", TI_FORGE_CUB_SOURCE_PROVIDER_QUERY_SYMBOL,
                 "source addon C ABI 1"},
                TI_TEST_CUB_SOURCE_PROVIDER_PATH}}) {
    SCOPED_TRACE(adapter.definition.provider_id);
    auto module = NativeProviderModule::load(adapter.definition, adapter.path);
    EXPECT_EQ(module->info().provider_id, adapter.definition.provider_id);
    EXPECT_FALSE(module->info().build_identity.empty());
    for (bool create : {false, true}) {
      try {
        const auto path = std::string(adapter.path) + ".missing-vendor";
        if (create) {
          module->create_runtime(path);
        } else {
          module->probe(path);
        }
        FAIL() << "missing vendor unexpectedly loaded";
      } catch (const NativeProviderError &error) {
        EXPECT_EQ(error.result(),
                  TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE);
        EXPECT_NE(std::string(error.what()).find("missing-vendor"),
                  std::string::npos);
      }
    }
  }
}
#endif

}  // namespace
}  // namespace taichi::lang
