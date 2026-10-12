#include "gtest/gtest.h"
#include <limits>
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif

namespace taichi::lang {
namespace {
struct TestDescriptor {
  static constexpr const char *kind = "test.operation";
  static constexpr std::uint32_t version = 1;
  bool valid{true};
};
struct ConflictingDescriptor {
  static constexpr const char *kind = TestDescriptor::kind;
  static constexpr std::uint32_t version = TestDescriptor::version;
};

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
