#include "gtest/gtest.h"
#include "taichi/runtime/native_operation_registry.h"
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
#endif
}  // namespace
}  // namespace taichi::lang
