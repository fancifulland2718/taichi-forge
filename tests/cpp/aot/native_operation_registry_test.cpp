#include "gtest/gtest.h"
#include <limits>
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/common/operation_json.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
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

#ifdef TI_WITH_CUDA
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
TEST(NativeOperationRegistry, CutensorDescriptionsValidateWithoutLoadingCode) {
  auto description = NativeOperationDescription::parse(liong::json::parse_complete(
      R"({"kind":"native","provider":"cutensor","operation":"contraction_plan_f32","version":1,"bindings":{},"parameters":{"a":{"shape":[2,3],"modes":[0,2]},"b":{"shape":[3,4],"modes":[2,1]},"c":{"shape":[2,4],"modes":[0,1]},"d":{"shape":[2,4],"modes":[0,1]}}})"));
  const auto &registry = native_operation_registry();
  EXPECT_NO_THROW(registry.decode(description, NativeObjectKind::resource));
  EXPECT_THROW(registry.decode(description), std::invalid_argument);
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
        R"({"split_k":0.0})", R"({"relu":1})",
        R"({"m":9223372036854775792})"}) {
    auto invalid = description;
    for (const auto &[name, value] :
         liong::json::parse_complete(patch).obj.inner)
      invalid.parameters.obj.inner[name] = value;
    EXPECT_THROW(registry.decode(invalid, NativeObjectKind::resource),
                 std::invalid_argument);
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
