#include <cstddef>
#include <new>

#include <gtest/gtest.h>
#include <taichi/aot/graph_data.h>
#include <taichi/aot/module_builder.h>

namespace taichi::lang::aot {

namespace {
class ControlCapture final : public CudaGraphCaptureCommand {
 public:
  explicit ControlCapture(bool qualified) : qualified_(qualified) {
  }
  const char *kind() const override {
    return "control_qualification_test";
  }
  Program *program() const override {
    return nullptr;
  }
  bool supports(const std::unordered_map<std::string, IValue> &,
                Program &) const override {
    return false;
  }
  void prepare(const std::unordered_map<std::string, IValue> &,
               Program &) override {
    ADD_FAILURE() << "Eligibility must not prepare a provider";
  }
  void record(const std::unordered_map<std::string, IValue> &,
              Program &,
              void *) override {
    ADD_FAILURE() << "Eligibility must not record a provider";
  }
  bool supports_conditional_body() const override {
    return qualified_;
  }

 private:
  bool qualified_;
};
}  // namespace

TEST(GraphControlEligibilityTest, RequiresExplicitProviderQualification) {
  CompiledGraph graph;
  graph.dispatches.emplace_back();
  EXPECT_FALSE(graph.cuda_conditional_body_eligible());
  EXPECT_FALSE(graph.cuda_expanded_control_eligible());
  auto &dispatch = graph.dispatches.front();
  dispatch.cuda_capture_command = std::make_shared<ControlCapture>(false);
  EXPECT_FALSE(graph.cuda_conditional_body_eligible());
  EXPECT_FALSE(graph.cuda_expanded_control_eligible());
  dispatch.cuda_capture_command = std::make_shared<ControlCapture>(true);
  EXPECT_TRUE(graph.cuda_conditional_body_eligible());
  EXPECT_FALSE(graph.cuda_expanded_control_eligible());
  dispatch.cuda_bounded_dispatch.emplace();
  EXPECT_FALSE(graph.cuda_conditional_body_eligible());
  dispatch.cuda_bounded_dispatch.reset();
  graph.dispatches.emplace_back();
  EXPECT_FALSE(graph.cuda_conditional_body_eligible());
}

TEST(GraphReplayIdentityTest, TokenChangesWhenHostAddressIsReused) {
  alignas(CompiledGraphReplayIdentity)
      std::byte storage[sizeof(CompiledGraphReplayIdentity)];

  auto *first = ::new (storage) CompiledGraphReplayIdentity();
  const uint64_t first_value = first->value();
  first->~CompiledGraphReplayIdentity();

  auto *second = ::new (storage) CompiledGraphReplayIdentity();
  const uint64_t second_value = second->value();

  EXPECT_EQ(static_cast<const void *>(first),
            static_cast<const void *>(second));
  EXPECT_NE(first_value, 0);
  EXPECT_NE(second_value, 0);
  EXPECT_NE(first_value, second_value);

  second->~CompiledGraphReplayIdentity();
}

TEST(GraphCaptureRetryStateTest, TransientBackoffIsBoundedAndRecoverable) {
  CompiledGraphCaptureRetryState retry;

  for (uint32_t failure = 0; failure < 10; ++failure) {
    EXPECT_TRUE(retry.should_attempt());
    retry.record_transient_failure();
    const uint32_t skipped = uint32_t{1} << std::min<uint32_t>(failure, 5);
    EXPECT_EQ(retry.retry_backoff_remaining(), skipped);
    EXPECT_LE(skipped, CompiledGraphCaptureRetryState::kMaxBackoffInvocations);
    for (uint32_t i = 0; i < skipped; ++i) {
      EXPECT_FALSE(retry.should_attempt());
    }
  }

  EXPECT_EQ(retry.consecutive_transient_failures(), 10);
  retry.record_success();
  EXPECT_EQ(retry.consecutive_transient_failures(), 0);
  EXPECT_EQ(retry.retry_backoff_remaining(), 0);
  EXPECT_TRUE(retry.should_attempt());
}

TEST(GraphCaptureRetryStateTest, StructuralFailureDoesNotRetry) {
  CompiledGraphCaptureRetryState retry;
  EXPECT_TRUE(retry.should_attempt());
  retry.record_structural_failure();
  EXPECT_TRUE(retry.structurally_disabled());
  EXPECT_FALSE(retry.should_attempt());
  retry.record_success();
  EXPECT_FALSE(retry.should_attempt());
}

namespace {
class ExportBuilder final : public AotModuleBuilder {
 public:
  int compilations{0};
  void dump(const std::string &, const std::string &) const override {
  }
  const CompiledGraph &graph(const std::string &name) const {
    return graphs_.at(name);
  }

 private:
  void add_per_backend(const std::string &, lang::Kernel *) override {
    ++compilations;
  }
  void add_with_launch_abi_per_backend(const std::string &,
                                       lang::Kernel *,
                                       const std::string &abi) override {
    if (abi != "cuda_graph_gate_v1")
      throw std::invalid_argument("Unsupported test launch ABI");
    ++compilations;
  }
  void add_per_backend_tmpl(const std::string &,
                            const std::string &,
                            lang::Kernel *) override {
    ++compilations;
  }
};

class AotExportIdentityTest : public ::testing::Test {
 protected:
  // The counting backend never dereferences these distinct identity tokens.
  int identities_[3]{};
  lang::Kernel *kernel(int index) {
    return reinterpret_cast<lang::Kernel *>(&identities_[index]);
  }
  CompiledGraph source(
      std::initializer_list<std::pair<std::string, int>> entries) {
    CompiledGraph graph;
    for (const auto &[name, index] : entries) {
      graph.dispatches.emplace_back();
      graph.dispatches.back().kernel_name = name;
      graph.dispatches.back().ti_kernel = kernel(index);
    }
    return graph;
  }
  ExportBuilder builder;
};
}  // namespace

TEST_F(AotExportIdentityTest, ReusesPublishedEntriesWithoutChangingSource) {
  builder.add("public", kernel(0));
  const auto original = source({{"generated", 0}, {"other_name", 0}});
  const auto entries = builder.add_graph("first", original);
  builder.add_graph("second", original);
  EXPECT_EQ(entries.size(), 1);
  EXPECT_EQ(entries.at("public"), kernel(0));
  EXPECT_EQ(builder.compilations, 1);
  EXPECT_EQ(original.dispatches[0].kernel_name, "generated");
  EXPECT_EQ(original.dispatches[1].kernel_name, "other_name");
  EXPECT_EQ(builder.graph("first").dispatches[0].kernel_name, "public");
  EXPECT_EQ(builder.graph("second").dispatches[1].kernel_name, "public");
  EXPECT_ANY_THROW(builder.add_graph("first", original));
}

TEST_F(AotExportIdentityTest, KeepsSpecializationsAndExplicitAliases) {
  const auto entries = builder.add_graph(
      "first",
      source({{"generated", 0}, {"duplicate", 0}, {"specialized", 1}}));
  EXPECT_EQ(entries.size(), 2);
  EXPECT_EQ(builder.compilations, 2);
  builder.add("generated", kernel(0));
  EXPECT_EQ(builder.compilations, 2);
  builder.add("public_alias", kernel(0));
  EXPECT_EQ(builder.compilations, 3);
  EXPECT_ANY_THROW(builder.add("generated", kernel(1)));
  EXPECT_EQ(builder.compilations, 3);
}

TEST_F(AotExportIdentityTest, PreflightsWholeGraphBeforePublishing) {
  builder.add("occupied", kernel(0));
  EXPECT_ANY_THROW(
      builder.add_graph("bad", source({{"new", 1}, {"occupied", 2}})));
  EXPECT_EQ(builder.compilations, 1);
  EXPECT_ANY_THROW(
      builder.add_graph("bad", source({{"duplicate", 1}, {"duplicate", 2}})));
  EXPECT_EQ(builder.compilations, 1);
  builder.add_graph("bad", source({{"new", 1}, {"distinct", 2}}));
  EXPECT_EQ(builder.compilations, 3);
}

TEST_F(AotExportIdentityTest, SeparatesLaunchAbisAndRejectsOverwrites) {
  builder.add_with_launch_abi("gated", kernel(0), "cuda_graph_gate_v1");
  builder.add_with_launch_abi("gated", kernel(0), "cuda_graph_gate_v1");
  EXPECT_EQ(builder.compilations, 1);
  EXPECT_ANY_THROW(builder.add_graph("graph", source({{"gated", 0}})));
  EXPECT_ANY_THROW(builder.add("gated", kernel(0)));
  builder.add_graph("graph", source({{"ordinary", 0}}));
  EXPECT_EQ(builder.graph("graph").dispatches[0].kernel_name, "ordinary");
  EXPECT_EQ(builder.compilations, 2);
  EXPECT_ANY_THROW(
      builder.add_with_launch_abi("ordinary", kernel(0), "cuda_graph_gate_v1"));
  EXPECT_ANY_THROW(
      builder.add_with_launch_abi("failed", kernel(1), "unsupported"));
  builder.add("failed", kernel(1));
  EXPECT_EQ(builder.compilations, 3);
}

TEST_F(AotExportIdentityTest, ReusesTemplateEntriesWithTheirPublishedNames) {
  builder.add_kernel_template("template", "x=1", kernel(0));
  builder.add_kernel_template("template", "x=1", kernel(0));
  builder.add_graph("graph", source({{"generated", 0}}));
  EXPECT_EQ(builder.compilations, 1);
  EXPECT_EQ(builder.graph("graph").dispatches[0].kernel_name,
            "template__tmpl__x=1");
  EXPECT_ANY_THROW(builder.add_kernel_template("template", "x=1", kernel(1)));
  EXPECT_EQ(builder.compilations, 1);
}

TEST_F(AotExportIdentityTest, RejectsMissingKernelBeforeExport) {
  auto graph = source({{"valid", 0}, {"missing", 1}});
  graph.dispatches[1].ti_kernel = nullptr;
  EXPECT_ANY_THROW(builder.add_graph("graph", graph));
  EXPECT_EQ(builder.compilations, 0);
}

}  // namespace taichi::lang::aot
