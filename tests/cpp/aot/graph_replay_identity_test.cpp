#include <cstddef>
#include <new>

#include <gtest/gtest.h>
#include <taichi/aot/graph_data.h>

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
    const uint32_t skipped =
        uint32_t{1} << std::min<uint32_t>(failure, 5);
    EXPECT_EQ(retry.retry_backoff_remaining(), skipped);
    EXPECT_LE(skipped,
              CompiledGraphCaptureRetryState::kMaxBackoffInvocations);
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

}  // namespace taichi::lang::aot
