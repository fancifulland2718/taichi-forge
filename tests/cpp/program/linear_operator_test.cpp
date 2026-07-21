#include "gtest/gtest.h"

#include <array>
#include <vector>

#include "taichi/program/linear_operator.h"

namespace taichi::lang {
namespace {

OperatorSpaceDesc scalar_space(DataType type, std::size_t extent) {
  return {type, extent};
}

class LeaseProbe {
 public:
  explicit LeaseProbe(int *release_count) : release_count_(release_count) {
  }
  LeaseProbe(const LeaseProbe &) = delete;
  LeaseProbe &operator=(const LeaseProbe &) = delete;
  LeaseProbe(LeaseProbe &&other) noexcept
      : release_count_(other.release_count_) {
    other.release_count_ = nullptr;
  }
  ~LeaseProbe() {
    if (release_count_) {
      (*release_count_)++;
    }
  }

 private:
  int *release_count_{nullptr};
};

TEST(LinearOperator, BindingTypeErasesProviderResourceLease) {
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f32, 2),
                                scalar_space(PrimitiveType::f32, 2)};
  int acquire_count = 0;
  int release_count = 0;
  OperatorBinding binding(
      make_dense_reference_operator_action(
          descriptor, {1.0, 0.0, 0.0, 1.0}),
      [&] {
        acquire_count++;
        return OperatorResourceLease::hold(LeaseProbe(&release_count));
      });
  OperatorPlan plan(nullptr, std::move(binding));

  EXPECT_EQ(plan.provider_name(), "dense_reference");
  {
    auto lease = plan.acquire_resource_lease();
    EXPECT_TRUE(static_cast<bool>(lease));
    EXPECT_EQ(acquire_count, 1);
    EXPECT_EQ(release_count, 0);
  }
  EXPECT_EQ(release_count, 1);
}

TEST(LinearOperator, DenseReferenceSupportsRectangularForwardAndAdjoint) {
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f64, 3),
                                scalar_space(PrimitiveType::f64, 2)};
  OperatorPlan plan(nullptr, make_dense_reference_operator_action(
                                 descriptor, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

  std::array<double, 3> input{1.0, -2.0, 0.5};
  std::array<double, 2> output{};
  plan.submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_const_host(input.data(), descriptor.domain),
       nullptr,
       OperatorVectorView::from_mutable_host(output.data(), descriptor.range)});
  EXPECT_DOUBLE_EQ(output[0], -1.5);
  EXPECT_DOUBLE_EQ(output[1], -3.0);

  std::array<double, 2> adjoint_input{2.0, -1.0};
  std::array<double, 3> adjoint_output{};
  plan.submit({OperatorApplyMode::adjoint,
               OperatorVectorView::from_const_host(adjoint_input.data(),
                                                   descriptor.range),
               nullptr,
               OperatorVectorView::from_mutable_host(adjoint_output.data(),
                                                     descriptor.domain)});
  EXPECT_DOUBLE_EQ(adjoint_output[0], -2.0);
  EXPECT_DOUBLE_EQ(adjoint_output[1], -1.0);
  EXPECT_DOUBLE_EQ(adjoint_output[2], 0.0);

  const auto statistics = plan.debug_runtime_statistics();
  EXPECT_EQ(statistics.submissions, 2u);
  EXPECT_EQ(statistics.primitive_apply_calls, 2u);
  EXPECT_EQ(statistics.generalized_lowerings, 0u);
  EXPECT_EQ(statistics.scratch_builds, 0u);
}

TEST(LinearOperator, GeneralizedApplyAllowsAddendOutputAlias) {
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f32, 2),
                                scalar_space(PrimitiveType::f32, 2)};
  OperatorPlan plan(nullptr, make_dense_reference_operator_action(
                                 descriptor, {2.0, 0.0, 0.0, 3.0}));
  std::array<float, 2> input{1.0f, 2.0f};
  std::array<float, 2> output_and_addend{10.0f, 20.0f};
  const auto input_view =
      OperatorVectorView::from_const_host(input.data(), descriptor.domain);
  const auto addend_view = OperatorVectorView::from_const_host(
      output_and_addend.data(), descriptor.range);
  const auto output_view = OperatorVectorView::from_mutable_host(
      output_and_addend.data(), descriptor.range);
  plan.submit({OperatorApplyMode::forward, input_view, &addend_view,
               output_view, 0.5, -1.0});
  EXPECT_FLOAT_EQ(output_and_addend[0], -9.0f);
  EXPECT_FLOAT_EQ(output_and_addend[1], -17.0f);

  std::array<float, 2> second_output{};
  OperatorVectorView unreadable_addend;
  plan.submit({OperatorApplyMode::forward, input_view, &unreadable_addend,
               OperatorVectorView::from_mutable_host(second_output.data(),
                                                     descriptor.range),
               2.0, 0.0});
  EXPECT_FLOAT_EQ(second_output[0], 4.0f);
  EXPECT_FLOAT_EQ(second_output[1], 12.0f);

  const auto statistics = plan.debug_runtime_statistics();
  EXPECT_EQ(statistics.submissions, 2u);
  EXPECT_EQ(statistics.primitive_apply_calls, 2u);
  EXPECT_EQ(statistics.generalized_lowerings, 2u);
  EXPECT_EQ(statistics.scratch_builds, 1u);
  EXPECT_EQ(statistics.scratch_reuses, 1u);
  EXPECT_EQ(statistics.scratch_reserved_bytes, 2u * sizeof(float));
}

TEST(LinearOperator, ZeroAlphaSkipsPrimitiveApplyAndInputOutputAliasFails) {
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f64, 2),
                                scalar_space(PrimitiveType::f64, 2)};
  OperatorPlan plan(nullptr, make_dense_reference_operator_action(
                                 descriptor, {1.0, 0.0, 0.0, 1.0}));
  std::array<double, 2> input{3.0, 4.0};
  std::array<double, 2> output_and_addend{5.0, 6.0};
  const auto input_view =
      OperatorVectorView::from_const_host(input.data(), descriptor.domain);
  const auto addend_view = OperatorVectorView::from_const_host(
      output_and_addend.data(), descriptor.range);
  const auto output_view = OperatorVectorView::from_mutable_host(
      output_and_addend.data(), descriptor.range);
  plan.submit({OperatorApplyMode::forward, input_view, &addend_view,
               output_view, 0.0, 1.0});
  EXPECT_DOUBLE_EQ(output_and_addend[0], 5.0);
  EXPECT_DOUBLE_EQ(output_and_addend[1], 6.0);
  EXPECT_EQ(plan.debug_runtime_statistics().primitive_apply_calls, 0u);

  auto alias_view =
      OperatorVectorView::from_mutable_host(input.data(), descriptor.domain);
  EXPECT_ANY_THROW(plan.submit(
      {OperatorApplyMode::forward, alias_view, nullptr, alias_view, 1.0, 0.0}));
  EXPECT_EQ(plan.debug_runtime_statistics().submissions, 1u);
}

}  // namespace
}  // namespace taichi::lang
