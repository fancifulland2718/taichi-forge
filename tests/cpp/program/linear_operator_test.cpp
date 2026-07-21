#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <thread>
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

OperatorAction make_scale_action(OperatorResourceStamp stamp, float scale) {
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f32, 2),
                                scalar_space(PrimitiveType::f32, 2)};
  return OperatorAction(
      descriptor, OperatorCapabilities{}, "scale_generation",
      [stamp] { return stamp; },
      [scale](OperatorApplyMode mode, const OperatorVectorView &input,
              const OperatorVectorView &output) {
        ASSERT_EQ(mode, OperatorApplyMode::forward);
        const auto *source = reinterpret_cast<const float *>(input.data);
        auto *target = reinterpret_cast<float *>(output.data);
        target[0] = scale * source[0];
        target[1] = scale * source[1];
      });
}

TEST(LinearOperator, BindingPinsLeaseBeforeReadingResourceStamp) {
  std::vector<int> order;
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f32, 1),
                                scalar_space(PrimitiveType::f32, 1)};
  OperatorAction action(
      descriptor, OperatorCapabilities{}, "ordered_binding",
      [&order] {
        order.push_back(2);
        return OperatorResourceStamp{};
      },
      [](OperatorApplyMode, const OperatorVectorView &,
         const OperatorVectorView &) {});
  OperatorBinding binding(
      std::move(action), [&order] {
        order.push_back(1);
        return OperatorResourceLease{};
      });

  auto generation = binding.pin();
  ASSERT_TRUE(generation);
  ASSERT_EQ(order.size(), 2u);
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
}

TEST(LinearOperator, DependencyMaskClassifiesPlanInvalidation) {
  OperatorResourceStamp planned{11, 101, 2, 3, 4, 5};
  auto current = planned;
  current.numeric_revision++;
  current.binding_revision++;

  auto schema_only = evaluate_operator_plan_invalidation(
      planned, current, operator_plan_schema_dependencies());
  EXPECT_EQ(schema_only.kind, OperatorPlanInvalidationKind::current);
  EXPECT_EQ(schema_only.relevant_changes, 0u);
  EXPECT_NE(schema_only.changes &
                operator_dependency(OperatorResourceDependency::numeric),
            0u);

  auto numeric_dependent = evaluate_operator_plan_invalidation(
      planned, current,
      operator_dependency(OperatorResourceDependency::numeric));
  EXPECT_EQ(numeric_dependent.kind, OperatorPlanInvalidationKind::rebuild);

  current = planned;
  current.binding_revision++;
  auto binding_dependent = evaluate_operator_plan_invalidation(
      planned, current,
      operator_dependency(OperatorResourceDependency::binding));
  EXPECT_EQ(binding_dependent.kind,
            OperatorPlanInvalidationKind::refresh_binding);

  current = planned;
  current.program_generation++;
  auto program_dependent = evaluate_operator_plan_invalidation(
      planned, current,
      operator_dependency(OperatorResourceDependency::program));
  EXPECT_EQ(program_dependent.kind,
            OperatorPlanInvalidationKind::program_invalid);
}

TEST(LinearOperator, PublishedGenerationRemainsUsableUntilPinRelease) {
  OperatorResourceGenerationPublisher publisher;
  int release_count = 0;
  OperatorResourceStamp first_stamp{11, 101, 1, 1, 1, 1};
  publisher.publish(
      make_scale_action(first_stamp, 2.0f),
      OperatorResourceLease::hold(LeaseProbe(&release_count)));
  auto first = publisher.acquire();

  OperatorResourceStamp second_stamp = first_stamp;
  second_stamp.numeric_revision++;
  second_stamp.binding_revision++;
  publisher.publish(
      make_scale_action(second_stamp, 3.0f),
      OperatorResourceLease::hold(LeaseProbe(&release_count)));
  auto second = publisher.acquire();
  auto statistics = publisher.debug_statistics();
  EXPECT_EQ(statistics.published, 2u);
  EXPECT_EQ(statistics.retired, 1u);
  EXPECT_EQ(statistics.released, 0u);
  EXPECT_EQ(statistics.active_leases, 2u);

  std::array<float, 2> input{1.0f, -2.0f};
  std::array<float, 2> old_output{};
  std::array<float, 2> new_output{};
  const auto space = scalar_space(PrimitiveType::f32, 2);
  first.apply_overwrite(
      OperatorApplyMode::forward,
      OperatorVectorView::from_const_host(input.data(), space),
      OperatorVectorView::from_mutable_host(old_output.data(), space));
  second.apply_overwrite(
      OperatorApplyMode::forward,
      OperatorVectorView::from_const_host(input.data(), space),
      OperatorVectorView::from_mutable_host(new_output.data(), space));
  EXPECT_EQ(old_output, (std::array<float, 2>{2.0f, -4.0f}));
  EXPECT_EQ(new_output, (std::array<float, 2>{3.0f, -6.0f}));

  first = OperatorPinnedAction{};
  statistics = publisher.debug_statistics();
  EXPECT_EQ(statistics.released, 1u);
  EXPECT_EQ(release_count, 1);

  publisher.retire_current();
  EXPECT_ANY_THROW(publisher.acquire());
  second.apply_overwrite(
      OperatorApplyMode::forward,
      OperatorVectorView::from_const_host(input.data(), space),
      OperatorVectorView::from_mutable_host(new_output.data(), space));
  EXPECT_EQ(new_output, (std::array<float, 2>{3.0f, -6.0f}));
  second = OperatorPinnedAction{};
  statistics = publisher.debug_statistics();
  EXPECT_EQ(statistics.released, 2u);
  EXPECT_EQ(release_count, 2);
}

TEST(LinearOperator, RetiredGenerationPinSurvivesConcurrentPublish) {
  OperatorResourceGenerationPublisher publisher;
  OperatorResourceStamp stamp{12, 102, 1, 1, 1, 1};
  publisher.publish(make_scale_action(stamp, 2.0f));
  auto pinned = publisher.acquire();
  std::atomic<bool> published{false};
  std::thread updater([&] {
    auto next = stamp;
    next.numeric_revision++;
    publisher.publish(make_scale_action(next, 4.0f));
    published.store(true, std::memory_order_release);
  });
  updater.join();
  ASSERT_TRUE(published.load(std::memory_order_acquire));

  std::array<float, 2> input{2.0f, 3.0f};
  std::array<float, 2> output{};
  const auto space = scalar_space(PrimitiveType::f32, 2);
  pinned.apply_overwrite(
      OperatorApplyMode::forward,
      OperatorVectorView::from_const_host(input.data(), space),
      OperatorVectorView::from_mutable_host(output.data(), space));
  EXPECT_EQ(output, (std::array<float, 2>{4.0f, 6.0f}));
}

TEST(LinearOperator, PlanPinsNewNumericGenerationWithoutSchemaRebuild) {
  OperatorResourceGenerationPublisher publisher;
  OperatorResourceStamp first_stamp{13, 103, 1, 1, 1, 1};
  auto first_action = make_scale_action(first_stamp, 2.0f);
  publisher.publish(first_action);
  OperatorPlan plan(
      nullptr, OperatorBinding::from_generation_publisher(
                   first_action, [&publisher] { return publisher.acquire(); }));

  auto first_pin = plan.pin();
  auto second_stamp = first_stamp;
  second_stamp.numeric_revision++;
  second_stamp.binding_revision++;
  publisher.publish(make_scale_action(second_stamp, 3.0f));
  auto second_pin = plan.pin();
  EXPECT_EQ(first_pin.resource_stamp().numeric_revision, 1u);
  EXPECT_EQ(second_pin.resource_stamp().numeric_revision, 2u);
  auto statistics = plan.debug_runtime_statistics();
  EXPECT_EQ(statistics.generation_pins, 2u);
  EXPECT_EQ(statistics.generation_changes, 1u);
  EXPECT_EQ(statistics.numeric_generation_changes, 1u);
  EXPECT_EQ(statistics.binding_generation_changes, 1u);
  EXPECT_EQ(statistics.invalidations, 0u);

  auto incompatible = second_stamp;
  incompatible.schema_revision++;
  publisher.publish(make_scale_action(incompatible, 4.0f));
  EXPECT_ANY_THROW(plan.pin());
  EXPECT_EQ(plan.debug_runtime_statistics().invalidations, 1u);
}

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
