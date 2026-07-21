#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include "taichi/program/linear_operator.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/sparse_fixed_bicgstab.h"

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
  OperatorBinding binding(std::move(action), [&order] {
    order.push_back(1);
    return OperatorResourceLease{};
  });

  auto generation = binding.pin();
  ASSERT_TRUE(generation);
  ASSERT_EQ(order.size(), 2u);
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
}

TEST(LinearOperator, SubmissionTicketRetainsPinnedGeneration) {
  int releases = 0;
  const OperatorResourceStamp stamp{0, 0, 1, 2, 3, 4};
  OperatorBinding binding(make_scale_action(stamp, 2.0f), [&releases] {
    return OperatorResourceLease::hold(LeaseProbe(&releases));
  });
  OperatorPlan plan(nullptr, std::move(binding));
  const auto &descriptor = plan.descriptor();
  std::array<float, 2> input{2.0f, 3.0f};
  std::array<float, 2> output{};

  {
    auto submission = plan.submit(
        {OperatorApplyMode::forward,
         OperatorVectorView::from_const_host(input.data(), descriptor.domain),
         nullptr,
         OperatorVectorView::from_mutable_host(output.data(),
                                               descriptor.range)});
    EXPECT_TRUE(submission.completed_synchronously);
    EXPECT_TRUE(submission.done());
    EXPECT_EQ(submission.resource_stamp.numeric_revision, 3u);
    EXPECT_EQ(releases, 0);
    auto moved = std::move(submission);
    EXPECT_TRUE(moved.done());
    EXPECT_EQ(releases, 0);
  }

  EXPECT_EQ(releases, 1);
  EXPECT_FLOAT_EQ(output[0], 4.0f);
  EXPECT_FLOAT_EQ(output[1], 6.0f);
}

#ifdef TI_WITH_LLVM
TEST(LinearOperator, AsyncCapableStandaloneSubmitRecordsCompletion) {
  Program program(Arch::x64);
  auto *input = program.create_ndarray(PrimitiveType::f32, {2},
                                       ExternalArrayLayout::kNull, false);
  auto *output = program.create_ndarray(PrimitiveType::f32, {2},
                                        ExternalArrayLayout::kNull, false);
  const OperatorSpaceDesc space = scalar_space(PrimitiveType::f32, 2);
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit = true;
  int releases = 0;
  OperatorAction action(
      {space, space}, capabilities, "cpu_async_probe",
      [&program] {
        return OperatorResourceStamp{reinterpret_cast<std::uintptr_t>(&program),
                                     program.runtime_program_generation(),
                                     1,
                                     1,
                                     1,
                                     1};
      },
      [](OperatorApplyMode mode, const OperatorVectorView &source,
         const OperatorVectorView &target) {
        ASSERT_EQ(mode, OperatorApplyMode::forward);
        const auto *src = reinterpret_cast<const float *>(source.data);
        auto *dst = reinterpret_cast<float *>(target.data);
        dst[0] = 3.0f * src[0];
        dst[1] = 3.0f * src[1];
      });
  OperatorBinding binding(std::move(action), [&releases] {
    return OperatorResourceLease::hold(LeaseProbe(&releases));
  });
  OperatorPlan plan(&program, std::move(binding));
  const std::array<float, 2> source{2.0f, -4.0f};
  program.copy_ndarray_from_host(input, source.data(), sizeof(source));

  {
    auto submission = plan.submit(
        {OperatorApplyMode::forward,
         OperatorVectorView::from_ndarray(&program, *input, space, false),
         nullptr,
         OperatorVectorView::from_ndarray(&program, *output, space, true)});
    EXPECT_TRUE(submission.completed_synchronously);
    EXPECT_TRUE(submission.done());
    EXPECT_EQ(releases, 0);
  }
  EXPECT_EQ(releases, 1);

  std::array<float, 2> result{};
  program.copy_ndarray_to_host(output, result.data(), sizeof(result));
  EXPECT_FLOAT_EQ(result[0], 6.0f);
  EXPECT_FLOAT_EQ(result[1], -12.0f);
  program.delete_ndarray(output);
  program.delete_ndarray(input);
}
#endif

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
  publisher.publish(make_scale_action(first_stamp, 2.0f),
                    OperatorResourceLease::hold(LeaseProbe(&release_count)));
  auto first = publisher.acquire();

  OperatorResourceStamp second_stamp = first_stamp;
  second_stamp.numeric_revision++;
  second_stamp.binding_revision++;
  publisher.publish(make_scale_action(second_stamp, 3.0f),
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

TEST(LinearOperator, FixedLinearPreconditionerTracksTargetGeneration) {
  OperatorResourceGenerationPublisher target_publisher;
  OperatorResourceGenerationPublisher preconditioner_publisher;
  OperatorResourceStamp stamp{13, 104, 1, 2, 3, 4};
  auto target_metadata = make_scale_action(stamp, 2.0f);
  auto preconditioner_metadata = make_scale_action(stamp, 0.5f);
  target_publisher.publish(target_metadata);
  preconditioner_publisher.publish(preconditioner_metadata);
  OperatorPlan target_plan(nullptr, OperatorBinding::from_generation_publisher(
                                        target_metadata, [&target_publisher] {
                                          return target_publisher.acquire();
                                        }));
  int refreshes = 0;
  PreconditionerPlan preconditioner_plan(
      nullptr, target_plan.descriptor(),
      OperatorBinding::from_generation_publisher(
          preconditioner_metadata,
          [&preconditioner_publisher] {
            return preconditioner_publisher.acquire();
          }),
      PreconditionerBehavior::fixed_linear, "inverse_scale",
      [&](const OperatorResourceStamp &target_stamp, bool changed) {
        if (!changed ||
            target_stamp.numeric_revision == stamp.numeric_revision) {
          return;
        }
        refreshes++;
        preconditioner_publisher.publish(
            make_scale_action(target_stamp, 0.25f));
      });

  auto first_target = target_plan.pin();
  preconditioner_plan.setup(first_target);
  auto first_preconditioner = preconditioner_plan.update_and_pin(first_target);
  EXPECT_EQ(first_preconditioner.resource_stamp().numeric_revision, 3u);

  auto next_stamp = stamp;
  next_stamp.numeric_revision++;
  next_stamp.binding_revision++;
  target_publisher.publish(make_scale_action(next_stamp, 4.0f));
  auto next_target = target_plan.pin();
  auto next_preconditioner = preconditioner_plan.update_and_pin(next_target);
  EXPECT_EQ(refreshes, 1);
  EXPECT_EQ(next_preconditioner.resource_stamp().numeric_revision, 4u);

  std::array<float, 2> input{4.0f, -8.0f};
  std::array<float, 2> first_output{};
  std::array<float, 2> next_output{};
  const auto space = scalar_space(PrimitiveType::f32, 2);
  first_preconditioner.apply_overwrite(
      OperatorApplyMode::forward,
      OperatorVectorView::from_const_host(input.data(), space),
      OperatorVectorView::from_mutable_host(first_output.data(), space));
  next_preconditioner.apply_overwrite(
      OperatorApplyMode::forward,
      OperatorVectorView::from_const_host(input.data(), space),
      OperatorVectorView::from_mutable_host(next_output.data(), space));
  EXPECT_EQ(first_output, (std::array<float, 2>{2.0f, -4.0f}));
  EXPECT_EQ(next_output, (std::array<float, 2>{1.0f, -2.0f}));

  const auto statistics = preconditioner_plan.debug_runtime_statistics();
  EXPECT_EQ(statistics.setup_calls, 1u);
  EXPECT_EQ(statistics.update_calls, 2u);
  EXPECT_EQ(statistics.update_successes, 1u);
  EXPECT_EQ(statistics.update_noops, 1u);
  EXPECT_EQ(statistics.target_generation_changes, 1u);
}

TEST(LinearOperator, PreconditionerRejectsUnsupportedBehaviorAndStaleAction) {
  const OperatorResourceStamp stamp{13, 105, 1, 1, 1, 1};
  auto target_action = make_scale_action(stamp, 2.0f);
  auto preconditioner_action = make_scale_action(stamp, 0.5f);
  OperatorPlan target_plan(nullptr, OperatorBinding(target_action));

  EXPECT_ANY_THROW(PreconditionerPlan(
      nullptr, target_plan.descriptor(), OperatorBinding(preconditioner_action),
      PreconditionerBehavior::variable_linear, "variable",
      [](const OperatorResourceStamp &, bool) {}));

  PreconditionerPlan stale_plan(nullptr, target_plan.descriptor(),
                                OperatorBinding(preconditioner_action),
                                PreconditionerBehavior::fixed_linear, "stale",
                                [](const OperatorResourceStamp &, bool) {});
  auto target = target_plan.pin();
  stale_plan.setup(target);

  auto changed_stamp = stamp;
  changed_stamp.numeric_revision++;
  auto changed_target_action = make_scale_action(changed_stamp, 3.0f);
  OperatorPlan changed_target(nullptr, std::move(changed_target_action));
  EXPECT_ANY_THROW(stale_plan.update_and_pin(changed_target.pin()));
  EXPECT_EQ(stale_plan.debug_runtime_statistics().update_failures, 1u);
}

TEST(LinearOperator, SolverGateRequiresTrustedMathematicalTraits) {
  const OperatorSpaceDesc space = scalar_space(PrimitiveType::f64, 3);
  const OperatorDescriptor square{space, space};
  OperatorMathematicalTraits unknown;
  EXPECT_ANY_THROW(validate_operator_solver_compatibility(
      square, unknown, OperatorSolverFamily::cg));
  EXPECT_NO_THROW(validate_operator_solver_compatibility(
      square, unknown, OperatorSolverFamily::bicgstab));

  const auto scope =
      operator_dependency(OperatorResourceDependency::schema) |
      operator_dependency(OperatorResourceDependency::topology) |
      operator_dependency(OperatorResourceDependency::numeric);
  const auto checked = make_spd_operator_traits(
      OperatorTraitProvenance::empirically_checked, scope);
  EXPECT_ANY_THROW(validate_operator_solver_compatibility(
      square, checked, OperatorSolverFamily::cg));

  const auto asserted = make_spd_operator_traits(
      OperatorTraitProvenance::asserted_by_user, scope);
  EXPECT_NO_THROW(validate_operator_solver_compatibility(
      square, asserted, OperatorSolverFamily::cg));
  EXPECT_NO_THROW(validate_operator_solver_compatibility(
      square, asserted, OperatorSolverFamily::pcg,
      PreconditionerBehavior::fixed_linear));
  EXPECT_ANY_THROW(validate_operator_solver_compatibility(
      square, asserted, OperatorSolverFamily::pcg,
      PreconditionerBehavior::variable_linear));

  const OperatorDescriptor rectangular{
      scalar_space(PrimitiveType::f64, 3),
      scalar_space(PrimitiveType::f64, 2)};
  EXPECT_ANY_THROW(validate_operator_solver_compatibility(
      rectangular, unknown, OperatorSolverFamily::bicgstab));
}

TEST(LinearOperator, TraitDecorationFollowsPublishedGenerations) {
  OperatorResourceGenerationPublisher publisher;
  OperatorResourceStamp stamp{17, 201, 1, 2, 3, 4};
  auto metadata = make_scale_action(stamp, 2.0f);
  publisher.publish(metadata);
  OperatorBinding source = OperatorBinding::from_generation_publisher(
      metadata, [&publisher] { return publisher.acquire(); });
  const auto scope =
      operator_dependency(OperatorResourceDependency::schema) |
      operator_dependency(OperatorResourceDependency::topology) |
      operator_dependency(OperatorResourceDependency::numeric);
  OperatorPlan plan(
      nullptr, source.with_mathematical_traits(make_spd_operator_traits(
                   OperatorTraitProvenance::derived_structurally, scope)));

  auto first = plan.pin();
  EXPECT_EQ(first.mathematical_traits().positive_definite.provenance,
            OperatorTraitProvenance::derived_structurally);
  auto next_stamp = stamp;
  next_stamp.numeric_revision++;
  publisher.publish(make_scale_action(next_stamp, 3.0f));
  auto next = plan.pin();
  EXPECT_TRUE(next.mathematical_traits().self_adjoint.value);
  EXPECT_EQ(next.mathematical_traits().self_adjoint.validity_scope, scope);
  EXPECT_EQ(next.resource_stamp().numeric_revision, 4u);
}

TEST(LinearOperator, BiCGSTABConsumesDenseOperatorAction) {
  const OperatorSpaceDesc space = scalar_space(PrimitiveType::f64, 3);
  OperatorBinding binding(make_dense_reference_operator_action(
      {space, space}, {4.0, 1.0, 0.0,
                       0.0, 3.0, 1.0,
                       1.0, 0.0, 2.0}));
  FixedSparseBiCGSTAB<Eigen::VectorXd, double> solver(
      nullptr, std::move(binding), 20, 1e-12, false);
  Eigen::VectorXd expected(3);
  expected << 1.0, -2.0, 0.5;
  Eigen::VectorXd rhs(3);
  rhs << 2.0, -5.5, 2.0;
  Eigen::VectorXd initial = Eigen::VectorXd::Zero(3);
  solver.set_b(rhs);
  solver.set_x(initial);

  solver.solve();

  EXPECT_TRUE(solver.is_success());
  EXPECT_LE((solver.get_x() - expected).norm(), 1e-11);
  const auto statistics = solver.debug_runtime_statistics();
  EXPECT_EQ(statistics.operator_action_provider, "dense_reference");
  EXPECT_EQ(statistics.operator_generation_pins, 1u);
  EXPECT_GT(statistics.operator_apply_calls, 0u);
}

TEST(LinearOperator, BindingTypeErasesProviderResourceLease) {
  OperatorDescriptor descriptor{scalar_space(PrimitiveType::f32, 2),
                                scalar_space(PrimitiveType::f32, 2)};
  int acquire_count = 0;
  int release_count = 0;
  OperatorBinding binding(
      make_dense_reference_operator_action(descriptor, {1.0, 0.0, 0.0, 1.0}),
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

TEST(LinearOperator, ExplicitAdjointBindingSatisfiesDotProductIdentity) {
  const OperatorDescriptor descriptor{
      scalar_space(PrimitiveType::f64, 3),
      scalar_space(PrimitiveType::f64, 2)};
  OperatorBinding source(make_dense_reference_operator_action(
      descriptor, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
  OperatorPlan forward(nullptr, source);
  OperatorPlan adjoint(nullptr,
                       make_adjoint_operator_binding(std::move(source)));

  EXPECT_EQ(adjoint.descriptor().domain, descriptor.range);
  EXPECT_EQ(adjoint.descriptor().range, descriptor.domain);
  EXPECT_EQ(adjoint.provider_name(), "adjoint(dense_reference)");
  EXPECT_TRUE(adjoint.capabilities().adjoint_apply);

  std::array<double, 3> x{1.0, -2.0, 0.5};
  std::array<double, 2> y{2.0, -1.0};
  std::array<double, 2> dx{};
  std::array<double, 3> d_star_y{};
  forward.submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_const_host(x.data(), descriptor.domain),
       nullptr,
       OperatorVectorView::from_mutable_host(dx.data(), descriptor.range)});
  adjoint.submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_const_host(y.data(), descriptor.range),
       nullptr,
       OperatorVectorView::from_mutable_host(d_star_y.data(),
                                             descriptor.domain)});
  const double left = dx[0] * y[0] + dx[1] * y[1];
  const double right = x[0] * d_star_y[0] + x[1] * d_star_y[1] +
                       x[2] * d_star_y[2];
  EXPECT_DOUBLE_EQ(left, right);

  std::array<double, 2> round_trip{};
  adjoint.submit(
      {OperatorApplyMode::adjoint,
       OperatorVectorView::from_const_host(x.data(), descriptor.domain),
       nullptr,
       OperatorVectorView::from_mutable_host(round_trip.data(),
                                             descriptor.range)});
  EXPECT_EQ(round_trip, dx);
}

TEST(LinearOperator, AdjointBindingRequiresExplicitProviderAction) {
  const OperatorResourceStamp stamp{0, 0, 1, 1, 1, 1};
  EXPECT_ANY_THROW(
      make_adjoint_operator_binding(OperatorBinding(make_scale_action(
          stamp, 2.0f))));
}

TEST(LinearOperator, MinimalCompositionBuildsRegularizedNormalOperator) {
  const OperatorSpaceDesc domain =
      scalar_space(PrimitiveType::f64, 3);
  const OperatorDescriptor descriptor{
      domain, scalar_space(PrimitiveType::f64, 2)};
  OperatorBinding derivative(make_dense_reference_operator_action(
      descriptor, {1.0, 2.0, 0.0, 0.0, -1.0, 3.0}));
  auto normal = make_composed_operator_binding(
      make_adjoint_operator_binding(derivative), derivative);
  auto regularizer =
      make_scaled_operator_binding(0.5,
                                   make_identity_operator_binding(domain));
  EXPECT_TRUE(
      regularizer.action().mathematical_traits().positive_definite.value);
  EXPECT_EQ(regularizer.action()
                .mathematical_traits()
                .positive_definite.provenance,
            OperatorTraitProvenance::derived_structurally);
  OperatorPlan plan(
      nullptr,
      make_sum_operator_binding(std::move(normal),
                                std::move(regularizer)));

  std::array<double, 3> input{1.0, -2.0, 0.5};
  std::array<double, 3> output{};
  plan.submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_const_host(input.data(), domain), nullptr,
       OperatorVectorView::from_mutable_host(output.data(), domain)});
  EXPECT_EQ(output, (std::array<double, 3>{-2.5, -10.5, 10.75}));

  std::array<double, 3> adjoint_output{};
  plan.submit(
      {OperatorApplyMode::adjoint,
       OperatorVectorView::from_const_host(input.data(), domain), nullptr,
       OperatorVectorView::from_mutable_host(adjoint_output.data(), domain)});
  EXPECT_EQ(adjoint_output, output);
  EXPECT_TRUE(plan.capabilities().adjoint_apply);
}

#ifdef TI_WITH_LLVM
TEST(LinearOperator, MinimalCompositionSupportsCpuProgramViews) {
  Program program(Arch::x64);
  const OperatorSpaceDesc space =
      scalar_space(PrimitiveType::f32, 3);
  auto *input = program.create_ndarray(PrimitiveType::f32, {3},
                                       ExternalArrayLayout::kNull, false);
  auto *output = program.create_ndarray(PrimitiveType::f32, {3},
                                        ExternalArrayLayout::kNull, false);
  auto product = make_composed_operator_binding(
      make_identity_operator_binding(space, &program),
      make_identity_operator_binding(space, &program));
  auto scaled =
      make_scaled_operator_binding(2.0, std::move(product));
  OperatorPlan plan(
      &program,
      make_sum_operator_binding(
          std::move(scaled),
          make_identity_operator_binding(space, &program)));
  const std::array<float, 3> source{1.0f, -2.0f, 0.5f};
  program.copy_ndarray_from_host(input, source.data(), sizeof(source));

  plan.submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(&program, *input, space, false),
       nullptr,
       OperatorVectorView::from_ndarray(&program, *output, space, true)});
  std::array<float, 3> result{};
  program.copy_ndarray_to_host(output, result.data(), sizeof(result));
  EXPECT_EQ(result, (std::array<float, 3>{3.0f, -6.0f, 1.5f}));

  program.delete_ndarray(output);
  program.delete_ndarray(input);
}
#endif

TEST(LinearOperator, BlockDiagonalSupportsForwardAndAdjointActions) {
  const OperatorSpaceDesc pair =
      scalar_space(PrimitiveType::f64, 2);
  const OperatorSpaceDesc scalar =
      scalar_space(PrimitiveType::f64, 1);
  std::vector<OperatorBinding> blocks;
  blocks.emplace_back(make_dense_reference_operator_action(
      {pair, pair}, {2.0, 1.0, 0.0, 3.0}));
  blocks.emplace_back(make_dense_reference_operator_action(
      {scalar, scalar}, {-1.0}));
  OperatorPlan plan(
      nullptr,
      make_block_diagonal_operator_binding(std::move(blocks)));
  const auto &descriptor = plan.descriptor();
  EXPECT_EQ(descriptor.domain.scalar_extent, 3u);
  EXPECT_EQ(descriptor.range.scalar_extent, 3u);

  std::array<double, 3> input{1.0, 2.0, 4.0};
  std::array<double, 3> forward{};
  plan.submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_const_host(input.data(), descriptor.domain),
       nullptr,
       OperatorVectorView::from_mutable_host(forward.data(),
                                             descriptor.range)});
  EXPECT_EQ(forward, (std::array<double, 3>{4.0, 6.0, -4.0}));

  std::array<double, 3> adjoint{};
  plan.submit(
      {OperatorApplyMode::adjoint,
       OperatorVectorView::from_const_host(input.data(), descriptor.range),
       nullptr,
       OperatorVectorView::from_mutable_host(adjoint.data(),
                                             descriptor.domain)});
  EXPECT_EQ(adjoint, (std::array<double, 3>{2.0, 7.0, -4.0}));
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
