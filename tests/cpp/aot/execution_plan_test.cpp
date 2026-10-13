#include "gtest/gtest.h"
#include "taichi/aot/execution_plan.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"

namespace taichi::lang::aot {
namespace {
void validate_provider_description(const std::string &document) {
  const auto plan = ExecutionPlan::parse(document);
  native_operation_registry().decode(
      std::get<ExecutionPlan::Native>(plan.operations.at(0)));
}
}  // namespace

TEST(ExecutionPlan, PreservesOrderedOperationsAndSymbolicBindings) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"kernel","entry":"forward","arguments":["x","scratch"]},
    {"kind":"copy","source":"scratch","destination":"y","source_offset":4},
    {"kind":"fill32","destination":"scratch","value":1065353216},
    {"kind":"kernel","entry":"backward","arguments":["y","x"]}]})");
  ASSERT_EQ(plan.operations.size(), 4);
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"scratch", "x", "y"}));
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(plan.operations[0]).entry,
            "forward");
  const auto &copy = std::get<ExecutionPlan::Copy>(plan.operations[1]);
  EXPECT_EQ(copy.source_offset, 4);
  EXPECT_FALSE(copy.bytes);
  EXPECT_EQ(std::get<ExecutionPlan::Fill>(plan.operations[2]).value,
            1065353216);
}

TEST(ExecutionPlan, InitializationKeepsOnlyReplayRuntimeDependencies) {
  const auto plan = ExecutionPlan::parse(R"({"version":2,
    "initialization":[{"kind":"copy","source":"input","destination":"scratch"}],
    "operations":[{"kind":"native","provider":"test","operation":"use",
      "version":1,"parameters":{},"bindings":{"plan":"p","output":"y"}}],
    "resources":[
      {"kind":"native","name":"p","provider":"test","operation":"plan",
       "version":1,"parameters":{},"bindings":{"values":"weights"}},
      {"kind":"ndarray","name":"weights","dtype":"f32",
       "shape":[{"array":"metadata","axis":0}]},
      {"kind":"ndarray","name":"scratch","dtype":"f32","shape":[3]}],
    "gradients":{"weights":"dw"}})");
  ASSERT_EQ(plan.initialization.size(), 1);
  ASSERT_EQ(plan.operations.size(), 1);
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"dw", "input", "metadata", "y"}));
  EXPECT_EQ(plan.replay_bindings,
            (std::set<std::string>{"dw", "p", "weights", "y"}));
  for (
      const auto *document :
      {R"({"version":1,"initialization":[],"operations":[{"kind":"fill32","destination":"x","value":0}]})",
       R"({"version":2,"initialization":[],"operations":[{"kind":"fill32","destination":"x","value":0}]})",
       R"({"version":2,"operations":[{"kind":"fill32","destination":"x","value":0}]})"})
    EXPECT_THROW(ExecutionPlan::parse(document), std::invalid_argument);
}

TEST(ExecutionPlan, RejectsUnsupportedOrAmbiguousDescriptions) {
  for (
      const auto *document :
      {R"({"version":2,"operations":[]})", R"({"version":1,"operations":[]})",
       R"({"version":1,"operations":[{"kind":"callback"}]})",
       R"({"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[],"gated":true}]})",
       R"({"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[""]}]})",
       R"({"version":1,"operations":[{"kind":"copy","source":"x","destination":"y","size_bytes":-1}]})",
       R"({"version":1,"operations":[{"kind":"fill32","destination":"y","value":4294967296}]})",
       R"({"version":1,"version":1,"operations":[{"kind":"fill32","destination":"y","value":0}]})",
       R"({"version":1,"operations":[{"kind":"fill32","destination":"y","value":1,"value":0}]})",
       R"({"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[]}] } {})"}) {
    EXPECT_THROW(ExecutionPlan::parse(document), std::invalid_argument)
        << document;
  }
}

TEST(ExecutionPlan, NativeCompactRetainsAllSymbolicDependencies) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","values":"x","flags":"active","output":"selected","count":"n"}]})");
  ASSERT_EQ(plan.operations.size(), 1);
  const auto &compact = std::get<ExecutionPlan::Compact>(plan.operations[0]);
  ASSERT_EQ(compact.columns.size(), 1);
  EXPECT_EQ(compact.columns[0].values, "x");
  EXPECT_EQ(compact.flags, "active");
  EXPECT_EQ(compact.columns[0].output, "selected");
  EXPECT_EQ(compact.count, "n");
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"active", "n", "selected", "x"}));
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","values":"x","flags":"active","output":"selected"}]})"),
               std::invalid_argument);
}

TEST(ExecutionPlan, FftPreservesExplicitDirectionAndDimensions) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"fft_c2c_f32","input":"x","output":"y","dimensions":[3,5],"batch_count":2,"direction":"inverse"}]})");
  const auto &fft = std::get<ExecutionPlan::Native>(plan.operations[0]);
  EXPECT_EQ(fft.provider, "cufft");
  EXPECT_EQ(fft.operation, "fft_c2c_f32");
  EXPECT_EQ(fft.bindings.at("input"), "x");
  EXPECT_EQ(fft.parameters["dimensions"].arr.inner[0].num_int, 3);
  EXPECT_EQ(fft.parameters["dimensions"].arr.inner[1].num_int, 5);
  EXPECT_EQ(fft.parameters["batch_count"].num_int, 2);
  EXPECT_EQ(fft.parameters["direction"].str, "inverse");
  EXPECT_NO_THROW(native_operation_registry().decode(fft));
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x", "y"}));
  for (const auto *fields :
       {R"("dimensions":[],"direction":"forward")",
        R"("dimensions":[true],"direction":"forward")",
        R"("dimensions":[1],"direction":"adjoint")",
        R"("dimensions":[1],"direction":"forward","batch_count":0)"})
    EXPECT_THROW(
        validate_provider_description(
            std::string(
                R"({"version":1,"operations":[{"kind":"fft_c2c_f32","input":"x","output":"y",)") +
            fields + "}]}"),
        std::invalid_argument);
}

TEST(ExecutionPlan, GemmPreservesScalarsAndPhysicalTransposeContract) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"gemm_f32","a":"x","b":"y","output":"z","rows":17,
     "columns":9,"inner":13,"alpha":-0.75,"beta":2,"transpose_b":true}]})");
  const auto &gemm = std::get<ExecutionPlan::Native>(plan.operations[0]);
  EXPECT_EQ(gemm.provider, "cublas");
  EXPECT_EQ(gemm.operation, "gemm_f32");
  EXPECT_EQ(gemm.parameters["rows"].num_int, 17);
  EXPECT_EQ(gemm.parameters["columns"].num_int, 9);
  EXPECT_EQ(gemm.parameters["inner"].num_int, 13);
  EXPECT_FLOAT_EQ(operation_json::optional_f32(gemm.parameters, "alpha", 1),
                  -0.75f);
  EXPECT_FLOAT_EQ(operation_json::optional_f32(gemm.parameters, "beta", 0), 2);
  EXPECT_FALSE(operation_json::optional_bool(gemm.parameters, "transpose_a"));
  EXPECT_TRUE(operation_json::optional_bool(gemm.parameters, "transpose_b"));
  EXPECT_NO_THROW(native_operation_registry().decode(gemm));
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x", "y", "z"}));
  for (const auto *extra : {R"(,"alpha":true)", R"(,"beta":1e100)",
                            R"(,"transpose_a":1)", R"(,"transpose_b":"N")"})
    EXPECT_THROW(
        validate_provider_description(
            std::string(
                R"({"version":1,"operations":[{"kind":"gemm_f32","a":"x","b":"y","output":"z","rows":2,"columns":3,"inner":4)") +
            extra + "}]}"),
        std::invalid_argument);
  EXPECT_THROW(validate_provider_description(R"({"version":1,"operations":[
    {"kind":"gemm_f32","a":"x","b":"y","output":"z","rows":0,"columns":3,"inner":4}]})"),
               std::invalid_argument);
}

TEST(ExecutionPlan, SortHasOptionalPayloadWithoutImplicitArguments) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"sort","keys":"x"},
    {"kind":"sort","keys":"x","values":"indices"}]})");
  ASSERT_EQ(plan.operations.size(), 2);
  EXPECT_FALSE(std::get<ExecutionPlan::Sort>(plan.operations[0]).values);
  EXPECT_EQ(*std::get<ExecutionPlan::Sort>(plan.operations[1]).values,
            "indices");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"indices", "x"}));
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"sort","keys":"x","descending":true}]})"),
               std::invalid_argument);
}

TEST(ExecutionPlan, NativeEnvelopeHasNoHardcodedProviderOrOperationList) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"native","provider":"application.library","operation":"custom_operation",
     "version":7,"bindings":{"input":"x","output":"scratch"},
     "parameters":{"nested":{"policy":"frozen"},"axes":[2,3]}}],
    "resources":[{"kind":"ndarray","name":"scratch","dtype":"f32","shape":[3]}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x"}));
  ASSERT_EQ(plan.resources.materialize, (std::vector<bool>{true}));
  const auto &native = std::get<ExecutionPlan::Native>(plan.operations[0]);
  EXPECT_EQ(native.provider, "application.library");
  EXPECT_EQ(native.version, 7);
  EXPECT_EQ(native.parameters["nested"]["policy"].str, "frozen");
  EXPECT_THROW(native_operation_registry().decode(native),
               std::invalid_argument);
  for (const auto *extra :
       {R"("version":0,"parameters":{},"bindings":{})",
        R"("version":true,"parameters":{},"bindings":{})",
        R"("version":1,"parameters":[],"bindings":{})",
        R"("version":1,"parameters":{},"bindings":{"input":""})",
        R"("version":1,"parameters":{},"bindings":{},"callback":1)"})
    EXPECT_THROW(
        ExecutionPlan::parse(
            std::string(
                R"({"version":1,"operations":[{"kind":"native","provider":"x","operation":"y",)") +
            extra + "}]}"),
        std::invalid_argument);
}

TEST(ExecutionPlan, CompactColumnsShareOnePredicateAndCount) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","flags":"active","count":"n","columns":[
      {"values":"keys","output":"selected_keys"},
      {"values":"values","output":"selected_values"}]}]})");
  const auto &compact = std::get<ExecutionPlan::Compact>(plan.operations[0]);
  ASSERT_EQ(compact.columns.size(), 2);
  EXPECT_EQ(compact.columns[1].values, "values");
  EXPECT_EQ(compact.columns[1].output, "selected_values");
  EXPECT_EQ(plan.arguments.size(), 6);
  for (
      const auto *description :
      {R"({"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[]}]})",
       R"({"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x"}]}]})",
       R"({"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x","output":"y"}],"values":"z","output":"w"}]})"})
    EXPECT_THROW(ExecutionPlan::parse(description), std::invalid_argument);
}
TEST(ExecutionPlan, ReconstructsOnlyLiveResourceStorageAndShapeClosure) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[
      {"kind":"ndarray","name":"scratch","dtype":"f32","shape":[{"array":"shape_only","axis":0}]},
      {"kind":"extent","name":"n","dtype":"i32","array":"scratch","axis":0},
      {"kind":"ndarray","name":"shape_only","dtype":"f32","shape":[{"array":"x","axis":0},3]},
      {"kind":"ndarray","name":"unused","dtype":"f32","shape":[{"array":"not_required","axis":0}]},
      {"kind":"ndarray","name":"dx","dtype":"f32","shape":[{"array":"scratch","axis":0}]}],
    "gradients":{"scratch":"dx"},
    "operations":[{"kind":"kernel","entry":"forward","arguments":["scratch","n","y"]}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(plan.resources.order, (std::vector<std::size_t>{2, 0, 1, 4}));
  EXPECT_EQ(plan.resources.materialize,
            (std::vector<bool>{true, true, false, false, true}));
}

TEST(ExecutionPlan, RejectsInvalidResourceGraphsBeforeBinding) {
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"fill32","destination":"x","value":0}],"resources":)";
  for (
      const auto *resources :
      {R"(null)",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[true]}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[17],"zero_initialize":1}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[],"element_shape":[0]}])",
       R"([{"kind":"extent","name":"n","dtype":"f32","array":"x","axis":0}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[]},{"kind":"ndarray","name":"x","dtype":"f32","shape":[]}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[{"array":"y","axis":0}]},{"kind":"ndarray","name":"y","dtype":"f32","shape":[{"array":"x","axis":0}]}])",
       R"([{"kind":"extent","name":"n","dtype":"i32","array":"input","axis":0},{"kind":"ndarray","name":"x","dtype":"f32","shape":[{"array":"n","axis":0}]}])"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + resources + "}"),
                 std::invalid_argument);
}

TEST(ExecutionPlan, ZeroInitializationDoesNotMaterializeShapeOnlyResources) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[
      {"kind":"ndarray","name":"state","dtype":"i8","shape":[{"array":"shape","axis":0}],"zero_initialize":true},
      {"kind":"ndarray","name":"shape","dtype":"f32","shape":[17,3],"zero_initialize":true},
      {"kind":"ndarray","name":"unused","dtype":"f32","shape":[123456789],"zero_initialize":true}],
    "operations":[{"kind":"copy","source":"state","destination":"out"}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"out"}));
  EXPECT_EQ(plan.resources.order, (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(plan.resources.materialize,
            (std::vector<bool>{true, false, false}));
  EXPECT_TRUE(
      std::get<ExecutionResources::Array>(plan.resources.declarations[0].value)
          .zero_initialize);
  EXPECT_EQ(plan.replay_bindings, (std::set<std::string>{"out", "state"}));
}

TEST(ExecutionPlan, ScalarResourcesPreserveBitsWithoutStorageDependencies) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[
      {"kind":"scalar","name":"wide","dtype":"u64","bits":"ffffffffffffffff"},
      {"kind":"scalar","name":"negative_zero","dtype":"f32","bits":"80000000"},
      {"kind":"scalar","name":"half","dtype":"f16","bits":"3fc00000"},
      {"kind":"scalar","name":"unused","dtype":"i8","bits":"ff"}],
    "operations":[{"kind":"kernel","entry":"k","arguments":["wide","negative_zero","half","out"]}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"out"}));
  EXPECT_EQ(plan.resources.order, (std::vector<std::size_t>{0, 1, 2}));
  EXPECT_EQ(
      std::get<ExecutionResources::Scalar>(plan.resources.declarations[0].value)
          .bits,
      std::uint64_t(-1));
  EXPECT_EQ(
      std::get<ExecutionResources::Scalar>(plan.resources.declarations[1].value)
          .bits,
      0x80000000u);
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"k","arguments":["x"]}],"resources":)";
  for (
      const auto *resources :
      {R"([{"kind":"scalar","name":"x","dtype":"i32","bits":"fffffffff"}])",
       R"([{"kind":"scalar","name":"x","dtype":"i8","bits":"-1"}])",
       R"([{"kind":"scalar","name":"x","dtype":"i16","bits":"0x01"}])",
       R"([{"kind":"scalar","name":"x","dtype":"f16","bits":"3e00"}])",
       R"([{"kind":"scalar","name":"x","dtype":"f32","bits":1}])",
       R"([{"kind":"scalar","name":"x","dtype":"u64","bits":"000000000000000g"}])",
       R"([{"kind":"scalar","name":"x","dtype":"bool","bits":"01"}])",
       R"([{"kind":"scalar","name":"x","dtype":"i32","bits":"00000001","value":1}])",
       R"([{"kind":"scalar","name":"n","dtype":"i32","bits":"00000001"},{"kind":"ndarray","name":"x","dtype":"f32","shape":[{"array":"n","axis":0}]}])"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + resources + "}"),
                 std::invalid_argument);
}

TEST(ExecutionPlan, InitialDataIsSelectedWithItsArrayStorage) {
  const std::string digest(64, 'a');
  const auto plan = ExecutionPlan::parse(
      R"({"version":1,"operations":[{"kind":"copy","source":"weights","destination":"out"}],"resources":[)"
      R"({"kind":"ndarray","name":"weights","dtype":"f32","shape":[{"array":"shape_only","axis":0}],"initial_data":")" +
      digest +
      R"("},{"kind":"ndarray","name":"shape_only","dtype":"f32","shape":[17],"initial_data":")" +
      digest + R"("}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"out"}));
  EXPECT_EQ(plan.resources.materialize, (std::vector<bool>{true, false}));
  EXPECT_EQ(
      std::get<ExecutionResources::Array>(plan.resources.declarations[0].value)
          .initial_data,
      digest);
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"copy","source":"x","destination":"out"}],"resources":[{"kind":"ndarray","name":"x","dtype":"f32","shape":[17],"initial_data":)";
  for (const auto &value :
       {std::string("null"), std::string("1"), std::string("\"../data\""),
        std::string("\"\""), "\"" + std::string(64, 'G') + "\"",
        "\"" + digest + "\",\"zero_initialize\":true"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + value + "}]}"),
                 std::invalid_argument);
}

TEST(ExecutionPlan, BindingConstraintsPreserveOnlySelectedBindingRequirements) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[{"kind":"ndarray","name":"scratch","dtype":"f32","shape":[17]}],
    "operations":[{"kind":"copy","source":"x","destination":"scratch"},
                  {"kind":"copy","source":"scratch","destination":"y"}],
    "constraints":[{"kind":"array_layout","array":"x","minimum_shape":[17],"record_bytes":4,"alignment":4},
                   {"kind":"disjoint","left":"scratch","right":"y"}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(plan.constraints.size(), 2);
  const auto &layout =
      std::get<ExecutionPlan::ArrayLayout>(plan.constraints[0]);
  EXPECT_EQ(layout.array, "x");
  EXPECT_EQ(layout.required.minimum_shape, (std::vector<std::uint64_t>{17}));
  EXPECT_TRUE(layout.required.element_shape.empty());
  EXPECT_EQ(layout.required.record_bytes, 4);
  EXPECT_EQ(layout.required.alignment, 4);
  EXPECT_EQ(std::get<ExecutionPlan::Disjoint>(plan.constraints[1]).left,
            "scratch");
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"copy","source":"x","destination":"y"}],"constraints":)";
  for (
      const auto *constraints :
      {"null", "{}", R"([{"kind":"disjoint","left":"x","right":"unused"}])",
       R"([{"kind":"disjoint","left":"x","right":"y","unknown":1}])",
       R"([{"kind":"array_layout","array":"x","minimum_shape":[-1],"record_bytes":4,"alignment":4}])",
       R"([{"kind":"array_layout","array":"x","minimum_shape":[true],"record_bytes":4,"alignment":4}])",
       R"([{"kind":"array_layout","array":"x","minimum_shape":[17],"record_bytes":0,"alignment":4}])",
       R"([{"kind":"array_layout","array":"x","minimum_shape":[17],"record_bytes":4,"alignment":0}])",
       R"([{"kind":"array_layout","array":"x","minimum_shape":[17],"element_shape":[0],"record_bytes":4,"alignment":4}])",
       R"([{"kind":"unknown"}])"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + constraints + "}"),
                 std::invalid_argument);
}

TEST(ExecutionPlan,
     PhysicalRangesDoNotConfuseImportedHandlesWithStorageIdentity) {
  auto *device = reinterpret_cast<Device *>(std::uintptr_t(1));
  NativeStorageRange a{device, 1, 32, 4, 0, 16};
  NativeStorageRange alias{device, 1, 40, 5, 0, 16};
  EXPECT_FALSE(
      a.proven_disjoint(alias));  // Two buffers bound to overlapping memory.
  alias.memory_offset = 48;
  EXPECT_TRUE(a.proven_disjoint(alias));
  alias.memory = 2;
  EXPECT_TRUE(a.proven_disjoint(alias));
  alias.memory = 0;
  EXPECT_FALSE(a.proven_disjoint(alias));  // No proof for an unknown import.
  alias.buffer = a.buffer;
  EXPECT_FALSE(
      a.proven_disjoint(alias));  // Same raw buffer, different imports.
  alias.buffer_offset = 16;
  EXPECT_TRUE(a.proven_disjoint(alias));
  alias.bytes = 0;
  EXPECT_TRUE(a.proven_disjoint(alias));
  alias.device = nullptr;
  EXPECT_FALSE(a.proven_disjoint(alias));
}

TEST(ExecutionPlan, KernelLabelsBelongToDispatches) {
  const auto plan = ExecutionPlan::parse(R"({"version":2,
    "initialization":[{"kind":"kernel","entry":"shared","arguments":["x"],"label":"init"}],
    "operations":[{"kind":"kernel","entry":"shared","arguments":["x"],"label":"solver/forward"},
                  {"kind":"kernel","entry":"shared","arguments":["x"],"label":"solver/reverse"},
                  {"kind":"kernel","entry":"shared","arguments":["x"]}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x"}));
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(plan.initialization[0]).label,
            "init");
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(plan.operations[0]).label,
            "solver/forward");
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(plan.operations[1]).label,
            "solver/reverse");
  EXPECT_TRUE(
      std::get<ExecutionPlan::Kernel>(plan.operations[2]).label.empty());
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"k","arguments":[],"label":)";
  for (const auto *label :
       {"null", "true", "1", "[]", "{}", R"("")", R"("invalid\u0000label")",
        R"("invalid\nlabel")", R"("invalid\rlabel")"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + label + "}]}"),
                 std::invalid_argument);
}

TEST(ExecutionPlan, NativeResourcesShareTheRequiredDependencyClosure) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[
      {"kind":"native","name":"child","provider":"test","operation":"plan","version":1,
       "parameters":{},"bindings":{"context":"parent","input":"scratch"}},
      {"kind":"native","name":"parent","provider":"test","operation":"context","version":1,
       "parameters":{},"bindings":{}},
      {"kind":"ndarray","name":"scratch","dtype":"f32","shape":[{"array":"x","axis":0}]},
      {"kind":"native","name":"unused","provider":"missing","operation":"unknown","version":1,
       "parameters":{},"bindings":{"input":"not_required"}}],
    "operations":[{"kind":"native","provider":"test","operation":"run","version":1,
                   "parameters":{},"bindings":{"plan":"child","output":"y"}}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(plan.resources.order, (std::vector<std::size_t>{1, 2, 0}));
  EXPECT_EQ(plan.resources.materialize,
            (std::vector<bool>{true, true, true, false}));
  const auto cyclic = R"({"version":1,"resources":[
    {"kind":"native","name":"x","provider":"test","operation":"plan","version":1,
     "parameters":{},"bindings":{"parent":"x"}}],
    "operations":[{"kind":"fill32","destination":"y","value":0}]})";
  EXPECT_THROW(ExecutionPlan::parse(cyclic), std::invalid_argument);
}
}  // namespace taichi::lang::aot
