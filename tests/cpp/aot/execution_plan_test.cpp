#include "gtest/gtest.h"
#include "taichi/aot/execution_plan.h"
#include "taichi/aot/root_storage.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"

namespace taichi::lang::aot {
TEST(ExecutionPlan, HostBranchesRequireV12AndKeepBackendSelection) {
  using Lowering = ExecutionPlan::Branch::Lowering;
  for (const auto &[name, lowering] :
       {std::pair{"cuda_host", Lowering::cuda_host},
        std::pair{"vulkan_host", Lowering::vulkan_host}}) {
    auto source = std::string(R"({"version":12,"operations":[
      {"kind":"branch","selector":"p","selection":"nonzero",
       "has_default":false,"condition":[],"bodies":[[]],"lowering":{"kind":")") +
                  name + R"(","version":1}}]})";
    const auto plan = ExecutionPlan::parse(source);
    EXPECT_EQ(std::get<std::shared_ptr<const ExecutionPlan::Branch>>(
                  plan.operations[0])->lowering, lowering);
    source.replace(source.find("12"), 2, "11");
    EXPECT_THROW(ExecutionPlan::parse(source), std::invalid_argument);
  }
}

TEST(ExecutionPlan, BranchKeepsEveryPathAndExplicitLowering) {
  const std::string source = R"({"version":11,"operations":[
    {"kind":"branch","selector":"selector","selection":"indexed",
     "has_default":true,"lowering":{"kind":"cuda_conditional","version":1},
     "condition":[{"kind":"kernel","entry":"condition","arguments":["input","selector"]}],
     "bodies":[[],[{"kind":"kernel","entry":"forward","arguments":["x","y"]}],
       [{"kind":"while","predicate":"p","max_iterations":2,"condition":[],
         "body":[{"kind":"kernel","entry":"reverse","arguments":["y","dx"]}]}]]}],
    "resources":[{"kind":"ndarray","name":"selector","dtype":"i32","shape":[1]}]})";
  const auto plan = ExecutionPlan::parse(source);
  const auto &branch = *std::get<std::shared_ptr<const ExecutionPlan::Branch>>(
      plan.operations[0]);
  EXPECT_EQ(branch.lowering, ExecutionPlan::Branch::Lowering::cuda_conditional);
  EXPECT_EQ(branch.selection.kind, BranchSelection::Kind::indexed);
  EXPECT_TRUE(branch.selection.has_default);
  ASSERT_EQ(branch.bodies.size(), 3);
  EXPECT_TRUE(branch.bodies[0].empty());
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"dx", "input", "p", "x", "y"}));
  EXPECT_EQ(plan.replay_bindings,
            (std::set<std::string>{"dx", "input", "p", "selector", "x", "y"}));
  for (const auto &change : std::vector<std::pair<std::string, std::string>>{
           {"\"version\":11", "\"version\":10"},
           {"\"selection\":\"indexed\"", "\"selection\":\"nonzero\""},
           {"\"selection\":\"indexed\"", "\"selection\":\"unknown\""},
           {"\"has_default\":true", "\"has_default\":1"},
           {"\"has_default\":true,", ""},
           {"cuda_conditional", "auto"},
           {"\"version\":1}", "\"version\":2}"},
           {"\"kind\":\"branch\"", "\"kind\":\"branch\",\"unknown\":0"},
           {"\"operations\":",
            "\"binding_strategy\":\"cuda_immutable_argument_frames\","
            "\"operations\":"}}) {
    auto invalid = source;
    invalid.replace(invalid.find(change.first), change.first.size(),
                    change.second);
    EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
  }
}

TEST(ExecutionPlan, BindingFamilyKeepsInitializationClosureSeparate) {
  const auto plan = ExecutionPlan::parse(R"({"version":10,
    "binding_strategy":"cuda_immutable_argument_frames",
    "initialization":[{"kind":"copy","source":"seed","destination":"state"}],
    "operations":[{"kind":"kernel","entry":"f","arguments":["state","out"]}],
    "resources":[{"kind":"ndarray","name":"state","dtype":"f32","shape":[17]}]})");
  EXPECT_TRUE(plan.cuda_binding_frames);
  EXPECT_EQ(plan.initialization_bindings,
            (std::set<std::string>{"seed", "state"}));
  EXPECT_EQ(plan.replay_bindings, (std::set<std::string>{"out", "state"}));
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"out", "seed"}));
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":9,
    "binding_strategy":"cuda_immutable_argument_frames",
    "operations":[{"kind":"fill32","destination":"out","value":0}]})"),
               std::invalid_argument);
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":10,
    "binding_strategy":"cuda_immutable_argument_frames",
    "operations":[{"kind":"while","predicate":"p","max_iterations":3,
      "condition":[],"body":[{"kind":"fill32","destination":"out","value":0}]}]})"),
               std::invalid_argument);
}

TEST(ExecutionPlan, ParallelPreservesBranchesAndBindings) {
  const std::string source = R"({"version":9,"operations":[
    {"kind":"cuda_parallel","branches":[
      [{"kind":"kernel","entry":"a","arguments":["x","shared"]}],
      [{"kind":"kernel","entry":"b","arguments":["y","shared"]}]]},
    {"kind":"kernel","entry":"join","arguments":["x","y"]}]})";
  const auto plan = ExecutionPlan::parse(source);
  const auto &parallel =
      *std::get<std::shared_ptr<const ExecutionPlan::Parallel>>(
          plan.operations[0]);
  ASSERT_EQ(parallel.branches.size(), 2);
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(parallel.branches[1][0]).entry,
            "b");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"shared", "x", "y"}));
  EXPECT_EQ(plan.replay_bindings, (std::set<std::string>{"shared", "x", "y"}));
  for (const auto &change : std::vector<std::pair<std::string, std::string>>{
           {"\"version\":9", "\"version\":8"},
           {"\"entry\":\"a\"", "\"entry\":\"a\",\"launch\":{}"},
           {"\"kind\":\"kernel\"", "\"kind\":\"while\""},
           {"\"branches\":", "\"unknown\":0,\"branches\":"}}) {
    auto invalid = source;
    invalid.replace(invalid.find(change.first), change.first.size(),
                    change.second);
    EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
  }
  for (const auto *branches : {"null", "[]", "[[]]", "[[],[]]", "[[{}],[{}]]"})
    EXPECT_THROW(
        ExecutionPlan::parse(
            std::string(
                R"({"version":9,"operations":[{"kind":"cuda_parallel","branches":)") +
            branches + "}]}"),
        std::invalid_argument);
}

TEST(ExecutionPlan, SelectedIndirectLaunchRetainsPacketAndVersion) {
  const std::string source = R"({"version":6,"resources":[
    {"kind":"ndarray","name":"grid","dtype":"u32","shape":[3]}],
    "operations":[{"kind":"kernel","entry":"payload","arguments":["x"],
      "launch":{"kind":"vulkan_indirect","packet":"grid"}}]})";
  const auto plan = ExecutionPlan::parse(source);
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x"}));
  EXPECT_EQ(plan.replay_bindings, (std::set<std::string>{"grid", "x"}));
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(plan.operations[0]).indirect_packet,
            std::optional<std::string>("grid"));
  for (const auto &change :
       std::vector<std::pair<std::string, std::string>>{
           {"\"version\":6", "\"version\":5"},
           {"vulkan_indirect", "cuda_indirect"},
           {"\"packet\":\"grid\"", "\"packet\":\"\""},
           {"\"packet\":\"grid\"", "\"packet\":\"grid\",\"address\":1"}}) {
    auto invalid = source;
    invalid.replace(invalid.find(change.first), change.first.size(), change.second);
    EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
  }
}

TEST(ExecutionPlan, SelectedExactLaunchRetainsExtentAndVersion) {
  for (const auto *kind : {"cpu_exact", "cuda_logical_exact"}) {
    const std::string source = std::string(R"({"version":7,"resources":[
      {"kind":"ndarray","name":"extent","dtype":"i32","shape":[2]}],
      "operations":[{"kind":"kernel","entry":"payload","arguments":["x"],
        "launch":{"kind":")") + kind + R"(","extent":"extent","capacity":65}}]})";
    const auto plan = ExecutionPlan::parse(source);
    EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x"}));
    EXPECT_EQ(plan.replay_bindings, (std::set<std::string>{"extent", "x"}));
    const auto &range = std::get<ExecutionPlan::Kernel>(plan.operations[0]).bounded_range;
    ASSERT_TRUE(range);
    EXPECT_EQ(range->cuda, std::string(kind) == "cuda_logical_exact");
    EXPECT_EQ(range->extent, "extent");
    EXPECT_EQ(range->capacity, 65);
    for (const auto &change : std::vector<std::pair<std::string, std::string>>{
             {"\"version\":7", "\"version\":6"},
             {"\"capacity\":65", "\"capacity\":0"},
             {"\"capacity\":65", "\"capacity\":-1"},
             {"\"capacity\":65", "\"capacity\":true"},
             {"\"capacity\":65", "\"capacity\":1.5"},
             {"\"capacity\":65", "\"capacity\":2147483648"},
             {"\"extent\":\"extent\"", "\"extent\":\"\""},
             {"\"extent\":\"extent\"", "\"extent\":\"extent\",\"address\":1"}}) {
      auto invalid = source;
      invalid.replace(invalid.find(change.first), change.first.size(), change.second);
      EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
    }
  }
}

TEST(ExecutionPlan, SelectedAdaptiveLaunchPreservesPolicyAndVersion) {
  for (const auto *policy : {"per_node", "grouped_stateful"}) {
    const std::string source = std::string(R"({"version":8,"resources":[
      {"kind":"ndarray","name":"extent","dtype":"i32","shape":[2]}],
      "operations":[{"kind":"kernel","entry":"payload","arguments":["x"],
        "launch":{"kind":"cuda_adaptive","extent":"extent","capacity":65,
                  "block_dim":32,"update_policy":")") + policy + R"("}}]})";
    const auto plan = ExecutionPlan::parse(source);
    EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x"}));
    EXPECT_EQ(plan.replay_bindings, (std::set<std::string>{"extent", "x"}));
    const auto &range = std::get<ExecutionPlan::Kernel>(plan.operations[0]).bounded_range;
    ASSERT_TRUE(range);
    EXPECT_TRUE(range->cuda);
    EXPECT_EQ(range->capacity, 65);
    EXPECT_EQ(range->block_dim, 32);
    EXPECT_EQ(range->grouped_update, std::string(policy) == "grouped_stateful");
    for (const auto &change : std::vector<std::pair<std::string, std::string>>{
             {"\"version\":8", "\"version\":7"},
             {"\"block_dim\":32", "\"block_dim\":0"},
             {"\"block_dim\":32", "\"block_dim\":true"},
             {"\"block_dim\":32", "\"block_dim\":1.5"},
             {"\"block_dim\":32", "\"block_dim\":4294967296"},
             {std::string("\"update_policy\":\"") + policy + "\"", "\"update_policy\":\"auto\""},
             {"\"extent\":\"extent\"", "\"extent\":\"extent\",\"device_node\":1"}}) {
      auto invalid = source;
      invalid.replace(invalid.find(change.first), change.first.size(), change.second);
      EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
    }
  }
}

namespace {
void validate_provider_description(const std::string &document) {
  const auto plan = ExecutionPlan::parse(document);
  native_operation_registry().decode(
      std::get<ExecutionPlan::Native>(plan.operations.at(0)));
}
}  // namespace

TEST(AotRootStorage, PackedLayoutsValidateSizesAndOffsets) {
  const auto parse = [](const std::string &layout) {
    return parse_root_storage_views(liong::json::parse_complete(
        "{\"version\":1,\"views\":{\"values\":" + layout + "}}"));
  };
  const auto values = parse(R"({"tree_id":2,"byte_offset":68,"dtype":"f32",
    "shape":[17],"element_shape":[2]})");
  const auto &value = values.at("values");
  EXPECT_EQ(value.tree_id, 2);
  EXPECT_EQ(value.byte_offset, 68);
  EXPECT_EQ(value.bytes(), 136);
  EXPECT_EQ(parse(R"({"tree_id":0,"byte_offset":0,"dtype":"i32",
    "shape":[],"element_shape":[]})")
                .at("values")
                .bytes(),
            4);
  EXPECT_EQ(parse(R"({"tree_id":0,"byte_offset":0,"dtype":"f64",
    "shape":[0],"element_shape":[]})")
                .at("values")
                .bytes(),
            0);
  for (
      const auto *layout :
      {R"({"tree_id":-1,"byte_offset":0,"dtype":"f32","shape":[17],"element_shape":[]})",
       R"({"tree_id":2147483648,"byte_offset":0,"dtype":"f32","shape":[17],"element_shape":[]})",
       R"({"tree_id":0,"byte_offset":1,"dtype":"f32","shape":[17],"element_shape":[]})",
       R"({"tree_id":0,"byte_offset":0,"dtype":"bad","shape":[17],"element_shape":[]})",
       R"({"tree_id":0,"byte_offset":0,"dtype":"f32","shape":[true],"element_shape":[]})",
       R"({"tree_id":0,"byte_offset":0,"dtype":"f32","shape":[17],"element_shape":[0]})",
       R"({"tree_id":0,"byte_offset":0,"dtype":"f64","shape":[4294967295,4294967295],"element_shape":[]})",
       R"({"tree_id":0,"byte_offset":0,"dtype":"f32","shape":[4294967296],"element_shape":[]})",
       R"({"tree_id":0,"byte_offset":0,"dtype":"f32","shape":[17],"element_shape":[],"address":123})"})
    EXPECT_THROW(parse(layout), std::invalid_argument) << layout;
}

TEST(ExecutionPlan, ModuleArraysSupplyStorageShapesAndGradients) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[
      {"kind":"ndarray_view","name":"tail","array":"values","offset":2,"shape":[15]},
      {"kind":"module_array","name":"values","view":"exported_values"},
      {"kind":"module_array","name":"gradient","view":"exported_gradient"},
      {"kind":"module_array","name":"metadata","view":"exported_shape"},
      {"kind":"extent","name":"count","dtype":"i32","array":"metadata","axis":0}],
    "gradients":{"values":"gradient"},
    "operations":[{"kind":"kernel","entry":"k","arguments":["values","tail","count"]}]})");
  EXPECT_TRUE(plan.arguments.empty());
  EXPECT_EQ(plan.resources.materialize,
            (std::vector<bool>{true, true, true, false, true}));
  EXPECT_EQ(std::get<ExecutionResources::ModuleArray>(
                plan.resources.declarations[1].value)
                .view,
            "exported_values");
  for (const auto *resource :
       {R"({"kind":"module_array","name":"values","view":""})",
        R"({"kind":"module_array","name":"values","view":123})",
        R"({"kind":"module_array","name":"values","view":"x","dtype":"f32"})"})
    EXPECT_THROW(
        ExecutionPlan::parse(
            std::string(
                R"({"version":1,"operations":[{"kind":"copy","source":"values","destination":"out"}],"resources":[)") +
            resource + "]}"),
        std::invalid_argument);
}

TEST(ExecutionPlan, PersistentControlKeepsNestedCodeAndRuntimeClosure) {
  const auto plan = ExecutionPlan::parse(R"({"version":3,
    "initialization":[{"kind":"copy","source":"initial","destination":"scratch"}],
    "resources":[{"kind":"ndarray","name":"scratch","dtype":"f32","shape":[17]}],
    "operations":[{"kind":"while","predicate":"predicate","max_iterations":8589934592,
      "condition":[{"kind":"kernel","entry":"check","arguments":["predicate","target"]}],
      "body":[{"kind":"while","predicate":"inner","max_iterations":2,
        "continue_while_nonzero":false,
        "condition":[{"kind":"kernel","entry":"inner_check","arguments":["inner"]}],
        "body":[{"kind":"kernel","entry":"advance","arguments":["scratch","x"]}]}]}]})");
  ASSERT_EQ(plan.operations.size(), 1);
  const auto &loop = *std::get<std::shared_ptr<const ExecutionPlan::While>>(
      plan.operations[0]);
  EXPECT_EQ(loop.max_iterations, 8589934592ull);
  EXPECT_TRUE(loop.continue_while_nonzero);
  const auto &inner =
      *std::get<std::shared_ptr<const ExecutionPlan::While>>(loop.body[0]);
  EXPECT_FALSE(inner.continue_while_nonzero);
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(inner.body[0]).entry, "advance");
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"initial", "inner", "predicate", "target",
                                      "x"}));
  EXPECT_EQ(
      plan.replay_bindings,
      (std::set<std::string>{"inner", "predicate", "scratch", "target", "x"}));
  EXPECT_EQ(plan.resources.materialize, (std::vector<bool>{true}));
  const std::string region =
      R"(,"condition":[{"kind":"fill32","destination":"p","value":0}],"body":[{"kind":"fill32","destination":"x","value":1}]}]})";
  for (
      const auto *fields :
      {R"({"version":2,"operations":[{"kind":"while","predicate":"p","max_iterations":1)",
       R"({"version":3,"operations":[{"kind":"while","predicate":"p","max_iterations":-1)",
       R"({"version":3,"operations":[{"kind":"while","predicate":"p","max_iterations":true)",
       R"({"version":3,"operations":[{"kind":"while","predicate":"p","max_iterations":1,"continue_while_nonzero":1)"})
    EXPECT_THROW(ExecutionPlan::parse(std::string(fields) + region),
                 std::invalid_argument);
}

TEST(ExecutionPlan, SelectedVulkanControlRetainsRecipeOnlyBindings) {
  const std::string source = R"({"version":4,"operations":[{"kind":"while",
    "predicate":"p","max_iterations":3,"condition":[],
    "body":[{"kind":"kernel","entry":"advance","arguments":["x","p"]}],
    "lowering":{"kind":"vulkan_structured","version":1,"counter":"counter",
      "status":"status","chunks":[{"iterations":2,"strategy":"compact"},
                                    {"iterations":1,"strategy":"coarse_conditional"}]}}]})";
  const auto plan = ExecutionPlan::parse(source);
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"counter", "p", "status", "x"}));
  EXPECT_EQ(plan.replay_bindings,
            (std::set<std::string>{"counter", "p", "status", "x"}));
  const auto &loop = *std::get<std::shared_ptr<const ExecutionPlan::While>>(
      plan.operations[0]);
  ASSERT_TRUE(loop.vulkan);
  EXPECT_EQ(loop.vulkan->recipe.chunks.size(), 2);
  EXPECT_EQ(loop.vulkan->recipe.chunks[1].strategy,
            VulkanControlStrategy::coarse_conditional);
  for (const auto &[from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"\"version\":4", "\"version\":3"},
           {"\"iterations\":2", "\"iterations\":3"},
           {"\"iterations\":2", "\"iterations\":4294967296"},
           {"\"strategy\":\"compact\"", "\"strategy\":\"auto\""}}) {
    auto invalid = source;
    invalid.replace(invalid.find(from), from.size(), to);
    EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
  }
}

TEST(ExecutionPlan, SelectedVulkanNestedControlRetainsPhysicalClosure) {
  const std::string source = R"({"version":5,"operations":[{"kind":"while",
    "predicate":"op","max_iterations":65,
    "condition":[{"kind":"kernel","entry":"initial","arguments":["x"]}],
    "body":[{"kind":"kernel","entry":"condition","arguments":["x"]},
            {"kind":"kernel","entry":"body","arguments":["x"]},
            {"kind":"kernel","entry":"tail","arguments":["x"]}],
    "lowering":{"kind":"vulkan_nested","version":1,"counter":"oc",
      "inners":[{"predicate":"ip","counter":"ic","status":"is",
        "condition_begin":1,"body_begin":2,"end":3,"max_iterations":65,"chunk_size":65}]}}],
    "resources":[{"kind":"ndarray","name":"ic","dtype":"i32","shape":[]}]})";
  const auto plan = ExecutionPlan::parse(source);
  const auto &loop = *std::get<std::shared_ptr<const ExecutionPlan::While>>(
      plan.operations[0]);
  ASSERT_TRUE(loop.vulkan && loop.vulkan->nested);
  EXPECT_EQ(loop.vulkan->nested->inners[0].chunk_size, 65);
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"ip", "is", "oc", "op", "x"}));
  EXPECT_EQ(plan.replay_bindings,
            (std::set<std::string>{"ic", "ip", "is", "oc", "op", "x"}));
  for (const auto &[from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"\"version\":5", "\"version\":4"},
           {"\"condition_begin\":1", "\"condition_begin\":0"},
           {"\"body_begin\":2", "\"body_begin\":1"},
           {"\"end\":3", "\"end\":4"},
           {"\"end\":3", "\"end\":true"},
           {"\"chunk_size\":65", "\"chunk_size\":0"},
           {"\"chunk_size\":65", "\"chunk_size\":66"},
           {"\"max_iterations\":65", "\"max_iterations\":4294967296"}}) {
    auto invalid = source;
    invalid.replace(invalid.find(from), from.size(), to);
    EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument);
  }
}

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

TEST(ExecutionPlan, HostBlobsFollowNativeDependenciesWithoutArrayStorage) {
  const std::string digest(64, 'a');
  const std::string blob = R"({"kind":"host_blob","name":"code","data":")" +
                           digest + R"(","byte_size":2096})";
  const auto plan = ExecutionPlan::parse(
      R"({"version":1,"operations":[{"kind":"native","provider":"test","operation":"run","version":1,"parameters":{},"bindings":{"plan":"p","out":"out"}}],"resources":[)" +
      blob +
      R"(,{"kind":"native","name":"p","provider":"test","operation":"plan","version":1,"parameters":{},"bindings":{"code":"code"}},)" +
      R"({"kind":"host_blob","name":"unused","data":")" + digest +
      R"(","byte_size":9223372036854775807}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"out"}));
  EXPECT_EQ(plan.resources.order, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(plan.resources.materialize, (std::vector<bool>{true, true, false}));
  const auto &data = std::get<ExecutionResources::HostBlob>(
      plan.resources.declarations[0].value);
  EXPECT_EQ(data.data, digest);
  EXPECT_EQ(data.byte_size, 2096);
  for (const auto *size : {"-1", "true", "1.5", "null"}) {
    const auto invalid =
        R"({"resources":[{"kind":"host_blob","name":"b","data":")" + digest +
        R"(","byte_size":)" + size + "}]}";
    EXPECT_THROW(
        ExecutionResources::parse(liong::json::parse_complete(invalid)),
        std::invalid_argument);
  }
  for (const auto &key :
       {std::string("../code"), std::string(64, 'A'), std::string(63, 'a')}) {
    const auto invalid =
        R"({"resources":[{"kind":"host_blob","name":"b","data":")" + key +
        R"(","byte_size":0}]})";
    EXPECT_THROW(
        ExecutionResources::parse(liong::json::parse_complete(invalid)),
        std::invalid_argument);
  }
  const auto invalid_shape =
      R"({"version":1,"operations":[{"kind":"copy","source":"a","destination":"out"}],"resources":[)" +
      blob +
      R"(,{"kind":"ndarray","name":"a","dtype":"f32","shape":[{"array":"code","axis":0}]}]})";
  EXPECT_THROW(ExecutionPlan::parse(invalid_shape), std::invalid_argument);
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

TEST(ExecutionPlan, ExactShapesAndOptionalSameViewRemainColdRequirements) {
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"copy","source":"x","destination":"y"}],"constraints":)";
  const auto plan = ExecutionPlan::parse(prefix + R"([
    {"kind":"array_layout","array":"x","exact_shape":[17],"record_bytes":4,"alignment":4,"dtype":"f32"},
    {"kind":"disjoint","left":"x","right":"y","allow_same_view":true}]})");
  const auto &required =
      std::get<ExecutionPlan::ArrayLayout>(plan.constraints[0]).required;
  EXPECT_TRUE(required.exact_shape);
  EXPECT_EQ(required.dtype, "f32");
  EXPECT_EQ(required.minimum_shape, (std::vector<std::uint64_t>{17}));
  EXPECT_TRUE(
      std::get<ExecutionPlan::Disjoint>(plan.constraints[1]).allow_same_view);
  for (
      const auto *constraint :
      {R"({"kind":"array_layout","array":"x","record_bytes":4,"alignment":4})",
       R"({"kind":"array_layout","array":"x","exact_shape":[17],"minimum_shape":[17],"record_bytes":4,"alignment":4})",
       R"({"kind":"array_layout","array":"x","exact_shape":null,"record_bytes":4,"alignment":4})",
       R"({"kind":"array_layout","array":"x","exact_shape":[true],"record_bytes":4,"alignment":4})",
       R"({"kind":"array_layout","array":"x","exact_shape":[17],"record_bytes":4,"alignment":4,"dtype":true})",
       R"({"kind":"array_layout","array":"x","exact_shape":[17],"record_bytes":4,"alignment":4,"dtype":"f128"})",
       R"({"kind":"disjoint","left":"x","right":"y","allow_same_view":1})",
       R"({"kind":"disjoint","left":"x","right":"y","allow_same_view":null})"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + "[" + constraint + "]}"),
                 std::invalid_argument);

  auto *device = reinterpret_cast<Device *>(std::uintptr_t(1));
  NativeStorageRange a{device, 1, 32, 4, 0, 16};
  auto alias = a;
  alias.buffer = 5;
  EXPECT_TRUE(
      a.proven_same_range(alias));  // Same memory through another buffer.
  alias.memory_offset++;
  EXPECT_FALSE(a.proven_same_range(alias));
  alias.memory_offset--;
  alias.bytes++;
  EXPECT_FALSE(a.proven_same_range(alias));
  alias.bytes--;
  alias.memory = 0;
  EXPECT_FALSE(a.proven_same_range(alias));
  alias.buffer = a.buffer;
  EXPECT_TRUE(
      a.proven_same_range(alias));  // Same buffer needs no memory import.
  alias.buffer_offset++;
  EXPECT_FALSE(a.proven_same_range(alias));
  alias.device = nullptr;
  EXPECT_FALSE(a.proven_same_range(alias));
}

TEST(ExecutionPlan, DisjointWriteGroupsValidatePortIndices) {
  const std::string prefix = R"({"version":1,"operations":[
    {"kind":"copy","source":"x","destination":"y"}],"constraints":[)";
  const auto plan = ExecutionPlan::parse(prefix + R"(
    {"kind":"disjoint_writes","reads":["x","x"],
     "writes":[{"array":"y","inplace_read":1}]}]})");
  const auto &group =
      std::get<ExecutionPlan::DisjointWrites>(plan.constraints[0]);
  EXPECT_EQ(group.reads, (std::vector<std::string>{"x", "x"}));
  ASSERT_EQ(group.writes.size(), 1);
  EXPECT_EQ(group.writes[0].inplace_read, 1);
  for (const auto *fields :
       {R"("reads":null,"writes":[{"array":"y"}])", R"("reads":[],"writes":[])",
        R"("reads":["unused"],"writes":[{"array":"y"}])",
        R"("reads":["x"],"writes":[{"array":"unused"}])",
        R"("reads":["x"],"writes":[{"array":"y","inplace_read":-1}])",
        R"("reads":["x"],"writes":[{"array":"y","inplace_read":1}])",
        R"("reads":["x"],"writes":[{"array":"y","inplace_read":true}])",
        R"("reads":["x"],"writes":[{"array":"y","inplace_read":0.0}])",
        R"("reads":["x"],"writes":[{"array":"y","inplace_read":null}])",
        R"("reads":["x"],"writes":[{"array":"y","unknown":0}])"})
    EXPECT_THROW(
        ExecutionPlan::parse(prefix + "{\"kind\":\"disjoint_writes\"," +
                             fields + "}]}"),
        std::invalid_argument);
}

TEST(ExecutionPlan, DisjointWriteRangesKeepOneInplaceReadAndNoSizeCap) {
  auto *device = reinterpret_cast<Device *>(std::uintptr_t(1));
  NativeStorageRange a{device, 1, 16, 0, 0, 16};
  auto b = a;
  b.memory_offset = 32;
  EXPECT_NO_THROW(validate_native_disjoint_writes({a, a}, {{b, std::nullopt}}));
  EXPECT_NO_THROW(
      validate_native_disjoint_writes({a}, {{a, 0}, {b, std::nullopt}}));
  EXPECT_THROW(validate_native_disjoint_writes({a, a}, {{a, 0}}),
               std::invalid_argument);
  EXPECT_THROW(validate_native_disjoint_writes({a}, {{a, 0}, {a, 0}}),
               std::invalid_argument);
  EXPECT_THROW(validate_native_disjoint_writes({a}, {{a, 1}}),
               std::invalid_argument);
  b.memory_offset = 17;
  EXPECT_THROW(validate_native_disjoint_writes({a}, {{b, 0}}),
               std::invalid_argument);
  b.bytes = 0;
  EXPECT_NO_THROW(validate_native_disjoint_writes({a}, {{b, std::nullopt}}));
  std::vector<NativeStorageRange> reads;
  std::vector<NativeStorageWrite> writes;
  for (std::size_t i = 0; i < 4097; ++i) {
    reads.push_back({device, 1, i * 32, 0, 0, 16});
    writes.push_back({{device, 1, i * 32 + 16, 0, 0, 16}, std::nullopt});
  }
  EXPECT_NO_THROW(validate_native_disjoint_writes(reads, writes));
  writes.back().range = reads.front();
  EXPECT_THROW(validate_native_disjoint_writes(reads, writes),
               std::invalid_argument);
}

TEST(ExecutionPlan, DisjointWriteGroupsCertifyImportedBufferRanges) {
  auto *device = reinterpret_cast<Device *>(std::uintptr_t(1));
  NativeStorageRange known{device, 3, 128, 9, 0, 16};
  auto unknown = known;
  unknown.memory = 0;
  EXPECT_NO_THROW(validate_native_disjoint_writes({unknown}, {{known, 0}}));
  EXPECT_NO_THROW(validate_native_disjoint_writes({known}, {{unknown, 0}}));
  auto adjacent = known;
  adjacent.memory_offset += 16;
  adjacent.buffer_offset += 16;
  EXPECT_NO_THROW(
      validate_native_disjoint_writes({unknown}, {{adjacent, std::nullopt}}));
  auto other = known;
  other.buffer = 10;
  EXPECT_THROW(validate_native_disjoint_writes({unknown}, {{other, 0}}),
               std::invalid_argument);
  // A known-memory alias through another buffer still participates in the
  // memory sweep when the allowed in-place read has no imported memory ID.
  EXPECT_THROW(validate_native_disjoint_writes({unknown, other}, {{known, 0}}),
               std::invalid_argument);
  other.memory_offset += 64;
  EXPECT_NO_THROW(
      validate_native_disjoint_writes({unknown, other}, {{known, 0}}));
  auto inconsistent = known;
  inconsistent.memory = 4;
  EXPECT_THROW(validate_native_disjoint_writes({known}, {{inconsistent, 0}}),
               std::invalid_argument);
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
TEST(ExecutionPlan, ViewStorageClosureIsSeparateFromShapeMetadata) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"resources":[
    {"kind":"ndarray_view","name":"selected","array":"middle","offset":1,"shape":[5]},
    {"kind":"ndarray_view","name":"middle","array":"parent","offset":2,"shape":[10]},
    {"kind":"ndarray","name":"parent","dtype":"f32","shape":[16]},
    {"kind":"ndarray_view","name":"metadata","array":"shape_parent","offset":0,"shape":[5]},
    {"kind":"ndarray","name":"shape_parent","dtype":"f32","shape":[1024]},
    {"kind":"ndarray","name":"scratch","dtype":"f32","shape":[{"array":"metadata","axis":0}]},
    {"kind":"native","name":"plan","provider":"test","operation":"plan","version":1,
     "parameters":{},"bindings":{"input":"selected"}}],
    "operations":[{"kind":"fill32","destination":"scratch","value":0},
                  {"kind":"native","provider":"test","operation":"run","version":1,
                   "parameters":{},"bindings":{"plan":"plan"}}]})");
  EXPECT_TRUE(plan.arguments.empty());
  EXPECT_EQ(plan.resources.materialize,
            (std::vector<bool>{true, true, true, false, false, true, true}));
  EXPECT_EQ(plan.replay_bindings,
            (std::set<std::string>{"scratch", "plan", "selected", "middle",
                                   "parent"}));
}

TEST(ExecutionPlan, ViewDescriptionsRejectCyclesAndReinterpretation) {
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"fill32","destination":"v","value":0}],"resources":[)";
  for (
      const auto *resource :
      {R"({"kind":"ndarray_view","name":"v","array":"v","offset":0,"shape":[1]})",
       R"({"kind":"ndarray_view","name":"v","array":"p","offset":0,"shape":[1],"dtype":"i32"})",
       R"({"kind":"ndarray_view","name":"v","array":"p","offset":-1,"shape":[1]})",
       R"({"kind":"ndarray_view","name":"v","array":"p","offset":0,"shape":[1]},
         {"kind":"scalar","name":"p","dtype":"i32","bits":"00000001"})"}) {
    EXPECT_THROW(ExecutionPlan::parse(prefix + resource + "]}"),
                 std::invalid_argument);
  }
}
TEST(ExecutionPlan, TypedLayoutRejectsEqualWidthReinterpretation) {
  NativeArrayBinding array;
  array.pointer = {};
  array.scalar_type = PrimitiveType::f32;
  array.shape = {17};
  const NativeArrayLayoutRequirement typed{{17}, {}, 4, 4, true, "f32"};
  EXPECT_NO_THROW(validate_native_array_layout(array, typed));
  array.scalar_type = PrimitiveType::i32;
  EXPECT_THROW(validate_native_array_layout(array, typed),
               std::invalid_argument);
  auto layout_only = typed;
  layout_only.dtype.clear();
  EXPECT_NO_THROW(validate_native_array_layout(array, layout_only));
}
}  // namespace taichi::lang::aot
