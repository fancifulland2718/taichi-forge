#include "gtest/gtest.h"
#include "taichi/aot/argument_origins.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/program/callable.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace taichi::lang::aot {
namespace {
ArgumentOrigins origins() {
  return {{{0}, {0}, "left", "argpack"},
          {{0, 0}, {0, 0}, "gain", "value"},
          {{0, 1}, {0, 1}, "child", "argpack"},
          {{0, 1, 0}, {3}, "values", "ndarray"},
          {{1}, {1}, "right", "argpack"},
          {{1, 0}, {4}, "values", "ndarray"},
          {{2}, {2}, "count", "value"}};
}

CallableBase compiled() {
  auto &factory = TypeFactory::get_instance();
  const auto empty = factory.get_struct_type({});
  const auto child = factory.get_struct_type(
      {{factory.get_pointer_type(const_cast<Type *>(empty)), "data_ptr"}});
  const auto body = factory.get_struct_type(
      {{PrimitiveType::f32, "arg_0_0"}, {child, "arg_0_1"}});
  const auto pack = factory.get_struct_type(
      {{factory.get_pointer_type(const_cast<Type *>(body)), "data_ptr"}});
  const auto array = factory.get_ndarray_struct_type(PrimitiveType::f32, 1, true);
  CallableBase kernel;
  kernel.args_type = factory.get_struct_type(
      {{pack, "arg_0"}, {child, "arg_1"}, {PrimitiveType::i32, "arg_2"},
       {array, "arg_3"}, {array, "arg_4"}})->as<StructType>();
  kernel.nested_parameters[{0}] = CallableBase::Parameter(pack, false, true);
  kernel.nested_parameters[{0, 0}] = CallableBase::Parameter(PrimitiveType::f32);
  kernel.nested_parameters[{0, 0}].ptype = ParameterType::kScalar;
  kernel.nested_parameters[{0, 1}] = CallableBase::Parameter(child, false, true);
  kernel.nested_parameters[{1}] = CallableBase::Parameter(child, false, true);
  kernel.nested_parameters[{2}] = CallableBase::Parameter(PrimitiveType::i32);
  kernel.nested_parameters[{2}].ptype = ParameterType::kScalar;
  for (int index : {3, 4}) {
    auto parameter = CallableBase::Parameter(array, true);
    parameter.ptype = ParameterType::kNdarray;
    parameter.name = "__argpack_values";  // Deliberately identical.
    kernel.nested_parameters[{index}] = parameter;
  }
  return kernel;
}

class OriginDir : public io::VirtualDir {
 public:
  std::string contents;
  bool get_file_size(const std::string &name, std::size_t &size) const override {
    if (name != "argument_origins.json")
      return false;
    size = contents.size();
    return true;
  }
  std::size_t load_file(const std::string &, void *out,
                        std::size_t size) const override {
    std::memcpy(out, contents.data(), size);
    return size;
  }
};
}  // namespace

TEST(AotArgumentOrigins, LiftedResourcesKeepDistinctPublicParents) {
  auto kernel = compiled();
  const auto members = origins();
  EXPECT_NO_THROW(validate_argument_origins(kernel, members));
  // GFX supplies a separate pack-storage map; LLVM can derive it from the
  // compiled pointer type. Neither route inspects generated member names.
  const auto outer = kernel.args_type->elements()[0].type->as<StructType>();
  kernel.argpack_types[{0}] =
      DataType(outer->elements()[0].type).ptr_removed()->as<StructType>();
  EXPECT_NO_THROW(validate_argument_origins(kernel, members));
  kernel.nested_parameters[{0, 0}] =
      CallableBase::Parameter(PrimitiveType::unknown);
  kernel.nested_parameters[{0, 0}].ptype = ParameterType::kScalar;
  EXPECT_NO_THROW(validate_argument_origins(kernel, members));
  kernel.argpack_types[{0}] = nullptr;
  EXPECT_THROW(validate_argument_origins(kernel, members), std::invalid_argument);
}

TEST(AotArgumentOrigins, AliasLayoutsAreCanonicalAndShared) {
  auto reversed = origins();
  std::reverse(reversed.begin(), reversed.end());
  const NamedArgumentOrigins original{{"entry", origins()}, {"alias", reversed}};
  const auto artifact = make_argument_origin_artifact(original);
  ASSERT_EQ(artifact.layouts.size(), 1u);
  EXPECT_EQ(artifact.kernels.at("entry"), artifact.kernels.at("alias"));
  OriginDir directory;
  directory.contents = liong::json::print(liong::json::serialize(artifact));
  const auto loaded = load_argument_origin_artifact(&directory);
  EXPECT_EQ(loaded.at("entry"), origins());
  EXPECT_EQ(loaded.at("alias"), origins());
  EXPECT_TRUE(load_argument_origin_artifact(nullptr).empty());
  directory.contents += " trailing";
  EXPECT_ANY_THROW(load_argument_origin_artifact(&directory));
}

TEST(AotArgumentOrigins, RejectsAmbiguousTreesAndPhysicalMismatches) {
  const auto kernel = compiled();
  auto expect_rejected = [&](auto mutate) {
    auto changed = origins();
    mutate(changed);
    EXPECT_THROW(validate_argument_origins(kernel, changed),
                 std::invalid_argument);
  };
  expect_rejected([](auto &o) { o.pop_back(); });
  expect_rejected([](auto &o) { o[5].physical = o[3].physical; });
  expect_rejected([](auto &o) { o[5].logical = o[3].logical; });
  expect_rejected([](auto &o) { o[2].name = "gain"; });
  expect_rejected([](auto &o) { o[3].logical = {0, 0, 0}; });
  expect_rejected([](auto &o) { o[3].kind = "value"; });
  expect_rejected([](auto &o) { o[3].kind = "texture"; });
  expect_rejected([](auto &o) { o[3].physical = {9}; });
  expect_rejected([](auto &o) { o[1].physical = {1, 0}; });
  expect_rejected([](auto &o) { o[6].logical = {3}; });
  expect_rejected([](auto &o) { o[0].name = std::string("a\0b", 3); });
  expect_rejected([](auto &o) { o[0].kind = "unknown"; });
}

TEST(AotArgumentOrigins, RejectsMalformedLayoutReferences) {
  auto artifact = make_argument_origin_artifact({{"entry", origins()}});
  auto parse = [&] {
    return parse_argument_origin_artifact(liong::json::serialize(artifact));
  };
  artifact.kernels["entry"] = 1;
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.kernels["entry"] = 0;
  artifact.layouts.push_back(artifact.layouts.front());
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.layouts.pop_back();
  artifact.kernels.clear();
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.kernels["entry"] = 0;
  artifact.version = 2;
  EXPECT_THROW(parse(), std::invalid_argument);
}

TEST(AotArgumentOrigins, MemberCountFollowsTheCompiledLayout) {
  auto &factory = TypeFactory::get_instance();
  std::vector<AbstractDictionaryMember> values;
  CallableBase kernel;
  ArgumentOrigins members{{{0}, {0}, "pack", "argpack"}};
  for (int index = 0; index < 65; ++index) {
    const auto name = "v" + std::to_string(index);
    values.push_back({PrimitiveType::i32, name});
    members.push_back({{0, index}, {0, index}, name, "value"});
    auto parameter = CallableBase::Parameter(PrimitiveType::i32);
    parameter.ptype = ParameterType::kScalar;
    kernel.nested_parameters[{0, index}] = parameter;
  }
  const auto body = factory.get_struct_type(values);
  const auto pack = factory.get_struct_type(
      {{factory.get_pointer_type(const_cast<Type *>(body)), "data_ptr"}});
  kernel.args_type = factory.get_struct_type({{pack, "arg_0"}})->as<StructType>();
  kernel.nested_parameters[{0}] = CallableBase::Parameter(pack, false, true);
  EXPECT_NO_THROW(validate_argument_origins(kernel, members));
  EXPECT_EQ(make_argument_origin_artifact({{"entry", members}})
                .layouts.front().size(), 66u);
}

}  // namespace taichi::lang::aot
