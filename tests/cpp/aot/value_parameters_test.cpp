#include "gtest/gtest.h"
#include "taichi/aot/value_parameters.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/program/callable.h"

#include <cstring>
#include <stdexcept>

namespace taichi::lang::aot {
namespace {
DataType lowered_vector(int count, DataType scalar = PrimitiveType::f32) {
  std::vector<AbstractDictionaryMember> members;
  for (int i = 0; i < count; ++i)
    members.push_back({scalar, "component_" + std::to_string(i)});
  return TypeFactory::get_instance().get_struct_type(members);
}

class ValueDir : public io::VirtualDir {
 public:
  std::string contents;
  bool get_file_size(const std::string &name, std::size_t &size) const override {
    if (name != "value_parameters.json")
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

TEST(AotValueParameters, LogicalShapesSurviveIdenticalLoweredStructures) {
  auto &factory = TypeFactory::get_instance();
  const auto vector = factory.get_tensor_type({3}, PrimitiveType::f32);
  const auto column = factory.get_tensor_type({3, 1}, PrimitiveType::f32);
  const auto lowered = lowered_vector(3);
  EXPECT_NE(vector, column);
  EXPECT_NO_THROW(validate_value_parameter_type(vector, lowered));
  EXPECT_NO_THROW(validate_value_parameter_type(column, lowered));
  EXPECT_THROW(validate_value_parameter_type(vector, lowered_vector(2)),
               std::invalid_argument);
  EXPECT_THROW(validate_value_parameter_type(vector,
                                              lowered_vector(3, PrimitiveType::i32)),
               std::invalid_argument);
  const auto logical = factory.get_struct_type(
      {{PrimitiveType::i8, "tag"}, {column, "column"}});
  const auto physical = factory.get_struct_type(
      {{PrimitiveType::i8, "tag"}, {lowered, "column"}});
  EXPECT_NO_THROW(validate_value_parameter_type(logical, physical));
  const auto renamed = factory.get_struct_type(
      {{PrimitiveType::i8, "tag"}, {lowered, "vector"}});
  EXPECT_THROW(validate_value_parameter_type(logical, renamed),
               std::invalid_argument);
}

TEST(AotValueParameters, ArtifactDeduplicatesAliasesAndKeepsWideShapes) {
  auto &factory = TypeFactory::get_instance();
  const auto vector = factory.get_tensor_type({65}, PrimitiveType::f32);
  const auto matrix = factory.get_tensor_type({9, 9}, PrimitiveType::f32);
  const ValueParameterTypes parameters{{{0}, vector}, {{1}, matrix}};
  NamedValueParameters original{{"entry", parameters}, {"alias", parameters}};
  const auto artifact = make_value_parameter_artifact(original);
  ASSERT_EQ(artifact.types.size(), 2);
  ASSERT_EQ(artifact.kernels.size(), 2);
  EXPECT_EQ(artifact.kernels.at("entry")[0].type,
            artifact.kernels.at("alias")[0].type);
  ValueDir directory;
  directory.contents = liong::json::print(liong::json::serialize(artifact));
  EXPECT_EQ(load_value_parameter_artifact(&directory), original);
  EXPECT_TRUE(load_value_parameter_artifact(nullptr).empty());
  directory.contents += " trailing";
  EXPECT_ANY_THROW(load_value_parameter_artifact(&directory));
}

TEST(AotValueParameters, InvalidReferencesAndResourceTypesAreRejected) {
  auto &factory = TypeFactory::get_instance();
  auto artifact = make_value_parameter_artifact(
      {{"entry", {{{0}, factory.get_tensor_type({3}, PrimitiveType::f32)}}}});
  auto parse = [&] {
    return parse_value_parameter_artifact(liong::json::serialize(artifact));
  };
  artifact.kernels["entry"].push_back(artifact.kernels["entry"][0]);
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.kernels["entry"].pop_back();
  artifact.kernels["entry"][0].type = 1;
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.kernels["entry"][0].type = 0;
  artifact.kernels["entry"][0].indices = {-1};
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.kernels["entry"][0].indices = {0};
  artifact.types.push_back(PrimitiveType::f32);
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.types.pop_back();
  artifact.types[0] = factory.get_pointer_type(PrimitiveType::f32);
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.version = 2;
  EXPECT_THROW(parse(), std::invalid_argument);
}

TEST(AotValueParameters, PhysicalTypesComeFromCompiledAbiAndArgumentPacks) {
  auto &factory = TypeFactory::get_instance();
  auto vector = lowered_vector(3);
  auto *inner = factory.get_struct_type({{vector, "vector"}})->as<StructType>();
  auto outer = factory.get_struct_type(
      {{factory.get_pointer_type(const_cast<StructType *>(inner)), "data_ptr"}});
  CallableBase kernel;
  kernel.args_type = factory.get_struct_type(
      {{vector, "arg_0"}, {outer, "arg_1"}})->as<StructType>();
  kernel.nested_parameters[{0}] = CallableBase::Parameter(PrimitiveType::unknown);
  kernel.nested_parameters[{1}] = CallableBase::Parameter(outer, false, true);
  EXPECT_EQ(compiled_value_parameter_type(kernel, {0}), vector);
  EXPECT_EQ(compiled_value_parameter_type(kernel, {1, 0}), vector);
  kernel.argpack_types[{1}] = inner;
  EXPECT_EQ(compiled_value_parameter_type(kernel, {1, 0}), vector);
  EXPECT_THROW(compiled_value_parameter_type(kernel, {2}), std::invalid_argument);
  EXPECT_THROW(compiled_value_parameter_type(kernel, {1, 1}), std::invalid_argument);
  kernel.argpack_types[{1}] = nullptr;
  EXPECT_THROW(compiled_value_parameter_type(kernel, {1, 0}), std::invalid_argument);
}

}  // namespace taichi::lang::aot
