#include "gtest/gtest.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/aot/struct_layout.h"

#include <cstring>
#include <stdexcept>

namespace taichi::lang::aot {
namespace {
DataType example_type() {
  auto &factory = TypeFactory::get_instance();
  auto inner = factory.get_struct_type(
      {{PrimitiveType::i16, "id"}, {PrimitiveType::f64, "weight"}});
  auto vector = factory.get_tensor_type({3}, PrimitiveType::f32);
  auto matrix = factory.get_tensor_type({2, 2}, PrimitiveType::f32);
  return factory.get_struct_type({{PrimitiveType::i8, "tag"},
                                  {inner, "inner"},
                                  {vector, "vector"},
                                  {matrix, "matrix"},
                                  {PrimitiveType::u8, "tail"}});
}

class JsonDir : public io::VirtualDir {
 public:
  std::string contents;
  bool get_file_size(const std::string &name, size_t &size) const override {
    if (name != "graphs.v2.json")
      return false;
    size = contents.size();
    return true;
  }
  size_t load_file(const std::string &, void *out, size_t size) const override {
    std::memcpy(out, contents.data(), size);
    return size;
  }
};
}  // namespace

TEST(AotStructLayout, NativeOffsetsPreserveNestedPaddingAndTensorShape) {
  StructLayout layout(example_type());
  ASSERT_EQ(layout.members.size(), 6);
  EXPECT_EQ(layout.stride, 56);
  EXPECT_EQ(layout.alignment, 8);
  EXPECT_EQ(layout.members[1].path, "inner.id");
  EXPECT_EQ(layout.members[1].offset, 8);
  EXPECT_EQ(layout.members[2].path, "inner.weight");
  EXPECT_EQ(layout.members[2].offset, 16);
  EXPECT_EQ(layout.members[2].alignment, 8);
  EXPECT_EQ(layout.members[3].offset, 24);
  EXPECT_EQ(layout.members[3].shape, std::vector<int>({3}));
  EXPECT_EQ(layout.members[4].offset, 36);
  EXPECT_EQ(layout.members[4].shape, std::vector<int>({2, 2}));
  EXPECT_EQ(layout.members[5].offset, 52);
}

TEST(AotStructLayout, InvalidMembersAndOverlappingOffsetsAreRejected) {
  auto &factory = TypeFactory::get_instance();
  auto invalid = factory.get_struct_type(
      {{PrimitiveType::f32, "a", 0}, {PrimitiveType::i32, "b", 2}});
  EXPECT_THROW(StructLayout{invalid}, std::invalid_argument);
  auto unresolved = factory.get_struct_type({{PrimitiveType::unknown, "a"}});
  EXPECT_THROW(StructLayout{unresolved}, std::invalid_argument);
  auto pointer = factory.get_struct_type(
      {{factory.get_pointer_type(PrimitiveType::f32), "ptr"}});
  EXPECT_THROW(StructLayout{pointer}, std::invalid_argument);
}

TEST(AotStructLayout, GraphV2RestoresTypesAndRejectsInconsistentMetadata) {
  GraphArtifact artifact;
  artifact.version = 2;
  CompiledDispatch dispatch;
  dispatch.kernel_name = "update";
  dispatch.symbolic_args.emplace_back(ArgKind::kNdarray, "records",
                                      example_type(), 1);
  artifact.graphs.emplace("step", CompiledGraph({dispatch}));
  artifact.struct_types["step"]["records"] = example_type();
  JsonDir directory;
  GraphMap restored;
  auto load = [&] {
    directory.contents = liong::json::print(liong::json::serialize(artifact));
    return load_struct_graph_artifact(directory, restored);
  };
  ASSERT_TRUE(load());
  EXPECT_EQ(restored.at("step").args.at("records").element_dtype(),
            example_type());
  EXPECT_EQ(restored.at("step").dispatches[0].symbolic_args[0],
            dispatch.symbolic_args[0]);
  artifact.version = 3;
  EXPECT_THROW(load(), std::invalid_argument);
  artifact.version = 2;
  artifact.struct_types.clear();
  EXPECT_THROW(load(), std::invalid_argument);
  artifact.struct_types["step"]["records"] = example_type();
  artifact.struct_types["step"]["unused"] = example_type();
  EXPECT_THROW(load(), std::invalid_argument);
}
}  // namespace taichi::lang::aot
