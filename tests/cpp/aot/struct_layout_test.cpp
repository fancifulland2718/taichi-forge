#include "gtest/gtest.h"
#include "taichi/aot/graph_artifact.h"
#include "taichi/aot/struct_layout.h"
#include "taichi/aot/value_layout.h"
#if defined(TI_WITH_VULKAN) || defined(TI_WITH_OPENGL)
#include "taichi/runtime/gfx/runtime.h"
#endif

#include <cstring>
#include <limits>
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
  std::string filename{"graphs.v2.json"};
  bool get_file_size(const std::string &name, size_t &size) const override {
    if (name != filename)
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

TEST(AotStructLayout, NamedCatalogDeduplicatesAndLoadsWithoutGraphs) {
  const auto type = example_type();
  auto artifact = make_struct_layout_artifact({{"record", type}, {"alias", type}});
  EXPECT_EQ(artifact.types.size(), 1u);
  EXPECT_EQ(artifact.layouts.at("record"), artifact.layouts.at("alias"));
  JsonDir directory;
  directory.filename = "array_layouts.json";
  directory.contents = liong::json::print(liong::json::serialize(artifact));
  const auto restored = load_struct_layout_artifact(&directory);
  EXPECT_EQ(restored.at("record"), type);
  EXPECT_EQ(restored.at("alias"), type);
  EXPECT_TRUE(load_struct_layout_artifact(nullptr).empty());
  directory.filename = "absent";
  EXPECT_TRUE(load_struct_layout_artifact(&directory).empty());
}

TEST(AotStructLayout, NamedCatalogRejectsInvalidTypesAndReferences) {
  const auto type = example_type();
  auto artifact = make_struct_layout_artifact({{"record", type}});
  auto parse = [&] {
    return parse_struct_layout_artifact(liong::json::serialize(artifact));
  };
  artifact.version = 2;
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.version = 1;
  artifact.layouts.at("record") = 1;
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.layouts.at("record") = 0;
  artifact.types.push_back(type);
  EXPECT_THROW(parse(), std::invalid_argument);
  artifact.layouts.emplace("alias", 1);
  EXPECT_THROW(parse(), std::invalid_argument);
  EXPECT_THROW(make_struct_layout_artifact({{"", type}}), std::invalid_argument);
  EXPECT_THROW(make_struct_layout_artifact({{"scalar", PrimitiveType::f32}}),
               std::invalid_argument);
}

TEST(AotStructLayout, NamedCatalogRejectsTrailingDataAndDuplicateObjectKeys) {
  JsonDir directory;
  directory.filename = "array_layouts.json";
  const auto document = liong::json::print(liong::json::serialize(
      make_struct_layout_artifact({{"record", example_type()}})));
  directory.contents = document + " \t\r\n";
  EXPECT_EQ(load_struct_layout_artifact(&directory).size(), 1u);
  for (const auto &suffix : {std::string("{}"), std::string("garbage"),
                             std::string(1, '\0')}) {
    directory.contents = document + suffix;
    EXPECT_THROW(load_struct_layout_artifact(&directory),
                 liong::json::JsonException);
  }
  directory.contents = "{\"version\":1," + document.substr(1);
  EXPECT_THROW(load_struct_layout_artifact(&directory),
               liong::json::JsonException);
}

TEST(AotValueLayout, ScalarTensorAndEmptyResultUseCompiledCapacity) {
  auto &factory = TypeFactory::get_instance();
  ValueLayout scalar(PrimitiveType::f64, 16, ValueStorage::Host);
  ASSERT_EQ(scalar.members.size(), 1u);
  EXPECT_EQ(scalar.buffer_size, 16u);
  EXPECT_EQ(scalar.members[0].size, 8u);
  EXPECT_TRUE(scalar.members[0].indices.empty());
  EXPECT_TRUE(scalar.members[0].shape.empty());
  auto tensor = factory.get_tensor_type({9, 9}, PrimitiveType::f32);
  ValueLayout matrix(tensor, 324, ValueStorage::Host);
  ASSERT_EQ(matrix.members.size(), 1u);
  EXPECT_EQ(matrix.members[0].shape, std::vector<int>({9, 9}));
  EXPECT_EQ(matrix.members[0].size, 324u);
  auto rank_three = factory.get_tensor_type({2, 3, 4}, PrimitiveType::i16);
  ValueLayout cube(rank_three, 48, ValueStorage::Host);
  EXPECT_EQ(cube.members[0].shape, std::vector<int>({2, 3, 4}));
  ValueLayout empty(factory.get_struct_type({}), 0, ValueStorage::Host);
  EXPECT_TRUE(empty.members.empty());
  EXPECT_EQ(empty.buffer_size, 0u);
}

TEST(AotValueLayout, GfxScalarWidthsPreserveExplicitPaddingAndIndices) {
  auto &factory = TypeFactory::get_instance();
  auto vector = factory.get_tensor_type({3}, PrimitiveType::f32);
  auto flags = factory.get_tensor_type({2}, PrimitiveType::u1);
  auto inner = factory.get_struct_type(
      {{PrimitiveType::u1, "enabled", 0}, {flags, "flags", 4}}, "4-");
  auto type = factory.get_struct_type(
      {{inner, "inner", 0}, {vector, "vector", 16},
       {PrimitiveType::i32, "tail", 32}}, "1-");
  ValueLayout layout(type, 48, ValueStorage::Gfx);
  ASSERT_EQ(layout.members.size(), 4u);
  EXPECT_EQ(layout.buffer_size, 48u);  // Preserve target trailing padding.
  EXPECT_EQ(layout.members[0].scalar_size, 4u);
  EXPECT_EQ(layout.members[1].path, "inner.flags");
  EXPECT_EQ(layout.members[1].indices, std::vector<int>({0, 1}));
  EXPECT_EQ(layout.members[1].offset, 4u);
  EXPECT_EQ(layout.members[1].size, 8u);
  EXPECT_EQ(layout.members[2].offset, 16u);
  EXPECT_EQ(layout.members[2].size, 12u);
  EXPECT_EQ(layout.members[3].offset, 32u);
  EXPECT_THROW(ValueLayout(type, 35, ValueStorage::Gfx),
               std::invalid_argument);
}

TEST(AotValueLayout, SharesNaturalRecordOffsetsWithoutLosingTensorMetadata) {
  const auto type = example_type();
  StructLayout record(type);
  ValueLayout value(type, record.stride, ValueStorage::Host);
  ASSERT_EQ(record.members.size(), value.members.size());
  for (std::size_t i = 0; i < record.members.size(); ++i) {
    EXPECT_EQ(record.members[i].path, value.members[i].path);
    EXPECT_EQ(record.members[i].offset, value.members[i].offset);
    EXPECT_EQ(record.members[i].size, value.members[i].size);
    EXPECT_EQ(record.members[i].alignment, value.members[i].scalar_alignment);
    EXPECT_EQ(record.members[i].shape, value.members[i].shape);
  }
  EXPECT_EQ(value.members[2].indices, std::vector<int>({1, 1}));
}

TEST(AotValueLayout, LogicalTensorUsesCompiledComponentSlots) {
  auto &factory = TypeFactory::get_instance();
  auto column = factory.get_tensor_type({3, 1}, PrimitiveType::f32);
  auto padded = factory.get_struct_type(
      {{PrimitiveType::f32, "a", 0}, {PrimitiveType::f32, "b", 8},
       {PrimitiveType::f32, "c", 16}}, "1-");
  ValueLayout layout(column, padded, 24, ValueStorage::Host);
  ASSERT_EQ(layout.members.size(), 1u);
  EXPECT_EQ(layout.members[0].shape, std::vector<int>({3, 1}));
  EXPECT_EQ(layout.members[0].component_offsets,
            std::vector<std::size_t>({0, 8, 16}));
  EXPECT_EQ(layout.members[0].size, 20u);
  EXPECT_EQ(layout.buffer_size, 24u);
  EXPECT_THROW(ValueLayout(column, padded, 19, ValueStorage::Host),
               std::invalid_argument);
  auto contiguous = factory.get_struct_type(
      {{PrimitiveType::f32, "a"}, {PrimitiveType::f32, "b"},
       {PrimitiveType::f32, "c"}});
  ValueLayout dense(column, contiguous, 12, ValueStorage::Host);
  EXPECT_TRUE(dense.members[0].component_offsets.empty());
  EXPECT_EQ(dense.members[0].size, 12u);
}

TEST(AotValueLayout, LogicalNestedFlagsUseGfxWidthsAndPhysicalOffsets) {
  auto &factory = TypeFactory::get_instance();
  auto flags = factory.get_struct_type(
      {{PrimitiveType::u1, "a"}, {PrimitiveType::u1, "b"}});
  auto logical = factory.get_struct_type(
      {{PrimitiveType::i8, "tag"}, {flags, "flags"}});
  auto physical_flags = factory.get_struct_type(
      {{PrimitiveType::u1, "a", 0}, {PrimitiveType::u1, "b", 4}}, "1-");
  auto physical = factory.get_struct_type(
      {{PrimitiveType::i8, "tag", 0}, {physical_flags, "flags", 16}}, "1-");
  ValueLayout layout(logical, physical, 32, ValueStorage::Gfx);
  ASSERT_EQ(layout.members.size(), 3u);
  EXPECT_EQ(layout.members[1].path, "flags.a");
  EXPECT_EQ(layout.members[1].offset, 16u);
  EXPECT_EQ(layout.members[2].offset, 20u);
  EXPECT_EQ(layout.members[2].scalar_size, 4u);
  EXPECT_EQ(layout.members[2].size, 4u);
}

#if defined(TI_WITH_VULKAN) || defined(TI_WITH_OPENGL)
TEST(AotValueLayout, UsesActualGfxUniformAndStorageBufferLayouts) {
  auto &factory = TypeFactory::get_instance();
  const auto *type = factory.get_struct_type(
      {{PrimitiveType::u1, "enabled"},
       {factory.get_tensor_type({2}, PrimitiveType::u1), "flags"},
       {factory.get_tensor_type({3}, PrimitiveType::f32), "vector"},
       {PrimitiveType::i16, "tail"}})->as<StructType>();
  for (const auto &name : {"1-", "4-"}) {
    const auto [compiled, bytes] =
        gfx::GfxRuntime::get_struct_type_with_data_layout(type, name);
    ValueLayout value(compiled, bytes, ValueStorage::Gfx);
    ASSERT_EQ(value.members.size(), 4u);
    EXPECT_EQ(value.members[0].size, 4u);
    EXPECT_EQ(value.members[1].size, 8u);
    EXPECT_EQ(value.members[2].size, 12u);
    EXPECT_EQ(value.members[3].size, 2u);
    for (std::size_t i = 0; i < value.members.size(); ++i)
      EXPECT_EQ(value.members[i].offset, compiled->elements()[i].offset);
    EXPECT_EQ(bytes, std::string(name) == "1-" ? 48u : 26u);
  }
}
#endif

TEST(AotValueLayout, RejectsOverlapsOverflowAndNonValueTypes) {
  auto &factory = TypeFactory::get_instance();
  auto zero_overlap = factory.get_struct_type(
      {{PrimitiveType::i32, "a", 0}, {PrimitiveType::f32, "b", 0}}, "4-");
  EXPECT_THROW(ValueLayout(zero_overlap, 8, ValueStorage::Gfx),
               std::invalid_argument);
  auto overlap = factory.get_struct_type(
      {{PrimitiveType::i32, "a", 0}, {PrimitiveType::f32, "b", 2}}, "4-");
  EXPECT_THROW(ValueLayout(overlap, 8, ValueStorage::Gfx),
               std::invalid_argument);
  auto overflow = factory.get_struct_type(
      {{PrimitiveType::i32, "a", std::numeric_limits<std::size_t>::max() - 3}},
      "4-");
  EXPECT_THROW(ValueLayout(overflow, std::numeric_limits<std::size_t>::max(),
                           ValueStorage::Gfx), std::invalid_argument);
  auto large = factory.get_tensor_type({std::numeric_limits<int>::max()},
                                       PrimitiveType::f64);
  EXPECT_THROW(ValueLayout(large, 0, ValueStorage::Host),
               std::invalid_argument);
  EXPECT_THROW(ValueLayout(PrimitiveType::unknown, 8, ValueStorage::Host),
               std::invalid_argument);
  EXPECT_THROW(ValueLayout(factory.get_pointer_type(PrimitiveType::f32), 8,
                           ValueStorage::Host), std::invalid_argument);
  EXPECT_THROW(ValueLayout(PrimitiveType::i32, 3, ValueStorage::Host),
               std::invalid_argument);
}

TEST(AotStructLayout, ValueReflectionDoesNotWidenArrayElementContract) {
  auto &factory = TypeFactory::get_instance();
  EXPECT_THROW(StructLayout{factory.get_struct_type({})},
               std::invalid_argument);
  auto duplicate = factory.get_struct_type(
      {{PrimitiveType::f32, "a"}, {PrimitiveType::f32, "a"}});
  EXPECT_THROW(StructLayout{duplicate}, std::invalid_argument);
  auto tensor = factory.get_tensor_type({2, 3, 4}, PrimitiveType::f32);
  auto rank_three = factory.get_struct_type({{tensor, "value"}});
  EXPECT_THROW(StructLayout{rank_three}, std::invalid_argument);
}
}  // namespace taichi::lang::aot
