#include "taichi/aot/root_storage.h"
#include "taichi/common/operation_json.h"
#include "taichi/program/storage_view.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <tuple>

namespace taichi::lang::aot {
namespace {
constexpr const char *kFilename = "root_storage.json";

DataType scalar_type(const std::string &dtype) {
  for (const auto &type : {PrimitiveType::i8, PrimitiveType::u8,
                          PrimitiveType::i16, PrimitiveType::u16, PrimitiveType::f16,
                          PrimitiveType::i32, PrimitiveType::u32, PrimitiveType::f32,
                          PrimitiveType::i64, PrimitiveType::u64, PrimitiveType::f64})
    if (type->to_string() == dtype)
      return type;
  throw std::invalid_argument("Unsupported AOT root storage scalar type");
}

std::vector<std::uint32_t> shape(const liong::json::JsonValue &value) {
  if (!value.is_arr())
    throw std::invalid_argument("AOT root storage shape must be an array");
  std::vector<std::uint32_t> result;
  for (const auto &dimension : value.arr.inner) {
    const auto size = operation_json::integer(dimension);
    if (size > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "AOT root storage dimensions must fit uint32");
    result.push_back(static_cast<std::uint32_t>(size));
  }
  return result;
}

std::vector<std::int64_t> strides(const liong::json::JsonValue &value) {
  if (!value.is_arr())
    throw std::invalid_argument("AOT root storage strides must be an array");
  std::vector<std::int64_t> result;
  for (const auto &stride : value.arr.inner) {
    if (stride.ty != liong::json::L_JSON_INT)
      throw std::invalid_argument("AOT root storage strides must be signed integers");
    result.push_back(stride.num_int);
  }
  return result;
}
}  // namespace

storage::DenseStorageProperties RootStorageView::properties() const {
  if (tree_id < 0 ||
      byte_offset > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument(
        "AOT root storage requires a valid tree and addressable offset");
  for (const auto extent : element_shape)
    if (!extent)
      throw std::invalid_argument(
          "AOT root storage element dimensions must be positive");
  storage::DenseStorageLayoutSpec layout;
  layout.scalar_type = scalar_type(dtype);
  layout.index_shape.assign(shape.begin(), shape.end());
  layout.element_shape.assign(element_shape.begin(), element_shape.end());
  layout.index_strides_bytes = index_strides_bytes;
  layout.element_strides_bytes = element_strides_bytes;
  layout.byte_offset = static_cast<std::int64_t>(byte_offset);
  const auto analysis = storage::analyze_dense_storage_layout(layout);
  if (!analysis || !analysis.properties.aligned ||
      analysis.properties.uniqueness !=
          storage::StorageMappingUniqueness::kProvenUnique)
    throw std::invalid_argument(
        "AOT root storage requires an aligned, unique affine layout");
  return analysis.properties;
}

std::uint64_t RootStorageView::bytes() const {
  const auto p = properties();
  if (p.scalar_count > std::numeric_limits<std::uint64_t>::max() / p.scalar_size)
    throw std::invalid_argument("AOT root storage payload size overflow");
  return p.scalar_count * p.scalar_size;
}

bool RootStorageView::operator==(const RootStorageView &other) const {
  return std::tie(tree_id, byte_offset, dtype, shape, element_shape,
                  index_strides_bytes, element_strides_bytes) ==
         std::tie(other.tree_id, other.byte_offset, other.dtype, other.shape,
                  other.element_shape, other.index_strides_bytes,
                  other.element_strides_bytes);
}

RootStorageViews parse_root_storage_views(const liong::json::JsonValue &json) {
  using namespace operation_json;
  fields(json, {"version", "views"});
  if (json.obj.inner.size() != 2 || integer(json["version"]) != 2 ||
      !json["views"].is_obj())
    throw std::invalid_argument("Unsupported AOT root storage metadata");
  RootStorageViews result;
  for (const auto &[key, value] : json["views"].obj.inner) {
    name(liong::json::JsonValue(key));
    fields(value,
           {"tree_id", "byte_offset", "dtype", "shape", "element_shape",
            "index_strides_bytes", "element_strides_bytes"});
    if (value.obj.inner.size() != 7)
      throw std::invalid_argument("AOT root storage is missing required layout fields");
    const auto tree = integer(value["tree_id"]);
    if (tree > std::numeric_limits<int>::max())
      throw std::invalid_argument("AOT root storage tree ID exceeds int32");
    RootStorageView view{static_cast<int>(tree), integer(value["byte_offset"]),
                         name(value["dtype"]), shape(value["shape"]),
                         shape(value["element_shape"]),
                         strides(value["index_strides_bytes"]),
                         strides(value["element_strides_bytes"])};
    view.properties();
    result.emplace(key, std::move(view));
  }
  return result;
}

RootStorageViews load_root_storage_views(const io::VirtualDir &directory) {
  std::size_t size = 0;
  if (!directory.get_file_size(kFilename, size))
    return {};
  std::vector<std::uint8_t> bytes;
  if (!directory.load_file(kFilename, bytes) || bytes.empty())
    throw std::invalid_argument("Unreadable AOT root storage metadata");
  const std::string text(bytes.begin(), bytes.end());
  return parse_root_storage_views(liong::json::parse_complete(text));
}

void write_root_storage_views(const RootStorageViews &views,
                              const std::map<int, int> &tree_mapping,
                              const std::string &directory) {
  const auto path = std::filesystem::path(directory) / kFilename;
  if (views.empty()) {
    std::filesystem::remove(path);
    return;
  }
  auto relocated = views;
  for (auto &[name, view] : relocated) {
    view.properties();
    const auto found = tree_mapping.find(view.tree_id);
    if (found == tree_mapping.end())
      throw std::invalid_argument(
          "AOT root storage references an unexported tree");
    view.tree_id = found->second;
  }
  liong::json::JsonObject named_views;
  for (const auto &[name, view] : relocated)
    named_views.inner.emplace(name, liong::json::serialize(view));
  const auto json =
      std::string("{\"version\":2,\"views\":") +
      liong::json::print(liong::json::JsonValue(std::move(named_views))) + "}";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(json.data(), json.size());
  if (!out)
    throw std::runtime_error("Unable to write AOT root storage metadata");
}
}  // namespace taichi::lang::aot
