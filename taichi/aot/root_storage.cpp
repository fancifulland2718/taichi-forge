#include "taichi/aot/root_storage.h"
#include "taichi/common/operation_json.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <tuple>

namespace taichi::lang::aot {
namespace {
constexpr const char *kFilename = "root_storage.json";

std::uint32_t scalar_bytes(const std::string &dtype) {
  if (dtype == "i8" || dtype == "u8")
    return 1;
  if (dtype == "i16" || dtype == "u16" || dtype == "f16")
    return 2;
  if (dtype == "i32" || dtype == "u32" || dtype == "f32")
    return 4;
  if (dtype == "i64" || dtype == "u64" || dtype == "f64")
    return 8;
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
}  // namespace

std::uint64_t RootStorageView::bytes() const {
  const auto width = scalar_bytes(dtype);
  if (tree_id < 0 || byte_offset % width)
    throw std::invalid_argument(
        "AOT root storage requires a valid tree and aligned offset");
  std::uint64_t size = width;
  const auto multiply = [&](std::uint32_t extent) {
    if (size && extent > std::numeric_limits<std::uint64_t>::max() / size)
      throw std::invalid_argument("AOT root storage size overflow");
    size *= extent;
  };
  for (const auto extent : element_shape) {
    if (!extent)
      throw std::invalid_argument(
          "AOT root storage element dimensions must be positive");
    multiply(extent);
  }
  for (const auto extent : shape)
    multiply(extent);
  if (size > std::numeric_limits<std::uint64_t>::max() - byte_offset)
    throw std::invalid_argument("AOT root storage range overflow");
  return size;
}

bool RootStorageView::operator==(const RootStorageView &other) const {
  return std::tie(tree_id, byte_offset, dtype, shape, element_shape) ==
         std::tie(other.tree_id, other.byte_offset, other.dtype, other.shape,
                  other.element_shape);
}

RootStorageViews parse_root_storage_views(const liong::json::JsonValue &json) {
  using namespace operation_json;
  fields(json, {"version", "views"});
  if (integer(json["version"]) != 1 || !json["views"].is_obj())
    throw std::invalid_argument("Unsupported AOT root storage metadata");
  RootStorageViews result;
  for (const auto &[key, value] : json["views"].obj.inner) {
    name(liong::json::JsonValue(key));
    fields(value,
           {"tree_id", "byte_offset", "dtype", "shape", "element_shape"});
    const auto tree = integer(value["tree_id"]);
    if (tree > std::numeric_limits<int>::max())
      throw std::invalid_argument("AOT root storage tree ID exceeds int32");
    RootStorageView view{static_cast<int>(tree), integer(value["byte_offset"]),
                         name(value["dtype"]), shape(value["shape"]),
                         shape(value["element_shape"])};
    view.bytes();
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
    view.bytes();
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
      std::string("{\"version\":1,\"views\":") +
      liong::json::print(liong::json::JsonValue(std::move(named_views))) + "}";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(json.data(), json.size());
  if (!out)
    throw std::runtime_error("Unable to write AOT root storage metadata");
}
}  // namespace taichi::lang::aot
