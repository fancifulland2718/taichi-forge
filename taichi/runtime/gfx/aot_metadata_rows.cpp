#include "taichi/runtime/gfx/aot_metadata_rows.h"

#include <unordered_map>
#include "taichi/ir/type_factory.h"

namespace taichi::lang::gfx {
namespace {
using namespace liong::json;
using Kernel = spirv::TaichiKernelAttributes;
using Context = spirv::KernelContextAttributes;
using Task = spirv::TaskAttributes;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

// All supported rows are objects in the ordinary codec. Within each group
// the first occurrence is written inline; an integer recalls an earlier row.
// Group order is significant only for args/returns sharing ABI elements.
class RowEncoder {
 public:
  void encode(JsonValue &kernel) {
    auto &ctx = kernel["ctx_attribs"];
    for (auto key : {"arg_attribs_vec_", "ret_attribs_vec_", "arr_access",
                     "grad_arr_access"})
      encode_rows(ctx[key], key);
    for (auto key : {"args_type_", "rets_type_"})
      if (!ctx[key].is_null())
        encode_rows(ctx[key]["content"]["elements_"], "abi_elements");
    for (auto &task : kernel["tasks_attribs"].arr.inner)
      for (auto key :
           {"buffer_binds", "texture_binds", "acceleration_structure_binds"})
        encode_rows(task[key], key);
    kernel.obj.inner.emplace("row_storage", JsonValue(1));
  }

 private:
  void encode_rows(JsonValue &rows, const char *group) {
    auto &index = indices_[group];
    for (auto &row : rows.arr.inner) {
      auto [it, added] = index.try_emplace(print(row), index.size());
      if (!added)
        row = JsonValue(it->second);
    }
  }

  std::unordered_map<std::string, std::unordered_map<std::string, size_t>>
      indices_;
};

// Consume each source array before the ordinary kernel/task serde visits it.
// The result and the short-lived dictionary contain typed rows, so repeated
// types aren't reparsed and the reader never retains a package-sized JSON tree.
template <typename T>
std::vector<T> read_rows(JsonValue &rows, std::vector<T> &dictionary) {
  require(rows.is_arr(), "Invalid GFX metadata row array");
  std::vector<T> result;
  result.reserve(rows.size());
  for (const auto &row : rows.elems()) {
    if (row.is_obj()) {
      T value{};
      deserialize(row, value, true);
      dictionary.push_back(value);
      result.emplace_back(std::move(value));
    } else {
      require(row.ty == L_JSON_INT && row.num_int >= 0 &&
                  uint64_t(row.num_int) < dictionary.size(),
              "Invalid GFX metadata row reference");
      result.push_back(dictionary[size_t(row.num_int)]);
    }
  }
  rows.arr.inner.clear();
  return result;
}
}  // namespace

class AotMetadataRowReader {
 public:
  void read(JsonValue &kernel, Kernel &out) {
    auto &marker = kernel["row_storage"];
    require(marker.ty == L_JSON_INT && marker.num_int == 1,
            "Unsupported GFX kernel row storage");
    auto &ctx = kernel["ctx_attribs"];
    auto args = read_rows(ctx["arg_attribs_vec_"], args_);
    auto rets = read_rows(ctx["ret_attribs_vec_"], rets_);
    auto access = read_rows(ctx["arr_access"], access_);
    auto grad = read_rows(ctx["grad_arr_access"], grad_access_);
    const auto *arg_type = read_type(ctx["args_type_"]);
    const auto *ret_type = read_type(ctx["rets_type_"]);
    auto tasks = std::move(kernel["tasks_attribs"]);
    kernel["tasks_attribs"] = JsonValue(JsonArray{});
    require(tasks.is_arr(), "Invalid GFX kernel task array");
    deserialize(kernel, out);
    auto &decoded = out.ctx_attribs;
    decoded.arg_attribs_vec_ = std::move(args);
    decoded.ret_attribs_vec_ = std::move(rets);
    decoded.arr_access = std::move(access);
    decoded.grad_arr_access = std::move(grad);
    decoded.args_type_ = arg_type;
    decoded.rets_type_ = ret_type;
    out.tasks_attribs.reserve(tasks.size());
    for (auto &task : tasks.arr.inner) {
      auto buffers = read_rows(task["buffer_binds"], buffers_);
      auto textures = read_rows(task["texture_binds"], textures_);
      auto acceleration =
          read_rows(task["acceleration_structure_binds"], acceleration_);
      auto &decoded_task = out.tasks_attribs.emplace_back();
      deserialize(task, decoded_task);
      decoded_task.buffer_binds = std::move(buffers);
      decoded_task.texture_binds = std::move(textures);
      decoded_task.acceleration_structure_binds = std::move(acceleration);
    }
  }

 private:
  const StructType *read_type(JsonValue &type) {
    if (type.is_null())
      return nullptr;
    const auto &kind = type["type_kind"];
    require(kind.ty == L_JSON_INT && kind.num_int == int(TypeKind::Struct),
            "Invalid GFX kernel argument struct type");
    auto &content = type["content"];
    auto elements = read_rows(content["elements_"], elements_);
    const auto *result =
        TypeFactory::get_instance()
            .get_struct_type(
                elements, static_cast<const std::string &>(content["layout_"]))
            ->as<StructType>();
    type = JsonValue(nullptr);
    return result;
  }
  decltype(Context::arg_attribs_vec_) args_;
  decltype(Context::ret_attribs_vec_) rets_;
  decltype(Context::arr_access) access_, grad_access_;
  std::vector<AbstractDictionaryMember> elements_;
  decltype(Task::buffer_binds) buffers_;
  decltype(Task::texture_binds) textures_;
  decltype(Task::acceleration_structure_binds) acceleration_;
};

void write_kernel_row_metadata(std::ostream &out, TaichiAotData &&artifact) {
  auto kernels = std::move(artifact.kernels);
  auto header = serialize(artifact);
  header.obj.inner.erase("kernels");
  header.obj.inner.emplace("source_metadata_version",
                           JsonValue(artifact.metadata_version));
  header["metadata_version"] = JsonValue(kRowStorageVersion);
  auto prefix = print(header);
  prefix.pop_back();  // Continue this object with the streamed kernel array.
  out << prefix << ",\"kernels\":[";
  RowEncoder rows;
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto value = serialize(kernels[i]);
    rows.encode(value);
    if (i)
      out << ',';
    out << print(value);
  }
  out << "]}";
}

void read_kernel_metadata(const char *begin,
                          const char *end,
                          TaichiAotData &out,
                          bool allow_row_storage) {
  std::vector<Kernel> kernels;
  AotMetadataRowReader rows;
  size_t packed_count = 0;
  const StreamedArrayField field{"kernels",
                                 [&] {
                                   kernels.clear();
                                   rows = {};
                                   packed_count = 0;
                                 },
                                 [&](JsonValue &value) {
                                   auto &kernel = kernels.emplace_back();
                                   if (value.obj.inner.count("row_storage")) {
                                     require(allow_row_storage,
                                             "Row storage requires Vulkan");
                                     rows.read(value, kernel);
                                     ++packed_count;
                                   } else {
                                     deserialize(value, kernel);
                                   }
                                 }};
  auto json = parse_complete(begin, end, field);
  const auto source = json.obj.inner.find("source_metadata_version");
  if (source != json.obj.inner.end() || packed_count ||
      (json.obj.inner.count("metadata_version") &&
       json["metadata_version"].num_int == kRowStorageVersion)) {
    require(allow_row_storage && source != json.obj.inner.end() &&
                source->second.ty == L_JSON_INT &&
                source->second.num_int >= 17 &&
                source->second.num_int <= TaichiAotData::kMetadataVersion &&
                json["metadata_version"].ty == L_JSON_INT &&
                json["metadata_version"].num_int == kRowStorageVersion &&
                json["kernels"].is_arr() && packed_count == kernels.size(),
            "Invalid GFX kernel row envelope");
    json["metadata_version"] = source->second;
  }
  deserialize(json, out);
  out.kernels = std::move(kernels);
  for (const auto &[name, value] : json.fields()) {
    if (name == "device_range_dispatch")
      aot::decode_device_range_dispatch(value, out.device_range_dispatch);
  }
}
}  // namespace taichi::lang::gfx
