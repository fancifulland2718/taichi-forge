#include "taichi/runtime/gfx/aot_snode_resources.h"

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace taichi::lang::gfx {
namespace {

using liong::json::JsonArray;
using liong::json::JsonObject;
using liong::json::JsonValue;

template <typename T>
T read_integer(const JsonValue &value) {
  if (value.ty != liong::json::L_JSON_INT)
    throw std::invalid_argument(
        "SNode resource integer has a noninteger value");
  const int64_t number = value.num_int;
  if constexpr (std::is_unsigned_v<T>) {
    if (number < 0 || uint64_t(number) > std::numeric_limits<T>::max())
      throw std::invalid_argument("SNode resource integer is out of range");
  } else {
    if (number < std::numeric_limits<T>::min() ||
        number > std::numeric_limits<T>::max())
      throw std::invalid_argument("SNode resource integer is out of range");
  }
  return static_cast<T>(number);
}

struct Reader {
  const JsonObject &object;
  template <typename T>
  void operator()(const char *name, T &value) const {
    const auto &json = object.inner.at(name);
    if constexpr (std::is_enum_v<T>) {
      value = static_cast<T>(read_integer<std::underlying_type_t<T>>(json));
    } else if constexpr (std::is_same_v<T, bool>) {
      value = static_cast<bool>(json);
    } else if constexpr (std::is_integral_v<T>) {
      value = read_integer<T>(json);
    } else {
      static_assert(std::is_floating_point_v<T>);
      if (json.ty == liong::json::L_JSON_INT)
        value = static_cast<T>(json.num_int);
      else if (json.ty == liong::json::L_JSON_FLOAT)
        value = static_cast<T>(json.num_float);
      else
        throw std::invalid_argument("SNode resource fraction is not numeric");
    }
  }
};

struct Writer {
  JsonObject object;
  template <typename T>
  void operator()(const char *name, const T &value) {
    if constexpr (std::is_enum_v<T>) {
      (*this)(name, static_cast<std::underlying_type_t<T>>(value));
    } else {
      if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        if (uint64_t(value) > uint64_t(std::numeric_limits<int64_t>::max()))
          throw std::invalid_argument(
              "SNode resource integer exceeds the JSON representation");
      }
      object.inner.emplace(name, JsonValue(value));
    }
  }
};

// Keep the allocator codec in this translation unit, without pulling JSON
// serializers into the SPIR-V codegen's shared POD header.
template <typename Io, typename Contract>
void pointer_fields(Io &io, Contract &c) {
  io("watermark_offset", c.watermark_offset);
  io("pool_data_offset", c.pool_data_offset);
  io("pool_capacity", c.pool_capacity);
  io("cell_stride_bytes", c.cell_stride_bytes);
  io("has_freelist", c.has_freelist);
  io("freelist_head_offset", c.freelist_head_offset);
  io("freelist_links_offset", c.freelist_links_offset);
  io("has_ambient_zone", c.has_ambient_zone);
  io("ambient_offset", c.ambient_offset);
  io("snode_id", c.snode_id);
  io("alloc_protocol", c.alloc_protocol);
  io("pool_fraction", c.pool_fraction);
  io("pool_buffer_binding_id", c.pool_buffer_binding_id);
  io("allocator_kind", c.allocator_kind);
  io("chunk_log2_capacity", c.chunk_log2_capacity);
  io("max_chunks", c.max_chunks);
  io("chunk_descriptor_array_first_binding",
     c.chunk_descriptor_array_first_binding);
  io("deterministic_slot", c.deterministic_slot);
}

template <typename Io, typename Watch>
void watch_fields(Io &io, Watch &watch) {
  io("snode_id", watch.snode_id);
  io("overflow_offset", watch.overflow_offset);
  io("active_count_offset", watch.active_count_offset);
}

const std::vector<JsonValue> &array(const JsonObject &object,
                                    const char *name) {
  const auto &value = object.inner.at(name);
  if (!value.is_arr())
    throw std::invalid_argument("SNode resource sequence is not an array");
  return value.arr.inner;
}

}  // namespace

void AotSNodeTreeResources::validate() const {
  if (producer_tree_id < 0 || resources.snode_ids.empty())
    throw std::invalid_argument("AOT SNode resource identity is missing");
  if (storage_kind != StorageKind::PlainDense &&
      storage_kind != StorageKind::PackedDense &&
      storage_kind != StorageKind::Sparse)
    throw std::invalid_argument("Unknown AOT SNode storage kind");
  if (storage_kind != StorageKind::Sparse &&
      (resources.listgen_capacity_entries != 0 ||
       !resources.pointer_pools.empty() || !resources.hash_watches.empty()))
    throw std::invalid_argument("Dense AOT storage contains sparse resources");
  validate_snode_tree_resources(resources);
}

JsonValue AotSNodeTreeResources::json_serialize_fields() const {
  validate();
  Writer writer;
  writer("producer_tree_id", producer_tree_id);
  writer("storage_kind", storage_kind);
  writer("root_size", resources.root_size);
  writer("listgen_capacity_entries", resources.listgen_capacity_entries);
  JsonArray ids, pools, watches;
  for (int id : resources.snode_ids)
    ids.inner.emplace_back(id);
  for (const auto &pool : resources.pointer_pools) {
    Writer entry, contract;
    pointer_fields(contract, pool.contract);
    entry("independent_size", pool.independent_size);
    entry.object.inner.emplace("contract",
                               JsonValue(std::move(contract.object)));
    pools.inner.emplace_back(std::move(entry.object));
  }
  for (const auto &watch : resources.hash_watches) {
    Writer entry;
    watch_fields(entry, watch);
    if (watch.tombstone_offset == static_cast<std::size_t>(-1))
      entry("tombstone_offset", int64_t(-1));
    else
      entry("tombstone_offset", watch.tombstone_offset);
    watches.inner.emplace_back(std::move(entry.object));
  }
  writer.object.inner.emplace("snode_ids", JsonValue(std::move(ids)));
  writer.object.inner.emplace("pointer_pools", JsonValue(std::move(pools)));
  writer.object.inner.emplace("hash_watches", JsonValue(std::move(watches)));
  return JsonValue(std::move(writer.object));
}

void AotSNodeTreeResources::json_deserialize_fields(const JsonObject &object,
                                                    bool strict) {
  // Every v6 envelope member is required, including under the legacy loader's
  // permissive outer parser. Do not silently synthesize a missing pool policy.
  AotSNodeTreeResources decoded;
  Reader reader{object};
  reader("producer_tree_id", decoded.producer_tree_id);
  reader("storage_kind", decoded.storage_kind);
  reader("root_size", decoded.resources.root_size);
  reader("listgen_capacity_entries",
         decoded.resources.listgen_capacity_entries);
  for (const auto &value : array(object, "snode_ids"))
    decoded.resources.snode_ids.push_back(read_integer<int>(value));
  for (const auto &value : array(object, "pointer_pools")) {
    SNodeTreeResources::PointerPool pool;
    Reader entry{static_cast<const JsonObject &>(value)};
    Reader contract{static_cast<const JsonObject &>(value["contract"])};
    pointer_fields(contract, pool.contract);
    entry("independent_size", pool.independent_size);
    decoded.resources.pointer_pools.push_back(pool);
  }
  for (const auto &value : array(object, "hash_watches")) {
    SNodeTreeResources::HashWatch watch;
    Reader entry{static_cast<const JsonObject &>(value)};
    watch_fields(entry, watch);
    const auto &offset = value["tombstone_offset"];
    watch.tombstone_offset =
        offset.ty == liong::json::L_JSON_INT && offset.num_int == -1
            ? static_cast<std::size_t>(-1)
            : read_integer<std::size_t>(offset);
    decoded.resources.hash_watches.push_back(watch);
  }
  decoded.validate();
  *this = std::move(decoded);
}

}  // namespace taichi::lang::gfx
