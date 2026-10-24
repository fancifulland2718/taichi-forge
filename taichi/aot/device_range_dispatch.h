#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "taichi/common/serialization.h"

namespace taichi::lang::aot {

// Entry-specific execution policy; never a code identity or a live device handle.
struct DeviceRangeDispatch {
  std::string name;
  std::uint32_t version{0};
  std::vector<std::uint32_t> tasks;
  TI_IO_DEF(name, version, tasks);
};

inline void decode_device_range_dispatch(
    const liong::json::JsonValue &value,
    std::vector<DeviceRangeDispatch> &entries) {
  using namespace liong::json;
  const auto check_index = [](const JsonValue &index) {
    if (index.ty != L_JSON_INT || index.num_int < 0 ||
        index.num_int > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("Device range policy requires a uint32 integer");
  };
  if (!value.is_arr())
    throw std::invalid_argument("Device range policies must be an array");
  for (const auto &entry : value.arr.inner) {
    if (!entry.is_obj())
      throw std::invalid_argument("Device range policy must be an object");
    const auto &fields = entry.obj.inner;
    // Inspect the wire values before arithmetic deserialization can narrow
    // them. A wrapped index must never select a different, valid task.
    check_index(fields.at("version"));
    if (!fields.at("name").is_str() || !fields.at("tasks").is_arr())
      throw std::invalid_argument("Invalid device range policy name or tasks");
    for (const auto &task : fields.at("tasks").arr.inner)
      check_index(task);
  }
  deserialize(value, entries, true);
}

}  // namespace taichi::lang::aot
