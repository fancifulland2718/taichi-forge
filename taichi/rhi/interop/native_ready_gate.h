#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace taichi::lang {

// Cold adapter for an exact native C gate structure. Its callbacks use status
// zero for success and a uint32_t 0/1 readiness result. The caller defines the
// epoch's backend-specific meaning; this helper never performs GPU work.
// Validation is separate so every other import argument can also be checked
// before the first native retain. No C API or Python headers are required.
template <typename Gate>
void validate_native_ready_gate(const Gate &gate) {
  static_assert(std::is_standard_layout_v<Gate> &&
                std::is_trivially_copyable_v<Gate>);
  if (gate.struct_size != sizeof(Gate) || gate.schema_version != 1)
    throw std::invalid_argument("Invalid native ready gate size or version");
  if (!gate.owner || !gate.retain_owner || !gate.release_owner || !gate.query)
    throw std::invalid_argument("Incomplete native ready gate callbacks");
}

template <typename Gate>
std::function<bool(std::uint64_t)> retain_native_ready_gate(
    const Gate &descriptor) {
  validate_native_ready_gate(descriptor);
  const auto gate = descriptor;
  if (gate.retain_owner(gate.owner) != 0)
    throw std::invalid_argument("Native ready gate retain failed");
  // shared_ptr invokes this deleter even if allocating its control block
  // throws. No successful native retain is left without an owning guard.
  auto owner = std::shared_ptr<void>(gate.owner, gate.release_owner);
  return [owner = std::move(owner), query = gate.query](std::uint64_t epoch) {
    if (!epoch)
      throw std::invalid_argument("Native ready gate epochs start at one");
    std::uint32_t result = 0;
    const auto error = query(owner.get(), epoch, &result);
    if (error != 0 || result > 1)
      throw std::runtime_error("Native ready gate query failed: " +
                               std::to_string(error));
    return result == 1;
  };
}

}  // namespace taichi::lang
