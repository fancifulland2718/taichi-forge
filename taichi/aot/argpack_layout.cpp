#include "taichi/aot/argpack_layout.h"
#include "taichi/aot/argument_origins.h"
#include "taichi/aot/value_parameters.h"
#include "taichi/program/callable.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace taichi::lang::aot {
namespace {
void require(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(message);
}
bool below(const std::vector<int> &path, const std::vector<int> &root) {
  return path.size() >= root.size() &&
         std::equal(root.begin(), root.end(), path.begin());
}
const StructType *pack_storage(const CallableBase &kernel,
                               const std::vector<int> &physical) {
  const auto found = kernel.argpack_types.find(physical);
  if (found != kernel.argpack_types.end())
    return found->second;
  const auto outer = compiled_value_parameter_type(kernel, physical)
                         ->as<StructType>();
  return DataType(outer->elements()[0].type).ptr_removed()->as<StructType>();
}
}  // namespace

ArgPackLayout::ArgPackLayout(const CallableBase &kernel,
                             const std::vector<int> &logical_root,
                             ValueStorage storage,
                             const Resolver &resolve)
    : value_storage_(storage) {
  require(kernel.argument_origins.has_value() && bool(resolve),
          "ArgPack storage requires member origins and a layout resolver");
  validate_argument_origins(kernel, *kernel.argument_origins);
  const auto origins = canonical_argument_origins(*kernel.argument_origins);
  const auto root = std::find_if(origins.begin(), origins.end(), [&](const auto &o) {
    return o.logical == logical_root;
  });
  require(root != origins.end() && root->kind == "argpack",
          "ArgPack storage root is not a public pack");
  std::map<std::vector<int>, std::size_t> packs;
  std::map<const StructType *, std::pair<const StructType *, std::size_t>> layouts;
  for (const auto &origin : origins) {
    if (!below(origin.logical, logical_root))
      continue;
    if (origin.kind != "argpack" && origin.kind != "value") {
      resources_.push_back(origin);  // Bound through the existing root ABI.
      continue;
    }
    if (origin.kind == "argpack") {
      const auto type = pack_storage(kernel, origin.physical);
      auto found = layouts.find(type);
      if (found == layouts.end())
        found = layouts.emplace(type, resolve(type)).first;
      const auto [resolved, bytes] = found->second;
      require(resolved && resolved->get_layout() != "none" &&
                  resolved->elements().size() == type->elements().size(),
              "ArgPack resolver did not supply the compiled storage layout");
      packs.emplace(origin.logical, storages_.size());
      storages_.push_back({origin, resolved, bytes, {}, {}});
    }
  }
  // Match each direct value/pack member to its compiled slot once. Resources
  // were lifted to roots and never consume an inline storage slot.
  std::vector<std::vector<bool>> used;
  for (const auto &pack : storages_)
    used.emplace_back(pack.type->elements().size(), false);
  for (const auto &origin : origins) {
    if (!below(origin.logical, logical_root) || origin.logical == logical_root ||
        (origin.kind != "argpack" && origin.kind != "value"))
      continue;
    auto parent = origin.logical;
    parent.pop_back();
    const auto index = packs.at(parent);
    auto &pack = storages_[index];
    const auto member = std::size_t(origin.physical.back());
    require(member < pack.type->elements().size() && !used[index][member],
            "ArgPack member does not have a unique storage slot");
    used[index][member] = true;
    const auto &elements = pack.type->elements();
    const auto offset = elements[member].offset;
    const auto end = member + 1 < elements.size() ? elements[member + 1].offset
                                                 : pack.bytes;
    require(offset <= end && end <= pack.bytes,
            "ArgPack member exceeds its compiled storage");
    if (origin.kind == "argpack") {
      const auto *outer = elements[member].type->cast<StructType>();
      require(outer && outer->elements().size() == 1 &&
                  outer->elements()[0].type->is<PointerType>() &&
                  outer->get_layout() != "none",
              "ArgPack child requires a resolved pointer slot");
      const auto pointer = outer->elements()[0].offset;
      // GFX binds children as separate uniforms; its pointer can be a u32
      // placeholder. Host/CUDA packets contain the actual 64-bit address.
      const auto width = storage == ValueStorage::Host ? sizeof(std::uint64_t)
                                                       : sizeof(std::uint32_t);
      require(pointer <= end - offset && width <= end - offset - pointer,
              "ArgPack child pointer exceeds its compiled slot");
      pack.children.push_back({packs.at(origin.logical), offset + pointer});
    } else {
      DataType logical = elements[member].type;
      if (kernel.value_parameter_types) {
        const auto found = kernel.value_parameter_types->find(origin.physical);
        if (found != kernel.value_parameter_types->end())
          logical = found->second;
      }
      pack.values.push_back({origin, offset,
                             ValueLayout(logical, elements[member].type,
                                         end - offset, storage)});
    }
  }
  for (const auto &members : used)
    require(std::find(members.begin(), members.end(), false) == members.end(),
            "ArgPack compiled storage contains an unmapped member");
}

}  // namespace taichi::lang::aot
