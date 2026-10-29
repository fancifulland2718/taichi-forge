#pragma once

#include "taichi/common/json.h"
#include "taichi/runtime/gfx/snode_tree_resources.h"

namespace taichi::lang {
class SNode;
}

namespace taichi::lang::gfx {

// Artifact-only envelope. Live JIT layout/allocator types remain independent
// of JSON and keep using the same validated resource projection.
struct AotSNodeTreeResources {
  enum class StorageKind : uint32_t {
    PlainDense = 0,
    PackedDense = 1,
    Sparse = 2,
  };

  // Shader overflow records contain the producer ID. Consumers map it through
  // this envelope instead of interpreting it as a runtime allocation slot.
  int producer_tree_id{-1};
  StorageKind storage_kind{StorageKind::PlainDense};
  SNodeTreeResources resources;

  static StorageKind classify_storage(const SNode *root);
  static AotSNodeTreeResources from_compiled(
      int tree_id, const spirv::CompiledSNodeStructs &compiled);
  // Query projection only: normalize binding identities, never offsets,
  // allocator policy, capacity or buffer sizes.
  std::string normalized_layout_json() const;
  void validate() const;
  liong::json::JsonValue json_serialize_fields() const;
  void json_deserialize_fields(const liong::json::JsonObject &object,
                               bool strict);
};

}  // namespace taichi::lang::gfx
