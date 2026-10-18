#pragma once

#include "taichi/common/json.h"
#include "taichi/runtime/gfx/snode_tree_resources.h"

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

  void validate() const;
  liong::json::JsonValue json_serialize_fields() const;
  void json_deserialize_fields(const liong::json::JsonObject &object,
                               bool strict);
};

}  // namespace taichi::lang::gfx
