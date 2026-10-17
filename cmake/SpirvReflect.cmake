# Patch only the private parser of the pinned SPIRV-Reflect dependency. Keep the
# submodule, public structs and SPIR-V bytes unchanged. The generated source is
# build-local, so parallel worktrees never mutate a shared dependency checkout.
function(_ti_reflect_replace old new)
  string(FIND "${_ti_reflect_source}" "${old}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "SPIRV-Reflect source changed; review its private ID index patch")
  endif()
  string(REPLACE "${old}" "${new}" _ti_reflect_source "${_ti_reflect_source}")
  set(_ti_reflect_source "${_ti_reflect_source}" PARENT_SCOPE)
endfunction()

function(ti_prepare_spirv_reflect output)
  set(_input "${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect/spirv_reflect.c")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_input}")
  file(READ "${_input}" _ti_reflect_source)
  _ti_reflect_replace(
    "  SpvReflectPrvNode*              nodes;"
    "  SpvReflectPrvNode*              nodes;
  uint32_t*                       node_id_slots;
  size_t                          node_id_mask;")
  _ti_reflect_replace([=[static SpvReflectPrvNode* FindNode(
  SpvReflectPrvParser* p_parser,
  uint32_t             result_id)
{
  SpvReflectPrvNode* p_node = NULL;
  for (size_t i = 0; i < p_parser->node_count; ++i) {
    SpvReflectPrvNode* p_elem = &(p_parser->nodes[i]);
    if (p_elem->result_id == result_id) {
      p_node = p_elem;
      break;
    }
  }
  return p_node;
}]=] [=[static SpvReflectPrvNode* FindNode(
  SpvReflectPrvParser* p_parser,
  uint32_t result_id)
{
  // Zero is not a valid SPIR-V ID. Preserve the old parser's sentinel lookup.
  if (result_id == 0) {
    for (size_t i = 0; i < p_parser->node_count; ++i) {
      if (p_parser->nodes[i].result_id == 0) return &p_parser->nodes[i];
    }
    return NULL;
  }
  size_t slot = (result_id * 2654435761u) & p_parser->node_id_mask;
  while (p_parser->node_id_slots[slot]) {
    SpvReflectPrvNode* node = &p_parser->nodes[p_parser->node_id_slots[slot] - 1];
    if (node->result_id == result_id) return node;
    slot = (slot + 1) & p_parser->node_id_mask;
  }
  return NULL;
}]=])
  _ti_reflect_replace("    SafeFree(p_parser->nodes);"
    "    SafeFree(p_parser->node_id_slots);
    SafeFree(p_parser->nodes);")
  _ti_reflect_replace("  // Mark all nodes with an invalid state" [=[  // Size storage by actual instruction count, never the untrusted ID bound.
  // At most one nonzero ID is inserted per node; keep the table below 1/2 full.
  if (p_parser->node_count > SIZE_MAX / (4 * sizeof(uint32_t))) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }
  size_t slot_count = 1;
  while (slot_count < p_parser->node_count * 2) slot_count *= 2;
  p_parser->node_id_slots = (uint32_t*)calloc(slot_count, sizeof(uint32_t));
  if (IsNull(p_parser->node_id_slots)) {
    return SPV_REFLECT_RESULT_ERROR_ALLOC_FAILED;
  }
  p_parser->node_id_mask = slot_count - 1;
  // Mark all nodes with an invalid state]=])
  _ti_reflect_replace("    ++node_index;" [=[    // Publish each parsed ID once. During ParseNodes, future nodes have not
    // received IDs yet; indexing them early would change access-chain parsing.
    if (p_node->result_id != 0) {
      size_t slot = (p_node->result_id * 2654435761u) & p_parser->node_id_mask;
      while (p_parser->node_id_slots[slot] &&
             p_parser->nodes[p_parser->node_id_slots[slot] - 1].result_id != p_node->result_id) {
        slot = (slot + 1) & p_parser->node_id_mask;
      }
      // Preserve first-definition lookup even on malformed duplicate IDs.
      if (!p_parser->node_id_slots[slot]) p_parser->node_id_slots[slot] = node_index + 1;
    }
    ++node_index;]=])
  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/spirv_reflect_indexed.c")
  file(WRITE "${_generated}.in" "${_ti_reflect_source}")
  configure_file("${_generated}.in" "${_generated}" COPYONLY)
  set(${output} "${_generated}" PARENT_SCOPE)
endfunction()
