// Include after the local_size declaration so gl_WorkGroupSize is finalized.
// Shared with the 14-word params array in vulkan_primitives.cpp. Descriptor ranges
// include alignment prefixes; logical extents and strides are explicit.
layout(push_constant) uniform BucketParams {
  uint n;
  uint bins;
  uint chunks;
  uint item_words;
  uint keys_offset;
  uint keys_stride;
  uint values_offset;
  uint values_stride;
  uint output_offset;
  uint output_stride;
  uint offsets_offset;
  uint offsets_stride;
  uint cursor_offset;
  uint write_cursor;
} params;

// Dispatch rows cover larger workloads without exceeding Vulkan's portable
// 65535-workgroup X dimension. This changes launch geometry, not capacity.
uint bucket_group_index() {
  return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

uint bucket_item_index() {
  return bucket_group_index() * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
}
