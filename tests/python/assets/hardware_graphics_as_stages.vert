#version 460
#extension GL_EXT_ray_query : require
// glslc --target-env=vulkan1.2 -O hardware_graphics_as_stages.vert -o hardware_graphics_as_stages.vert.spv
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(location = 0) in vec2 position;
layout(location = 0) out vec3 receiver_position;
layout(location = 1) flat out float vertex_custom;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
  receiver_position = vec3(position, 0.0);
  rayQueryEXT query;
  rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT, 255,
                       vec3(0.0), 0.001, vec3(0.0, 0.0, 1.0), 1.0);
  while (rayQueryProceedEXT(query)) {}
  bool hit = rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
  vertex_custom = hit ? float(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)) : -1.0;
}
