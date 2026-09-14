#version 460
#extension GL_EXT_ray_query : require

// glslc --target-env=vulkan1.2 -O hardware_graphics_ray_query.frag
//       -o hardware_graphics_ray_query.frag.spv
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(location = 0) in vec3 receiver_position;
layout(location = 0) out vec4 result;

void main() {
  rayQueryEXT query;
  rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT, 255,
                        receiver_position, 0.001,
                        vec3(0.0, 0.0, 1.0), 1.0);
  while (rayQueryProceedEXT(query)) {}
  bool hit = rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
  result = hit ? vec4(1.0, float(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)),
                       float(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)), 1.0)
               : vec4(0.0, -1.0, -1.0, 1.0);
}
