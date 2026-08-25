#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 in_position;
layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0, std430) readonly buffer Color {
  vec4 value;
} colors[4];

layout(set = 0, binding = 1, std140) uniform Parameters {
  uint selector;
} parameters;

void main() {
  gl_Position = vec4(in_position, 0.0, 1.0);
  out_color = colors[nonuniformEXT(parameters.selector)].value.rgb;
}
