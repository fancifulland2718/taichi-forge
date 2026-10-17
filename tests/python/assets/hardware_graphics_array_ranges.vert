#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(location = 0) in vec2 in_position;
layout(location = 0) out vec3 out_color;
layout(set = 0, binding = 0, std430) readonly buffer Color {
  float values[];
} colors[4];
layout(set = 0, binding = 1, std140) uniform Parameters {
  uint selector;
} parameters;
void main() {
  uint selected = nonuniformEXT(parameters.selector);
  gl_Position = vec4(in_position, 0.0, 1.0);
  out_color = vec3(colors[selected].values[0], colors[selected].values[1],
                   colors[selected].values[2] + float(colors[selected].values.length()) / 8.0);
}
