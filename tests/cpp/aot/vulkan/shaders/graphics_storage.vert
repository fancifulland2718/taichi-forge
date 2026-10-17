#version 450
layout(set = 0, binding = 0, std430) readonly buffer Color {
  vec4 value;
} colors[4];
void main() {
  gl_Position = colors[0].value;
}
