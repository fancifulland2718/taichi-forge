#version 450
layout(set = 0, binding = 0, std430) buffer State { uint value; } state;
layout(location = 0) out vec3 out_color;
void main() {
  state.value = uint(gl_VertexIndex);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  out_color = vec3(1.0);
}
