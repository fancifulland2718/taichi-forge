#version 460

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_color;

layout(location = 0) out vec3 color;

void main() {
  gl_Position = vec4(v_position, 1.0);
  color = v_color;
}
