#version 450
layout(location = 0) in vec2 position;
layout(set = 0, binding = 0, std140) uniform Parameters { uint base; } params;
layout(location = 0) flat out float value;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
  value = float(params.base);
}
