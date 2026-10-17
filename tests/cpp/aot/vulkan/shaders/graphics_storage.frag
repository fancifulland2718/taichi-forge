#version 450
#ifndef COLOR_BINDING
#define COLOR_BINDING 1
#endif
#ifndef COLOR_COUNT
#define COLOR_COUNT 4
#endif
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = COLOR_BINDING, std430) readonly buffer Color {
  vec4 value;
} colors[COLOR_COUNT];
void main() {
  out_color = colors[0].value;
}
